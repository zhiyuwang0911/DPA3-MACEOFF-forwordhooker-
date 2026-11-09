#!/usr/bin/env python3
"""Train continuous energy/force error regression on frozen DPA3 embeddings."""
import argparse
import json
from pathlib import Path
import torch
from probe.regression import ErrorRegressor, run_epoch
from probe.pca import ModelWithPCA, make_pca, restore_model
from probe.backends.dpa3_regression import (file_digest, load_backbone, read_frames,
    split_frames, make_loader, RegressionDPA3Processor)
from probe.train import atomic_torch_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', '--dpa3-model', dest='dpa3_model', required=True)
    p.add_argument('--train-xyz', required=True)
    p.add_argument('--output-dir', default='probe_regression_outputs')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--valid-fraction', type=float, default=.1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cache-dir', help='Persist raw DPA3 outputs between jobs')
    p.add_argument('--no-cache-dpa3', action='store_true', help='Recompute DPA3 instead of caching')
    p.add_argument('--loss-weights', type=float, nargs=3, default=[1., 1., 1.],
                   metavar=('ENERGY', 'FORCE_ATOM', 'FORCE_STRUCTURE'))
    p.add_argument('--pca-dim', type=int, help='Reduced atomic embedding dimension (e.g. 8, 32, 64)')
    p.add_argument('--pca-model', help='Prefit torch or sklearn PCA artifact; requires --pca-dim')
    p.add_argument('--pca-max-atoms', type=int, help='Maximum training atoms used to fit PCA')
    p.add_argument('--resume', help='Path to regression last_checkpoint.pt')
    p.add_argument('--atom-encoder-hidden', type=int, nargs='+', default=[256, 128])
    p.add_argument('--atom-encoder-output-dim', type=int, default=256)
    p.add_argument('--mol-attention-heads', type=int, default=32)
    p.add_argument('--classifier-hidden', type=int, nargs='+', default=[256, 128, 32])
    p.add_argument('--atom-force-head-hidden', type=int, nargs='+', default=[128, 32])
    p.add_argument('--dropout', type=float, default=.1)
    args = p.parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        p.error('--batch-size, --epochs and --patience must be positive')
    if not __import__('math').isfinite(args.lr) or args.lr <= 0:
        p.error('--lr must be positive and finite')
    if not 0 < args.valid_fraction < 1:
        p.error('--valid-fraction must be between zero and one')
    if any(w < 0 or not __import__('math').isfinite(w) for w in args.loss_weights) or not any(args.loss_weights):
        p.error('--loss-weights must be finite, nonnegative and include a positive weight')
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(backbone_sha256=file_digest(args.dpa3_model),
                      training_data_sha256=file_digest(args.train_xyz),
                      valid_fraction=args.valid_fraction, seed=args.seed,
                      loss_weights=args.loss_weights)
    frames = read_frames(args.train_xyz)
    train_frames, val_frames = split_frames(frames, args.valid_fraction, args.seed)
    train = make_loader(train_frames, args.batch_size, shuffle=True)
    val = make_loader(val_frames, args.batch_size)
    extractor = load_backbone(args.dpa3_model, args.device)
    processor = RegressionDPA3Processor(extractor, provenance['backbone_sha256'],
                                       args.cache_dir, cache=not args.no_cache_dpa3)
    # Discover the raw descriptor width from training data, including disk-only hits.
    processor([train_frames[0]], args.device)
    provenance['representation'] = processor.identity
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        if ckpt.get('task') != 'dpa3_error_regression':
            raise ValueError('Expected a regression checkpoint')
        if ckpt.get('provenance') != provenance:
            raise ValueError('Resume requires the same backbone, training data, split, seed and loss weights')
        if ckpt.get('original_dim', extractor.feat_dim) != extractor.feat_dim:
            raise ValueError('DPA3 feature dimension differs from checkpoint')
        if args.pca_dim is not None and args.pca_dim != ckpt.get('pca_dim'):
            raise ValueError('Resume requires unchanged PCA dimension')
        if args.pca_model or args.pca_max_atoms is not None:
            raise ValueError('Resume restores checkpoint PCA; omit --pca-model and --pca-max-atoms')
        args.pca_dim = ckpt.get('pca_dim')
        config = ckpt['model_config']
        model = restore_model(ErrorRegressor, ckpt, args.device)
        pca_info = ckpt.get('pca_info')
    else:
        projection, pca_info = make_pca(processor, train, args.device, args.pca_dim,
                                        extractor.feat_dim, args.pca_model, args.pca_max_atoms)
        config = dict(backbone_dim=args.pca_dim or extractor.feat_dim,
                      atom_encoder_hidden=args.atom_encoder_hidden,
                      atom_encoder_output_dim=args.atom_encoder_output_dim,
                      mol_attention_heads=args.mol_attention_heads,
                      classifier_hidden=args.classifier_hidden,
                      atom_force_head_hidden=args.atom_force_head_hidden,
                      dropout=args.dropout)
        model = ErrorRegressor(**config)
        if projection is not None:
            model = ModelWithPCA(model, projection)
        model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.9)
    best, stale, start, history = float('inf'), 0, 0, []
    if ckpt is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best, stale, start = ckpt['best'], ckpt['stale'], ckpt['epoch'] + 1
        history = ckpt['history']
        torch.set_rng_state(ckpt['rng_state'].cpu())
        if torch.cuda.is_available() and ckpt.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
    print(f'Raw embedding width: {extractor.feat_dim}; model input width: {config["backbone_dim"]}')
    for epoch in range(start, args.epochs):
        train_loss, _ = run_epoch(model, processor, train, args.device, args.loss_weights, optimizer)
        val_loss, metrics = run_epoch(model, processor, val, args.device, args.loss_weights)
        scheduler.step(val_loss)
        improved = val_loss < best
        best, stale = (val_loss, 0) if improved else (best, stale + 1)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'metrics': metrics})
        ckpt = dict(task='dpa3_error_regression', provenance=provenance, model_config=config, model_state_dict=model.state_dict(),
                    pca_dim=args.pca_dim, original_dim=extractor.feat_dim, pca_info=pca_info,
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch,
                    best=best, stale=stale, history=history, args=vars(args),
                    rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        for name in (['last_checkpoint.pt', 'best_regressor.pt'] if improved else ['last_checkpoint.pt']):
            atomic_torch_save(ckpt, out / name)
        (out / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False))
        print(f'Epoch {epoch + 1}: train={train_loss:.6g} val={val_loss:.6g} {metrics}', flush=True)
        print(f'DPA3 cache hits={processor.hits}, misses={processor.misses}', flush=True)
        if stale >= args.patience:
            break


if __name__ == '__main__':
    main()
