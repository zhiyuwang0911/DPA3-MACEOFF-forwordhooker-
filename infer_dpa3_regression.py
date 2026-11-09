#!/usr/bin/env python3
"""Export continuous predicted errors; references are optional at inference."""
import argparse
import csv
import json
from pathlib import Path
import torch
from probe.regression import ErrorRegressor, error_targets, regression_metrics
from probe.pca import restore_model
from probe.backends.dpa3_regression import (file_digest, load_backbone, read_frames,
    make_loader, RegressionDPA3Processor)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', '--dpa3-model', dest='dpa3_model', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-xyz', required=True)
    p.add_argument('--output-dir', default='probe_regression_inference')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cache-dir', help='Persist raw DPA3 outputs for this model/geometry')
    p.add_argument('--no-cache-dpa3', action='store_true')
    args = p.parse_args()
    if args.batch_size <= 0:
        p.error('--batch-size must be positive')
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if ckpt.get('task') != 'dpa3_error_regression':
        raise ValueError('Expected a dpa3_error_regression checkpoint, not classifier or MACE weights')
    model = restore_model(ErrorRegressor, ckpt, args.device)
    model.eval()
    digest = file_digest(args.dpa3_model)
    if ckpt['provenance']['backbone_sha256'] != digest:
        raise ValueError('Inference requires the same DPA3 model as training')
    extractor = load_backbone(args.dpa3_model, args.device)
    processor = RegressionDPA3Processor(extractor, digest, args.cache_dir,
                                       cache=not args.no_cache_dpa3)
    if processor.identity != ckpt['provenance']['representation']:
        raise ValueError('DPA3 descriptor representation differs from checkpoint')
    frames = read_frames(args.test_xyz)
    loader = make_loader(frames, args.batch_size)
    structures, atoms = [], []
    store = {k: [[], []] for k in ('energy', 'force_atom', 'force_structure')}
    offset = 0
    for batch in loader:
        feats, mask, pe, te, pf, tf, _ = processor(batch, args.device)
        if feats.shape[-1] != ckpt['original_dim']:
            raise ValueError('DPA3 embedding width differs from checkpoint')
        valid = torch.isfinite(pe) & mask.any(1)
        valid &= torch.isfinite(feats).all(-1).masked_fill(~mask, True).all(1)
        valid &= torch.isfinite(pf).all(-1).masked_fill(~mask, True).all(1)
        if not valid.all():
            raise ValueError(f'Nonfinite backbone output or empty structure in batch starting at {offset}')
        with torch.no_grad():
            predictions = model(feats.float(), mask, pe.float())
        targets = error_targets(pe, te, pf, tf, mask)
        if not all(torch.isfinite(x).all() for x in predictions):
            raise ValueError('Nonfinite regressor output')
        def number(x):
            return x.item() if torch.isfinite(x) else None
        for b in range(len(pe)):
            structures.append(dict(structure_index=offset + b,
                predicted_energy_error_eV=number(predictions[0][b]),
                reference_energy_error_eV=number(targets[0][b]),
                predicted_mean_force_error_eV_A=number(predictions[2][b]),
                reference_mean_force_error_eV_A=number(targets[2][b])))
            for a in mask[b].nonzero().flatten().tolist():
                atoms.append(dict(structure_index=offset + b, atom_index=a,
                    predicted_force_error_eV_A=number(predictions[1][b, a]),
                    reference_force_error_eV_A=number(targets[1][b, a])))
        for i, key in enumerate(store):
            pred, target = predictions[i].detach().cpu(), targets[i].detach().cpu()
            if i == 1:
                pred, target = pred[mask.cpu()], target[mask.cpu()]
            store[key][0].append(pred)
            store[key][1].append(target)
        offset += len(pe)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not structures:
        raise ValueError('No structures available for inference')
    for name, rows in [('structure', structures), ('atom', atoms)]:
        with (out / f'predictions_{name}_regression.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metrics = {key: regression_metrics(torch.cat(pred), torch.cat(target))
               for key, (pred, target) in store.items()}
    (out / 'metrics_regression.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
