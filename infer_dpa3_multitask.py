#!/usr/bin/env python3
"""
Inference for multitask PROBE trained on DPA3 embeddings.

Also can cache DPA3 atomic embeddings for the test set in one pass
(``--cache-dir``).

Example:
  python infer_dpa3_multitask.py \\
    --model /path/to/DPA3-L6.pt \\
    --checkpoint /path/to/best_multitask_model_*.pt \\
    --test-xyz /path/to/test.xyz \\
    --output-dir ./probe_dpa3_inference \\
    --cache-dir ./dpa3_cache_test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from probe.model import MultitaskPROBEModel
from probe.backends.dpa3 import (
    load_dpa3,
    load_extxyz_dataloader,
    CachedDPA3Processor,
    process_batch_dpa3,
)
from probe.labels import (
    atom_force_component_mae,
    structure_mean_force_error,
    scalar_to_bin_index,
)
from probe.metrics import compute_all_metrics, confusion_matrix_torch


def parse_args():
    p = argparse.ArgumentParser(description="Infer multitask PROBE on DPA3")
    p.add_argument("--model", required=True, help="DPA3 backbone checkpoint")
    p.add_argument("--checkpoint", default=None, help="PROBE best_*.pt (not needed with --cache-only)")
    p.add_argument("--test-xyz", required=True)
    p.add_argument("--output-dir", default="./probe_dpa3_inference")
    p.add_argument("--cache-dir", default=None,
                   help="Optional: write/read DPA3 {idx}.pt embedding cache")
    p.add_argument("--cache-only", action="store_true",
                   help="Only run DPA3 and save embeddings (no PROBE head)")
    p.add_argument("--embeddings-only", action="store_true",
                   help="With --cache-only, store only node_feats")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--hook-module", default=None)
    p.add_argument("--max-structures", type=int, default=None)
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--atom-encoder-hidden", nargs="+", type=int, default=[256, 128])
    p.add_argument("--atom-encoder-output-dim", type=int, default=256)
    p.add_argument("--mol-attention-heads", type=int, default=32)
    p.add_argument("--classifier-hidden", nargs="+", type=int, default=[256, 128, 32])
    p.add_argument("--atom-force-head-hidden", nargs="+", type=int, default=[128, 32])
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


def _bins(ckpt, key, device):
    if key not in ckpt:
        raise KeyError(f"Checkpoint missing {key!r}")
    return torch.tensor(ckpt[key], device=device, dtype=torch.float32)


@torch.no_grad()
def run_inference(model, process_fn, loader, device, bins_e, bins_fa, bins_fm,
                  compute_metrics: bool):
    model.eval()
    struct_rows, atom_rows = [], []
    store = {
        "energy": {"logits": [], "targets": []},
        "force_mol": {"logits": [], "targets": []},
        "force_atom": {"logits": [], "targets": []},
    }
    next_struct = 0
    for batch in tqdm(loader, desc="Inference"):
        (atom_feats, atom_mask, pred_e, true_e,
         pred_f, true_f, n_atoms) = process_fn(batch, device)
        B = atom_feats.shape[0]
        logits_e, logits_fa, logits_fm = model(
            atom_feats, atom_mask, energy=pred_e
        )
        pe = F.softmax(logits_e, dim=-1)
        pfa = F.softmax(logits_fa, dim=-1)
        pfm = F.softmax(logits_fm, dim=-1)

        atom_err = atom_force_component_mae(pred_f, true_f)
        struct_err = structure_mean_force_error(atom_err, atom_mask)

        for i in range(B):
            sid = int(batch[i].get("structure_idx", next_struct + i))
            n = int(n_atoms[i].item())
            struct_rows.append({
                "structure_idx": sid,
                "n_atoms": n,
                "pred_energy": float(pred_e[i].cpu()),
                "true_energy": float(true_e[i].cpu()),
                "abs_energy_error": float(torch.abs(true_e[i] - pred_e[i]).cpu()),
                "mean_force_mae": float(struct_err[i].cpu()),
                "p_unreliable_energy": float(pe[i, 1].cpu()),
                "p_unreliable_force_mol": float(pfm[i, 1].cpu()),
                "pred_energy_class": int(pe[i].argmax().cpu()),
                "pred_force_mol_class": int(pfm[i].argmax().cpu()),
            })
            for a in range(n):
                atom_rows.append({
                    "structure_idx": sid,
                    "atom_idx": a,
                    "force_mae": float(atom_err[i, a].cpu()),
                    "p_unreliable_force_atom": float(pfa[i, a, 1].cpu()),
                    "pred_force_atom_class": int(pfa[i, a].argmax().cpu()),
                })

            if compute_metrics and not torch.isnan(true_e[i]):
                store["energy"]["logits"].append(logits_e[i].cpu())
                store["energy"]["targets"].append(
                    scalar_to_bin_index(torch.abs(true_e[i] - pred_e[i]), bins_e).cpu()
                )
                store["force_mol"]["logits"].append(logits_fm[i].cpu())
                store["force_mol"]["targets"].append(
                    scalar_to_bin_index(struct_err[i], bins_fm).cpu()
                )
                store["force_atom"]["logits"].append(logits_fa[i, :n].cpu())
                store["force_atom"]["targets"].append(
                    scalar_to_bin_index(atom_err[i, :n], bins_fa).cpu()
                )
        next_struct += B

    metrics = {}
    if compute_metrics and store["energy"]["logits"]:
        for task in ("energy", "force_mol", "force_atom"):
            logits = torch.cat(store[task]["logits"], dim=0)
            targets = torch.cat(store[task]["targets"], dim=0)
            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            cm = confusion_matrix_torch(preds, targets, num_classes=2)
            metrics[task] = compute_all_metrics(cm)
            metrics[task]["n"] = int(len(targets))
    return struct_rows, atom_rows, metrics


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    extractor = load_dpa3(
        args.model, device=device, hook_module=args.hook_module
    )
    loader = load_extxyz_dataloader(
        args.test_xyz,
        batch_size=args.batch_size,
        shuffle=False,
        max_structures=args.max_structures,
    )

    if args.cache_only:
        from probe.backends.dpa3 import cache_embeddings_to_dir
        cache_dir = args.cache_dir or str(out_dir / "embedding_cache")
        cache_embeddings_to_dir(
            args.test_xyz,
            extractor,
            cache_dir,
            batch_size=args.batch_size,
            compute_force=not args.embeddings_only,
            max_structures=args.max_structures,
            embeddings_only=args.embeddings_only,
        )
        print(f"Cache-only done → {cache_dir}")
        extractor.close()
        return

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --cache-only")

    if args.cache_dir:
        process_fn = CachedDPA3Processor(
            extractor, compute_force=True, cache_dir=args.cache_dir
        )
    else:
        process_fn = lambda batch, dev: process_batch_dpa3(
            batch, dev, extractor, compute_force=True
        )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing model_state_dict")

    bins_e = _bins(ckpt, "error_bins_energy", device)
    bins_fa = _bins(ckpt, "error_bins_force_atom", device)
    bins_fm = _bins(ckpt, "error_bins_force_mol", device)

    # backbone_dim from first live batch if not in ckpt
    sample = next(iter(loader))
    live0 = process_fn(sample, device)
    backbone_dim = int(live0[0].shape[-1])

    model = MultitaskPROBEModel(
        backbone_dim=backbone_dim,
        atom_encoder_hidden=args.atom_encoder_hidden,
        atom_encoder_output_dim=args.atom_encoder_output_dim,
        mol_attention_heads=args.mol_attention_heads,
        classifier_hidden=args.classifier_hidden,
        atom_force_head_hidden=args.atom_force_head_hidden,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # rebuild full loader (sample consumed one batch — reload)
    loader = load_extxyz_dataloader(
        args.test_xyz,
        batch_size=args.batch_size,
        shuffle=False,
        max_structures=args.max_structures,
    )

    struct_rows, atom_rows, metrics = run_inference(
        model, process_fn, loader, device, bins_e, bins_fa, bins_fm,
        compute_metrics=not args.no_metrics,
    )

    with open(out_dir / "predictions_structure.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(struct_rows[0].keys()))
        w.writeheader()
        w.writerows(struct_rows)
    with open(out_dir / "predictions_atom.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(atom_rows[0].keys()))
        w.writeheader()
        w.writerows(atom_rows)

    if metrics:
        # convert tensors in metrics if any
        def _to_jsonable(o):
            if isinstance(o, dict):
                return {k: _to_jsonable(v) for k, v in o.items()
                        if k not in ("probabilities", "predictions", "targets", "errors")}
            if isinstance(o, (np.floating, float)):
                return float(o)
            if isinstance(o, (np.integer, int)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(_to_jsonable(metrics), f, indent=2)
        print("Metrics:", json.dumps(_to_jsonable(metrics), indent=2))

    print(f"Wrote results → {out_dir}")
    extractor.close()


if __name__ == "__main__":
    main()
