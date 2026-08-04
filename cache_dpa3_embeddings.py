#!/usr/bin/env python3
"""
One-pass DPA3 inference: save last-layer atomic embeddings via forward hook.

Like PROBE's MACE cache, each structure is written once as ``{idx}.pt``.

Default (for PROBE training labels) stores:
  node_feats, pred_energy, pred_forces

With ``--embeddings-only`` stores only:
  node_feats [n_atoms, D]

Example:
  python cache_dpa3_embeddings.py \\
    --model /path/to/DPA3-L6.pt \\
    --xyz /path/to/train.xyz \\
    --cache-dir ./dpa3_cache_train \\
    --device cuda

  # embeddings only
  python cache_dpa3_embeddings.py \\
    --model /path/to/DPA3-L6.pt \\
    --xyz /path/to/test.xyz \\
    --cache-dir ./dpa3_cache_test \\
    --embeddings-only
"""

from __future__ import annotations

import argparse

from probe.backends.dpa3 import load_dpa3, cache_embeddings_to_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="Cache DPA3 last-layer atomic embeddings (forward hook)"
    )
    p.add_argument(
        "--model", required=True,
        help="Path to DPA3-SPICE-MACE-OFF checkpoint (DPA3-L6 .pt/.pth)",
    )
    p.add_argument("--xyz", required=True, help="Input extxyz with energy+forces")
    p.add_argument("--cache-dir", required=True, help="Output directory for {idx}.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-structures", type=int, default=None)
    p.add_argument(
        "--hook-module", default=None,
        help="Optional dot-path of module to hook (auto-detect if omitted)",
    )
    p.add_argument(
        "--type-map", nargs="+", default=None,
        help="Override type_map symbols if not in checkpoint",
    )
    p.add_argument(
        "--embeddings-only", action="store_true",
        help="Save only node_feats (no energy/force preds)",
    )
    p.add_argument(
        "--no-force", action="store_true",
        help="Do not compute forces (ignored if --embeddings-only)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or (
        "cuda" if __import__("torch").cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    extractor = load_dpa3(
        args.model,
        device=device,
        hook_module=args.hook_module,
        type_map=args.type_map,
    )

    n = cache_embeddings_to_dir(
        xyz_path=args.xyz,
        extractor=extractor,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        compute_force=not args.no_force,
        max_structures=args.max_structures,
        embeddings_only=args.embeddings_only,
    )
    print(f"Done. {n} structures. feat_dim={extractor.feat_dim} "
          f"hook={extractor.hook_name}")
    extractor.close()


if __name__ == "__main__":
    main()
