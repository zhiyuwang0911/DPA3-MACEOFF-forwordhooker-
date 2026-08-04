#!/usr/bin/env python3
"""
Train multitask PROBE on frozen DPA3 (DPA3-SPICE-MACE-OFF / DPA3-L6) embeddings.

Energy + per-atom force + structure force reliability heads.
DPA3 runs once per structure (CachedDPA3Processor); later epochs reuse cache.

Example:
  python train_dpa3_multitask.py \\
    --model /path/to/DPA3-L6.pt \\
    --train-xyz /path/to/train.xyz \\
    --output-dir ./probe_dpa3_outputs \\
    --cache-dir ./dpa3_cache_train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from probe.model import MultitaskPROBEModel
from probe.backends.dpa3 import (
    load_dpa3,
    train_val_split_loader,
    CachedDPA3Processor,
    process_batch_dpa3,
    scan_force_error_boundaries,
)
from probe.train import run_multitask_training, compute_error_boundary

CONFIG = {
    "model_path": "/path/to/DPA3-L6.pt",
    "train_xyz": "/path/to/train.xyz",
    "output_dir": "./probe_dpa3_multitask_outputs",
    "cache_dir": None,  # set path to persist {idx}.pt embeddings

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hook_module": None,  # auto-detect last repflow / descriptor layer
    "type_map": None,
    "ev_to_kcalmol": 23.06,

    "batch_size": 32,
    "valid_fraction": 0.1,

    "error_boundary_percentile": 50,

    "lambda_energy": 1.0,
    "lambda_force_atom": 1.0,
    "lambda_force_mol": 1.0,

    "lr": 5e-5,
    "weight_decay": 1e-4,
    "epochs": 1000,
    "early_stopping_patience": 10,
    "scheduler_patience": 5,
    "scheduler_factor": 0.9,
    "min_lr": 5e-6,
    "gradient_clip_norm": 1.0,
    "checkpoint_every": 0,

    "cache_dpa3": True,

    "atom_encoder_hidden": [256, 128],
    "atom_encoder_output_dim": 256,
    "mol_attention_heads": 32,
    "classifier_hidden": [256, 128, 32],
    "atom_force_head_hidden": [128, 32],
    "dropout": 0.1,

    "high_conf_cutoffs": {0: 0.8, 1: 0.8},
}


def parse_args():
    p = argparse.ArgumentParser(description="Train multitask PROBE on DPA3")
    p.add_argument("--model", type=str, default=None, help="DPA3-L6 checkpoint")
    p.add_argument("--train-xyz", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Persist DPA3 embedding cache for resume")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--hook-module", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lambda-energy", type=float, default=None)
    p.add_argument("--lambda-force-atom", type=float, default=None)
    p.add_argument("--lambda-force-mol", type=float, default=None)
    p.add_argument("--resume", nargs="?", const="AUTO", default=None)
    p.add_argument("--no-cache-dpa3", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=None)
    return p.parse_args()


def _cfg(cli, key, cast=lambda x: x):
    if cli is not None:
        return cast(cli)
    return CONFIG[key]


def main():
    args = parse_args()
    if args.model:
        CONFIG["model_path"] = args.model
    if args.train_xyz:
        CONFIG["train_xyz"] = args.train_xyz
    if args.output_dir:
        CONFIG["output_dir"] = args.output_dir
    if args.cache_dir is not None:
        CONFIG["cache_dir"] = args.cache_dir
    if args.device:
        CONFIG["device"] = args.device
    if args.hook_module:
        CONFIG["hook_module"] = args.hook_module
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size

    device = CONFIG["device"]
    lambda_energy = _cfg(args.lambda_energy, "lambda_energy", float)
    lambda_force_atom = _cfg(args.lambda_force_atom, "lambda_force_atom", float)
    lambda_force_mol = _cfg(args.lambda_force_mol, "lambda_force_mol", float)
    checkpoint_every = (
        args.checkpoint_every
        if args.checkpoint_every is not None
        else CONFIG.get("checkpoint_every", 0)
    )

    resume_path = None
    if args.resume is not None:
        resume_path = (
            Path(CONFIG["output_dir"]) / "last_checkpoint.pt"
            if args.resume == "AUTO"
            else Path(args.resume)
        )
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume not found: {resume_path}")

    print(
        f"Loss weights: λE={lambda_energy}, λFa={lambda_force_atom}, "
        f"λFs={lambda_force_mol}"
    )

    # 1. Frozen DPA3 backbone + forward hook
    extractor = load_dpa3(
        CONFIG["model_path"],
        device=device,
        hook_module=CONFIG.get("hook_module"),
        type_map=CONFIG.get("type_map"),
    )

    # 2. Data
    print("Loading data...")
    train_loader, val_loader = train_val_split_loader(
        CONFIG["train_xyz"],
        CONFIG["batch_size"],
        CONFIG["valid_fraction"],
    )

    # 3. Processor / cache
    use_cache = CONFIG.get("cache_dpa3", True) and not args.no_cache_dpa3
    if use_cache:
        cache_dir = CONFIG.get("cache_dir")
        if cache_dir:
            print(f"DPA3 cache enabled (RAM + disk) → {cache_dir}")
        else:
            print("DPA3 cache enabled (RAM only; set --cache-dir to persist)")
        process_fn = CachedDPA3Processor(
            extractor, compute_force=True, cache_dir=cache_dir
        )
    else:
        print("DPA3 cache disabled")
        process_fn = lambda batch, dev: process_batch_dpa3(
            batch, dev, extractor, compute_force=True
        )

    # 4. Error boundaries
    if resume_path is not None:
        print(f"Loading error bins from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        error_bins_e = torch.tensor(
            ckpt["error_bins_energy"], device=device, dtype=torch.float32
        )
        error_bins_f_atom = torch.tensor(
            ckpt["error_bins_force_atom"], device=device, dtype=torch.float32
        )
        error_bins_f_mol = torch.tensor(
            ckpt["error_bins_force_mol"], device=device, dtype=torch.float32
        )
    else:
        print("Scanning energy errors (fills DPA3 cache on first pass)...")
        errors_kcal = []
        for batch in tqdm(train_loader, desc="Scan energy"):
            (_, _, pred_e, true_e, _, _, _) = process_fn(batch, device)
            err = torch.abs(true_e - pred_e)
            valid = ~torch.isnan(err)
            errors_kcal.extend(
                (err[valid].cpu().numpy() * CONFIG["ev_to_kcalmol"]).tolist()
            )
        boundary_kcal = compute_error_boundary(
            np.array(errors_kcal), CONFIG["error_boundary_percentile"]
        )
        boundary_ev = boundary_kcal / CONFIG["ev_to_kcalmol"]
        error_bins_e = torch.tensor([0.0, boundary_ev], device=device)

        print("Scanning force errors...")
        b_fa, b_fm = scan_force_error_boundaries(
            train_loader, device, extractor,
            CONFIG["error_boundary_percentile"],
            process_batch_fn=process_fn,
        )
        error_bins_f_atom = torch.tensor([0.0, b_fa], device=device)
        error_bins_f_mol = torch.tensor([0.0, b_fm], device=device)
        if use_cache and isinstance(process_fn, CachedDPA3Processor):
            print(
                f"Cache size={len(process_fn)} "
                f"(hits={process_fn.hits}, misses={process_fn.misses})"
            )

    # Warm feat_dim if only resumed from full cache
    if extractor.feat_dim is None:
        sample = next(iter(train_loader))
        live = process_fn(sample, device)
        extractor.feat_dim = int(live[0].shape[-1])

    # 5. Multitask PROBE
    model = MultitaskPROBEModel(
        backbone_dim=extractor.feat_dim,
        atom_encoder_hidden=CONFIG["atom_encoder_hidden"],
        atom_encoder_output_dim=CONFIG["atom_encoder_output_dim"],
        mol_attention_heads=CONFIG["mol_attention_heads"],
        classifier_hidden=CONFIG["classifier_hidden"],
        atom_force_head_hidden=CONFIG["atom_force_head_hidden"],
        dropout=CONFIG["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Multitask PROBE parameters: {n_params:,} | backbone_dim={extractor.feat_dim}")

    history = run_multitask_training(
        model=model,
        process_batch_fn=process_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        error_bins_e=error_bins_e,
        error_bins_f_atom=error_bins_f_atom,
        error_bins_f_mol=error_bins_f_mol,
        device=device,
        output_dir=CONFIG["output_dir"],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        epochs=CONFIG["epochs"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
        scheduler_patience=CONFIG["scheduler_patience"],
        scheduler_factor=CONFIG["scheduler_factor"],
        min_lr=CONFIG["min_lr"],
        gradient_clip_norm=CONFIG["gradient_clip_norm"],
        lambda_energy=lambda_energy,
        lambda_force_atom=lambda_force_atom,
        lambda_force_mol=lambda_force_mol,
        high_conf_cutoffs=CONFIG["high_conf_cutoffs"],
        resume_path=str(resume_path) if resume_path else None,
        checkpoint_every=checkpoint_every,
    )
    print(f"Done. Best epoch={history['best_epoch']} → {CONFIG['output_dir']}/")
    extractor.close()


if __name__ == "__main__":
    main()
