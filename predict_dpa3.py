#!/usr/bin/env python3
"""
Single-point DPA3 energy / force predictions on a concatenated XYZ file.

Reads a multi-frame ``.xyz`` / ``.extxyz`` (geometry-only or with reference
energy/forces) and writes DPA3 predictions. Units: energy eV, forces eV/Å.

Example:
  python predict_dpa3.py \\
    --model /path/to/DPA3-L6.pt \\
    --xyz /path/to/concat.xyz \\
    --output-dir ./dpa3_single_point \\
    --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from probe.backends.dpa3 import load_dpa3, load_extxyz_dataloader
from probe.io_extxyz import format_extxyz_frame


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-point DPA3 energy/force predictions from concatenated XYZ"
    )
    p.add_argument("--model", required=True, help="DPA3-L6 .pt/.pth checkpoint")
    p.add_argument("--xyz", required=True, help="Concatenated .xyz / .extxyz")
    p.add_argument("--output-dir", default="./dpa3_single_point")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-structures", type=int, default=None)
    p.add_argument("--hook-module", default=None)
    p.add_argument("--type-map", nargs="+", default=None)
    p.add_argument(
        "--energy-only", action="store_true",
        help="Write energies only (still obtained from a force-capable eval)",
    )
    p.add_argument(
        "--no-xyz", action="store_true",
        help="Do not write predicted concatenated XYZ",
    )
    return p.parse_args()


def _finite_mae(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def _finite_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compute_force = not args.energy_only
    print(f"Device: {device}")

    extractor = load_dpa3(
        args.model,
        device=device,
        hook_module=args.hook_module,
        type_map=args.type_map,
    )
    loader = load_extxyz_dataloader(
        args.xyz,
        batch_size=args.batch_size,
        shuffle=False,
        max_structures=args.max_structures,
    )

    energy_csv = out_dir / "predictions_energy.csv"
    force_csv = out_dir / "predictions_forces.csv"
    xyz_path = out_dir / "predictions.xyz"

    energy_fields = [
        "structure_idx", "n_atoms", "pred_energy_eV", "true_energy_eV",
        "abs_energy_error_eV",
    ]
    force_fields = [
        "structure_idx", "atom_idx", "symbol", "x", "y", "z",
        "pred_fx", "pred_fy", "pred_fz",
        "true_fx", "true_fy", "true_fz",
    ]

    pred_e_all, true_e_all = [], []
    pred_f_all, true_f_all = [], []
    n_atoms_all = []
    n_struct = 0

    energy_fh = open(energy_csv, "w", newline="")
    energy_w = csv.DictWriter(energy_fh, fieldnames=energy_fields)
    energy_w.writeheader()
    force_fh = force_w = None
    xyz_fh = None
    if compute_force:
        force_fh = open(force_csv, "w", newline="")
        force_w = csv.DictWriter(force_fh, fieldnames=force_fields)
        force_w.writeheader()
    if not args.no_xyz:
        xyz_fh = open(xyz_path, "w")

    try:
        # DeePMD energy/force eval needs autograd on coordinates — no no_grad.
        for batch in tqdm(loader, desc="DPA3 single-point"):
            for fr in batch:
                sid = int(fr.get("structure_idx", n_struct))
                pos = np.asarray(fr["positions"], dtype=np.float64)
                zs = np.asarray(fr["atomic_numbers"], dtype=np.int32)
                symbols = list(fr["symbols"])
                n = int(pos.shape[0])
                pred = extractor.predict_energy_force(
                    pos, zs, cell=fr.get("cell"), compute_force=compute_force,
                )
                pred_e = float(pred["pred_energy"])
                true_e = fr.get("true_energy")
                true_e_f = float(true_e) if true_e is not None else float("nan")
                abs_e = abs(true_e_f - pred_e) if np.isfinite(true_e_f) else float("nan")

                energy_w.writerow({
                    "structure_idx": sid,
                    "n_atoms": n,
                    "pred_energy_eV": pred_e,
                    "true_energy_eV": true_e_f,
                    "abs_energy_error_eV": abs_e,
                })
                pred_e_all.append(pred_e)
                true_e_all.append(true_e_f)
                n_atoms_all.append(n)

                pred_f = None
                if compute_force:
                    pred_f = np.asarray(pred["pred_forces"], dtype=np.float64).reshape(n, 3)
                    true_f = np.asarray(fr["true_forces"], dtype=np.float64).reshape(n, 3)
                    pred_f_all.append(pred_f)
                    true_f_all.append(true_f)
                    for a in range(n):
                        force_w.writerow({
                            "structure_idx": sid,
                            "atom_idx": a,
                            "symbol": symbols[a],
                            "x": pos[a, 0],
                            "y": pos[a, 1],
                            "z": pos[a, 2],
                            "pred_fx": pred_f[a, 0],
                            "pred_fy": pred_f[a, 1],
                            "pred_fz": pred_f[a, 2],
                            "true_fx": true_f[a, 0],
                            "true_fy": true_f[a, 1],
                            "true_fz": true_f[a, 2],
                        })

                if xyz_fh is not None:
                    extra = {}
                    if np.isfinite(true_e_f):
                        extra["REF_energy"] = true_e_f
                    xyz_fh.write(format_extxyz_frame(
                        symbols, pos,
                        energy=pred_e,
                        forces=pred_f,
                        cell=fr.get("cell"),
                        pbc=fr.get("pbc"),
                        extra_info=extra,
                    ))
                n_struct += 1
    finally:
        energy_fh.close()
        if force_fh is not None:
            force_fh.close()
        if xyz_fh is not None:
            xyz_fh.close()
        extractor.close()

    pred_e_arr = np.asarray(pred_e_all, dtype=np.float64)
    true_e_arr = np.asarray(true_e_all, dtype=np.float64)
    summary = {
        "model": args.model,
        "xyz": args.xyz,
        "n_structures": n_struct,
        "n_atoms_total": int(np.sum(n_atoms_all)) if n_atoms_all else 0,
        "units": {"energy": "eV", "forces": "eV/A"},
        "energy_mae_eV": _finite_mae(pred_e_arr, true_e_arr),
        "energy_rmse_eV": _finite_rmse(pred_e_arr, true_e_arr),
        "n_structures_with_ref_energy": int(np.isfinite(true_e_arr).sum()),
    }
    if compute_force and pred_f_all:
        pred_f_cat = np.concatenate(pred_f_all, axis=0)
        true_f_cat = np.concatenate(true_f_all, axis=0)
        summary["force_mae_eV_A"] = _finite_mae(pred_f_cat, true_f_cat)
        summary["force_rmse_eV_A"] = _finite_rmse(pred_f_cat, true_f_cat)
        atom_mae = np.mean(np.abs(pred_f_cat - true_f_cat), axis=1)
        mask = np.isfinite(true_f_cat).all(axis=1)
        summary["force_atom_mae_eV_A"] = (
            float(np.mean(atom_mae[mask])) if mask.any() else float("nan")
        )
        summary["n_atoms_with_ref_forces"] = int(mask.sum())

    np.savez_compressed(
        out_dir / "predictions.npz",
        structure_idx=np.arange(n_struct, dtype=np.int64),
        n_atoms=np.asarray(n_atoms_all, dtype=np.int64),
        pred_energy_eV=pred_e_arr,
        true_energy_eV=true_e_arr,
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {n_struct} structures → {out_dir}")
    print(f"  {energy_csv}")
    if compute_force:
        print(f"  {force_csv}")
    if not args.no_xyz:
        print(f"  {xyz_path}")
    print(f"  {out_dir / 'predictions.npz'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
