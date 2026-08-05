#!/usr/bin/env python
"""
Save DPA3 last-layer atomic embeddings for GMM/KDE.

Same output layout as save_atomic_092.py (MACE / PROBE), but backbone is
DPA3-SPICE-MACE-OFF (e.g. DPA3-L6) with a forward hook on the last
atomic embedding layer.

Saves:
1. variable-length atomic embeddings per molecule
2. fixed-length pooled molecular embeddings for GMM/KDE
3. DPA3 predicted energy, true energy, absolute error, and n_atoms

Example:
  python save_atomic_dpa3.py \\
    --model /path/to/DPA3-L6.pt \\
    --xyz /path/to/test.xyz \\
    --output /path/to/test_dpa3.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm.auto import tqdm

from probe.backends.dpa3 import (
    load_dpa3,
    load_extxyz_dataloader,
    process_batch_dpa3,
)

CONFIG = {
    "model_path": "/path/to/DPA3-L6.pt",
    "xyz_file": "/path/to/test.xyz",
    "output_npz": "/path/to/test_dpa3.npz",
    "batch_size": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hook_module": None,  # auto-detect; or e.g. "descriptor.repflows.5"
}


def masked_mean_max(atom_feats, atom_mask):
    """
    atom_feats: [B, Nmax, D]
    atom_mask:  [B, Nmax], bool

    returns:
        mean_feat: [B, D]
        max_feat:  [B, D]
    """
    mask_f = atom_mask.unsqueeze(-1).float()
    n_valid = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).float()

    mean_feat = (atom_feats * mask_f).sum(dim=1) / n_valid

    tmp = atom_feats.clone()
    tmp[~atom_mask.unsqueeze(-1).expand_as(tmp)] = float("-inf")
    max_feat = tmp.max(dim=1)[0]
    max_feat[max_feat == float("-inf")] = 0.0

    return mean_feat, max_feat


def parse_args():
    p = argparse.ArgumentParser(
        description="Save DPA3 last-layer atomic embeddings (.npz)"
    )
    p.add_argument("--model", default=None, help="DPA3-L6 .pt/.pth checkpoint")
    p.add_argument("--xyz", default=None, help="Input .xyz / .extxyz")
    p.add_argument("--output", default=None, help="Output .npz path")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--hook-module", default=None)
    p.add_argument("--max-structures", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model or CONFIG["model_path"]
    xyz_file = args.xyz or CONFIG["xyz_file"]
    output_npz = args.output or CONFIG["output_npz"]
    batch_size = args.batch_size or CONFIG["batch_size"]
    device = args.device or CONFIG["device"]
    hook_module = args.hook_module or CONFIG.get("hook_module")

    print(f"Using device: {device}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Frozen DPA3 + forward hook on last atomic layer
    extractor = load_dpa3(model_path, device=device, hook_module=hook_module)

    loader = load_extxyz_dataloader(
        xyz_path=xyz_file,
        batch_size=batch_size,
        shuffle=False,
        max_structures=args.max_structures,
    )

    all_atom_feats = []
    all_mean_feats = []
    all_max_feats = []
    all_mean_max_energy_natoms = []

    all_pred_energy = []
    all_true_energy = []
    all_abs_error = []
    all_n_atoms = []

    extractor.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting DPA3 embeddings"):
            # energy-only path is enough for this npz (same as MACE script)
            atom_feats, atom_mask, pred_energy, true_energy, n_atoms = process_batch_dpa3(
                batch=batch,
                device=device,
                extractor=extractor,
                compute_force=False,
            )

            mean_feat, max_feat = masked_mean_max(atom_feats, atom_mask)
            pooled = torch.cat(
                [
                    mean_feat,
                    max_feat,
                    pred_energy.unsqueeze(-1),
                    n_atoms.unsqueeze(-1),
                ],
                dim=-1,
            )

            B = atom_feats.shape[0]
            for i in range(B):
                n = int(n_atoms[i].detach().cpu().item())

                atom_i = atom_feats[i, :n, :].detach().cpu().numpy().astype(np.float32)
                mean_i = mean_feat[i].detach().cpu().numpy().astype(np.float32)
                max_i = max_feat[i].detach().cpu().numpy().astype(np.float32)
                pooled_i = pooled[i].detach().cpu().numpy().astype(np.float32)

                pred_i = float(pred_energy[i].detach().cpu().item())
                true_i = float(true_energy[i].detach().cpu().item())
                err_i = abs(true_i - pred_i)

                all_atom_feats.append(atom_i)
                all_mean_feats.append(mean_i)
                all_max_feats.append(max_i)
                all_mean_max_energy_natoms.append(pooled_i)

                all_pred_energy.append(pred_i)
                all_true_energy.append(true_i)
                all_abs_error.append(err_i)
                all_n_atoms.append(n)

    os.makedirs(os.path.dirname(os.path.abspath(output_npz)) or ".", exist_ok=True)
    np.savez_compressed(
        output_npz,
        atom_feats=np.array(all_atom_feats, dtype=object),
        mean_feats=np.stack(all_mean_feats),
        max_feats=np.stack(all_max_feats),
        mean_max_energy_natoms=np.stack(all_mean_max_energy_natoms),
        pred_energy=np.array(all_pred_energy, dtype=np.float64),
        true_energy=np.array(all_true_energy, dtype=np.float64),
        abs_error=np.array(all_abs_error, dtype=np.float64),
        n_atoms=np.array(all_n_atoms, dtype=np.int64),
    )

    print("\nSaved:", output_npz)
    print("Number of structures:", len(all_atom_feats))
    print("Atomic feature dimension D:", all_atom_feats[0].shape[1])
    print("mean_feats shape:", np.stack(all_mean_feats).shape)
    print("max_feats shape:", np.stack(all_max_feats).shape)
    print("mean_max_energy_natoms shape:", np.stack(all_mean_max_energy_natoms).shape)
    print("hook module:", extractor.hook_name)
    extractor.close()


if __name__ == "__main__":
    main()
