"""
DPA3 (DeePMD-kit) backbone for PROBE.

Captures last-layer *atomic* embeddings with a PyTorch forward hook
(same idea as PROBE's MACE products[-1] hook), then freezes the backbone.

Supports:
  - DPA3-SPICE-MACE-OFF checkpoints (e.g. DPA3-L6) from AIS Square
  - frozen DeePMD `.pt` / `.pth` models and training-format model dumps

Typical usage:
    extractor = load_dpa3("/path/to/DPA3-L6.pt", device="cuda")
    # process_batch_dpa3 / CachedDPA3Processor feed MultitaskPROBEModel
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Optional DeePMD imports
# ---------------------------------------------------------------------------

_DP_AVAILABLE = False
_DeepPot = None
_DeepEval = None

try:
    from deepmd.infer import DeepPot as _DeepPot  # type: ignore
    _DP_AVAILABLE = True
except ImportError:
    pass

try:
    from deepmd.infer.deep_eval import DeepEval as _DeepEval  # type: ignore
    _DP_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ELEMENT_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "I": 53,
}
_Z_TO_SYM = {v: k for k, v in _ELEMENT_Z.items()}


def _symbols_to_z(symbols: Sequence[str]) -> np.ndarray:
    zs = []
    for s in symbols:
        if s not in _ELEMENT_Z:
            raise KeyError(f"Unknown element symbol: {s}")
        zs.append(_ELEMENT_Z[s])
    return np.asarray(zs, dtype=np.int32)


def _z_to_atype(atomic_numbers: np.ndarray, type_map: Sequence[str]) -> np.ndarray:
    """Map nuclear charges to DeePMD atom-type indices via type_map."""
    z_of_type = []
    for sym in type_map:
        sym = str(sym)
        if sym not in _ELEMENT_Z:
            raise KeyError(f"type_map entry {sym!r} not in known elements")
        z_of_type.append(_ELEMENT_Z[sym])
    z_to_t = {z: i for i, z in enumerate(z_of_type)}
    atype = []
    for z in atomic_numbers:
        z = int(z)
        if z not in z_to_t:
            raise KeyError(
                f"Atom Z={z} ({_Z_TO_SYM.get(z, '?')}) not in model type_map={list(type_map)}"
            )
        atype.append(z_to_t[z])
    return np.asarray(atype, dtype=np.int32)


def _large_box(n: int = 100.0) -> np.ndarray:
    """Non-periodic molecular box (Å) for DeePMD."""
    return np.diag([float(n)] * 3).astype(np.float64)


def _flatten_feat(t: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Force hooked tensor to shape [n_atoms, D]."""
    if not isinstance(t, torch.Tensor):
        if isinstance(t, (tuple, list)):
            # prefer first tensor-like that looks like atom features
            for x in t:
                if isinstance(x, torch.Tensor) and x.ndim >= 2:
                    t = x
                    break
            else:
                t = t[0]
        else:
            raise TypeError(f"Hook output type not supported: {type(t)}")

    x = t.detach()
    # [nf, nat, D] or [nat, D] or [nf*nat, D]
    if x.ndim == 3:
        # take first frame
        x = x.reshape(-1, x.shape[-1]) if x.shape[0] * x.shape[1] == n_atoms else x[0]
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if x.shape[0] != n_atoms and x.numel() % n_atoms == 0:
        x = x.reshape(n_atoms, -1)
    if x.shape[0] != n_atoms:
        raise RuntimeError(
            f"Hooked atomic features shape {tuple(x.shape)} incompatible with "
            f"n_atoms={n_atoms}"
        )
    return x.float()


def _module_name_score(name: str) -> int:
    """Higher = better candidate for last-layer atomic embedding."""
    n = name.lower()
    score = 0
    if "repflow" in n:
        score += 50
    if "descriptor" in n:
        score += 20
    if any(k in n for k in ("layer", "block", "msg", "update")):
        score += 10
    if any(k in n for k in ("fit", "fitting", "head", "out", "energy")):
        score -= 30
    if any(k in n for k in ("edge", "angle", "tri")):
        score -= 5
    return score


def find_hook_module(
    model: nn.Module,
    hook_module: Optional[str] = None,
) -> Tuple[str, nn.Module]:
    """Locate the module for last-layer atomic embeddings.

    If ``hook_module`` is given (dot path, e.g. ``descriptor.repflows.5``),
    resolve it. Otherwise auto-select the last high-scoring named module.
    """
    if hook_module:
        mod = model
        for part in hook_module.split("."):
            if part.isdigit():
                mod = mod[int(part)]  # type: ignore[index]
            else:
                if not hasattr(mod, part):
                    raise AttributeError(
                        f"Cannot resolve hook path {hook_module!r} at {part!r}. "
                        f"Children: {list(dict(mod.named_children()).keys())}"
                    )
                mod = getattr(mod, part)
        return hook_module, mod

    ranked: List[Tuple[int, str, nn.Module]] = []
    for name, mod in model.named_modules():
        if name == "" or not isinstance(mod, nn.Module):
            continue
        # Skip pure containers without parameters / buffers sometimes empty
        score = _module_name_score(name)
        if score <= 0:
            continue
        ranked.append((score, name, mod))

    if not ranked:
        # fall back: last leaf module with parameters
        leaves = [(n, m) for n, m in model.named_modules()
                  if n and any(True for _ in m.parameters(recurse=False))]
        if not leaves:
            raise RuntimeError(
                "Could not auto-detect a hook module. Pass hook_module= explicitly "
                "(print(model) and pick the last repflow / descriptor layer)."
            )
        name, mod = leaves[-1]
        return name, mod

    ranked.sort(key=lambda x: (x[0], x[1]))
    # Prefer the last among highest scores (deepest layer index)
    best_score = ranked[-1][0]
    top = [r for r in ranked if r[0] == best_score]
    # among top score, take the alphabetically last path (often higher index)
    _, name, mod = top[-1]
    return name, mod


def _unwrap_deepmd_nn(wrapper: Any) -> nn.Module:
    """Best-effort access to the underlying nn.Module from DeePMD loaders.

    DeePMD-kit 3.1 high-level ``DeepEval`` stores the PT backend at
    ``wrapper.deep_eval`` and exposes ``get_model()``. The PT backend keeps
    the runnable module at ``deep_eval.dp.model['Default']``.
    """
    candidates: List[Any] = [
        wrapper,
        getattr(wrapper, "model", None),
        getattr(wrapper, "deep_eval", None),
        getattr(wrapper, "_model", None),
        getattr(wrapper, "module", None),
    ]

    # Official API (3.1+)
    if hasattr(wrapper, "get_model") and callable(wrapper.get_model):
        try:
            candidates.insert(0, wrapper.get_model())
        except Exception:  # noqa: BLE001
            pass

    deep_eval = getattr(wrapper, "deep_eval", None)
    if deep_eval is not None:
        candidates.extend([
            deep_eval,
            getattr(deep_eval, "module", None),
            getattr(deep_eval, "model", None),
            getattr(deep_eval, "_model", None),
            getattr(deep_eval, "pt_model", None),
            getattr(deep_eval, "dp", None),
        ])
        if hasattr(deep_eval, "get_model") and callable(deep_eval.get_model):
            try:
                candidates.insert(0, deep_eval.get_model())
            except Exception:  # noqa: BLE001
                pass
        dp = getattr(deep_eval, "dp", None)
        if dp is not None:
            # ModelWrapper: dp.model is often a ModuleDict with "Default"
            mdict = getattr(dp, "model", None)
            if isinstance(mdict, nn.ModuleDict) and "Default" in mdict:
                candidates.insert(0, mdict["Default"])
            elif isinstance(mdict, dict) and "Default" in mdict:
                candidates.insert(0, mdict["Default"])
            candidates.append(mdict)

    for c in candidates:
        if isinstance(c, nn.Module):
            return c
        m = getattr(c, "model", None) if c is not None else None
        if isinstance(m, nn.Module):
            return m
        if isinstance(m, nn.ModuleDict) and "Default" in m:
            return m["Default"]
        if isinstance(m, dict) and "Default" in m and isinstance(m["Default"], nn.Module):
            return m["Default"]
    raise TypeError(
        "Could not unwrap a torch.nn.Module from the loaded DPA3 object. "
        "Ensure you load a PyTorch DeePMD model (not a pure C++ backend)."
    )


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class DPA3FeatureExtractor(nn.Module):
    """Frozen DPA3/DeePMD model; last-layer atomic embeddings via
    ``eval_descriptor`` (preferred) and/or a forward hook.
    """

    def __init__(
        self,
        model: Optional[nn.Module],
        type_map: Sequence[str],
        hook_module: Optional[str] = None,
        device: str = "cpu",
        infer_wrapper: Any = None,
        prefer_eval_descriptor: bool = True,
    ):
        super().__init__()
        self.model = model
        self.type_map = list(type_map)
        self.device = device
        self.infer_wrapper = infer_wrapper  # DeepPot / DeepEval
        self._last_feats: Optional[torch.Tensor] = None
        self.feat_dim: Optional[int] = None
        self.hook_name = None
        self._hook_handle = None
        self.prefer_eval_descriptor = prefer_eval_descriptor and (
            infer_wrapper is not None and hasattr(infer_wrapper, "eval_descriptor")
        )

        if self.prefer_eval_descriptor:
            print("DPA3FeatureExtractor: using DeepEval.eval_descriptor() "
                  "for atomic embeddings (recommended for DPA3 / JIT models)")
        elif model is not None:
            self.hook_name, hook_mod = find_hook_module(self.model, hook_module)
            self._hook_handle = hook_mod.register_forward_hook(self._hook_fn)
            print(f"DPA3FeatureExtractor: forward hook on '{self.hook_name}'")
            self.model.to(device).eval()
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            raise RuntimeError(
                "Need either DeepEval.eval_descriptor or an nn.Module for hooks."
            )

    def _hook_fn(self, module, inputs, output):
        self._last_feats = output

    def close(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def eval(self):
        if self.model is not None:
            self.model.eval()
        return self

    def _eval_descriptor(
        self,
        coords: np.ndarray,
        atype: np.ndarray,
        cell: Optional[np.ndarray],
    ) -> torch.Tensor:
        """Return atomic descriptor [natoms, D] via DeePMD official API."""
        wrap = self.infer_wrapper
        nat = coords.shape[0]
        coord = coords.reshape(1, nat, 3)
        box = None if cell is None else np.asarray(cell, dtype=np.float64).reshape(1, 3, 3)
        at = atype.reshape(1, nat)
        try:
            desc = wrap.eval_descriptor(coord, box, at)
        except TypeError:
            coord_flat = coords.reshape(1, -1)
            box_flat = None if cell is None else np.asarray(cell).reshape(1, 9)
            desc = wrap.eval_descriptor(coord_flat, box_flat, atype)
        desc = np.asarray(desc, dtype=np.float32)
        # shapes seen: [nframes, natoms, D] or [natoms, D]
        if desc.ndim == 3:
            desc = desc[0]
        if desc.ndim != 2 or desc.shape[0] != nat:
            raise RuntimeError(
                f"eval_descriptor returned shape {desc.shape}, expected [{nat}, D]"
            )
        return torch.from_numpy(np.ascontiguousarray(desc))

    def _eval_deepmd_energy_force(
        self,
        coords: np.ndarray,
        atype: np.ndarray,
        cell: Optional[np.ndarray],
    ) -> Tuple[float, np.ndarray]:
        """Energy (eV) and forces (eV/Å) via DeePMD infer API."""
        wrap = self.infer_wrapper
        if wrap is None:
            raise RuntimeError(
                "No DeePMD infer wrapper; cannot compute energy/forces."
            )
        nat = coords.shape[0]
        coord = coords.reshape(1, nat, 3)
        box = None if cell is None else np.asarray(cell, dtype=np.float64).reshape(1, 3, 3)
        at = atype.reshape(1, nat)

        try:
            out = wrap.eval(coord, box, at) if hasattr(wrap, "eval") else wrap(coord, box, at)
        except TypeError:
            coord_flat = coords.reshape(1, -1)
            box_flat = None if cell is None else np.asarray(cell).reshape(1, 9)
            out = wrap.eval(coord_flat, box_flat, atype)

        if isinstance(out, dict):
            e = out.get("energy", out.get("energies"))
            f = out.get("force", out.get("forces"))
            energy = float(np.asarray(e).reshape(-1)[0])
            forces = np.asarray(f, dtype=np.float64).reshape(nat, 3)
            return energy, forces

        if isinstance(out, (tuple, list)):
            energy = float(np.asarray(out[0]).reshape(-1)[0])
            forces = np.asarray(out[1], dtype=np.float64).reshape(nat, 3)
            return energy, forces

        raise RuntimeError(f"Unexpected DeePMD eval return type: {type(out)}")

    @torch.no_grad()
    def forward_structure(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: Optional[np.ndarray] = None,
        compute_force: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run one structure.

        Returns dict with:
          node_feats [nat, D], pred_energy scalar, pred_forces [nat, 3] (optional)
        """
        positions = np.asarray(positions, dtype=np.float64)
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int32)
        n_atoms = positions.shape[0]
        atype = _z_to_atype(atomic_numbers, self.type_map)

        # Energy / forces
        if self.infer_wrapper is not None:
            energy, forces = self._eval_deepmd_energy_force(positions, atype, cell)
        else:
            energy, forces = 0.0, np.zeros((n_atoms, 3), dtype=np.float64)

        # Atomic embeddings
        if self.prefer_eval_descriptor:
            node_feats = self._eval_descriptor(positions, atype, cell)
        else:
            self._last_feats = None
            self._run_eager_for_hook(positions, atype, cell)
            if self._last_feats is None:
                raise RuntimeError(
                    f"Forward hook on '{self.hook_name}' did not fire. "
                    "Try --hook-module or rely on eval_descriptor."
                )
            node_feats = _flatten_feat(self._last_feats, n_atoms).cpu()

        if self.feat_dim is None:
            self.feat_dim = int(node_feats.shape[-1])
            print(f"DPA3FeatureExtractor: feat_dim={self.feat_dim}")

        out = {
            "node_feats": node_feats.float().contiguous(),
            "pred_energy": torch.tensor(float(energy), dtype=torch.float32),
        }
        if compute_force:
            out["pred_forces"] = torch.tensor(forces, dtype=torch.float32)
        return out

    def _run_eager_for_hook(
        self,
        positions: np.ndarray,
        atype: np.ndarray,
        cell: Optional[np.ndarray],
        return_energy_force: bool = False,
    ):
        """Attempt an eager nn.Module forward so the hook fires."""
        device = self.device
        pos = torch.tensor(positions, dtype=torch.float64, device=device)
        at = torch.tensor(atype, dtype=torch.long, device=device)
        n = pos.shape[0]

        if cell is None:
            box = torch.tensor(_large_box(), dtype=torch.float64, device=device)
        else:
            box = torch.tensor(np.asarray(cell, dtype=np.float64).reshape(3, 3),
                               dtype=torch.float64, device=device)

        model = self.model
        call_errors = []
        attempts = [
            lambda: model(
                coord=pos.unsqueeze(0),
                atype=at.unsqueeze(0),
                box=box.unsqueeze(0),
            ),
            lambda: model(
                pos.reshape(1, -1),
                at.unsqueeze(0),
                box.reshape(1, -1),
            ),
            lambda: model.forward_common_batched(  # type: ignore[attr-defined]
                {"coord": pos.unsqueeze(0), "atype": at.unsqueeze(0),
                 "box": box.unsqueeze(0)},
            ),
        ]
        result = None
        for fn in attempts:
            try:
                result = fn()
                break
            except Exception as e:  # noqa: BLE001
                call_errors.append(str(e))

        if result is None and self._last_feats is None:
            raise RuntimeError(
                "Eager DPA3 forward failed and hook did not fire.\n"
                + "\n".join(call_errors[:5])
            )
        if not return_energy_force:
            return None, None
        energy, forces = 0.0, np.zeros((n, 3), dtype=np.float64)
        if isinstance(result, dict):
            if "energy" in result:
                energy = float(torch.as_tensor(result["energy"]).reshape(-1)[0])
            if "force" in result or "forces" in result:
                f = result.get("force", result.get("forces"))
                forces = torch.as_tensor(f).detach().cpu().numpy().reshape(n, 3)
        return energy, forces


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_dpa3(
    model_path: str,
    device: str = "cuda",
    hook_module: Optional[str] = None,
    type_map: Optional[Sequence[str]] = None,
) -> DPA3FeatureExtractor:
    """
    Load a frozen DPA3 checkpoint (DPA3-L6 / DPA3-SPICE-MACE-OFF etc.).

    Uses DeePMD ``DeepEval`` / ``DeepPot``. Atomic embeddings come from
    ``eval_descriptor`` (robust with JIT). Forward hooks are optional fallback.
    """
    if not _DP_AVAILABLE:
        raise ImportError(
            "deepmd-kit is required. Install e.g.\n"
            "  pip install deepmd-kit\n"
            "or follow https://docs.deepmodeling.com/"
        )

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    model_path = str(model_path)
    print(f"Loading DPA3 model from {model_path}")

    infer_wrapper = None
    nn_model: Optional[nn.Module] = None
    resolved_type_map = list(type_map) if type_map is not None else None
    load_errors: List[str] = []

    # 1) High-level DeepEval (preferred in deepmd 3.1)
    try:
        from deepmd.infer import DeepEval as HighLevelDeepEval  # type: ignore
        # DeepPot is the energy model interface; DeepEval auto-selects
        try:
            from deepmd.infer import DeepPot  # type: ignore
            infer_wrapper = DeepPot(model_path)
            print("Loaded via DeepPot")
        except Exception as e_dp:  # noqa: BLE001
            load_errors.append(f"DeepPot: {e_dp}")
            infer_wrapper = HighLevelDeepEval(model_path)
            print("Loaded via DeepEval")
        if resolved_type_map is None and hasattr(infer_wrapper, "get_type_map"):
            resolved_type_map = list(infer_wrapper.get_type_map())
        try:
            nn_model = _unwrap_deepmd_nn(infer_wrapper)
            print(f"Unwrapped nn.Module: {type(nn_model).__name__}")
        except TypeError as e:
            load_errors.append(f"DeepEval unwrap: {e}")
            # OK if eval_descriptor is available
            if not hasattr(infer_wrapper, "eval_descriptor"):
                raise
            print("No unwrap for hooks; will use eval_descriptor() only")
    except Exception as e:  # noqa: BLE001
        load_errors.append(f"DeepEval/DeepPot: {e}")
        infer_wrapper = None

    # 2) Fallback: older DeepEval ctor used elsewhere
    if infer_wrapper is None and _DeepEval is not None:
        try:
            infer_wrapper = _DeepEval(model_path, device=device)
            if resolved_type_map is None and hasattr(infer_wrapper, "get_type_map"):
                resolved_type_map = list(infer_wrapper.get_type_map())
            try:
                nn_model = _unwrap_deepmd_nn(infer_wrapper)
            except TypeError as e:
                load_errors.append(f"DeepEval unwrap: {e}")
        except Exception as e:  # noqa: BLE001
            load_errors.append(f"DeepEval: {e}")
            infer_wrapper = None

    # 3) Raw torch.load fallback
    if nn_model is None and infer_wrapper is None:
        try:
            obj = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(obj, nn.Module):
                nn_model = obj
            elif isinstance(obj, dict):
                for key in ("model", "module"):
                    if key in obj and isinstance(obj[key], nn.Module):
                        nn_model = obj[key]
                        break
                if nn_model is None and "model" in obj and isinstance(obj["model"], dict):
                    try:
                        from deepmd.pt.model.model import BaseModel  # type: ignore
                        nn_model = BaseModel.deserialize(obj["model"])
                    except Exception as e:  # noqa: BLE001
                        load_errors.append(f"BaseModel.deserialize: {e}")
                if resolved_type_map is None:
                    for k in ("type_map", "model_params"):
                        if k in obj:
                            tm = obj[k]
                            if isinstance(tm, dict) and "type_map" in tm:
                                resolved_type_map = list(tm["type_map"])
                            elif isinstance(tm, (list, tuple)):
                                resolved_type_map = list(tm)
            else:
                load_errors.append(f"torch.load type: {type(obj)}")
        except Exception as e:  # noqa: BLE001
            load_errors.append(f"torch.load: {e}")

    can_descriptor = infer_wrapper is not None and hasattr(infer_wrapper, "eval_descriptor")
    if nn_model is None and not can_descriptor:
        hint = ""
        joined = "\n".join(load_errors)
        if "mpich" in joined.lower() and "metadata" in joined.lower():
            hint = (
                "\n\nHint: HPC MPI/metadata issue. Try:\n"
                "  conda install -y -c conda-forge mpich mpi4py\n"
            )
        raise RuntimeError(
            "Failed to load DPA3 (need DeepEval.eval_descriptor or nn.Module).\n"
            + joined
            + hint
        )

    if resolved_type_map is None:
        resolved_type_map = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        print(
            "WARNING: type_map not found in checkpoint; using SPICE-like default:\n"
            f"  {resolved_type_map}"
        )

    extractor = DPA3FeatureExtractor(
        model=nn_model,
        type_map=resolved_type_map,
        hook_module=hook_module,
        device=device,
        infer_wrapper=infer_wrapper,
        prefer_eval_descriptor=can_descriptor,
    )
    print(f"type_map={extractor.type_map}")
    return extractor


# ---------------------------------------------------------------------------
# Dataset / loaders (extxyz → simple dicts; no MACE AtomicData)
# ---------------------------------------------------------------------------

class StructureDataset(Dataset):
    def __init__(self, frames: List[dict]):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        fr = self.frames[idx]
        out = dict(fr)
        # Keep global structure_idx (set when building frames) for disk cache keys.
        if "structure_idx" not in out:
            out["structure_idx"] = idx
        return out


def _collate_structures(batch: List[dict]) -> List[dict]:
    return batch


def _frames_from_xyz(xyz_path: str, max_structures: Optional[int] = None) -> List[dict]:
    from ..io_extxyz import iter_probe_extxyz

    frames = []
    for fr in iter_probe_extxyz(xyz_path, max_structures=max_structures):
        zs = _symbols_to_z(fr["symbols"])
        frames.append({
            "positions": fr["positions"].astype(np.float64),
            "atomic_numbers": zs,
            "symbols": fr["symbols"],
            "true_energy": fr["true_energy"],
            "true_forces": fr["true_forces"].astype(np.float64),
            "cell": fr.get("cell"),
            "pbc": fr.get("pbc"),
        })
    return frames


def train_val_split_loader(
    xyz_path: str,
    batch_size: int,
    valid_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    frames = _frames_from_xyz(xyz_path)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(frames))
    rng.shuffle(idx)
    n_val = max(1, int(len(frames) * valid_fraction))
    val_set = set(idx[:n_val].tolist())

    train_frames, val_frames = [], []
    for i, fr in enumerate(frames):
        fr = dict(fr)
        fr["structure_idx"] = i  # stable index for caching
        (val_frames if i in val_set else train_frames).append(fr)

    # Remap structure_idx is already global from full set — good for disk cache.
    train_loader = DataLoader(
        StructureDataset(train_frames), batch_size=batch_size,
        shuffle=True, collate_fn=_collate_structures,
    )
    val_loader = DataLoader(
        StructureDataset(val_frames), batch_size=batch_size,
        shuffle=False, collate_fn=_collate_structures,
    )
    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}")
    return train_loader, val_loader


def load_extxyz_dataloader(
    xyz_path: str,
    batch_size: int,
    shuffle: bool = False,
    max_structures: Optional[int] = None,
) -> DataLoader:
    frames = _frames_from_xyz(xyz_path, max_structures=max_structures)
    for i, fr in enumerate(frames):
        fr["structure_idx"] = i
    return DataLoader(
        StructureDataset(frames), batch_size=batch_size,
        shuffle=shuffle, collate_fn=_collate_structures,
    )


# ---------------------------------------------------------------------------
# Batch processing + cache (only atomic embeddings + preds)
# ---------------------------------------------------------------------------

def process_batch_dpa3(
    batch: List[dict],
    device: str,
    extractor: DPA3FeatureExtractor,
    compute_force: bool = True,
) -> Tuple:
    """
    Returns PROBE multitask tensors:
      atom_feats, atom_mask, pred_energy, true_energy,
      pred_forces, true_forces, n_atoms
    """
    entries = []
    true_energies = []
    true_forces_list = []
    for fr in batch:
        cell = fr.get("cell")
        out = extractor.forward_structure(
            fr["positions"], fr["atomic_numbers"],
            cell=cell, compute_force=compute_force,
        )
        entries.append(out)
        te = fr["true_energy"]
        true_energies.append(
            torch.tensor(float(te) if te is not None else float("nan"))
        )
        true_forces_list.append(
            torch.tensor(fr["true_forces"], dtype=torch.float32)
        )

    return _pad_cached_structures(
        entries, true_energies, true_forces_list, device, compute_force
    )


def _pad_cached_structures(entries, true_energies, true_forces_list, device,
                           compute_force: bool):
    B = len(entries)
    D = entries[0]["node_feats"].shape[1]
    sizes = [e["node_feats"].shape[0] for e in entries]
    N_max = max(sizes)

    atom_feats = torch.zeros(B, N_max, D, device=device)
    atom_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)
    pred_energy = torch.empty(B, device=device)
    true_energy = torch.empty(B, device=device)
    pred_forces = true_forces = None
    if compute_force:
        pred_forces = torch.zeros(B, N_max, 3, device=device)
        true_forces = torch.zeros(B, N_max, 3, device=device)

    for i, entry in enumerate(entries):
        n = sizes[i]
        atom_feats[i, :n] = entry["node_feats"].to(device=device, dtype=torch.float32)
        atom_mask[i, :n] = True
        pred_energy[i] = entry["pred_energy"].to(device=device, dtype=torch.float32)
        true_energy[i] = true_energies[i].to(device=device, dtype=torch.float32)
        if compute_force:
            pred_forces[i, :n] = entry["pred_forces"].to(
                device=device, dtype=torch.float32)
            true_forces[i, :n] = true_forces_list[i].to(
                device=device, dtype=torch.float32)

    n_atoms = atom_mask.sum(dim=1).float()
    if compute_force:
        return (atom_feats, atom_mask, pred_energy, true_energy,
                pred_forces, true_forces, n_atoms)
    return atom_feats, atom_mask, pred_energy, true_energy, n_atoms


class CachedDPA3Processor:
    """Run DPA3 once per structure; cache atomic embeddings (+ energy/force preds).

    Disk format matches PROBE MACE cache: ``{structure_idx}.pt`` with keys
    ``node_feats``, ``pred_energy``, optional ``pred_forces``.
    """

    def __init__(
        self,
        extractor: DPA3FeatureExtractor,
        compute_force: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.extractor = extractor
        self.compute_force = compute_force
        self._mem: Dict[int, dict] = {}
        self.cache_dir: Optional[Path] = None
        self.hits = 0
        self.misses = 0
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            n_disk = sum(1 for _ in self.cache_dir.glob("*.pt"))
            if n_disk:
                print(f"DPA3 cache dir {self.cache_dir}: {n_disk} entries on disk")

    def _disk_path(self, sid: int) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{sid}.pt"

    def _load(self, sid: int) -> Optional[dict]:
        if sid in self._mem:
            return self._mem[sid]
        if self.cache_dir is not None:
            path = self._disk_path(sid)
            if path.exists():
                entry = torch.load(path, map_location="cpu", weights_only=False)
                self._mem[sid] = entry
                return entry
        return None

    def _store(self, sid: int, entry: dict):
        # embeddings-only disk option still stores preds needed for labels
        self._mem[sid] = entry
        if self.cache_dir is not None:
            torch.save(entry, self._disk_path(sid))

    def __len__(self):
        return len(self._mem)

    def __call__(self, batch: List[dict], device: str):
        ids = [int(fr["structure_idx"]) for fr in batch]
        entries = [self._load(sid) for sid in ids]
        if all(e is not None for e in entries):
            self.hits += len(ids)
            true_energies = [
                torch.tensor(
                    float(fr["true_energy"]) if fr["true_energy"] is not None
                    else float("nan")
                )
                for fr in batch
            ]
            true_forces_list = [
                torch.tensor(fr["true_forces"], dtype=torch.float32) for fr in batch
            ]
            return _pad_cached_structures(
                entries, true_energies, true_forces_list, device, self.compute_force
            )

        self.misses += len(ids)
        live = process_batch_dpa3(
            batch, device, self.extractor, compute_force=self.compute_force
        )
        if self.compute_force:
            (atom_feats, atom_mask, pred_energy, true_energy,
             pred_forces, true_forces, n_atoms) = live
        else:
            atom_feats, atom_mask, pred_energy, true_energy, n_atoms = live
            pred_forces = true_forces = None

        for i, sid in enumerate(ids):
            if self._load(sid) is not None:
                continue
            n = int(atom_mask[i].sum().item())
            entry = {
                "node_feats": atom_feats[i, :n].detach().cpu().contiguous(),
                "pred_energy": pred_energy[i].detach().cpu(),
            }
            if self.compute_force:
                entry["pred_forces"] = pred_forces[i, :n].detach().cpu().contiguous()
            self._store(sid, entry)
        return live


def cache_embeddings_to_dir(
    xyz_path: str,
    extractor: DPA3FeatureExtractor,
    cache_dir: str,
    batch_size: int = 1,
    compute_force: bool = True,
    max_structures: Optional[int] = None,
    embeddings_only: bool = False,
) -> int:
    """
    One-pass inference: save per-structure ``{idx}.pt`` caches.

    If ``embeddings_only`` is True, each file contains only
    ``node_feats`` [n_atoms, D] (and metadata). Otherwise also stores
    ``pred_energy`` / ``pred_forces`` (needed for PROBE training labels).
    """
    from tqdm.auto import tqdm

    loader = load_extxyz_dataloader(
        xyz_path, batch_size=batch_size, shuffle=False,
        max_structures=max_structures,
    )
    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for batch in tqdm(loader, desc="Caching DPA3 embeddings"):
        for fr in batch:
            sid = int(fr["structure_idx"])
            path = cache_dir_p / f"{sid}.pt"
            if path.exists():
                n_saved += 1
                continue
            out = extractor.forward_structure(
                fr["positions"], fr["atomic_numbers"],
                cell=fr.get("cell"), compute_force=compute_force and not embeddings_only,
            )
            if embeddings_only:
                entry = {
                    "node_feats": out["node_feats"].cpu().contiguous(),
                    "structure_idx": sid,
                    "n_atoms": int(out["node_feats"].shape[0]),
                    "feat_dim": int(out["node_feats"].shape[1]),
                }
            else:
                entry = {
                    "node_feats": out["node_feats"].cpu().contiguous(),
                    "pred_energy": out["pred_energy"].cpu(),
                    "structure_idx": sid,
                }
                if "pred_forces" in out:
                    entry["pred_forces"] = out["pred_forces"].cpu().contiguous()
            torch.save(entry, path)
            n_saved += 1
    # write a small manifest
    meta = {
        "n_structures": n_saved,
        "feat_dim": extractor.feat_dim,
        "type_map": extractor.type_map,
        "hook_module": extractor.hook_name,
        "embeddings_only": embeddings_only,
    }
    torch.save(meta, cache_dir_p / "cache_meta.pt")
    print(f"Saved {n_saved} caches → {cache_dir_p}")
    return n_saved


def scan_force_error_boundaries(
    train_loader, device, extractor, percentile: float = 50, process_batch_fn=None
):
    from ..labels import (
        atom_force_component_mae,
        structure_mean_force_error,
        compute_percentile_boundary,
    )

    if process_batch_fn is None:
        process_batch_fn = (
            lambda batch, dev: process_batch_dpa3(batch, dev, extractor, True)
        )

    atom_errors, struct_errors = [], []
    for batch in train_loader:
        (_, atom_mask, _, _, pred_forces, true_forces, _) = process_batch_fn(
            batch, device
        )
        atom_err = atom_force_component_mae(pred_forces, true_forces)
        struct_err = structure_mean_force_error(atom_err, atom_mask)
        mask = atom_mask.bool()
        atom_errors.extend(atom_err[mask].cpu().numpy().tolist())
        struct_errors.extend(struct_err.cpu().numpy().tolist())

    boundary_atom = compute_percentile_boundary(
        np.array(atom_errors), percentile, unit="eV/Å (per atom)"
    )
    boundary_mol = compute_percentile_boundary(
        np.array(struct_errors), percentile, unit="eV/Å (per structure mean)"
    )
    return boundary_atom, boundary_mol
