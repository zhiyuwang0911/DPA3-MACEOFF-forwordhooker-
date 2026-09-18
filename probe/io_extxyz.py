"""
Extended XYZ I/O for PROBE training data.

Reads structure inputs and reference labels from extxyz:
  - species, positions  → model input (required)
  - energy              → reference energy (optional; used for labeling)
  - forces              → reference forces (optional; used for labeling)

Geometry-only XYZ (species x y z) is valid for embedding dump scripts
such as save_atomic_dpa3.py. Training / force-reliability still needs
reference energy + forces when those labels are used.

Extra columns (e.g. MACE_forces) and info fields (e.g. MACE_energy) are
ignored — model predictions always come from a live forward pass,
matching the original PROBE design.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

_INFO_RE = re.compile(r'(\w+)=(?:"([^"]*)"|([^\s"]+))')

_ENERGY_KEYS = ('energy', 'REF_energy', 'DFT_energy')


def parse_extxyz_info(info_line: str) -> dict:
    """Parse key=value pairs from an extxyz comment line."""
    info = {}
    for match in _INFO_RE.finditer(info_line.strip()):
        key = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        info[key] = value
    return info


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_reference_energy(info: dict) -> Optional[float]:
    for key in _ENERGY_KEYS:
        val = _parse_float(info.get(key))
        if val is not None:
            return val
    return None


def _parse_lattice(info: dict) -> Optional[np.ndarray]:
    raw = info.get("Lattice") or info.get("lattice")
    if raw is None:
        return None
    vals = [float(x) for x in str(raw).replace(",", " ").split()]
    if len(vals) != 9:
        return None
    return np.asarray(vals, dtype=np.float64).reshape(3, 3)


def _parse_pbc(info: dict) -> Optional[np.ndarray]:
    raw = info.get("pbc") or info.get("PBC")
    if raw is None:
        return None
    tokens = str(raw).replace(",", " ").split()
    if len(tokens) != 3:
        return None
    truthy = {"t", "true", "1"}
    return np.array([t.lower() in truthy for t in tokens], dtype=bool)


def read_probe_extxyz_frame(lines: list[str], start: int = 0) -> tuple[dict, int]:
    """
    Parse one extxyz frame starting at `start` (line index of natoms line).

    Atom columns:
      required: species  x  y  z
      optional: fx  fy  fz  [extra columns ignored]
    """
    n_atoms = int(lines[start].strip())
    info = parse_extxyz_info(lines[start + 1])

    symbols, positions, true_forces = [], [], []
    has_forces = None
    for i in range(n_atoms):
        parts = lines[start + 2 + i].split()
        if len(parts) < 4:
            raise ValueError(
                f"Expected >=4 columns per atom (species x y z), "
                f"got {len(parts)} at line {start + 3 + i}"
            )
        row_has_forces = len(parts) >= 7
        if has_forces is None:
            has_forces = row_has_forces
        elif row_has_forces != has_forces:
            raise ValueError(
                f"Inconsistent atom columns in one frame: expected "
                f"{'forces' if has_forces else 'positions only'}, "
                f"got {len(parts)} columns at line {start + 3 + i}"
            )
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if row_has_forces:
            true_forces.append([float(parts[4]), float(parts[5]), float(parts[6])])

    if has_forces:
        forces = np.asarray(true_forces, dtype=np.float64)
    else:
        forces = np.full((n_atoms, 3), np.nan, dtype=np.float64)

    frame = {
        'n_atoms': n_atoms,
        'symbols': symbols,
        'positions': np.asarray(positions, dtype=np.float64),
        'true_forces': forces,
        'has_forces': bool(has_forces),
        'true_energy': _get_reference_energy(info),
        'cell': _parse_lattice(info),
        'pbc': _parse_pbc(info),
        'config_type': info.get('config_type', ''),
        'smiles': info.get('smiles', ''),
        'info': info,
    }
    return frame, start + 2 + n_atoms


def iter_probe_extxyz(path: str | Path,
                      max_structures: Optional[int] = None) -> Iterator[dict]:
    """Yield frame dicts from an extended XYZ file."""
    path = Path(path)
    with path.open() as fh:
        lines = fh.readlines()

    idx = 0
    n_frames = 0
    while idx < len(lines):
        if not lines[idx].strip():
            idx += 1
            continue
        frame, idx = read_probe_extxyz_frame(lines, idx)
        yield frame
        n_frames += 1
        if max_structures is not None and n_frames >= max_structures:
            break


def load_probe_extxyz(path: str | Path,
                      max_structures: Optional[int] = None) -> list[dict]:
    """Load all frames from an extended XYZ file."""
    return list(iter_probe_extxyz(path, max_structures=max_structures))


def _fmt_xyz_value(value) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    text = str(value)
    if any(c.isspace() for c in text) or not text:
        return f'"{text}"'
    return text


def format_extxyz_frame(
    symbols: list[str],
    positions: np.ndarray,
    *,
    energy: Optional[float] = None,
    forces: Optional[np.ndarray] = None,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[np.ndarray] = None,
    extra_info: Optional[dict] = None,
) -> str:
    """Serialize one concatenated-XYZ / extxyz frame."""
    n_atoms = len(symbols)
    positions = np.asarray(positions, dtype=np.float64).reshape(n_atoms, 3)
    info_parts: list[str] = []
    if cell is not None:
        lat = " ".join(f"{x:.12g}" for x in np.asarray(cell, dtype=np.float64).reshape(-1))
        info_parts.append(f'Lattice="{lat}"')
    props = "species:S:1:pos:R:3"
    if forces is not None:
        props += ":forces:R:3"
    info_parts.append(f"Properties={props}")
    if energy is not None:
        info_parts.append(f"energy={float(energy):.12g}")
    if pbc is not None:
        flags = " ".join("T" if bool(x) else "F" for x in np.asarray(pbc).reshape(-1))
        info_parts.append(f'pbc="{flags}"')
    skip = {"Lattice", "lattice", "Properties", "energy", "pbc", "PBC"}
    if extra_info:
        for key, value in extra_info.items():
            if value is None or key in skip:
                continue
            info_parts.append(f"{key}={_fmt_xyz_value(value)}")
    lines = [str(n_atoms), " ".join(info_parts)]
    forces_arr = None if forces is None else np.asarray(forces, dtype=np.float64).reshape(n_atoms, 3)
    for i, sym in enumerate(symbols):
        x, y, z = positions[i]
        row = f"{sym} {x:.12g} {y:.12g} {z:.12g}"
        if forces_arr is not None:
            fx, fy, fz = forces_arr[i]
            row += f" {fx:.12g} {fy:.12g} {fz:.12g}"
        lines.append(row)
    return "\n".join(lines) + "\n"
