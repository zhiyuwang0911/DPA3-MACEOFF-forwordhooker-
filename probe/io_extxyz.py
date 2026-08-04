"""
Extended XYZ I/O for PROBE training data.

Reads structure inputs and reference labels from extxyz:
  - species, positions  → MACE graph input
  - energy              → reference energy (labeling)
  - forces              → reference forces (labeling)

Extra columns (e.g. MACE_forces) and info fields (e.g. MACE_energy) are
ignored — MACE predictions always come from a live model forward pass,
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


def read_probe_extxyz_frame(lines: list[str], start: int = 0) -> tuple[dict, int]:
    """
    Parse one extxyz frame starting at `start` (line index of natoms line).

    Atom columns: species  x  y  z  fx  fy  fz  [optional extra columns ignored]
    """
    n_atoms = int(lines[start].strip())
    info = parse_extxyz_info(lines[start + 1])

    symbols, positions, true_forces = [], [], []
    for i in range(n_atoms):
        parts = lines[start + 2 + i].split()
        if len(parts) < 7:
            raise ValueError(
                f"Expected >=7 columns per atom (species pos forces), "
                f"got {len(parts)} at line {start + 3 + i}"
            )
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        true_forces.append([float(parts[4]), float(parts[5]), float(parts[6])])

    frame = {
        'n_atoms': n_atoms,
        'symbols': symbols,
        'positions': np.asarray(positions, dtype=np.float64),
        'true_forces': np.asarray(true_forces, dtype=np.float64),
        'true_energy': _get_reference_energy(info),
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
