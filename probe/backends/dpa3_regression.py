"""Regression data and validated, content-addressed DPA3 output cache.

Keeps legacy classifier I/O/cache behavior unchanged. Reference labels are never
cached: geometry/backbone identify predictions; current labels supply targets.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_backbone(path, device):
    # DeePMD PT reads these before import; explicit CPU must also affect backbone.
    target = torch.device(device)
    if target.type not in ('cpu', 'cuda'):
        raise ValueError('DPA3 supports cpu or cuda devices')
    os.environ['DEVICE'] = target.type
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        index = target.index if target.index is not None else torch.cuda.current_device()
        os.environ['LOCAL_RANK'] = str(index)
        torch.cuda.set_device(index)
    from .dpa3 import load_dpa3
    extractor = load_dpa3(str(path), device=str(target))
    if extractor.model is not None:
        extractor.model.eval()
        for p in extractor.model.parameters():
            p.requires_grad_(False)
    # Fail rather than silently selecting an arbitrary raw hook/type mapping.
    if not extractor.prefer_eval_descriptor:
        raise ValueError('Regression requires the DeePMD eval_descriptor API (PyTorch DPA3 model)')
    wrapper = extractor.infer_wrapper
    if not hasattr(wrapper, 'get_type_map') or not wrapper.get_type_map():
        raise ValueError('DPA3 model must supply its element type map')
    extractor.type_map = list(wrapper.get_type_map())
    from importlib.metadata import version
    extractor.runtime_version = version("deepmd-kit")
    return extractor


def read_frames(path):
    from ase.io import iread
    frames = []
    for i, atoms in enumerate(iread(str(path), index=':')):
        if not len(atoms):
            raise ValueError(f'Empty structure at index {i}')
        pbc = np.asarray(atoms.pbc, dtype=bool)
        if pbc.any() and not pbc.all():
            raise ValueError('Partially periodic cells are unsupported; use molecular or fully periodic data')
        cell = np.asarray(atoms.cell, dtype=np.float64) if pbc.all() else None
        if cell is not None and abs(np.linalg.det(cell)) < 1e-12:
            raise ValueError(f'Invalid periodic cell at index {i}')
        results = {} if atoms.calc is None else atoms.calc.results
        energy = next((atoms.info[k] for k in ('REF_energy', 'DFT_energy', 'energy')
                       if k in atoms.info), results.get('energy', float('nan')))
        forces = next((atoms.arrays[k] for k in ('REF_forces', 'DFT_forces', 'forces')
                       if k in atoms.arrays), results.get('forces', np.full((len(atoms), 3), np.nan)))
        frame = dict(positions=np.asarray(atoms.positions, dtype=np.float64),
                     atomic_numbers=np.asarray(atoms.numbers, dtype=np.int32),
                     cell=cell, true_energy=float(energy),
                     true_forces=np.asarray(forces, dtype=np.float64))
        if not np.isfinite(frame['positions']).all():
            raise ValueError(f'Nonfinite positions at index {i}')
        frames.append(frame)
    if not frames:
        raise ValueError('No structures found')
    return frames


def collate_frames(frames):
    return frames


def make_loader(frames, batch_size, shuffle=False):
    return DataLoader(frames, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_frames)


def split_frames(frames, valid_fraction, seed):
    if len(frames) < 2 or not 0 < valid_fraction < 1:
        raise ValueError('Need at least two structures and 0 < valid_fraction < 1')
    indices = np.random.default_rng(seed).permutation(len(frames))
    nval = max(1, min(len(frames) - 1, int(len(frames) * valid_fraction)))
    return ([frames[i] for i in indices[nval:]], [frames[i] for i in indices[:nval]])


def geometry_digest(frame):
    h = hashlib.sha256()
    for key in ('positions', 'atomic_numbers', 'cell'):
        value = frame[key]
        h.update(key.encode())
        if value is None:
            h.update(b'none')
        else:
            a = np.ascontiguousarray(value)
            h.update(str(a.shape).encode())
            h.update(a.dtype.str.encode())
            h.update(a.tobytes())
    return h.hexdigest()


class RegressionDPA3Processor:
    """Cache each geometry separately, including mixed cache-hit/miss batches."""
    def __init__(self, extractor, backbone_digest, cache_dir=None, cache=True):
        self.extractor = extractor
        self.cache = cache
        self.memory = {}
        self.hits = self.misses = 0
        identity = dict(version=1, backbone_sha256=backbone_digest,
                        representation='dpa3_eval_descriptor', type_map=extractor.type_map,
                        deepmd_version=getattr(extractor, 'runtime_version', 'unknown'))
        self.identity = identity
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.cache_dir = Path(cache_dir)/namespace if cache_dir and cache else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir/'manifest.json').write_text(json.dumps(identity, indent=2))

    def _validate(self, entry, natoms):
        x = torch.as_tensor(entry['node_feats']).detach().cpu().float()
        e = torch.as_tensor(entry['pred_energy']).detach().cpu().float()
        f = torch.as_tensor(entry['pred_forces']).detach().cpu().float()
        if x.ndim != 2 or x.shape[0] != natoms or not x.shape[1] or e.numel() != 1 or f.shape != (natoms, 3):
            raise ValueError('Invalid DPA3 embedding/energy/force shapes')
        if not all(torch.isfinite(v).all() for v in (x, e, f)):
            raise ValueError('Nonfinite DPA3 output')
        if self.extractor.feat_dim is not None and self.extractor.feat_dim != x.shape[1]:
            raise ValueError('DPA3 embedding width changed')
        self.extractor.feat_dim = x.shape[1]
        return dict(node_feats=x.contiguous(), pred_energy=e.reshape(()), pred_forces=f.contiguous())

    def _entry(self, frame):
        key = geometry_digest(frame)
        entry = self.memory.get(key) if self.cache else None
        path = self.cache_dir/(key+'.pt') if self.cache_dir else None
        if entry is None and path is not None and path.exists():
            entry = torch.load(path, map_location='cpu', weights_only=True)
        if entry is None:
            self.misses += 1
            with torch.enable_grad():
                entry = self.extractor.forward_structure(frame['positions'], frame['atomic_numbers'],
                                                         cell=frame['cell'], compute_force=True)
            entry = self._validate(entry, len(frame['positions']))
            if path is not None:
                tmp = path.with_suffix(f'.tmp.{os.getpid()}')
                try:
                    torch.save(entry, tmp)
                    tmp.replace(path)
                finally:
                    if tmp.exists():
                        tmp.unlink()
        else:
            self.hits += 1
            entry = self._validate(entry, len(frame['positions']))
        if self.cache:
            self.memory[key] = entry
        return entry

    def __call__(self, frames, device):
        entries = [self._entry(fr) for fr in frames]
        n = max(len(e['node_feats']) for e in entries)
        b, d = len(entries), self.extractor.feat_dim
        x = torch.zeros(b, n, d, device=device)
        mask = torch.zeros(b, n, dtype=torch.bool, device=device)
        pe, te = torch.empty(b, device=device), torch.empty(b, device=device)
        pf, tf = torch.zeros(b, n, 3, device=device), torch.zeros(b, n, 3, device=device)
        for i, (entry, frame) in enumerate(zip(entries, frames)):
            k = len(entry['node_feats'])
            x[i, :k] = entry['node_feats'].to(device)
            mask[i, :k] = True
            pe[i], te[i] = entry['pred_energy'].to(device), frame['true_energy']
            pf[i, :k] = entry['pred_forces'].to(device)
            tf[i, :k] = torch.as_tensor(frame['true_forces'], dtype=torch.float32, device=device)
        return x, mask, pe, te, pf, tf, mask.sum(1).float()
