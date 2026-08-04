# DPA3-MACEOFF-forwardhooker

PROBE-style **multitask reliability classifier** on frozen
**[DPA3-SPICE-MACE-OFF](https://www.aissquare.com/models/detail?pageType=models&name=DPA3-SPICE-MACE-OFF&id=388)**
atomic embeddings (checkpoint **DPA3-L6**).

Same idea as PROBE-multitask + MACE-OFF23, but the backbone is **DPA3**
(DeePMD-kit). Last-layer *atomic* embeddings are captured with a
**PyTorch forward hook** (analogous to MACE `products[-1]`).

## What you get

| Script | Role |
|--------|------|
| `cache_dpa3_embeddings.py` | **One-pass** DPA3 forward; save `{structure_idx}.pt` embeddings (cache once) |
| `train_dpa3_multitask.py` | Train PROBE heads (energy + Fa + Fs) on frozen DPA3 feats |
| `infer_dpa3_multitask.py` | Test / metrics; optional embedding cache |

Cache file (`{idx}.pt`) contents (full training mode):

```text
node_feats     # [n_atoms, D]  last-layer atomic embedding (hook)
pred_energy    # scalar
pred_forces    # [n_atoms, 3]  (optional)
```

With `--embeddings-only` only `node_feats` is stored.

## Install

```bash
# DeePMD-kit ≥ 3.1 (PyTorch backend) + torch
conda env create -f environment.yml
conda activate dpa3_probe

# or pip
pip install deepmd-kit torch ase numpy tqdm
```

Download **DPA3-SPICE-MACE-OFF** from
[AIS Square](https://www.aissquare.com/models/detail?pageType=models&name=DPA3-SPICE-MACE-OFF&id=388)
and use the **DPA3-L6** checkpoint file (`.pt` / `.pth`).

## 1) Cache atomic embeddings (once)

```bash
python cache_dpa3_embeddings.py \
  --model /path/to/DPA3-L6.pt \
  --xyz /path/to/train.xyz \
  --cache-dir ./dpa3_cache_train \
  --device cuda
```

Embeddings only:

```bash
python cache_dpa3_embeddings.py \
  --model /path/to/DPA3-L6.pt \
  --xyz /path/to/test.xyz \
  --cache-dir ./dpa3_cache_test \
  --embeddings-only
```

Inspect a cache entry:

```python
import torch
x = torch.load("dpa3_cache_train/0.pt", weights_only=False)
print(x["node_feats"].shape)  # [n_atoms, feat_dim]
```

Optional: pin the hooked module if auto-detect is wrong:

```bash
python cache_dpa3_embeddings.py ... --hook-module descriptor.repflows.5
```

List module names:

```python
from probe.backends.dpa3 import load_dpa3
ext = load_dpa3("/path/to/DPA3-L6.pt", device="cpu")
for n, _ in ext.model.named_modules():
    if n:
        print(n)
```

## 2) Train multitask PROBE

```bash
python train_dpa3_multitask.py \
  --model /path/to/DPA3-L6.pt \
  --train-xyz /path/to/train.xyz \
  --output-dir ./probe_dpa3_outputs \
  --cache-dir ./dpa3_cache_train \
  --lambda-energy 1.0 \
  --lambda-force-atom 1.0 \
  --lambda-force-mol 0.3
```

- First epoch / boundary scan runs DPA3 and **fills the cache**.
- Later epochs read embeddings from RAM/disk (no DPA3 recompute).

Resume:

```bash
python train_dpa3_multitask.py --resume --cache-dir ./dpa3_cache_train ...
```

## 3) Infer

```bash
python infer_dpa3_multitask.py \
  --model /path/to/DPA3-L6.pt \
  --checkpoint ./probe_dpa3_outputs/best_multitask_model_*.pt \
  --test-xyz /path/to/test.xyz \
  --output-dir ./probe_dpa3_inference \
  --cache-dir ./dpa3_cache_test
```

Cache embeddings only (no classifier):

```bash
python infer_dpa3_multitask.py \
  --model /path/to/DPA3-L6.pt \
  --test-xyz /path/to/test.xyz \
  --cache-only \
  --cache-dir ./dpa3_cache_test \
  --embeddings-only
```

Prefer `cache_dpa3_embeddings.py` for a dedicated cache job.

## Architecture (PROBE side, unchanged)

```
Frozen DPA3 (forward hook → h_i)
        │
        ▼  atom feats [B, N, D]
  Atom encoder + self-attention
        ├─ energy reliability [B, 2]
        ├─ force-atom reliability [B, N, 2]
        └─ force-structure = mean-agg atom logits [B, 2]
```

## Input data

Extended XYZ with reference energy + forces (same as PROBE):

```
natoms
Lattice="..." Properties=... energy=...
C x y z fx fy fz
...
```

## Notes

- Units: DeePMD/DPA3 energy **eV**, forces **eV/Å** (PROBE labels use the same).
- Energy reliability errors for the boundary are reported in kcal/mol using `ev_to_kcalmol=23.06` (PROBE convention); bins are stored in eV.
- Model card: [DPA3-SPICE-MACE-OFF](https://www.aissquare.com/models/detail?pageType=models&name=DPA3-SPICE-MACE-OFF&id=388).
- DPA3 paper / DeePMD-kit ≥ 3.1: [deepmd-kit DPA3 docs](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa3.html).

## License

MIT — see [LICENSE](LICENSE). PROBE classifier code adapted from
[PROBE](https://github.com/isayevlab/PROBE) / multitask PROBE-forces.
DPA3 weights are under their AIS Square / publishers’ license (LGPL-3.0 on card).
