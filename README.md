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
| `save_atomic_dpa3.py` | Same as MACE `save_atomic_092.py`: dump last-layer feats to one `.npz` (GMM/KDE) |
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

DPA3 **does support GPU acceleration** through DeePMD-kit’s PyTorch + CUDA
backend. Use the B200 recipe below on HyperGator B200 nodes.

### B200 / GPU install (recommended)

Blackwell (B200, `sm_100`) needs **CUDA ≥ 12.8**, a matching **PyTorch CUDA
wheel**, and DeePMD built with:

```bash
export DP_VARIANT=cuda
export DP_ENABLE_PYTORCH=1
```

On HyperGator:

```bash
module load cuda/12.8.0   # or newest >=12.8 on the cluster
# optional: module load gcc/... if the deepmd build needs it

cd DPA3-MACEOFF-forwardhooker-
bash scripts/install_b200.sh
conda activate dpa3_probe_b200
```

Sanity check:

```bash
python - <<'PY'
import torch, deepmd
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("deepmd", deepmd.__version__)
PY
```

Manual alternative (same stack):

```bash
conda env create -f environment_b200.yml
conda activate dpa3_probe_b200

# PyTorch for CUDA 12.8 (change cu128 → cu129 if needed)
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

export DP_VARIANT=cuda DP_ENABLE_PYTORCH=1
export CUDAToolkit_ROOT=$CUDA_HOME CUDA_PATH=$CUDA_HOME
pip install "deepmd-kit[torch]>=3.1.0"
# If GPU ops fail on B200, build from source:
#   pip install -e "git+https://github.com/deepmodeling/deepmd-kit.git#egg=deepmd-kit[torch]"
```

Files:

| File | When to use |
|------|-------------|
| `environment_b200.yml` + `scripts/install_b200.sh` | **B200 / CUDA GPU** |
| `environment.yml` | CPU / quick smoke test only |

### CPU / generic (not for B200)

```bash
conda env create -f environment.yml
conda activate dpa3_probe
```

Download **DPA3-SPICE-MACE-OFF** from
[AIS Square](https://www.aissquare.com/models/detail?pageType=models&name=DPA3-SPICE-MACE-OFF&id=388)
and use the **DPA3-L6** checkpoint file (`.pt` / `.pth`).

Always pass `--device cuda` (or set `CONFIG["device"]="cuda"`) when running on GPU.

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
