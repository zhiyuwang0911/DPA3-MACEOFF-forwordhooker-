# DPA3 PROBE error regressor: full embeddings or PCA

These new entry points adapt the continuous error regressor from
[PROBE-regression](https://github.com/zhiyuwang0911/PROBE-regression) to this
repository's frozen DPA3 descriptor extractor. Existing multitask classification
scripts are unchanged. MACE weights and PROBE classifier weights cannot be
loaded as DPA3 regressor checkpoints.

## Environment and data

Use the existing `environment.yml` or B200 environment/install instructions in
README.md. The regressor needs Python >=3.10, PyTorch, NumPy, ASE, tqdm and a
PyTorch DeePMD-kit DPA3 checkpoint (e.g. DPA3-L6). Optional sklearn PCA artifact
loading needs `joblib`; the torch PCA fit itself does not require sklearn.

The regression adapter requires DeePMD's `eval_descriptor` interface and a model
with a type map. It uses the same per-atom DPA3 descriptor representation as this
repository's default extractor, not an energy head output or a fixed assumed
embedding width. Full dimensionality is discovered from a training structure.

Use standard extended XYZ with `Properties` specifying atom columns, positions
in Angstrom, reference total energy in eV and forces in eV/Angstrom. Example:

```text
2
Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.23 pbc="F F F"
H 0.0 0.0 0.0 0.1 0.0 0.0
H 0.0 0.0 0.7 -0.1 0.0 0.0
```

ASE reads the declared properties, including reordered/extra columns. Reference
aliases `REF_energy`/`DFT_energy` and `REF_forces`/`DFT_forces` are supported.
Reference aliases take priority over generic energy/force fields. Backbone
predictions are always computed by frozen DPA3, never taken from input labels.
Missing references are excluded separately for each regression task; each
training/validation split needs at least one finite target for a meaningful run.
No-label inference is supported. Nonperiodic frames use no cell even when a
Lattice is present; fully periodic frames need a nonsingular cell. Partially
periodic cells are rejected rather than silently treated as 3D-periodic.

## Train using the full embedding

```bash
python train_dpa3_regression.py \
  --model /path/to/DPA3-L6.pt \
  --train-xyz /path/to/train.xyz \
  --output-dir ./dpa3_regression_full \
  --cache-dir ./dpa3_regression_cache \
  --batch-size 64 --epochs 1000 --patience 20 \
  --loss-weights 1.0 1.0 0.3 \
  --device cuda
```

No PCA flag means full-dimensional atomic descriptors. The pretrained DPA3
backbone is frozen; only PROBE is trained. Outputs are nonnegative predictions
of absolute total energy error, per-atom mean absolute force component error,
and structure mean force error (the mean of per-atom predictions). These are
continuous errors, not reliability classes or corrected energy/force predictions.
The three loss weights apply in that order. Default weights are all 1.0.

## Train with PCA

```bash
python train_dpa3_regression.py \
  --model /path/to/DPA3-L6.pt \
  --train-xyz /path/to/train.xyz \
  --pca-dim 32 \
  --output-dir ./dpa3_regression_pca32 \
  --cache-dir ./dpa3_regression_cache \
  --loss-weights 1.0 1.0 0.3 --device cuda
```

Choose 8, 32, 64 or another positive width no greater than the raw descriptor
width. Each width requires its own new training run/output directory. Existing
full-dimensional weights cannot be converted simply by adding `--pca-dim`.

PCA fits only training atoms after the seeded train/validation split. Padded and
nonfinite embeddings are excluded; validation/test data never fit the basis.
The streaming float64 covariance algorithm requires more training atoms than
retained components. `--pca-max-atoms 1000000` caps the fit to the first training
atoms encountered, not a uniformly random sample.

To use an existing training-only basis, add `--pca-model /path/to/pca.pt` with
`--pca-dim`. The torch artifact contains `mean` (raw width) and `components`
(rows of PCA directions); trusted sklearn `.joblib`/`.pkl` PCA artifacts also
work with joblib installed. The first requested rows are retained. A MACE PCA
basis is not a DPA3 basis, even if the widths happen to match.

## Inference: identical command for full and PCA checkpoints

```bash
python infer_dpa3_regression.py \
  --model /path/to/DPA3-L6.pt \
  --checkpoint ./dpa3_regression_pca32/best_regressor.pt \
  --test-xyz /path/to/test.xyz \
  --output-dir ./dpa3_regression_predictions \
  --cache-dir ./dpa3_regression_cache \
  --device cuda
```

For a full-dimensional run, select its `best_regressor.pt` instead. Architecture
and PCA mean/components are stored in each checkpoint and restored automatically.
Inference never fits PCA or needs a separate projection file. The checkpoint
checks the model file SHA-256, descriptor type map, embedding width, extraction
format and DeePMD version to prevent mixing incompatible backbones/coordinates.
Keep the same DeePMD version when reusing a checkpoint.

Outputs:

- `predictions_structure_regression.csv`: predicted/reference total energy error
  and mean force error, with explicit native units.
- `predictions_atom_regression.csv`: predicted/reference per-atom force errors.
- `metrics_regression.json`: MAE, RMSE, R2 and finite sample counts for each task.
  Missing references are blank in CSV; undefined metrics are JSON null.

## Fast cache and resume

RAM caching is on by default. Optional `--cache-dir` persists raw descriptors,
DPA3 predicted energies and forces. Each geometry is computed on its first cache
miss; mixed-hit batches compute only missing structures. Later epochs reuse
outputs, including during PCA fitting. Some DeePMD versions require a separate
descriptor forward pass on the first miss; no DPA3 pass is needed for cache hits.
The cache holds raw descriptors so full/PCA runs can share it.

Cache directories are namespaced by model file hash, type map, DeePMD version
and representation. Entries are keyed by atomic numbers, positions and cell,
not row index. Changing dataset order cannot attach another geometry's embedding
to a target. Reference labels are read from the current dataset and never cached.
These regression caches are separate from the legacy numbered classifier caches;
old `{idx}.pt` entries are not imported. RAM holds all encountered raw outputs;
large datasets may require substantial host memory. `--no-cache-dpa3` disables
both RAM and disk caching and recomputes every pass.

```bash
python train_dpa3_regression.py \
  --model /path/to/DPA3-L6.pt --train-xyz /path/to/train.xyz \
  --output-dir ./dpa3_regression_pca32 --cache-dir ./dpa3_regression_cache \
  --loss-weights 1.0 1.0 0.3 --device cuda \
  --resume ./dpa3_regression_pca32/last_checkpoint.pt
```

Resume restores model/PCA, optimizer, scheduler, history and torch RNG state.
Keep the same model/data contents, split fraction, seed and loss weights.
Architecture comes from the checkpoint. `--pca-dim` may be omitted or must match;
omit `--pca-model` and `--pca-max-atoms` on resume. Use the original batch size for
consistent optimization. Checkpoints are saved atomically after each epoch as
`last_checkpoint.pt`; improved validation loss also writes `best_regressor.pt`.

All flags and architecture overrides:

```bash
python train_dpa3_regression.py --help
python infer_dpa3_regression.py --help
```

Defaults: batch size 64, epochs 1000, validation fraction 0.1, seed 42,
learning rate 5e-5, early-stopping patience 20, dropout 0.1. Device defaults to
CUDA if available, otherwise CPU. `--model` also accepts the alias `--dpa3-model`.

## Validation

```bash
python -m pip install pytest
python -m pytest tests -q
```

Tests exercise regression math/gradients, projection checkpoint restoration,
training-only PCA, full/PCA training-resume-inference with a deterministic mock
DPA3 extractor, disk cache hits, mixed-hit batches, changed-model rejection,
missing references, extxyz Properties parsing and periodicity handling.
Real DPA3-L6/DeePMD execution and GPU scientific accuracy must be validated with
your pretrained model and representative train/test data. No accuracy claim is
made from the mocked integration tests.

API references: [DeePMD Python inference](https://docs.deepmodeling.com/projects/deepmd/en/stable/inference/python.html)
and [DeepEval descriptor API](https://docs.deepmodeling.org/projects/deepmd/en/latest/autoapi/deepmd/infer/deep_eval/index.html).
