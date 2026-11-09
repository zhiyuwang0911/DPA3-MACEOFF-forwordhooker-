"""Deterministic, train-only PCA for frozen atomic representations.

Store PCA buffers in the model checkpoint; inference never refits PCA.
"""
import torch
from torch import nn


class PCAProjection(nn.Module):
    def __init__(self, mean, components):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        components = torch.as_tensor(components, dtype=torch.float32)
        if mean.ndim != 1 or components.ndim != 2 or components.shape[1] != len(mean):
            raise ValueError("PCA mean/components have incompatible shapes")
        self.register_buffer("mean", mean)
        self.register_buffer("components", components)

    @property
    def output_dim(self):
        return self.components.shape[0]

    def forward(self, features, mask=None):
        if features.shape[-1] != self.mean.numel():
            raise ValueError(f"Expected raw features of dimension {self.mean.numel()}, got {features.shape[-1]}")
        # Compute in float32, matching PCA training; zero padding AFTER centering.
        reduced = (features.float() - self.mean) @ self.components.T
        if mask is not None:
            reduced = reduced.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        return reduced


class ModelWithPCA(nn.Module):
    def __init__(self, probe, projection):
        super().__init__()
        self.probe = probe
        self.pca = projection

    @property
    def n_classes(self):
        return self.probe.n_classes

    def forward(self, atom_feats, atom_mask, *args, **kwargs):
        return self.probe(self.pca(atom_feats, atom_mask), atom_mask, *args, **kwargs)

    def get_atom_importance(self, atom_feats, atom_mask):
        return self.probe.get_atom_importance(self.pca(atom_feats, atom_mask), atom_mask)


def fit_streaming_pca(processor, loader, device, n_components, feat_dim,
                      max_atoms=None):
    """Fit PCA from training atoms only, via streaming covariance (no sklearn needed).

    A shared PCA file is advisable when comparing classifier/regressor coordinates.
    """
    if not 1 <= n_components <= feat_dim:
        raise ValueError(f"PCA dimension must be in [1, {feat_dim}]")
    total = 0
    sums = torch.zeros(feat_dim, dtype=torch.float64)
    cross = torch.zeros((feat_dim, feat_dim), dtype=torch.float64)
    for batch in loader:
        feats, mask, *_ = processor(batch, device)
        x = feats.detach().cpu()[mask.detach().cpu().bool()].to(torch.float64)
        x = x[torch.isfinite(x).all(1)]
        if max_atoms is not None:
            x = x[:max(max_atoms - total, 0)]
        if x.numel():
            sums += x.sum(0)
            cross += x.T @ x
            total += len(x)
        if max_atoms is not None and total >= max_atoms:
            break
    if total <= n_components:
        raise ValueError(f"PCA needs more than {n_components} training atoms; got {total}")
    mean = sums / total
    cov = (cross - total * torch.outer(mean, mean)) / (total - 1)
    cov = (cov + cov.T) / 2
    vals, vecs = torch.linalg.eigh(cov)
    components = vecs[:, -n_components:].flip(1).T.contiguous()
    # Eigenvector signs are arbitrary: choose deterministic orientation.
    indexes = components.abs().argmax(1)
    signs = torch.sign(components[torch.arange(n_components), indexes])
    components *= signs[:, None]
    return PCAProjection(mean, components), dict(
        n_training_atoms=total, explained_variance=vals[-n_components:].flip(0).tolist(),
        total_variance=max(vals.sum().item(), 0.0))


def load_pca_artifact(path, n_components, original_dim):
    """Load a prefit sklearn PCA joblib or torch PCA artifact; never fit on test data."""
    from pathlib import Path
    if Path(path).suffix in ('.joblib', '.pkl'):
        import joblib
        obj = joblib.load(path)
        mean = obj.mean_
        components = obj.components_
    else:
        obj = torch.load(path, map_location='cpu', weights_only=True)
        mean, components = obj['mean'], obj['components']
    if len(mean) != original_dim or len(components) < n_components:
        raise ValueError("PCA artifact dimension incompatible with raw atomic embeddings or requested PCA dimension")
    return PCAProjection(mean, components[:n_components])


def make_pca(processor, train_loader, device, n_components, original_dim,
             pca_model=None, max_atoms=None):
    if n_components is None:
        if pca_model:
            raise ValueError("--pca-model also requires --pca-dim")
        return None, None
    if not 1 <= n_components <= original_dim:
        raise ValueError(f"PCA dimension must be in [1, {original_dim}]")
    if max_atoms is not None and max_atoms <= n_components:
        raise ValueError("--pca-max-atoms must exceed --pca-dim")
    if pca_model:
        return load_pca_artifact(pca_model, n_components, original_dim), {'source': str(pca_model)}
    return fit_streaming_pca(processor, train_loader, device, n_components,
                             original_dim, max_atoms=max_atoms)


def restore_model(class_type, checkpoint, device):
    """Rebuild with saved projection; legacy checkpoints without PCA remain supported."""
    cfg = checkpoint['model_config']
    probe = class_type(**cfg)
    if checkpoint.get('pca_dim') is not None:
        state = checkpoint['model_state_dict']
        projection = PCAProjection(state['pca.mean'], state['pca.components'])
        if cfg['backbone_dim'] != projection.output_dim:
            raise ValueError("PCA dimension and model input dimension disagree")
        model = ModelWithPCA(probe, projection)
    else:
        model = probe
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)
