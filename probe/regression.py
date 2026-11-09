"""Continuous error prediction; all outputs and targets use native eV / eV/Å."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import PROBEModel, build_mlp


class ErrorRegressor(PROBEModel):
    def __init__(self, backbone_dim, atom_encoder_hidden=(256, 128),
                 atom_encoder_output_dim=256, mol_attention_heads=32,
                 classifier_hidden=(256, 128, 32),
                 atom_force_head_hidden=(128, 32), dropout=0.1):
        super().__init__(backbone_dim, n_classes=1,
                         atom_encoder_hidden=atom_encoder_hidden,
                         atom_encoder_output_dim=atom_encoder_output_dim,
                         mol_attention_heads=mol_attention_heads,
                         classifier_hidden=classifier_hidden, dropout=dropout)
        self.atom_force_head = build_mlp(atom_encoder_output_dim,
                                        atom_force_head_hidden, 1, dropout)

    def forward(self, atom_feats, atom_mask, energy=None):
        attended = self.encode_atoms(atom_feats, atom_mask)
        energy_error = F.softplus(self.pool_and_classify(
            attended, atom_mask, energy).squeeze(-1))
        atom_error = F.softplus(self.atom_force_head(attended).squeeze(-1))
        atom_error = atom_error.masked_fill(~atom_mask, 0)
        structure_error = atom_error.sum(1) / atom_mask.sum(1).clamp(min=1)
        return energy_error, atom_error, structure_error


def error_targets(pred_e, true_e, pred_f, true_f, mask):
    """Absolute total energy error and mean absolute force component error."""
    e = (pred_e - true_e).abs()
    fa = (pred_f - true_f).abs().mean(-1).masked_fill(~mask, 0)
    fm = fa.sum(1) / mask.sum(1).clamp(min=1)
    return e, fa, fm


def regression_loss(predictions, targets, mask, weights=(1., 1., 1.)):
    """MSE per task; missing labels excluded independently before arithmetic."""
    losses = []
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        valid = torch.isfinite(target)
        if i == 1:
            valid = valid & mask
        if valid.any():
            losses.append(weights[i] * F.mse_loss(pred[valid], target[valid]))
    if not losses:
        return None
    return sum(losses)


def regression_metrics(pred, target):
    valid = torch.isfinite(target) & torch.isfinite(pred)
    pred, target = pred[valid].double(), target[valid].double()
    if not target.numel():
        return {'n': 0, 'mae': None, 'rmse': None, 'r2': None}
    residual = pred - target
    denominator = ((target - target.mean()) ** 2).sum()
    return {'n': target.numel(), 'mae': residual.abs().mean().item(),
            'rmse': residual.square().mean().sqrt().item(),
            'r2': (1 - residual.square().sum() / denominator).item()
                  if target.numel() > 1 and denominator > 0 else None}


def run_epoch(model, processor, loader, device, weights, optimizer=None):
    model.train(optimizer is not None)
    count = 0
    collected = {k: [[], []] for k in ('energy', 'force_atom', 'force_structure')}
    for batch in loader:
        # MACE forces require positional autograd even during evaluation.
        feats, mask, pe, te, pf, tf, _ = processor(batch, device)
        feats, pe, pf = feats.detach().float(), pe.detach().float(), pf.detach().float()
        valid = torch.isfinite(pe) & mask.any(1)
        valid &= torch.isfinite(feats).all(-1).masked_fill(~mask, True).all(1)
        valid &= torch.isfinite(pf).all(-1).masked_fill(~mask, True).all(1)
        if not valid.any():
            continue
        feats, mask, pe, te, pf, tf = [x[valid] for x in (feats, mask, pe, te, pf, tf)]
        targets = tuple(x.detach() for x in error_targets(pe, te, pf, tf, mask))
        with torch.set_grad_enabled(optimizer is not None):
            predictions = model(feats, mask, pe)
            loss = regression_loss(predictions, targets, mask, weights)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite regression loss')
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
        count += 1
        for i, key in enumerate(collected):
            p, t = predictions[i].detach().cpu(), targets[i].cpu()
            if i == 1:
                p, t = p[mask.cpu()], t[mask.cpu()]
            collected[key][0].append(p)
            collected[key][1].append(t)
    if not count:
        raise ValueError('No batches with finite inputs and reference targets')
    metrics = {key: regression_metrics(torch.cat(p), torch.cat(t))
               for key, (p, t) in collected.items()}
    # Global task MSE avoids validation depending on minibatch sizes.
    objective = sum(w * m['rmse'] ** 2 for w, m in zip(weights, metrics.values())
                    if m['n'])
    return objective, metrics
