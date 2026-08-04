"""
Training utilities for PROBE.

  - uncertainty_loss_fn   — size-normalized cross-entropy
  - scalar_to_bin_index   — maps per-molecule errors → class indices
  - train_epoch           — one training epoch
  - evaluate              — evaluation loop
  - run_training          — full training loop with early stopping

These functions are backend-agnostic. They accept a `process_batch_fn`
callable that hides the MLIP-specific forward pass. See probe_aimnet2.py
and probe_mace.py for backend-specific implementations.
"""

from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from .metrics import confusion_matrix_torch, compute_all_metrics
from .plotting import update_training_curves


def atomic_torch_save(obj, path) -> None:
    """Write a checkpoint via temp file + os.replace (safe on job TIMEOUT)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def save_resumable_checkpoint(resumable: dict, output_dir: Path,
                              epoch: int, checkpoint_every: int = 0) -> Path:
    """Always rewrite last_checkpoint.pt; optionally keep epoch-named copies.

    Args:
        checkpoint_every: 0 = only last_checkpoint.pt (recommended on HPG).
            N > 0 also writes checkpoint_epoch_XXXX.pt every N epochs.
    """
    output_dir = Path(output_dir)
    last_ckpt_path = output_dir / 'last_checkpoint.pt'
    atomic_torch_save(resumable, last_ckpt_path)
    if checkpoint_every and epoch % checkpoint_every == 0:
        epoch_ckpt_path = output_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        atomic_torch_save(resumable, epoch_ckpt_path)
        print(f"  saved epoch → {epoch_ckpt_path.name}  |  "
              f"rewrote → {last_ckpt_path.name}")
    else:
        print(f"  rewrote → {last_ckpt_path.name} (epoch {epoch})")
    return last_ckpt_path


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def uncertainty_loss_fn(logits: torch.Tensor, targets: torch.Tensor,
                        n_atoms: torch.Tensor,
                        class_weights: Optional[torch.Tensor] = None,
                        label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy normalized by sqrt(n_atoms).

    Prevents large molecules from dominating the gradient.
    """
    if class_weights is not None:
        class_weights = class_weights.to(dtype=logits.dtype, device=logits.device)
    targets = targets.to(device=logits.device)
    ce = F.cross_entropy(logits, targets, weight=class_weights,
                         reduction='none', label_smoothing=label_smoothing)
    normalized = ce / n_atoms.to(dtype=logits.dtype).sqrt().clamp(min=1.0)
    return normalized.mean()


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def scalar_to_bin_index(errors: torch.Tensor,
                        bin_edges: torch.Tensor) -> torch.Tensor:
    """Map per-molecule absolute errors to class indices.

    bin_edges: 1-D tensor [0, boundary]  →  class 0 = reliable, class 1 = unreliable
    """
    bin_idx = torch.bucketize(errors, bin_edges, right=False) - 1
    bin_idx = torch.clamp(bin_idx, 0, len(bin_edges) - 1)
    return bin_idx


def compute_error_boundary(errors_kcal: np.ndarray,
                           percentile: float = 50) -> float:
    """Return the p-th percentile of the error distribution in kcal/mol."""
    boundary = float(np.percentile(errors_kcal, percentile))
    print(f"Error boundary ({percentile}th percentile): {boundary:.4f} kcal/mol")
    return boundary


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_epoch(model, process_batch_fn: Callable, dataloader,
                optimizer, error_bins: torch.Tensor, device,
                class_weights=None, label_smoothing: float = 0.0,
                gradient_clip_norm: float = 1.0) -> float:
    """Run one training epoch.

    Args:
        process_batch_fn: callable(batch, device) →
            (atom_feats [B,N,D], atom_mask [B,N], energy [B], true_energy [B], n_atoms [B])
    Returns:
        mean training loss for the epoch
    """
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        atom_feats, atom_mask, pred_energy, true_energy, n_atoms = \
            process_batch_fn(batch, device)

        abs_errors = torch.abs(true_energy - pred_energy)
        target_classes = scalar_to_bin_index(abs_errors, error_bins)
        valid = ~torch.isnan(pred_energy)
        if not valid.any():
            continue

        atom_feats_v  = atom_feats[valid]
        atom_mask_v   = atom_mask[valid]
        pred_energy_v = pred_energy[valid]
        target_v      = target_classes[valid]
        n_atoms_v     = n_atoms[valid]

        optimizer.zero_grad()
        logits = model(atom_feats_v, atom_mask_v, energy=pred_energy_v)
        loss = uncertainty_loss_fn(logits, target_v, n_atoms_v,
                                   class_weights, label_smoothing)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, process_batch_fn: Callable, dataloader,
             error_bins: torch.Tensor, device,
             n_classes: int = 2,
             high_conf_cutoffs: Optional[Dict] = None) -> dict:
    """Evaluate the model on a dataloader.

    Returns a dict with accuracy, MCC, F1, probabilities, predictions, errors, etc.
    """
    model.eval()
    all_logits, all_targets, all_errors, all_n_atoms = [], [], [], []

    for batch in tqdm(dataloader, desc='Evaluating', leave=False):
        atom_feats, atom_mask, pred_energy, true_energy, n_atoms = \
            process_batch_fn(batch, device)

        abs_errors = torch.abs(true_energy - pred_energy)
        target_classes = scalar_to_bin_index(abs_errors, error_bins)
        valid = ~torch.isnan(pred_energy)
        if not valid.any():
            continue

        atom_feats_v  = atom_feats[valid]
        atom_mask_v   = atom_mask[valid]
        pred_energy_v = pred_energy[valid]
        target_v      = target_classes[valid]
        errors_v      = abs_errors[valid]
        n_atoms_v     = n_atoms[valid]

        logits = model(atom_feats_v, atom_mask_v, energy=pred_energy_v)

        all_logits.append(logits.cpu())
        all_targets.append(target_v.cpu())
        all_errors.append(errors_v.cpu())
        all_n_atoms.append(n_atoms_v.cpu())

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_errors  = torch.cat(all_errors)
    all_n_atoms = torch.cat(all_n_atoms)

    all_probs = F.softmax(all_logits, dim=-1)
    all_preds = all_probs.argmax(dim=-1)

    cm   = confusion_matrix_torch(all_preds, all_targets, n_classes)
    loss = (F.cross_entropy(all_logits, all_targets, reduction='none') /
            all_n_atoms.float().sqrt().clamp(min=1.0)).mean().item()

    results = compute_all_metrics(cm)
    results['loss']          = loss
    results['probabilities'] = all_probs.numpy()
    results['predictions']   = all_preds.numpy()
    results['targets']       = all_targets.numpy()
    results['errors']        = all_errors.numpy()

    if high_conf_cutoffs is not None:
        from .metrics import high_confidence_analysis
        results['high_conf'] = high_confidence_analysis(
            all_probs, all_preds, all_targets, high_conf_cutoffs, n_classes)

    return results


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training(model, process_batch_fn: Callable,
                 train_loader, val_loader,
                 error_bins: torch.Tensor, device,
                 output_dir: str = './probe_outputs',
                 lr: float = 5e-5, weight_decay: float = 1e-4,
                 epochs: int = 1000,
                 early_stopping_patience: int = 25,
                 scheduler_patience: int = 5,
                 scheduler_factor: float = 0.9,
                 min_lr: float = 5e-6,
                 gradient_clip_norm: float = 1.0,
                 class_weights=None,
                 label_smoothing: float = 0.0,
                 high_conf_cutoffs: Optional[Dict] = None) -> dict:
    """
    Full training loop with validation, LR scheduling, and early stopping.

    Saves:
        best_model_<timestamp>.pt  — best model by validation loss

    Returns:
        history dict with per-epoch train/val metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=scheduler_factor,
                                  patience=scheduler_patience,
                                  min_lr=min_lr)

    best_val_loss = float('inf')
    best_state    = None
    best_epoch    = 0
    patience_ctr  = 0
    history: dict = {'train_loss': [], 'val_loss': [],
                     'val_acc': [], 'val_mcc': [], 'val_f1': []}

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, process_batch_fn, train_loader, optimizer,
            error_bins, device, class_weights, label_smoothing, gradient_clip_norm
        )
        val_results = evaluate(
            model, process_batch_fn, val_loader, error_bins, device,
            n_classes=model.n_classes, high_conf_cutoffs=high_conf_cutoffs
        )
        val_loss = val_results['loss']
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_results['accuracy'])
        history['val_mcc'].append(val_results['mcc'])
        history['val_f1'].append(val_results['f1'])

        print(f"Epoch {epoch:4d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | acc={val_results['accuracy']:.4f} | "
              f"mcc={val_results['mcc']:.4f} | f1={val_results['f1']:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
            ckpt_path = output_dir / f'best_model_{timestamp}.pt'
            torch.save({
                'model_state_dict': best_state,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_results,
                'error_bins': error_bins.cpu().tolist(),
            }, ckpt_path)
        else:
            patience_ctr += 1

        plot_path = update_training_curves(
            history, output_dir, best_epoch=best_epoch or None)
        if plot_path is not None and epoch == 1:
            print(f"  training curves → {plot_path}")

        if patience_ctr >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(best epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history['best_epoch'] = best_epoch
    return history


# ---------------------------------------------------------------------------
# Multitask (energy + force) training
# ---------------------------------------------------------------------------

def atom_force_loss_fn(logits_atom: torch.Tensor, targets_atom: torch.Tensor,
                       atom_mask: torch.Tensor, n_atoms: torch.Tensor,
                       class_weights=None, label_smoothing: float = 0.0) -> torch.Tensor:
    """Size-normalized cross-entropy over valid atoms per molecule."""
    B, N, C = logits_atom.shape
    logits_flat = logits_atom.reshape(B * N, C)
    targets_flat = targets_atom.reshape(B * N).to(device=logits_atom.device)

    if class_weights is not None:
        class_weights = class_weights.to(dtype=logits_atom.dtype,
                                        device=logits_atom.device)
    ce = F.cross_entropy(
        logits_flat, targets_flat, weight=class_weights,
        reduction='none', label_smoothing=label_smoothing,
    ).reshape(B, N)
    ce = ce.masked_fill(~atom_mask, 0.0)
    per_mol = ce.sum(dim=1) / atom_mask.sum(dim=1).clamp(min=1).float()
    normalized = per_mol / n_atoms.to(dtype=logits_atom.dtype).sqrt().clamp(min=1.0)
    return normalized.mean()


def multitask_loss_fn(logits_energy: torch.Tensor, logits_force_atom: torch.Tensor,
                      logits_force_mol: torch.Tensor,
                      target_energy: torch.Tensor, target_force_atom: torch.Tensor,
                      target_force_mol: torch.Tensor,
                      atom_mask: torch.Tensor, n_atoms: torch.Tensor,
                      lambda_energy: float = 1.0,
                      lambda_force_atom: float = 1.0,
                      lambda_force_mol: float = 1.0,
                      class_weights=None, label_smoothing: float = 0.0) -> dict:
    """Combined multitask loss with per-task breakdown."""
    loss_e = uncertainty_loss_fn(
        logits_energy, target_energy, n_atoms, class_weights, label_smoothing)
    loss_fa = atom_force_loss_fn(
        logits_force_atom, target_force_atom, atom_mask, n_atoms,
        class_weights, label_smoothing)
    loss_fs = uncertainty_loss_fn(
        logits_force_mol, target_force_mol, n_atoms, class_weights, label_smoothing)

    total = (lambda_energy * loss_e + lambda_force_atom * loss_fa +
             lambda_force_mol * loss_fs)
    return {
        'total': total,
        'energy': loss_e,
        'force_atom': loss_fa,
        'force_mol': loss_fs,
    }


def _multitask_targets_from_batch(pred_energy, true_energy, pred_forces, true_forces,
                                  atom_mask, error_bins_e, error_bins_f_atom,
                                  error_bins_f_mol):
    from .labels import (atom_force_component_mae, structure_mean_force_error,
                         scalar_to_bin_index)

    abs_e = torch.abs(true_energy - pred_energy)
    atom_err = atom_force_component_mae(pred_forces, true_forces)
    struct_err = structure_mean_force_error(atom_err, atom_mask)

    target_e = scalar_to_bin_index(abs_e, error_bins_e)
    target_f_atom = scalar_to_bin_index(atom_err, error_bins_f_atom)
    target_f_mol = scalar_to_bin_index(struct_err, error_bins_f_mol)
    return target_e, target_f_atom, target_f_mol, abs_e, struct_err, atom_err


def train_epoch_multitask(model, process_batch_fn, dataloader, optimizer,
                          error_bins_e, error_bins_f_atom, error_bins_f_mol,
                          device, lambda_energy=1.0, lambda_force_atom=1.0,
                          lambda_force_mol=1.0, class_weights=None,
                          label_smoothing=0.0, gradient_clip_norm=1.0) -> dict:
    """One multitask training epoch."""
    model.train()
    totals = {'total': 0.0, 'energy': 0.0, 'force_atom': 0.0, 'force_mol': 0.0}
    n_batches = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        (atom_feats, atom_mask, pred_energy, true_energy,
         pred_forces, true_forces, n_atoms) = process_batch_fn(batch, device)

        valid = ~torch.isnan(pred_energy)
        if not valid.any():
            continue

        atom_feats = atom_feats[valid]
        atom_mask = atom_mask[valid]
        pred_energy = pred_energy[valid]
        pred_forces = pred_forces[valid]
        true_forces = true_forces[valid]
        n_atoms = n_atoms[valid]

        target_e, target_fa, target_fm, _, _, _ = _multitask_targets_from_batch(
            pred_energy, true_energy[valid], pred_forces, true_forces,
            atom_mask, error_bins_e, error_bins_f_atom, error_bins_f_mol)

        optimizer.zero_grad()
        logits_e, logits_fa, logits_fm = model(atom_feats, atom_mask, energy=pred_energy)
        losses = multitask_loss_fn(
            logits_e, logits_fa, logits_fm,
            target_e, target_fa, target_fm,
            atom_mask, n_atoms,
            lambda_energy, lambda_force_atom, lambda_force_mol,
            class_weights, label_smoothing,
        )
        losses['total'].backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in totals.items()}


@torch.no_grad()
def evaluate_multitask(model, process_batch_fn, dataloader,
                       error_bins_e, error_bins_f_atom, error_bins_f_mol,
                       device, n_classes: int = 2,
                       lambda_energy: float = 1.0,
                       lambda_force_atom: float = 1.0,
                       lambda_force_mol: float = 1.0,
                       high_conf_cutoffs: Optional[Dict] = None) -> dict:
    """Evaluate multitask model; returns per-task metrics."""
    model.eval()
    tasks = ('energy', 'force_atom', 'force_mol')
    store = {t: {'logits': [], 'targets': [], 'errors': []} for t in tasks}
    store['force_atom']['batch_losses'] = []
    all_n_atoms = []

    for batch in tqdm(dataloader, desc='Evaluating', leave=False):
        (atom_feats, atom_mask, pred_energy, true_energy,
         pred_forces, true_forces, n_atoms) = process_batch_fn(batch, device)

        valid = ~torch.isnan(pred_energy)
        if not valid.any():
            continue

        atom_feats = atom_feats[valid]
        atom_mask = atom_mask[valid]
        pred_energy = pred_energy[valid]
        pred_forces = pred_forces[valid]
        true_forces = true_forces[valid]
        n_atoms_v = n_atoms[valid]

        target_e, target_fa, target_fm, err_e, err_fm, err_fa = \
            _multitask_targets_from_batch(
                pred_energy, true_energy[valid], pred_forces, true_forces,
                atom_mask, error_bins_e, error_bins_f_atom, error_bins_f_mol)

        logits_e, logits_fa, logits_fm = model(atom_feats, atom_mask, energy=pred_energy)

        store['energy']['logits'].append(logits_e.cpu())
        store['energy']['targets'].append(target_e.cpu())
        store['energy']['errors'].append(err_e.cpu())

        store['force_mol']['logits'].append(logits_fm.cpu())
        store['force_mol']['targets'].append(target_fm.cpu())
        store['force_mol']['errors'].append(err_fm.cpu())

        # Flatten valid atoms — N_max varies across batches so [B, N, *] cannot be cat'd.
        mask_cpu = atom_mask.cpu()
        store['force_atom']['logits'].append(logits_fa.cpu()[mask_cpu])
        store['force_atom']['targets'].append(target_fa.cpu()[mask_cpu])
        store['force_atom']['errors'].append(err_fa.cpu()[mask_cpu])
        store['force_atom']['batch_losses'].append(
            atom_force_loss_fn(logits_fa, target_fa, atom_mask, n_atoms_v).item())
        all_n_atoms.append(n_atoms_v.cpu())

    all_n_atoms = torch.cat(all_n_atoms)
    results = {'per_task': {}, 'loss': 0.0}
    loss_sum = 0.0
    task_lambdas = {
        'energy': lambda_energy,
        'force_atom': lambda_force_atom,
        'force_mol': lambda_force_mol,
    }

    for task in tasks:
        if task == 'force_atom':
            logits = torch.cat(store['force_atom']['logits'], dim=0)
            targets = torch.cat(store['force_atom']['targets'], dim=0)
            errors = torch.cat(store['force_atom']['errors'], dim=0)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            cm = confusion_matrix_torch(preds, targets, n_classes)
            task_loss = float(np.mean(store['force_atom']['batch_losses']))
        else:
            logits = torch.cat(store[task]['logits'])
            targets = torch.cat(store[task]['targets'])
            errors = torch.cat(store[task]['errors'])
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            cm = confusion_matrix_torch(preds, targets, n_classes)
            task_loss = (F.cross_entropy(logits, targets, reduction='none') /
                         all_n_atoms.float().sqrt().clamp(min=1.0)).mean().item()

        metrics = compute_all_metrics(cm)
        metrics['loss'] = task_loss
        metrics['probabilities'] = probs.numpy()
        metrics['predictions'] = preds.numpy()
        metrics['targets'] = targets.numpy()
        metrics['errors'] = errors.numpy()

        if high_conf_cutoffs is not None and task != 'force_atom':
            from .metrics import high_confidence_analysis
            metrics['high_conf'] = high_confidence_analysis(
                probs, preds, targets, high_conf_cutoffs, n_classes)

        results['per_task'][task] = metrics
        loss_sum += task_lambdas[task] * task_loss

    results['loss'] = loss_sum
    results['accuracy'] = results['per_task']['energy']['accuracy']
    results['mcc'] = results['per_task']['energy']['mcc']
    results['f1'] = results['per_task']['energy']['f1']
    return results


def run_multitask_training(model, process_batch_fn, train_loader, val_loader,
                           error_bins_e, error_bins_f_atom, error_bins_f_mol,
                           device, output_dir='./probe_outputs',
                           lr=5e-5, weight_decay=1e-4, epochs=1000,
                           early_stopping_patience=25, scheduler_patience=5,
                           scheduler_factor=0.9, min_lr=5e-6,
                           gradient_clip_norm=1.0,
                           lambda_energy=1.0, lambda_force_atom=1.0,
                           lambda_force_mol=1.0,
                           class_weights=None, label_smoothing=0.0,
                           high_conf_cutoffs: Optional[Dict] = None,
                           resume_path: Optional[str] = None,
                           checkpoint_every: int = 1) -> dict:
    """Full multitask training with best + periodic last checkpoints and resume."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor,
                                  patience=scheduler_patience, min_lr=min_lr)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    patience_ctr = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_loss_energy': [], 'train_loss_force_atom': [], 'train_loss_force_mol': [],
        'val_acc_energy': [], 'val_acc_force_atom': [], 'val_acc_force_mol': [],
    }
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_epoch = 1

    if resume_path:
        resume_path = Path(resume_path)
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        best_val_loss = float(ckpt.get('best_val_loss', float('inf')))
        best_epoch = int(ckpt.get('best_epoch', 0))
        patience_ctr = int(ckpt.get('patience_ctr', 0))
        if ckpt.get('best_state') is not None:
            best_state = ckpt['best_state']
        if ckpt.get('history'):
            history = ckpt['history']
        if ckpt.get('timestamp'):
            timestamp = ckpt['timestamp']
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        print(f"Resume: next epoch={start_epoch}, best_epoch={best_epoch}, "
              f"best_val_loss={best_val_loss:.4f}, patience={patience_ctr}")

    for epoch in range(start_epoch, epochs + 1):
        train_losses = train_epoch_multitask(
            model, process_batch_fn, train_loader, optimizer,
            error_bins_e, error_bins_f_atom, error_bins_f_mol, device,
            lambda_energy, lambda_force_atom, lambda_force_mol,
            class_weights, label_smoothing, gradient_clip_norm,
        )
        val_results = evaluate_multitask(
            model, process_batch_fn, val_loader,
            error_bins_e, error_bins_f_atom, error_bins_f_mol, device,
            n_classes=model.n_classes,
            lambda_energy=lambda_energy,
            lambda_force_atom=lambda_force_atom,
            lambda_force_mol=lambda_force_mol,
            high_conf_cutoffs=high_conf_cutoffs,
        )
        val_loss = val_results['loss']
        scheduler.step(val_loss)

        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_loss)
        history['train_loss_energy'].append(train_losses['energy'])
        history['train_loss_force_atom'].append(train_losses['force_atom'])
        history['train_loss_force_mol'].append(train_losses['force_mol'])
        for task in ('energy', 'force_atom', 'force_mol'):
            history[f'val_acc_{task}'].append(val_results['per_task'][task]['accuracy'])

        print(
            f"Epoch {epoch:4d} | train={train_losses['total']:.4f} "
            f"(E={train_losses['energy']:.4f}, Fa={train_losses['force_atom']:.4f}, "
            f"Fs={train_losses['force_mol']:.4f}) | val={val_loss:.4f} | "
            f"acc E/Fa/Fs="
            f"{val_results['per_task']['energy']['accuracy']:.3f}/"
            f"{val_results['per_task']['force_atom']['accuracy']:.3f}/"
            f"{val_results['per_task']['force_mol']['accuracy']:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            ckpt_path = output_dir / f'best_multitask_model_{timestamp}.pt'
            torch.save({
                'model_state_dict': best_state,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': {
                    k: v for k, v in val_results.items()
                    if k in ('loss', 'accuracy', 'mcc', 'f1')
                },
                'error_bins_energy': error_bins_e.cpu().tolist(),
                'error_bins_force_atom': error_bins_f_atom.cpu().tolist(),
                'error_bins_force_mol': error_bins_f_mol.cpu().tolist(),
                'lambda_energy': lambda_energy,
                'lambda_force_atom': lambda_force_atom,
                'lambda_force_mol': lambda_force_mol,
            }, ckpt_path)
            print(f"  saved best → {ckpt_path}")
        else:
            patience_ctr += 1

        # Rewrite last_checkpoint.pt every epoch (atomic) for HPG TIMEOUT resume.
        resumable = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'best_state': best_state,
            'patience_ctr': patience_ctr,
            'history': history,
            'timestamp': timestamp,
            'error_bins_energy': error_bins_e.cpu().tolist(),
            'error_bins_force_atom': error_bins_f_atom.cpu().tolist(),
            'error_bins_force_mol': error_bins_f_mol.cpu().tolist(),
            'lambda_energy': lambda_energy,
            'lambda_force_atom': lambda_force_atom,
            'lambda_force_mol': lambda_force_mol,
        }
        save_resumable_checkpoint(
            resumable, output_dir, epoch, checkpoint_every=checkpoint_every)

        plot_path = update_training_curves(
            history, output_dir, best_epoch=best_epoch or None)
        if plot_path is not None and epoch == start_epoch:
            print(f"  training curves → {plot_path}")

        if patience_ctr >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(best epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history['best_epoch'] = best_epoch
    return history
