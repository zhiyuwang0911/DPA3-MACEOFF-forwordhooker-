"""Training curve plots rewritten after each epoch."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def update_training_curves(history: dict, output_dir,
                           best_epoch: Optional[int] = None,
                           filename: str = 'training_curves.png') -> Optional[Path]:
    """Overwrite a loss/accuracy figure in ``output_dir`` from ``history``.

    Supports energy-only, multitask, and force-only history keys.
    Returns the saved path, or None if matplotlib is unavailable / plot fails.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (skip training plot: matplotlib not installed)')
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loss = history.get('train_loss') or []
    val_loss = history.get('val_loss') or []
    n = len(train_loss)
    if n == 0:
        return None
    epochs = list(range(1, n + 1))

    acc_series = []
    if history.get('val_acc'):
        acc_series.append(('val acc', history['val_acc']))
    for key, label in (
        ('val_acc_energy', 'val acc E'),
        ('val_acc_force_atom', 'val acc Fa'),
        ('val_acc_force_mol', 'val acc Fs'),
    ):
        if history.get(key):
            acc_series.append((label, history[key]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(epochs, train_loss, label='train loss', color='#1f77b4', lw=1.8)
    if val_loss:
        ax.plot(epochs[:len(val_loss)], val_loss, label='val loss',
                color='#d62728', lw=1.8)
    # Optional per-task train losses (lighter)
    for key, label, color in (
        ('train_loss_energy', 'train E', '#aec7e8'),
        ('train_loss_force_atom', 'train Fa', '#ffbb78'),
        ('train_loss_force_mol', 'train Fs', '#98df8a'),
    ):
        vals = history.get(key) or []
        if vals:
            ax.plot(epochs[:len(vals)], vals, label=label, color=color,
                    lw=1.0, alpha=0.85)
    if best_epoch and 1 <= best_epoch <= n:
        ax.axvline(best_epoch, color='gray', ls='--', lw=1.0, label=f'best ({best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if acc_series:
        colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
        for i, (label, vals) in enumerate(acc_series):
            ax.plot(epochs[:len(vals)], vals, label=label,
                    color=colors[i % len(colors)], lw=1.8)
        if best_epoch and 1 <= best_epoch <= n:
            ax.axvline(best_epoch, color='gray', ls='--', lw=1.0,
                       label=f'best ({best_epoch})')
        ax.set_ylim(0.0, 1.05)
    else:
        ax.text(0.5, 0.5, 'no accuracy in history', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation accuracy')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / filename
    tmp_path = output_dir / f'.{filename}.partial.png'
    try:
        fig.savefig(tmp_path, dpi=120)
        tmp_path.replace(out_path)
    finally:
        plt.close(fig)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    # Also dump numeric history for resume/debug
    try:
        import json
        hist_path = output_dir / 'training_history.json'
        dump = {k: v for k, v in history.items()
                if isinstance(v, (list, int, float, str))}
        hist_path.write_text(json.dumps(dump, indent=2))
    except Exception:
        pass

    return out_path
