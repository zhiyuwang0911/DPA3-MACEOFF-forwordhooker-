"""Classification metrics for PROBE."""

import torch


def confusion_matrix_torch(pred: torch.Tensor, target: torch.Tensor,
                            num_classes: int) -> torch.Tensor:
    pred   = pred.flatten().long()
    target = target.flatten().long()
    mask   = (target >= 0) & (target < num_classes) & \
             (pred >= 0) & (pred < num_classes)
    indices = target[mask] * num_classes + pred[mask]
    return torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def accuracy_from_cm(cm):
    return cm.diag().sum().float() / (cm.sum().float() + 1e-12)

def precision_from_cm(cm):
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    return (tp / (tp + fp + 1e-12)).mean()

def recall_from_cm(cm):
    tp = cm.diag().float()
    fn = cm.sum(dim=1).float() - tp
    return (tp / (tp + fn + 1e-12)).mean()

def f1_from_cm(cm):
    p = precision_from_cm(cm)
    r = recall_from_cm(cm)
    return 2 * p * r / (p + r + 1e-12)

def mcc_from_cm(cm):
    t_sum = cm.sum(dim=1).float()
    p_sum = cm.sum(dim=0).float()
    n = cm.sum().float()
    c = cm.diag().sum().float()
    s = (p_sum * t_sum).sum()
    num = c * n - s
    den = torch.sqrt((n**2 - (p_sum**2).sum()) * (n**2 - (t_sum**2).sum()))
    return num / (den + 1e-12)

def compute_all_metrics(cm: torch.Tensor) -> dict:
    cwa = cm.diag().float() / (cm.sum(dim=1).float() + 1e-12)
    return {
        'accuracy':          accuracy_from_cm(cm).item(),
        'precision':         precision_from_cm(cm).item(),
        'recall':            recall_from_cm(cm).item(),
        'f1':                f1_from_cm(cm).item(),
        'mcc':               mcc_from_cm(cm).item(),
        'class_wise_accuracy': cwa.cpu().numpy().tolist(),
        'confusion_matrix':  cm.cpu().numpy().tolist(),
    }


def high_confidence_analysis(probs: torch.Tensor, preds: torch.Tensor,
                              targets: torch.Tensor, high_conf_cutoffs: dict,
                              n_classes: int) -> dict:
    """Filter predictions by per-class confidence threshold and compute metrics."""
    thresholds = torch.zeros(len(preds), device=probs.device)
    for cls_idx, cutoff in high_conf_cutoffs.items():
        thresholds[preds == cls_idx] = cutoff

    pred_probs = probs[torch.arange(len(preds), device=probs.device), preds]
    hc_mask = pred_probs > thresholds
    n_hc = hc_mask.sum().item()
    n_total = len(preds)

    if n_hc == 0:
        return {'n_high_conf': 0, 'n_total': n_total, 'fraction_high_conf': 0.0}

    cm_hc = confusion_matrix_torch(preds[hc_mask], targets[hc_mask], n_classes)
    metrics = compute_all_metrics(cm_hc)
    metrics.update({'n_high_conf': n_hc, 'n_total': n_total,
                    'fraction_high_conf': n_hc / n_total})
    return metrics
