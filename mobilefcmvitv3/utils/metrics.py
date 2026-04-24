"""
mobilefcmvitv3/utils/metrics.py

Full standalone metrics module for MobileFCMViTv3 — duplicated from
baseline/mambavision_fcm/utils/extended_metrics.py so the two pipelines
are fully independent with no cross-package imports.

Computes the full suite of classification metrics for BUSI:
  - Per-class and macro: accuracy, precision, recall, F1, AUC-ROC, AUC-PR
  - 95% bootstrap CIs for macro-F1 and per-class recall/AUC
  - Sensitivity at fixed specificity (90%)
  - Confusion matrix
  - Model efficiency: Params (M), FLOPs (G), Inference Time (ms)
"""

import time
import json
import numpy as np
import torch
from collections import OrderedDict
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
)

CLASS_NAMES = ['benign', 'malignant', 'normal']


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(metric_fn, N, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    scores = [metric_fn(rng.choice(idx, size=N, replace=True)) for _ in range(n_boot)]
    return (float(np.mean(scores)),
            float(np.percentile(scores, 100 * alpha / 2)),
            float(np.percentile(scores, 100 * (1 - alpha / 2))))


def sensitivity_at_fixed_specificity(y_true_bin, y_score, target_spec=0.90):
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)
    specificity = 1.0 - fpr
    idx = np.argmin(np.abs(specificity - target_spec))
    return float(tpr[idx]), float(specificity[idx]), float(thresholds[idx])


# ── Efficiency metrics ────────────────────────────────────────────────────────

def compute_efficiency(model, in_chans=3, input_size=224,
                       n_warmup=10, n_runs=50, device='cuda'):
    """Returns params_M, flops_G, inference_ms."""
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    flops_G = float('nan')
    try:
        from thop import profile as thop_profile
        dummy = torch.randn(1, in_chans, input_size, input_size).to(device)
        with torch.no_grad():
            flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        flops_G = flops / 1e9
    except Exception:
        pass

    inference_ms = float('nan')
    try:
        dummy = torch.randn(1, in_chans, input_size, input_size).to(device)
        model.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model(dummy)
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - t0) / n_runs * 1000
    except Exception:
        pass

    return round(params_M, 3), round(flops_G, 3), round(inference_ms, 3)


# ── Main metrics computation ──────────────────────────────────────────────────

def compute_extended_metrics(all_targets, all_preds, all_probs,
                              class_names=None, n_boot=1000):
    """
    Args:
        all_targets: np.ndarray [N] int
        all_preds:   np.ndarray [N] int
        all_probs:   np.ndarray [N, C] float (softmax probabilities)
        class_names: list of str, length C
        n_boot:      bootstrap iterations for CIs
    Returns:
        OrderedDict with all metrics
    """
    N, C = all_probs.shape
    if class_names is None:
        class_names = CLASS_NAMES[:C] if C <= len(CLASS_NAMES) else [str(i) for i in range(C)]

    labels = list(range(C))

    accuracy = float(accuracy_score(all_targets, all_preds)) * 100

    prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels, average=None, zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0)

    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    roc_auc_cls, pr_auc_cls = {}, {}
    for c, name in enumerate(class_names):
        y_bin = (all_targets == c).astype(np.uint8)
        y_sc = all_probs[:, c]
        fpr, tpr, _ = roc_curve(y_bin, y_sc)
        roc_auc_cls[name] = round(float(auc(fpr, tpr)), 4)
        prec_c, rec_c, _ = precision_recall_curve(y_bin, y_sc)
        pr_auc_cls[name] = round(float(auc(rec_c, prec_c)), 4)

    roc_auc_macro = float(np.mean(list(roc_auc_cls.values())))
    pr_auc_macro  = float(np.mean(list(pr_auc_cls.values())))

    sens_at_spec = {}
    for c, name in enumerate(class_names):
        y_bin = (all_targets == c).astype(np.uint8)
        sens, spec, thr = sensitivity_at_fixed_specificity(y_bin, all_probs[:, c])
        sens_at_spec[name] = {
            'sensitivity@90%spec': round(sens, 4),
            'achieved_spec':       round(spec, 4),
            'threshold':           round(thr, 4),
        }

    def _macro_f1(b):
        return precision_recall_fscore_support(
            all_targets[b], all_preds[b], average='macro', zero_division=0)[2]

    f1_mean, f1_lo, f1_hi = bootstrap_ci(_macro_f1, N, n_boot=n_boot)

    recall_ci, auc_ci, f1_ci = {}, {}, {}
    for c, name in enumerate(class_names):
        def _rec(b, c=c):
            return precision_recall_fscore_support(
                all_targets[b], all_preds[b], labels=[c], average=None, zero_division=0)[1][0]
        m, lo, hi = bootstrap_ci(_rec, N, n_boot=n_boot)
        recall_ci[name] = {'mean': round(m, 4), 'ci_low': round(lo, 4), 'ci_high': round(hi, 4)}

        def _auc(b, c=c):
            y_bin = (all_targets[b] == c).astype(np.uint8)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                return float('nan')
            fpr, tpr, _ = roc_curve(y_bin, all_probs[b, c])
            return auc(fpr, tpr)
        m, lo, hi = bootstrap_ci(_auc, N, n_boot=n_boot)
        auc_ci[name] = {'mean': round(m, 4), 'ci_low': round(lo, 4), 'ci_high': round(hi, 4)}

        def _f1(b, c=c):
            return precision_recall_fscore_support(
                all_targets[b], all_preds[b], labels=[c], average=None, zero_division=0)[2][0]
        m, lo, hi = bootstrap_ci(_f1, N, n_boot=n_boot)
        f1_ci[name] = {'mean': round(m, 4), 'ci_low': round(lo, 4), 'ci_high': round(hi, 4)}

    metrics = OrderedDict([
        ('accuracy',         round(accuracy, 4)),
        ('precision_macro',  round(float(prec_macro), 4)),
        ('recall_macro',     round(float(rec_macro), 4)),
        ('f1_macro',         round(float(f1_macro), 4)),
        ('f1_macro_mean',    round(f1_mean, 4)),
        ('f1_macro_ci_low',  round(f1_lo, 4)),
        ('f1_macro_ci_high', round(f1_hi, 4)),
        ('roc_auc_macro',    round(roc_auc_macro, 4)),
        ('pr_auc_macro',     round(pr_auc_macro, 4)),
        ('per_class', {
            name: {
                'accuracy':       round(float(per_class_acc[c]), 4),
                'precision':      round(float(prec_cls[c]), 4),
                'recall':         round(float(rec_cls[c]), 4),
                'f1':             round(float(f1_cls[c]), 4),
                'f1_mean':        f1_ci[name]['mean'],
                'f1_ci_low':      f1_ci[name]['ci_low'],
                'f1_ci_high':     f1_ci[name]['ci_high'],
                'roc_auc':        roc_auc_cls[name],
                'pr_auc':         pr_auc_cls[name],
                'recall_mean':    recall_ci[name]['mean'],
                'recall_ci_low':  recall_ci[name]['ci_low'],
                'recall_ci_high': recall_ci[name]['ci_high'],
                'auc_mean':       auc_ci[name]['mean'],
                'auc_ci_low':     auc_ci[name]['ci_low'],
                'auc_ci_high':    auc_ci[name]['ci_high'],
                'sens_at_90spec': sens_at_spec[name],
                'support':        int(support[c]),
            }
            for c, name in enumerate(class_names)
        }),
        ('confusion_matrix', cm.tolist()),
    ])
    return metrics


def flatten_for_wandb(metrics, prefix='val'):
    """Flatten nested metrics dict into a flat key->value dict for wandb.log()."""
    flat = {}
    skip = {'confusion_matrix', 'per_class'}
    for k, v in metrics.items():
        if k in skip:
            continue
        flat[f'{prefix}/{k}'] = v
    for cls_name, cls_vals in metrics.get('per_class', {}).items():
        for metric_name, val in cls_vals.items():
            if isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    flat[f'{prefix}/{cls_name}/{metric_name}/{sub_k}'] = sub_v
            else:
                flat[f'{prefix}/{cls_name}/{metric_name}'] = val
    return flat


def save_metrics_json(metrics, path):
    """Write metrics dict to a JSON file."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


__all__ = [
    'compute_extended_metrics',
    'flatten_for_wandb',
    'save_metrics_json',
    'compute_efficiency',
    'bootstrap_ci',
    'sensitivity_at_fixed_specificity',
]
