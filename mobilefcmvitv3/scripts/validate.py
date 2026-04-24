"""
mobilefcmvitv3/scripts/validate.py

Validation script for MobileFCMViTv3 on BUSI.
Loads a trained checkpoint and computes full extended metrics.

Usage:
    cd /path/to/ultrascannet_reproduced
    PYTHONPATH=. python mobilefcmvitv3/scripts/validate.py \\
        -c mobilefcmvitv3/configs/mobilefcmvitv3_s_busi.yaml \\
        --checkpoint checkpoints/mobilefcmvitv3_s_BUSI/model_best.pth.tar \\
        --data_dir datasets/BUSI_split \\
        --output_json results/mobilefcmvitv3_s_BUSI/evaluation_metrics.json
"""

import argparse
import json
import logging
import sys
from contextlib import suppress
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mobilefcmvitv3.models import MobileFCMViTv3
from mobilefcmvitv3.utils.dataset import BUSIDataset
from mobilefcmvitv3.utils.augmentation import build_val_transform, build_tta_transforms
from mobilefcmvitv3.utils.metrics import (
    compute_extended_metrics, save_metrics_json, compute_efficiency
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
_logger = logging.getLogger('validate')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='', type=str)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_dir', type=str, default='datasets/BUSI_split')
    p.add_argument('--output_json', type=str, default='')
    p.add_argument('--split', type=str, default='validation',
                   choices=['train', 'validation'])
    p.add_argument('--tta', action='store_true', default=False,
                   help='Enable Test-Time Augmentation (8 views averaged)')
    p.add_argument('--use_ema', action='store_true', default=False,
                   help='Use EMA weights instead of live model weights')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, amp_autocast):
    """Standard single-view evaluation."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss().cuda()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_targets, all_probs = [], [], []

    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        with amp_autocast():
            outputs, _, _ = model(inputs)
        loss = loss_fn(outputs, targets)
        preds = outputs.argmax(1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += (preds == targets).sum().item()
        total_n += inputs.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_probs.append(torch.softmax(outputs.float(), 1).cpu())

    return {
        'loss':    total_loss / total_n,
        'top1':    100.0 * total_correct / total_n,
        'preds':   torch.cat(all_preds).numpy(),
        'targets': torch.cat(all_targets).numpy(),
        'probs':   torch.cat(all_probs).numpy(),
    }


@torch.no_grad()
def evaluate_tta(model, dataset, tta_transforms, amp_autocast, batch_size=16):
    """
    Test-Time Augmentation: run each TTA view through the model and
    average the softmax probabilities before taking argmax.
    """
    model.eval()
    n = len(dataset)
    n_views = len(tta_transforms)
    all_probs = torch.zeros(n, 3)   # accumulate averaged probs
    all_targets = []

    for view_idx, tf in enumerate(tta_transforms):
        dataset.transform = tf
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
        view_probs = []
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            with amp_autocast():
                logits, _, _ = model(inputs)
            view_probs.append(torch.softmax(logits.float(), 1).cpu())
            if view_idx == 0:
                all_targets.append(targets)
        all_probs += torch.cat(view_probs) / n_views
        _logger.info(f"  TTA view {view_idx+1}/{n_views} done")

    targets_np = torch.cat(all_targets).numpy()
    probs_np   = all_probs.numpy()
    preds_np   = all_probs.argmax(1).numpy()
    correct    = (preds_np == targets_np).sum()

    return {
        'loss':    float('nan'),
        'top1':    100.0 * correct / n,
        'preds':   preds_np,
        'targets': targets_np,
        'probs':   probs_np,
    }


def main():
    args = parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    # Model
    in_chans = 3
    model = MobileFCMViTv3(
        num_classes=cfg.get('num_classes', 3),
        in_chans=in_chans,
        fcm_k=cfg.get('fcm_k', 3),
        fcm_proj_dim=cfg.get('fcm_proj_dim', 32),
        tau=cfg.get('tau', 1.0),
        fcm_m=cfg.get('fcm_m', 2.0),
        membership=cfg.get('membership', 'softmax'),
        normalize_feat=cfg.get('normalize_feat', True),
        fusion_type=cfg.get('fusion_type', 'attention'),
        dropout=cfg.get('dropout', 0.1),
        attn_dropout=cfg.get('attn_dropout', 0.0),
        ffn_dropout=cfg.get('ffn_dropout', 0.0),
    ).cuda()

    # Load checkpoint — prefer live state_dict; use EMA only if explicitly requested
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.use_ema and ckpt.get('ema_state_dict') is not None:
        state = ckpt['ema_state_dict']
        _logger.info("Using EMA weights")
    else:
        state = ckpt.get('state_dict', ckpt)
        _logger.info("Using live model weights")
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    _logger.info(f"Loaded checkpoint: {args.checkpoint}")
    if missing:
        _logger.warning(f"  Missing keys ({len(missing)}): {missing[:3]}...")

    # Efficiency
    params_M, flops_G, inf_ms = compute_efficiency(model, in_chans=3)
    _logger.info(f"Params: {params_M}M | FLOPs: {flops_G}G | Inference: {inf_ms}ms")

    # Data
    val_dir = str(Path(args.data_dir) / args.split)
    amp_autocast = partial(torch.amp.autocast, 'cuda') if cfg.get('amp', True) else suppress

    if args.tta:
        _logger.info("Running TTA (8 views)...")
        tta_tfs = build_tta_transforms(img_size=224)
        # Dataset with placeholder transform — evaluate_tta swaps it per view
        dataset = BUSIDataset(val_dir, transform=tta_tfs[0])
        _logger.info(f"Evaluating on {args.split}: {len(dataset)} samples")
        results = evaluate_tta(model, dataset, tta_tfs, amp_autocast,
                               batch_size=cfg.get('validation_batch_size', 16))
    else:
        dataset = BUSIDataset(val_dir, transform=build_val_transform(img_size=224))
        loader = DataLoader(dataset, batch_size=cfg.get('validation_batch_size', 16),
                            shuffle=False, num_workers=cfg.get('workers', 8),
                            pin_memory=True)
        _logger.info(f"Evaluating on {args.split}: {len(dataset)} samples")
        results = evaluate(model, loader, amp_autocast)

    _logger.info(f"Top-1: {results['top1']:.2f}%" +
                 (f" | Loss: {results['loss']:.4f}" if not args.tta else " (TTA)"))

    # Extended metrics
    class_names = getattr(dataset, 'classes', None)
    ext = compute_extended_metrics(
        results['targets'], results['preds'], results['probs'],
        class_names=class_names, n_boot=500
    )

    _logger.info(f"Precision: {ext.get('precision_macro', 0):.4f} | "
                 f"Recall: {ext.get('recall_macro', 0):.4f} | "
                 f"F1: {ext.get('f1_macro', 0):.4f} | "
                 f"AUC: {ext.get('roc_auc_macro', 0):.4f}")

    cm = ext.get('confusion_matrix')
    if cm is not None:
        _logger.info(f"Confusion matrix:\n{np.array(cm)}")

    record = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'tta': args.tta,
        'loss': results['loss'],
        'top1': results['top1'],
        'params_M': params_M,
        'flops_G': flops_G,
        'inference_ms': inf_ms,
        **{k: v for k, v in ext.items() if k != 'confusion_matrix'},
        'confusion_matrix': ext.get('confusion_matrix'),
    }

    out_path = args.output_json
    if not out_path:
        suffix = '_tta' if args.tta else ''
        out_path = str(Path(args.checkpoint).parent / f'evaluation_metrics{suffix}.json')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_metrics_json(record, out_path)
    _logger.info(f"Metrics saved to: {out_path}")


if __name__ == '__main__':
    main()
