"""
mobilefcmvitv3/scripts/test.py

Test / inference script for MobileFCMViTv3 on BUSI.
Runs the model on a data split and writes per-sample predictions to a CSV,
plus a summary JSON with top-level metrics.

Usage:
    cd /path/to/ultrascannet_reproduced
    PYTHONPATH=. python mobilefcmvitv3/scripts/test.py \\
        -c mobilefcmvitv3/configs/mobilefcmvitv3_s_busi.yaml \\
        --checkpoint checkpoints/mobilefcmvitv3_s_BUSI/model_best.pth.tar \\
        --data_dir datasets/BUSI_split \\
        --split validation \\
        --output_dir results/mobilefcmvitv3_s_BUSI
"""

import argparse
import csv
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
from mobilefcmvitv3.utils.augmentation import build_val_transform
from mobilefcmvitv3.utils.metrics import (
    compute_extended_metrics, save_metrics_json, compute_efficiency
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
_logger = logging.getLogger('test')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='', type=str)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_dir', type=str, default='datasets/BUSI_split')
    p.add_argument('--split', type=str, default='validation',
                   choices=['train', 'validation'])
    p.add_argument('--output_dir', type=str, default='results/mobilefcmvitv3_s_BUSI')
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, class_names, amp_autocast):
    model.eval()
    loss_fn = nn.CrossEntropyLoss().cuda()
    rows = []
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_targets, all_probs = [], [], []

    sample_idx = 0
    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        with amp_autocast():
            outputs, _, _ = model(inputs)   # unpack (logits, c_loss, e_loss)
        loss = loss_fn(outputs, targets)
        probs = torch.softmax(outputs.float(), 1)
        preds = probs.argmax(1)

        total_loss += loss.item() * inputs.size(0)
        total_correct += (preds == targets).sum().item()
        total_n += inputs.size(0)

        for i in range(inputs.size(0)):
            gt_name  = class_names[targets[i].item()] if class_names else str(targets[i].item())
            pred_name = class_names[preds[i].item()]  if class_names else str(preds[i].item())
            row = {
                'sample_idx': sample_idx,
                'gt_label': targets[i].item(),
                'gt_name': gt_name,
                'pred_label': preds[i].item(),
                'pred_name': pred_name,
                'correct': int(preds[i].item() == targets[i].item()),
            }
            for c, cn in enumerate(class_names or [str(j) for j in range(probs.shape[1])]):
                row[f'prob_{cn}'] = round(probs[i, c].item(), 6)
            rows.append(row)
            sample_idx += 1

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())

    preds_np   = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()
    probs_np   = torch.cat(all_probs).numpy()

    return rows, {
        'loss': total_loss / total_n,
        'top1': 100.0 * total_correct / total_n,
        'preds': preds_np,
        'targets': targets_np,
        'probs': probs_np,
    }


def main():
    args = parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Load checkpoint — prefer EMA weights
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('ema_state_dict') or ckpt.get('state_dict', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    _logger.info(f"Loaded: {args.checkpoint}")

    # Efficiency
    params_M, flops_G, inf_ms = compute_efficiency(model, in_chans=3)
    _logger.info(f"Params: {params_M}M | FLOPs: {flops_G}G | Inference: {inf_ms}ms")

    # Data
    val_tf  = build_val_transform(img_size=224)
    split_dir = str(Path(args.data_dir) / args.split)
    from mobilefcmvitv3.utils.dataset import BUSIDataset
    dataset = BUSIDataset(split_dir, transform=val_tf)

    loader = DataLoader(
        dataset,
        batch_size=cfg.get('validation_batch_size', 16),
        shuffle=False,
        num_workers=cfg.get('workers', 8),
        pin_memory=True,
    )
    class_names = getattr(dataset, 'classes', None)
    _logger.info(f"Testing on {args.split}: {len(dataset)} samples | classes: {class_names}")

    amp_autocast = partial(torch.amp.autocast, 'cuda') if cfg.get('amp', True) else suppress

    # Inference
    rows, summary = run_inference(model, loader, class_names, amp_autocast)
    _logger.info(f"Loss: {summary['loss']:.4f} | Top-1: {summary['top1']:.2f}%")

    # Extended metrics
    ext = compute_extended_metrics(
        summary['targets'], summary['preds'], summary['probs'],
        class_names=class_names, n_boot=500
    )
    _logger.info(f"Precision: {ext.get('precision_macro', 0):.4f} | "
                 f"Recall: {ext.get('recall_macro', 0):.4f} | "
                 f"F1: {ext.get('f1_macro', 0):.4f}")

    # Write per-sample CSV
    csv_path = out_dir / f'predictions_{args.split}.csv'
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    _logger.info(f"Per-sample predictions: {csv_path}")

    # Write summary JSON
    record = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'loss': summary['loss'],
        'top1': summary['top1'],
        'params_M': params_M,
        'flops_G': flops_G,
        'inference_ms': inf_ms,
        **{k: v for k, v in ext.items() if k != 'confusion_matrix'},
        'confusion_matrix': ext.get('confusion_matrix'),
    }
    json_path = out_dir / f'test_metrics_{args.split}.json'
    save_metrics_json(record, str(json_path))
    _logger.info(f"Summary metrics: {json_path}")

    # Print confusion matrix
    cm = ext.get('confusion_matrix')
    if cm is not None:
        _logger.info(f"Confusion matrix:\n{np.array(cm)}")


if __name__ == '__main__':
    main()
