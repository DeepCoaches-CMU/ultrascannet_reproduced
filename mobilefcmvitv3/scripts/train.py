"""
mobilefcmvitv3/scripts/train.py

Training script for MobileFCMViTv3 on BUSI.
Follows the same flow as mambavision_fcm/train.py.

Usage:
    cd /path/to/ultrascannet_reproduced
    PYTHONPATH=. python mobilefcmvitv3/scripts/train.py \
        -c mobilefcmvitv3/configs/mobilefcmvitv3_s_busi.yaml \
        --data_dir datasets/BUSI_split
"""

import argparse
import os
import sys
import time
import yaml
import json
import logging
from contextlib import suppress
from collections import OrderedDict
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from timm.data import Mixup
from timm.utils import ModelEmaV2, AverageMeter, accuracy
from timm.scheduler import CosineLRScheduler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mobilefcmvitv3.models import MobileFCMViTv3
from mobilefcmvitv3.utils.dataset import BUSIDataset
from mobilefcmvitv3.utils.augmentation import build_train_transform, build_val_transform
from mobilefcmvitv3.utils.class_imbalance import build_loss_fn
from mobilefcmvitv3.utils.metrics import (
    compute_extended_metrics, flatten_for_wandb, save_metrics_json, compute_efficiency
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
_logger = logging.getLogger('train')

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='', type=str)
    p.add_argument('--data_dir', type=str, default='datasets/BUSI_split')
    p.add_argument('--experiment', type=str, default='')
    p.add_argument('--seed', type=int, default=None)
    args, remaining = p.parse_known_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    # CLI overrides: --key=value or --key value (bool flags)
    i = 0
    while i < len(remaining):
        item = remaining[i]
        if item.startswith('--'):
            if '=' in item:
                key, val = item.lstrip('-').split('=', 1)
            else:
                key = item.lstrip('-')
                val = remaining[i + 1] if (i + 1 < len(remaining) and not remaining[i + 1].startswith('--')) else True
                if val is not True:
                    i += 1
            key = key.replace('-', '_')
            # Type coercion: try int, float, bool, then string
            if val is not True:
                if val.lower() in ('true', 'false'):
                    val = val.lower() == 'true'
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass  # keep as string
            cfg[key] = val
        i += 1
    return args, cfg


# ── Training / validation loops ───────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, mixup_fn,
                    epoch, cfg, amp_autocast):
    model.train()
    losses = AverageMeter()
    task_losses = AverageMeter()
    clust_losses = AverageMeter()
    entropy_losses = AverageMeter()
    lambda_clust   = cfg.get('lambda_clust',   0.1)
    lambda_entropy = cfg.get('lambda_entropy', 0.01)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        with amp_autocast():
            logits, c_loss, e_loss = model(inputs)
            task_loss = loss_fn(logits, targets)
            loss = task_loss + lambda_clust * c_loss + lambda_entropy * e_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.get('clip_grad'):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.get('clip_grad'):
                nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
            optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        task_losses.update(task_loss.item(), inputs.size(0))
        clust_losses.update(c_loss.item(), inputs.size(0))
        entropy_losses.update(e_loss.item(), inputs.size(0))

    return {
        'loss':          losses.avg,
        'task_loss':     task_losses.avg,
        'clust_loss':    clust_losses.avg,
        'entropy_loss':  entropy_losses.avg,
    }


@torch.no_grad()
def validate(model, loader, loss_fn, amp_autocast, log_suffix=''):
    model.eval()
    losses, top1 = AverageMeter(), AverageMeter()
    all_preds, all_targets, all_probs = [], [], []

    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        with amp_autocast():
            logits, _, _ = model(inputs)   # auxiliary losses not needed at eval
        loss = loss_fn(logits, targets)
        acc1, _ = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        all_preds.append(logits.argmax(1).cpu())
        all_targets.append(targets.cpu())
        all_probs.append(torch.softmax(logits.float(), 1).cpu())

    return {
        'loss': losses.avg, 'top1': top1.avg,
        '_preds':   torch.cat(all_preds).numpy(),
        '_targets': torch.cat(all_targets).numpy(),
        '_probs':   torch.cat(all_probs).numpy(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args, cfg = parse_args()
    seed = cfg.get('seed', 42) if args.seed is None else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Dirs
    exp_name = args.experiment or cfg.get('experiment',
        f"mobilefcmvitv3_s_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ckpt_dir = Path(cfg.get('checkpoint_dir', f'checkpoints/{exp_name}'))
    res_dir  = Path(cfg.get('results_dir',    f'results/{exp_name}'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # WandB
    if cfg.get('log_wandb') and HAS_WANDB:
        wandb.init(project='mobilefcmvitv3', name=exp_name, config=cfg)

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
        drop_path_rate=cfg.get('drop_path_rate', 0.1),
        dropout=cfg.get('dropout', 0.1),
        attn_dropout=cfg.get('attn_dropout', 0.0),
        ffn_dropout=cfg.get('ffn_dropout', 0.0),
    )

    ckpt_path = cfg.get('pretrained_weights', '')
    if ckpt_path:
        model.load_pretrained(ckpt_path)

    model = model.cuda()

    # EMA
    model_ema = None
    if cfg.get('model_ema', True):
        model_ema = ModelEmaV2(model, decay=cfg.get('model_ema_decay', 0.9998))

    # Efficiency
    params_M, flops_G, inf_ms = compute_efficiency(model, in_chans=3)
    _logger.info(f"Params: {params_M}M | FLOPs: {flops_G}G | Inference: {inf_ms}ms")

    # Data — plain RGB, no FCM channel at input
    img_size = 224
    train_tf = build_train_transform(
        img_size=img_size,
        scale=cfg.get('scale', [0.08, 1.0]),
        ratio=cfg.get('ratio', [0.75, 1.3333]),
        hflip=cfg.get('hflip', 0.5),
        color_jitter=cfg.get('color_jitter', 0.2),
    )
    val_tf = build_val_transform(img_size=img_size)

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir   = os.path.join(args.data_dir, 'validation')

    dataset_train = BUSIDataset(train_dir, transform=train_tf)
    dataset_val   = BUSIDataset(val_dir,   transform=val_tf)

    bs = cfg.get('batch_size', 32)
    loader_train = DataLoader(dataset_train, batch_size=bs, shuffle=True,
                              num_workers=cfg.get('workers', 8),
                              pin_memory=cfg.get('pin_mem', True), drop_last=True)
    loader_val   = DataLoader(dataset_val,
                              batch_size=cfg.get('validation_batch_size', 16),
                              shuffle=False, num_workers=cfg.get('workers', 8),
                              pin_memory=cfg.get('pin_mem', True))

    _logger.info(f"Train: {len(dataset_train)} | Val: {len(dataset_val)}")

    # Mixup
    mixup_active = cfg.get('mixup', 0) > 0 or cfg.get('cutmix', 0) > 0
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=cfg.get('mixup', 0.2),
            cutmix_alpha=cfg.get('cutmix', 0.2),
            prob=cfg.get('mixup_prob', 0.3),
            switch_prob=cfg.get('mixup_switch_prob', 0.5),
            mode=cfg.get('mixup_mode', 'batch'),
            label_smoothing=cfg.get('smoothing', 0.1),
            num_classes=cfg.get('num_classes', 3),
        )

    # Loss
    train_loss_fn = build_loss_fn(
        smoothing=cfg.get('smoothing', 0.1),
        mixup_active=mixup_active,
        use_class_weights=cfg.get('use_class_weights', True),
        num_classes=cfg.get('num_classes', 3),
    )
    val_loss_fn = nn.CrossEntropyLoss().cuda()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get('lr', 6.25e-5),
        betas=tuple(cfg.get('opt_betas', [0.9, 0.999])),
        eps=cfg.get('opt_eps', 1e-8),
        weight_decay=cfg.get('weight_decay', 0.01),
    )

    # Scheduler
    num_epochs = cfg.get('epochs', 150)
    warmup_epochs = cfg.get('warmup_epochs', 20)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=cfg.get('min_lr', 6.25e-7),
        warmup_lr_init=cfg.get('warmup_lr', 6.25e-8),
        warmup_t=warmup_epochs,
        cycle_limit=1,
        t_in_epochs=True,
    )

    # AMP
    amp_autocast = suppress
    scaler = None
    if cfg.get('amp', True):
        amp_autocast = partial(torch.amp.autocast, 'cuda')
        scaler = torch.cuda.amp.GradScaler()
        _logger.info("Using native AMP.")

    # Training loop
    best_top1 = float('-inf')
    best_epoch = 0
    epochs_no_improve = 0
    patience = cfg.get('patience_epochs', 40)
    summary_rows = []

    for epoch in range(num_epochs):
        scheduler.step(epoch)

        train_metrics = train_one_epoch(
            model, loader_train, optimizer, train_loss_fn,
            scaler, mixup_fn, epoch, cfg, amp_autocast)

        eval_metrics = validate(model, loader_val, val_loss_fn, amp_autocast)

        if model_ema is not None:
            model_ema.update(model)
            ema_metrics = validate(model_ema.module, loader_val,
                                   val_loss_fn, amp_autocast, log_suffix=' (EMA)')
            if ema_metrics['top1'] > eval_metrics['top1']:
                eval_metrics = ema_metrics

        top1 = eval_metrics['top1']
        preds, targets, probs = (eval_metrics.pop('_preds'),
                                 eval_metrics.pop('_targets'),
                                 eval_metrics.pop('_probs'))

        ext = compute_extended_metrics(targets, preds, probs, n_boot=200)

        # Mirror mambavision: surface macro metrics into eval_metrics so
        # the checkpoint saver and summary CSV see them.
        eval_metrics['precision'] = ext['precision_macro']
        eval_metrics['recall']    = ext['recall_macro']
        eval_metrics['f1']        = ext['f1_macro']

        _logger.info(
            f"Epoch {epoch:3d} | "
            f"task={train_metrics['task_loss']:.4f} "
            f"clust={train_metrics['clust_loss']:.4f} "
            f"entropy={train_metrics['entropy_loss']:.4f} "
            f"total={train_metrics['loss']:.4f} "
            f"| val_top1={top1:.2f}%  prec={ext['precision_macro']:.4f}  "
            f"rec={ext['recall_macro']:.4f}  f1={ext['f1_macro']:.4f}  "
            f"auc={ext['roc_auc_macro']:.4f}"
        )

        # Save per-epoch JSON
        record = {
            'epoch': epoch,
            'train_loss':         train_metrics['loss'],
            'train_task_loss':    train_metrics['task_loss'],
            'train_clust_loss':   train_metrics['clust_loss'],
            'train_entropy_loss': train_metrics['entropy_loss'],
            'val_loss':      eval_metrics['loss'],
            'val_top1':      top1,
            'val_precision': ext['precision_macro'],
            'val_recall':    ext['recall_macro'],
            'val_f1':        ext['f1_macro'],
            'val_roc_auc':   ext['roc_auc_macro'],
            'val_pr_auc':    ext['pr_auc_macro'],
            'params_M': params_M, 'flops_G': flops_G, 'inference_ms': inf_ms,
            **{k: v for k, v in ext.items() if k != 'confusion_matrix'},
            'confusion_matrix': ext.get('confusion_matrix'),
        }
        save_metrics_json(record, str(res_dir / f'metrics_epoch_{epoch:04d}.json'))

        # WandB
        if cfg.get('log_wandb') and HAS_WANDB and wandb.run:
            wandb.log({
                'epoch': epoch,
                'train/loss':         train_metrics['loss'],
                'train/task_loss':    train_metrics['task_loss'],
                'train/clust_loss':   train_metrics['clust_loss'],
                'train/entropy_loss': train_metrics['entropy_loss'],
                'val/loss':      eval_metrics['loss'],
                'val/top1':      top1,
                'val/precision': ext['precision_macro'],
                'val/recall':    ext['recall_macro'],
                'val/f1':        ext['f1_macro'],
                'val/roc_auc':   ext['roc_auc_macro'],
                'val/pr_auc':    ext['pr_auc_macro'],
                'lr': optimizer.param_groups[0]['lr'],
                **flatten_for_wandb(ext, prefix='val'),
            })

        summary_rows.append({
            'epoch': epoch,
            'train_loss':         train_metrics['loss'],
            'train_task_loss':    train_metrics['task_loss'],
            'train_clust_loss':   train_metrics['clust_loss'],
            'train_entropy_loss': train_metrics['entropy_loss'],
            'val_loss':      eval_metrics['loss'],
            'val_top1':      top1,
            'val_precision': ext['precision_macro'],
            'val_recall':    ext['recall_macro'],
            'val_f1':        ext['f1_macro'],
            'val_roc_auc':   ext['roc_auc_macro'],
            'val_pr_auc':    ext['pr_auc_macro'],
        })

        # Checkpoint
        if top1 > best_top1:
            best_top1 = top1
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt = {
                'epoch': epoch, 'top1': top1,
                'state_dict': model.state_dict(),
                'ema_state_dict': model_ema.module.state_dict() if model_ema else None,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(ckpt, str(ckpt_dir / 'model_best.pth.tar'))
            save_metrics_json(record, str(res_dir / 'best_metrics.json'))
            _logger.info(f"  ✓ New best: {best_top1:.2f}% at epoch {best_epoch}")
        else:
            epochs_no_improve += 1

        # Save latest
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   str(ckpt_dir / 'last.pth.tar'))

        if patience > 0 and epochs_no_improve >= patience:
            _logger.info(f"Early stopping: no improvement for {patience} epochs.")
            break

    # Final summary
    import csv
    with open(str(res_dir / 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['epoch', 'train_loss', 'train_task_loss',
                           'train_clust_loss', 'train_entropy_loss',
                           'val_loss', 'val_top1', 'val_precision',
                           'val_recall', 'val_f1', 'val_roc_auc', 'val_pr_auc'])
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(str(ckpt_dir / 'best.txt'), 'w') as f:
        f.write(f"best_top1={best_top1:.4f} at epoch={best_epoch}\n")

    _logger.info(f"*** Best top1: {best_top1:.2f}% (epoch {best_epoch})")


if __name__ == '__main__':
    main()
