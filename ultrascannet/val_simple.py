""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examplesf
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2023 Ross Wightman (https://github.com/rwightman)
"""
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress

from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import ImageDataset, create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm import utils
from timm.loss import JsdCrossEntropy, BinaryCrossEntropy, SoftTargetCrossEntropy,\
    LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import *
from timm.utils import ApexScaler, NativeScaler
from scheduler.scheduler_factory import create_scheduler
import shutil
from utils.datasets import imagenet_lmdb_dataset
from tensorboard import TensorboardLogger
from models.ultra_scan_net import *
from models.mamba_vision_baseline import mamba_vision_T2_baseline
from torchinfo import summary
from sklearn.metrics import precision_recall_fscore_support
from thop import profile
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset')
group.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--tag', default='exp', type=str, metavar='TAG')
# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='gc_vit_tiny', type=str, metavar='MODEL',
                    help='Name of model to train (default: "gc_vit_tiny"')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--loadcheckpoint', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=0.875, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                    help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
group.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
group.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8, use opt default)')
group.add_argument('--opt-betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
group.add_argument('--clip-grad', type=float, default=5.0, metavar='NORM',
                    help='Clip gradient norm (default: 5.0, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr-ep', action='store_true', default=False,
                        help='using the epoch-based scheduler')
group.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=1.0, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
group.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (5e-6)')
group.add_argument('--epochs', type=int, default=310, metavar='N',
                    help='number of epochs to train (default: 310)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
group.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
group.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop-rate', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
group.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT',
                    help='Drop of the attention, gaussian std')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=1, metavar='N',
                    help='number of checkpoints to keep (default: 3)')
group.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 8)')
group.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
group.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
group.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
group.add_argument('--log_dir', default='./log_dir/', type=str,
                    help='where to store tensorboard')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument("--data_len", default=1281167, type=int,help='size of the dataset')

group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
group.add_argument('--validate_only', action='store_true', default=False,
                    help='run model validation only')
group.add_argument('--infer_only', action='store_true', default=False,
                    help='run only inference')

group.add_argument('--no_saver', action='store_true', default=False,
                    help='Save checkpoints')
group.add_argument('--ampere_sparsity', action='store_true', default=False,
                    help='Save checkpoints')
group.add_argument('--lmdb_dataset', action='store_true', default=False,
                    help='use lmdb dataset')
group.add_argument('--bfloat', action='store_true', default=False,
                    help='use bfloat datatype')
group.add_argument('--mesa',  type=float, default=0.0,
                    help='use memory efficient sharpness optimization, enabled if >0.0')
group.add_argument('--mesa-start-ratio',  type=float, default=0.25,
                    help='when to start MESA, ratio to total training time, def 0.25')

kl_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()

def count_params(model):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total, trainable


def kdloss(y, teacher_scores):
    T = 3
    p = torch.nn.functional.log_softmax(y/T, dim=1)
    q = torch.nn.functional.softmax(teacher_scores/T, dim=1)
    l_kl = 50.0*kl_loss(p, q)
    return l_kl

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def sensitivity_at_fixed_specificity(y_true_bin, y_score, target_specificity=0.90):
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)
    specificity = 1.0 - fpr
    # pick closest specificity >= target_specificity (or closest overall)
    idx = np.argmin(np.abs(specificity - target_specificity))
    sens = tpr[idx]
    thr = thresholds[idx]
    return sens, specificity[idx], thr

def bootstrap_ci(metric_fn, N, n_boot=1000, alpha=0.05, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    scores = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=N, replace=True)
        scores.append(metric_fn(b))
    mean = float(np.mean(scores))
    lo = float(np.percentile(scores, 100*(alpha/2)))
    hi = float(np.percentile(scores, 100*(1 - alpha/2)))
    return mean, lo, hi



def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()


    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        # torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d. Local rank %d'
                     % (args.rank, args.world_size, args.local_rank))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    utils.random_seed(args.seed, args.rank)

    if '_baseline' in args.model:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint,
            attn_drop_rate=args.attn_drop_rate,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path,
        )
    elif 'ultra_scan_net' == args.model:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint,
            attn_drop_rate=args.attn_drop_rate,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path,
            patch_embed = args.patch_embed if hasattr(args, "patch_embed") else None,
            first_layer = args.first_layer if  hasattr(args, "first_layer") else None,
            second_layer = args.second_layer if  hasattr(args, "second_layer") else None,
            mixer=0
        )
    elif 'resnet50' in args.model:
        model = create_model("resnet50", pretrained=True, num_classes=3)
    elif 'mobilenet' in args.model:
        model = create_model("mobilenetv2_100", pretrained=True, num_classes=3)
    elif 'vit_small' in args.model:
        model = create_model("vit_small_patch16_224", pretrained=True, num_classes=3)
    elif 'densenet121' in args.model:
        model = create_model("densenet121", pretrained=True, num_classes=3)
    elif 'efficientnet_b0' in args.model:
        model = create_model("efficientnet_b0", pretrained=True, num_classes=3)
    elif 'convnext' in args.model:
        model = create_model("convnext_tiny", pretrained=True, num_classes=3)
    elif 'swin' in args.model:
        model = create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=3)
    elif 'deit' in args.model:
        model = create_model("deit_tiny_patch16_224", pretrained=True, num_classes=3)
    elif 'maxvit' in args.model:
        model = create_model("maxvit_tiny_rw_224", pretrained=True, num_classes=3)

    total, trainable = count_params(model)
    print(f"{args.model:<20}: {total:,} | {trainable:,}")

    # print(model)
    
    if args.bfloat:
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float16

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    # print("filter_bias_and_bn")
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')


    _logger.info(f"Loading checkpoint {args.loadcheckpoint}, checking for existing parameters if their shape match")
    new_model_weights = torch.load(args.loadcheckpoint, weights_only=False)["state_dict"]
    current_model = model.state_dict()

    new_state_dict = OrderedDict()
    for k in current_model.keys():
        if k in new_model_weights.keys():
            if new_model_weights[k].size() == current_model[k].size():
                # print(f"loading weights {k} {new_model_weights[k].size()}")
                new_state_dict[k] = new_model_weights[k]

    model.load_state_dict(new_state_dict, strict=False)


    if args.infer_only:
        from glob import glob
        image_paths = sorted(glob("/home/alexandra/Documents/MambaVision/brease_demo/val/*/*.png"))  # load 3 images
        print(args.model)
        all_preds, all_confs = run_inference(model, image_paths, args)

        print(all_preds, all_confs)
        exit()


    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

 
    validate_loss_fn = nn.CrossEntropyLoss().cuda()



    eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
    print(args.model)
    print(eval_metrics)
    # print("\n".join([f"{k:<10}: {v:.4f}" for k, v in eval_metrics.items()]))
    # Format the metrics as string


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    feature_maps = {}

    def save_activation(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()
        return hook

    # Choose a deep feature block (you can experiment with others too)
    # model.levels[3].blocks[-1].register_forward_hook(save_activation("cam_features"))

    model.eval()

    if args.ampere_sparsity:
        model.enforce_mask()

    all_preds = []
    all_targets = []
    all_probs = [] 

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                start = time.time()
                output = model(input)
                elapsed = (time.time() - start) 

            # if batch_idx == 0 and args.local_rank == 0:
            #     fmap = feature_maps["cam_features"]  # shape: [B, HW, C]
            #     fmap = fmap[0]  # take the first image → [49, 640]

            #     # Transpose to [C, HW]
            #     fmap = fmap.permute(1, 0)  # → [640, 49]

            #     # Reshape to spatial map [C, H, W]
            #     S = int(fmap.shape[1] ** 0.5)  # 49 → 7
            #     fmap = fmap.view(fmap.shape[0], S, S)  # → [640, 7, 7]

            #     # Get predicted class index
            #     pred_class = output[0].argmax().item()

            #     # Get classifier weights for that class
            #     class_weights = model.head.weight[pred_class]  # shape: [640]

            #     # Compute CAM: weighted sum over channels
            #     cam = (class_weights.view(-1, 1, 1) * fmap).sum(0)  # → [7, 7]

            #     # Normalize
            #     cam = (cam - cam.min()) / (cam.max() - cam.min())

            #     # Upsample to input image size
            #     import torch.nn.functional as F
            #     cam_resized = F.interpolate(
            #         cam.unsqueeze(0).unsqueeze(0),  # [1, 1, 7, 7]
            #         size=(input.shape[2], input.shape[3]),  # input image size
            #         mode="bilinear", align_corners=False
            #     )[0, 0]  # → [H, W]

            #     # Convert to numpy
            #     cam_np = cam_resized.cpu().numpy()
            #     input_np = input[0].permute(1, 2, 0).cpu().numpy()
            #     input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())

            #     # Overlay CAM on input
            #     import matplotlib.pyplot as plt
            #     import matplotlib.cm as cm

            #     cmap = cm.get_cmap("jet")
            #     cam_rgb = cmap(cam_np)[:, :, :3]  # drop alpha
            #     overlay = 0.5 * cam_rgb + 0.5 * input_np
            #     overlay = (overlay * 255).astype("uint8")

            #     plt.imsave(f"cam_class_{pred_class}.png", overlay)

            if isinstance(output, (tuple, list)):
                output = output[0]

            probs = torch.softmax(output, dim=1)  # NEW
            preds = probs.argmax(dim=1)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            # Collect for precision/recall/F1
            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())  # NEW

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    # Stack all predictions and targets
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()  # shape: [N, C]

    x = all_targets
    unique_values, counts = np.unique(x, return_counts=True)
    for val, count in zip(unique_values, counts):
        print(f"Value {val}: {count} occurrences")
    


    # Class‑wise precision/recall/F1
    target_names = getattr(loader.dataset, 'classes', [str(i) for i in range(all_probs.shape[1])])
    prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(len(target_names))), average=None, zero_division=0
    )

    # ROC/PR curves + AUC (one-vs-rest)
    roc_auc_per_class = {}
    pr_auc_per_class = {}
    roc_points = {}
    pr_points = {}

    for c in range(all_probs.shape[1]):
        y_true_bin = (all_targets == c).astype(np.uint8)
        y_score = all_probs[:, c]

        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc_per_class[target_names[c]] = auc(fpr, tpr)
        roc_points[target_names[c]] = (fpr, tpr)

        prec, rec, _ = precision_recall_curve(y_true_bin, y_score)
        pr_auc_per_class[target_names[c]] = auc(rec, prec)
        pr_points[target_names[c]] = (rec, prec)


    #  Sensitivity at fixed specificity (per class)
    sens_at_spec = {}
    for c in range(all_probs.shape[1]):
        y_true_bin = (all_targets == c).astype(np.uint8)
        y_score = all_probs[:, c]
        sens, spec, thr = sensitivity_at_fixed_specificity(y_true_bin, y_score, target_specificity=0.90)
        sens_at_spec[target_names[c]] = {"sensitivity@90%spec": float(sens), "achieved_spec": float(spec), "threshold": float(thr)}

    # Bootstrap 95% CIs (any metric)

    # Macro F1 CI
    def macro_f1_on_idx(b):
        return precision_recall_fscore_support(
            all_targets[b], all_preds[b], average='macro', zero_division=0
        )[2]
    f1_macro_mean, f1_macro_lo, f1_macro_hi = bootstrap_ci(macro_f1_on_idx, N=len(all_targets))

    def recall_class_on_idx(c):
        def fn(b):
            _, rec, _, _ = precision_recall_fscore_support(all_targets[b], all_preds[b], labels=[c], average=None, zero_division=0)
            return rec[0]
        return fn

    recall_ci_per_class = {}
    for c, name in enumerate(target_names):
        m, lo, hi = bootstrap_ci(recall_class_on_idx(c), N=len(all_targets))
        recall_ci_per_class[name] = (m, lo, hi)

    # AUC

    def auc_class_on_idx(c):
        def fn(b):
            y_true_bin = (all_targets[b] == c).astype(np.uint8)
            y_score = all_probs[b, c]
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            return auc(fpr, tpr)
        return fn

    auc_ci_per_class = {}
    for c, name in enumerate(target_names):
        m, lo, hi = bootstrap_ci(auc_class_on_idx(c), N=len(all_targets))
        auc_ci_per_class[name] = (m, lo, hi)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(target_names))))


    

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )

    metrics = OrderedDict([
        ('loss', losses_m.avg),
        ('top1', top1_m.avg),
        ('top5', top5_m.avg),
        ('precision', precision_macro),
        ('recall', recall_macro),
        ('f1', f1_macro),
        ('classwise', {
            'names': target_names,
            'precision': prec_cls.tolist(),
            'recall': rec_cls.tolist(),
            'f1': f1_cls.tolist(),
            'support': support.tolist(),
        }),
        ('roc_auc_per_class', roc_auc_per_class),
        ('pr_auc_per_class', pr_auc_per_class),
        ('sens_at_90spec', sens_at_spec),
        ('ci', {
            'f1_macro_mean': f1_macro_mean,
            'f1_macro_95ci': [f1_macro_lo, f1_macro_hi],
            'recall_ci_per_class': {k: {'mean': v[0], 'lo': v[1], 'hi': v[2]} for k, v in recall_ci_per_class.items()},
            'auc_ci_per_class': {k: {'mean': v[0], 'lo': v[1], 'hi': v[2]} for k, v in auc_ci_per_class.items()},
        }),
        ('confusion_matrix', cm.tolist()),
    ])
    return metrics


def register_cam_hook(model, model_name, feature_map):
    def hook_fn(module, input, output):
        feature_map["feat"] = output.detach()

    if "mamba" in model_name:
        model.levels[3].blocks[-1].register_forward_hook(hook_fn)
    elif "resnet" in model_name:
        model.layer4.register_forward_hook(hook_fn)
    elif "mobilenetv2" in model_name:
        model.conv_head.register_forward_hook(hook_fn)
    elif "densenet" in model_name:
        model.features[-1].register_forward_hook(hook_fn)
    elif "efficientnet" in model_name:
        model.conv_head.register_forward_hook(hook_fn)
    elif "convnext" in model_name:
        model.stages[-1].register_forward_hook(hook_fn)
    elif "vit_small_patch16_224" in model_name or "deit" in model_name:
    # Hook into the last transformer block's output — we skip the CLS token later
        model.blocks[-1].register_forward_hook(hook_fn)
    elif "swin" in model_name:
        # Last normalization layer before head: model.norm
        model.norm.register_forward_hook(hook_fn)
    elif "maxvit" in model_name:
         model.stages[-1].blocks[-1].register_forward_hook(hook_fn)
    else:
        raise NotImplementedError(f"CAM hook not implemented for model: {model_name}")
    

def generate_and_save_cam(
    args,
    inputs,              # Tensor: [B, C, H, W]
    feature_map,         # Tensor: [B, C, H_f, W_f]
    preds,               # Tensor: [B]
    model,               # The model (assumes model.head.weight exists)
    image_paths,         # List of strings, image filenames
    save_dir="cam_outputs"
):
    
    save_dir = os.path.join(save_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    def get_classifier_weights(model, model_name):
        if 'resnet' in model_name:
            return model.fc.weight
        elif 'densenet' in model_name or 'efficientnet' in model_name or 'mobilenetv2' in model_name:
            return model.classifier.weight
        elif 'convnext' in model_name:
            return model.head.fc.weight  # ✅ correct for ConvNeXt
        elif 'swin' in model_name:
             return model.head.fc.weight
        elif "maxvit" in model_name:
            return model.head.fc.weight
        elif  'deit' in model_name or 'vit_small_patch16_224' in model_name:
            return model.head.weight  # usually nn.Linear
        elif 'mamba' in model_name:
            return model.head.weight
        else:
            raise NotImplementedError(f"No known classifier mapping for model: {model_name}")
        

    for i in range(inputs.shape[0]):
        try:

            feat = feature_map[i]

            if feat.ndim == 2 and 'mamba' in args.model:
                # [Tokens, Channels] → [Channels, H, W]
                num_tokens, channels = feat.shape
                h = w = int(num_tokens ** 0.5)
                feat = feat.permute(1, 0).reshape(channels, h, w)

            # Handle transformer-style features (ViT / Swin)
            if "deit" in args.model or "vit_small_patch16_224" in args.model:
                if feat.dim() != 2:
                    raise ValueError(f"Expected 2D features for transformer-style models, got: {feat.shape}")
                feat = feat[1:]  # remove CLS token → shape [N, C]
                h = w = int(feat.shape[0] ** 0.5)
                feat = feat.permute(1, 0).reshape(-1, h, w)  # [C, H, W]

            elif "swin" in args.model:
                if feat.dim() == 3 and feat.shape[0] == feat.shape[1] and feat.shape[2] > 32:  # [H, W, C]
                    feat = feat.permute(2, 0, 1)  # → [C, H, W]

            # CNN-style (e.g., ResNet, EfficientNet)
            elif feat.dim() == 3:
                pass  # already [C, H, W]
            else:
                raise ValueError(f"Unexpected feature shape: {feat.shape}")



            pred_class = preds[i].item()
            
            all_weights = get_classifier_weights(model, model_name=args.model)  # ← pass args.model
            class_weights = all_weights[pred_class]

            # Compute raw CAM
            cam = (class_weights.view(-1, 1, 1) * feat).sum(0)  # [H_f, W_f]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-5)  # Normalize

            # Resize to original input size
            cam_resized = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                size=(inputs.shape[2], inputs.shape[3]),
                mode="bilinear", align_corners=False
            )[0, 0]  # [H, W]

            # Convert input image to numpy
            input_np = inputs[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-5)

            # Convert CAM to RGB heatmap
            cam_np = cam_resized.cpu().numpy()
            cam_rgb = cm.get_cmap("jet")(cam_np)[..., :3]  # Drop alpha channel

            # Overlay CAM on input image
            overlay = (0.5 * cam_rgb + 0.5 * input_np)
            overlay = np.clip(overlay, 0, 1)
            overlay = (overlay * 255).astype("uint8")

            # Save
            base_name = os.path.basename(image_paths[i])
            class_name = f"class{pred_class}"
            save_path = os.path.join(save_dir, f"{os.path.splitext(base_name)[0]}_{class_name}_cam.png")
            plt.imsave(save_path, overlay)
        except Exception as e:
            print(f"[WARN] Failed to generate CAM for image {image_paths[i]}: {e}")


def run_inference(model, image_paths, args, batch_size=3, input_size=(224, 224), normalize=True):
    """
    Args:
        model: Trained PyTorch model
        image_paths: List of image file paths
        args: Arguments object with `.device`, `.channels_last`, `.native_amp`, `.ampere_sparsity`
        batch_size: Number of images per batch
        input_size: Expected model input size (H, W)
        normalize: Whether to apply (x - 0.5) / 0.5 normalization
    
    Returns:
        all_preds: List of predicted class indices
        all_confs: List of confidence scores
    """

    feature_map = {}
    register_cam_hook(model, args.model, feature_map)

    model.eval()
    if args.ampere_sparsity:
        model.enforce_mask()

    all_preds = []
    all_confs = []

    def preprocess(path):
        img = Image.open(path).convert("RGB").resize(input_size)
        img = np.array(img).astype(np.float32) / 255.0  # [H, W, C]
        if normalize:
            img = (img - 0.5) / 0.5  # normalize to [-1, 1]
        tensor = torch.tensor(img).permute(2, 0, 1)  # [C, H, W]
        return tensor

    tensors = [preprocess(p) for p in image_paths]

    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i + batch_size]
            inputs = torch.stack(batch).to(args.device)

            if args.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            with torch.cuda.amp.autocast() if args.native_amp else torch.inference_mode():
                outputs = model(inputs)

                x = torch.randn(1, 3, 224, 224).to(args.device)
                flops, params = profile(model, inputs=(x,))
                print(f"FLOPS: {flops} PARAMS: {params}")
                start = time.time()
                _ = model(x)
                elapsed = (time.time() - start)
                print('Inference TIME', elapsed * 1000, "ms")

            probs = F.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            if "feat" in feature_map:
                generate_and_save_cam(
                    args,
                    inputs=inputs,
                    feature_map=feature_map["feat"],
                    preds=preds,
                    model=model,
                    image_paths=image_paths[i:i + batch_size],
                    save_dir="/home/alexandra/Documents/MambaVision/brease_demo/cam_outputs/"
                )

            for j, path in enumerate(image_paths[i:i + batch_size]):
                print(f"{os.path.basename(path)} → Class: {preds[j].item()}, Confidence: {confidences[j].item():.4f}")


            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confidences.cpu().tolist())

    return all_preds, all_confs


if __name__ == '__main__':
    main()


#  python3 val_simple.py -c /home/alexandra/Documents/MambaVision/best-model-article-1/args.yaml --loadcheckpoint /home/alexandra/Documents/MambaVision/best-model-article-1/model_best.pth.tar --validate_only


# find /home/alexandra/Documents/Datasets/BUSI_split/train/benign -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l
# find /home/alexandra/Documents/Datasets/BUSI_split/train/malignant -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l
# find /home/alexandra/Documents/Datasets/BUSI_split/train/normal -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l

# find /home/alexandra/Documents/Datasets/BUSI_split/val/benign -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l
# find /home/alexandra/Documents/Datasets/BUSI_split/val/malignant -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l
# find /home/alexandra/Documents/Datasets/BUSI_split/val/normal -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l


# python3 val_simple.py -c /home/alexandra/Documents/MambaVision/mambavision/output_busi/mamba_vision_T2_BUSI/args.yaml --loadcheckpoint /home/alexandra/Documents/MambaVision/mambavision/output_busi/mamba_vision_T2_BUSI/checkpoint-87.pth.tar --validate_only