from timm import create_model
from models.ultra_scan_net import *
from models.mamba_vision_baseline import mamba_vision_T2_baseline
import torch 

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def check_model_params():
    model_names = {
        "resnet50": "resnet50",
        "mobilenetv2_100": "mobilenetv2_100",
        "vit_small": "vit_small_patch16_224",
        "densenet121": "densenet121",
        "efficientnet_b0": "efficientnet_b0",
        "convnext_tiny": "convnext_tiny",
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "deit_tiny": "deit_tiny_patch16_224",
        "maxvit_tiny": "maxvit_tiny_rw_224"
    }

    print("Model Parameters (Total | Trainable)")
    print("-" * 40)

    for name, model_id in model_names.items():
        model = create_model(model_id, pretrained=True, num_classes=3)
        total, trainable = count_params(model)
        print(f"{name:<20}: {total:,} | {trainable:,}")

def check_mamba_params():
    # mamba_baseline_path = '/home/alexandra/Documents/MambaVision/mambavision/ckpts/mambavision_tiny2_1k.pth.tar'
    mamba_updated_path = '/home/alexandra/Documents/MambaVision/mambavision/output_busi/mamba_vision_T2_BUSI/checkpoint-87.pth.tar'
    checkpoint = torch.load(mamba_updated_path, weights_only=False)
    # Extract state dict (if wrapped in a dict)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Total param count
    total_params = sum(v.numel() for v in state_dict.values())

    print(f"Total parameters in checkpoint: {total_params:,}")

    # Optional: list all tensor names and shapes
    for k, v in state_dict.items():
        print(f"{k:60} {tuple(v.shape)}  -> {v.numel():,} params")



check_model_params()

# Baseline Mamba 35,107,763
# Updated Mamba 37,204,206

# resnet50            : 23,514,179 | 23,514,179
# mobilenetv2_100     : 2,227,715 | 2,227,715
# vit_small           : 21,666,819 | 21,666,819
# densenet121         : 6,956,931 | 6,956,931
# efficientnet_b0     : 4,011,391 | 4,011,391
# convnext_tiny       : 27,822,435 | 27,822,435
# swin_tiny           : 27,521,661 | 27,521,661
# deit_tiny           : 5,524,995 | 5,524,995
# maxvit_tiny         : 28,545,851 | 28,545,851
