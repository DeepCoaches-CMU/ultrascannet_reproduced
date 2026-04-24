"""
mobilefcmvitv3/utils/augmentation.py

Training and validation transforms for MobileFCMViTv3 on BUSI.
Mirrors the augmentation strategy from the paper config.
"""

import torchvision.transforms as T


def build_train_transform(img_size: int = 224,
                          scale: tuple = (0.08, 1.0),
                          ratio: tuple = (0.75, 1.3333),
                          hflip: float = 0.5,
                          color_jitter: float = 0.2,
                          mean: tuple = (0.485, 0.456, 0.406),
                          std: tuple = (0.229, 0.224, 0.225)) -> T.Compose:
    """
    Training transform pipeline matching the paper's augmentation config:
      - RandomResizedCrop
      - RandomHorizontalFlip
      - ColorJitter (brightness/contrast/saturation)
      - ToTensor + Normalize
    """
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=scale, ratio=ratio,
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=hflip),
        T.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                      saturation=color_jitter),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_val_transform(img_size: int = 224,
                        crop_pct: float = 0.875,
                        mean: tuple = (0.485, 0.456, 0.406),
                        std: tuple = (0.229, 0.224, 0.225)) -> T.Compose:
    """
    Validation transform: resize → centre-crop → normalize.
    """
    resize_size = int(img_size / crop_pct)
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
