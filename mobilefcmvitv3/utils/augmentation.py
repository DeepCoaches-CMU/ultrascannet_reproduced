"""
mobilefcmvitv3/utils/augmentation.py

Training and validation transforms for MobileFCMViTv3 on BUSI.
Mirrors the augmentation strategy from the paper config.
"""

import torchvision.transforms as T
from typing import List


def build_train_transform(img_size: int = 224,
                          scale: tuple = (0.08, 1.0),
                          ratio: tuple = (0.75, 1.3333),
                          hflip: float = 0.5,
                          color_jitter: float = 0.2,
                          mean: tuple = (0.485, 0.456, 0.406),
                          std: tuple = (0.229, 0.224, 0.225)) -> T.Compose:
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
    resize_size = int(img_size / crop_pct)
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_tta_transforms(img_size: int = 224,
                         mean: tuple = (0.485, 0.456, 0.406),
                         std: tuple = (0.229, 0.224, 0.225)) -> List[T.Compose]:
    """
    Returns a list of deterministic transforms for Test-Time Augmentation.
    Predictions are averaged over all views.

    Views:
      0. centre crop (standard val)
      1. horizontal flip + centre crop
      2. vertical flip + centre crop
      3. horizontal + vertical flip
      4. 5-crop top-left
      5. 5-crop top-right
      6. 5-crop bottom-left
      7. 5-crop bottom-right
    """
    resize_size = int(img_size / 0.875)
    normalize = T.Normalize(mean=mean, std=std)

    def _make(flips, crop_fn):
        ops = [T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC)]
        ops += flips
        ops += [crop_fn, T.ToTensor(), normalize]
        return T.Compose(ops)

    centre = T.CenterCrop(img_size)
    tl = T.Lambda(lambda img: T.functional.crop(img, 0, 0, img_size, img_size))
    tr = T.Lambda(lambda img: T.functional.crop(img, 0, img.size[0] - img_size, img_size, img_size))
    bl = T.Lambda(lambda img: T.functional.crop(img, img.size[1] - img_size, 0, img_size, img_size))
    br = T.Lambda(lambda img: T.functional.crop(
        img, img.size[1] - img_size, img.size[0] - img_size, img_size, img_size))

    hflip = T.RandomHorizontalFlip(p=1.0)
    vflip = T.RandomVerticalFlip(p=1.0)

    return [
        _make([],            centre),   # 0: standard
        _make([hflip],       centre),   # 1: h-flip
        _make([vflip],       centre),   # 2: v-flip
        _make([hflip, vflip], centre),  # 3: both flips
        _make([],            tl),       # 4: top-left
        _make([],            tr),       # 5: top-right
        _make([],            bl),       # 6: bottom-left
        _make([],            br),       # 7: bottom-right
    ]
