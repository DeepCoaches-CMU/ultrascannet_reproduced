"""Code for getting the data loaders."""

import io
import os
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import accumulate as _accumulate
from PIL import Image
from timm.data import IterableImageDataset, ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from .imagefolder_with_fcm import ImageFolderWithFCM


def load_image_with_fcm(image_path, fcm_dir, transform=None, fcm_transform=None):
    """
    Loads an image and its corresponding FCM map, concatenates as 4 channels.
    Args:
        image_path: Path to the original image (PNG/JPG).
        fcm_dir: Directory containing FCM PNGs with same basename as image_path.
        transform: Transform to apply to the image (should output 3xHxW tensor).
        fcm_transform: Transform to apply to the FCM map (should output 1xHxW tensor).
    Returns:
        4xHxW tensor (image + FCM map)
    """
    img = Image.open(image_path).convert('RGB')
    base = os.path.splitext(os.path.basename(image_path))[0]
    fcm_path = os.path.join(fcm_dir, base + '.png')
    fcm = Image.open(fcm_path).convert('L')
    if transform:
        img = transform(img)
    else:
        img = transforms.ToTensor()(img)
    if fcm_transform:
        fcm = fcm_transform(fcm)
    else:
        fcm = transforms.ToTensor()(fcm)
    if fcm.ndim == 2:
        fcm = fcm.unsqueeze(0)
    return torch.cat([img, fcm], dim=0)


def get_loaders(args, mode='eval', dataset=None):
    """Get data loaders for required dataset."""
    if dataset is None:
        dataset = args.dataset
    if dataset == 'imagenet':
        return get_imagenet_loader(args, mode)
    else:
        if mode == 'search':
            return get_loaders_search(args)
        elif mode == 'eval':
            return get_loaders_eval(dataset, args)


class Subset_imagenet(torch.utils.data.Dataset):
    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = None

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.indices)


def get_loaders_eval(dataset, args):
    """Get train and valid loaders for cifar10/tiny imagenet."""
    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'cifar100':
        num_classes = 100
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        sampler=valid_sampler, pin_memory=True, num_workers=16)
    return train_queue, valid_queue, num_classes


def get_loaders_search(args):
    """Get train and valid loaders for cifar10/tiny imagenet."""
    if args.dataset == 'cifar10':
        num_classes = 10
        train_transform, _ = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_transform, _ = _data_transforms_cifar10(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    sub_num_train = int(np.floor(args.train_portion * num_train))
    sub_num_valid = num_train - sub_num_train
    sub_train_data, sub_valid_data = my_random_split(train_data, [sub_num_train, sub_num_valid], seed=0)

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(sub_train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(sub_valid_data)

    train_queue = torch.utils.data.DataLoader(
        sub_train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=16, drop_last=True)
    valid_queue = torch.utils.data.DataLoader(
        sub_valid_data, batch_size=args.batch_size, shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=16, drop_last=True)
    return train_queue, valid_queue, num_classes


def get_imagenet_loader(args, mode='eval', testdir=""):
    """Get train/val for imagenet."""
    traindir = os.path.join(args.data_dir, 'train')
    validdir = os.path.join(args.data_dir, 'validation')
    if len(testdir) < 2:
        testdir = os.path.join("../ImageNetV2/", 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    downscale = 1

    val_transform = transforms.Compose([
        transforms.Resize(256 // downscale),
        transforms.CenterCrop(224 // downscale),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224 // downscale),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    fcm_dir = getattr(args, 'fcm_dir', None)

    if mode == 'eval':
        if fcm_dir:
            train_data = ImageFolderWithFCM(traindir, fcm_dir, transform=train_transform)
            valid_data = ImageFolderWithFCM(validdir, fcm_dir, transform=val_transform)
        elif getattr(args, 'lmdb_dataset', False):
            train_data = imagenet_lmdb_dataset(traindir, transform=train_transform)
            valid_data = imagenet_lmdb_dataset(validdir, transform=val_transform)
        else:
            train_data = dset.ImageFolder(traindir, transform=train_transform)
            valid_data = dset.ImageFolder(validdir, transform=val_transform)

        train_sampler, valid_sampler = None, None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
            pin_memory=True, num_workers=args.workers, sampler=train_sampler, drop_last=True)
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=(valid_sampler is None),
            pin_memory=True, num_workers=args.workers, sampler=valid_sampler)

    return train_queue, valid_queue, 1000


def my_random_split(dataset, lengths, seed=0):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(sum(lengths), generator=g)
    return [Subset_imagenet(dataset, indices[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)]


def imagenet_lmdb_dataset(root, transform=None, target_transform=None, loader=None):
    try:
        import lmdb
        from torchvision import datasets

        def lmdb_loader(path, lmdb_data):
            with lmdb_data.begin(write=False, buffers=True) as txn:
                bytedata = txn.get(path.encode('ascii'))
            img = Image.open(io.BytesIO(bytedata))
            return img.convert('RGB')

        if root.endswith('/'):
            root = root[:-1]
        pt_path = root + '_faster_imagefolder.lmdb.pt'
        lmdb_path = root + '_faster_imagefolder.lmdb'
        if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
            data_set = torch.load(pt_path)
        else:
            data_set = datasets.ImageFolder(root, None, None, None)
            torch.save(data_set, pt_path, pickle_protocol=4)
            env = lmdb.open(lmdb_path, map_size=1e12)
            with env.begin(write=True) as txn:
                for path, class_index in data_set.imgs:
                    with open(path, 'rb') as f:
                        data = f.read()
                    txn.put(path.encode('ascii'), data)
        data_set.lmdb_data = lmdb.open(lmdb_path, readonly=True, max_readers=1, lock=False,
                                        readahead=False, meminit=False)
        data_set.samples = data_set.imgs
        data_set.transform = transform
        data_set.target_transform = target_transform
        data_set.loader = lambda path: lmdb_loader(path, data_set.lmdb_data)
        return data_set
    except ImportError:
        raise ImportError("lmdb is required for lmdb datasets. Install with: pip install lmdb")
