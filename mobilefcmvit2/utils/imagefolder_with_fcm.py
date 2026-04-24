import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderWithFCM(Dataset):
    """ImageFolder-style dataset that concatenates a precomputed FCM map as a 4th channel."""

    def __init__(self, root, fcm_dir, transform=None, fcm_transform=None):
        self.root = root
        self.fcm_dir = fcm_dir
        self.transform = transform
        # FCM transform: resize to match image output size, then ToTensor
        if fcm_transform is not None:
            self.fcm_transform = fcm_transform
        else:
            # Infer target size from the image transform if possible
            target_size = 224  # default
            if transform is not None:
                for t in getattr(transform, 'transforms', []):
                    if hasattr(t, 'size'):  # CenterCrop / Resize
                        s = t.size
                        target_size = s if isinstance(s, int) else s[0]
                    elif hasattr(t, 'crop_size'):  # RandomResizedCrop
                        s = t.crop_size
                        target_size = s if isinstance(s, int) else s[0]
            self.fcm_transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
            ])
        self.samples = self._find_samples()

    def _find_samples(self):
        classes = sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)))
        samples = []
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.root, class_name)
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((os.path.join(class_dir, fname), class_idx))
        return samples

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        base = os.path.splitext(os.path.basename(path))[0]
        fcm_path = os.path.join(self.fcm_dir, base + '.png')

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        if os.path.exists(fcm_path):
            fcm = Image.open(fcm_path).convert('L')
            fcm_tensor = self.fcm_transform(fcm)
        else:
            # Fallback: zero channel if FCM map not found
            fcm_tensor = torch.zeros(1, img_tensor.shape[1], img_tensor.shape[2])

        if fcm_tensor.ndim == 2:
            fcm_tensor = fcm_tensor.unsqueeze(0)

        return torch.cat([img_tensor, fcm_tensor], dim=0), target

    def __len__(self):
        return len(self.samples)
