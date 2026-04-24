"""
mobilefcmvitv3/utils/dataset.py

Standard ImageFolder-style dataset for BUSI (3-channel RGB only).
FCM is now applied in feature space inside the model, so no 4th input
channel is needed here.
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class BUSIDataset(Dataset):
    """
    ImageFolder-style dataset returning (3, H, W) RGB tensors.

    Args:
        root:      path to split root (contains class subdirectories)
        transform: torchvision transform applied to the PIL image
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples, self.classes = self._find_samples()

    def _find_samples(self):
        classes = sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )
        samples = []
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.root, class_name)
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((os.path.join(class_dir, fname), class_idx))
        return samples, classes

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)


# Keep the old name as an alias so existing imports don't break
BUSIDatasetWithFCM = BUSIDataset
