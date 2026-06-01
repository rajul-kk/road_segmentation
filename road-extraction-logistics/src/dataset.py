import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image
import numpy as np

from src.preprocessing import apply_clahe


class RoadSegmentationDataset(Dataset):
    """
    Custom dataset for road segmentation.
    Expects DeepGlobe format: images (*_sat.jpg) and masks (*_mask.png) in the same directory.
    """

    def __init__(self, data_dir, image_transform=None, mask_transform=None,
                 normalize=True, augment=True, use_clahe=False, file_list=None,
                 img_size=None):
        """
        Args:
            data_dir: Directory containing satellite images (*_sat.jpg) and masks (*_mask.png).
            image_transform: Optional transform applied to the image only (e.g. colour jitter).
            mask_transform: Optional transform applied to the mask only.
            normalize: Normalize image to ImageNet stats required by DeepLabV3.
            augment: Apply joint spatial augmentations (flips + 90/180/270 rotations).
            use_clahe: Apply CLAHE contrast enhancement before any other processing.
            file_list: Optional explicit list of *_sat.jpg filenames. When provided the
                       directory is not scanned — useful for reproducible train/val splits.
            img_size: If set, resize both image and mask to (img_size, img_size) before
                      augmentation. Recommended: 512 (halves memory vs native 1024×1024).
        """
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.normalize = normalize
        self.augment = augment
        self.use_clahe = use_clahe
        self.resize = transforms.Resize((img_size, img_size)) if img_size else None

        if file_list is not None:
            self.image_files = sorted(file_list)
        else:
            self.image_files = sorted(
                f for f in os.listdir(data_dir) if f.endswith('_sat.jpg')
            )

        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.data_dir, img_name)).convert('RGB')

        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        mask = Image.open(os.path.join(self.data_dir, mask_name)).convert('L')

        # Resize before any processing (joint, so mask stays aligned)
        if self.resize:
            image = self.resize(image)
            mask  = self.resize(mask)

        if self.use_clahe:
            image = apply_clahe(image)

        # Joint spatial augmentation (image and mask transformed identically)
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # Per-modality transforms (kept separate to avoid corrupting mask values)
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        image = self.to_tensor(image)
        if self.normalize:
            image = self.normalize_transform(image)

        mask = self.to_tensor(mask)
        mask = (mask > 0.5).long().squeeze(0)  # (H, W) with class indices 0/1

        return image, mask
