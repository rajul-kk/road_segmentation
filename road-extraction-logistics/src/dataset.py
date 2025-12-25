"""
Custom DataLoader for Road Segmentation Dataset

This module handles:
- Loading satellite images and their corresponding binary masks
- Image normalization (typically to [0, 1] or standardized values)
- Data augmentation (optional)
- Batching and shuffling for training

Libraries to Use:
- torch.utils.data.Dataset: Base class for custom datasets
- torch.utils.data.DataLoader: For batching and loading data
- torchvision.transforms: For image preprocessing and augmentation
- PIL / PIL.Image: For image loading
- numpy: For array operations
- os / pathlib: For file path handling
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from src.mask_loader import load_and_process_mask

class RoadSegmentationDataset(Dataset):
    """
    Custom dataset for road segmentation.
    Expects images in data/raw/train/images/ and masks in data/raw/train/masks/
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, normalize=True):
        """
        Args:
            image_dir: Path to directory containing satellite images
            mask_dir: Path to directory containing binary masks
            transform: Optional torchvision transforms for augmentation
            normalize: If True, normalize to ImageNet stats for DeepLabV3
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.normalize = normalize
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # ImageNet normalization stats (required for pre-trained DeepLabV3)
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Basic transform: convert to tensor
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load and process mask using mask_loader
        mask_path = os.path.join(self.mask_dir, img_name)
        mask_class = load_and_process_mask(
            mask_path, 
            transform=self.transform, 
            to_tensor=self.to_tensor
        )
        
        # Apply transforms to image if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor
            image = self.to_tensor(image)
        
        # Normalize image for DeepLabV3
        if self.normalize:
            image = self.normalize_transform(image)
        
        return image, mask_class
