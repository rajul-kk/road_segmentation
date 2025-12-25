"""
Mask Loading and Processing Utilities

This module handles loading and processing binary masks for road segmentation.
"""

import os
import torch
from torchvision import transforms
from PIL import Image


def load_mask(mask_path, transform=None):
    """
    Load a binary mask from file path.
    
    Args:
        mask_path: Path to the mask image file
        transform: Optional torchvision transform to apply
    
    Returns:
        PIL Image (grayscale) if transform is None, otherwise transformed tensor
    """
    mask = Image.open(mask_path).convert('L')  # Grayscale for binary mask
    
    if transform:
        mask = transform(mask)
    
    return mask


def process_mask_for_training(mask, to_tensor=None):
    """
    Process mask for training: convert to tensor, binarize, and convert to class indices.
    
    Args:
        mask: PIL Image or tensor of the mask
        to_tensor: transforms.ToTensor() instance (if mask is PIL Image)
    
    Returns:
        Class indices tensor: (height, width) with values 0 (background) or 1 (road)
    """
    # Convert to tensor if PIL Image
    if isinstance(mask, Image.Image):
        if to_tensor is None:
            to_tensor = transforms.ToTensor()
        mask = to_tensor(mask)
    
    # Convert mask to binary (0 or 1)
    mask = (mask > 0.5).float()
    
    # Convert to class indices: 0 for background, 1 for road
    mask_class = mask.squeeze(0).long()  # Shape: (height, width) with values 0 or 1
    
    return mask_class


def load_and_process_mask(mask_path, transform=None, to_tensor=None):
    """
    Load and process a mask in one step.
    
    Args:
        mask_path: Path to the mask image file
        transform: Optional torchvision transform to apply before processing
        to_tensor: transforms.ToTensor() instance (used if transform is None)
    
    Returns:
        Class indices tensor: (height, width) with values 0 (background) or 1 (road)
    """
    mask = load_mask(mask_path, transform)
    
    # If transform was applied, mask is already a tensor
    # Otherwise, we need to convert it
    if transform is None:
        mask = process_mask_for_training(mask, to_tensor)
    else:
        # Transform already converted to tensor, just process it
        mask = (mask > 0.5).float()
        mask_class = mask.squeeze(0).long()
        return mask_class
    
    return mask

