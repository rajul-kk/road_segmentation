"""
DeepLabV3 Architecture for Road Segmentation

DeepLabV3 is a state-of-the-art semantic segmentation model that uses:
- Atrous (dilated) convolutions to capture multi-scale context
- Atrous Spatial Pyramid Pooling (ASPP) for robust feature extraction
- Pre-trained backbone (typically ResNet) for feature extraction

How to Use DeepLabV3:

1. Import from torchvision:
   from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
   from torchvision.models.segmentation.deeplabv3 import DeepLabV3

2. Load Pre-trained Model:
   model = deeplabv3_resnet50(pretrained=True, num_classes=2)  # 2 classes: road and background
   # or
   model = deeplabv3_resnet101(pretrained=True, num_classes=2)

3. Modify for Binary Segmentation:
   - Set num_classes=2 (road vs background)
   - The model outputs logits, apply sigmoid for binary classification
   - Or use softmax if treating as 2-class classification

4. Input Requirements:
   - Input shape: (batch_size, 3, height, width) - RGB images
   - Normalized to ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - Or normalize to [0, 1] range and adjust accordingly

5. Output:
   - Output shape: (batch_size, num_classes, height, width)
   - For binary: Apply torch.sigmoid() or torch.softmax(dim=1)
   - Resize output to match input image size if needed

6. Training:
   - Freeze backbone initially (optional): model.backbone.requires_grad_(False)
   - Use appropriate loss: BCEWithLogitsLoss, CrossEntropyLoss, or Dice Loss
   - Fine-tune on road segmentation dataset

Libraries Required:
- torch: Core PyTorch library
- torch.nn: Neural network modules
- torchvision: Contains pre-trained DeepLabV3 models
- torchvision.models.segmentation: DeepLabV3 implementations
- torchvision.transforms: For image preprocessing and normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

# Load DeepLabV3 model for road segmentation (2 classes: road and background)
model = deeplabv3_resnet50(pretrained=True, num_classes=2)

def forward_with_binary_output(model, x, use_sigmoid=True):
    """
    Forward pass with binary segmentation output.
    
    Args:
        model: DeepLabV3 model
        x: Input tensor (batch_size, 3, height, width)
        use_sigmoid: If True, use sigmoid for binary classification. 
                    If False, use softmax for 2-class classification.
    
    Returns:
        Binary mask predictions (batch_size, 1, height, width) if use_sigmoid=True
        or (batch_size, 2, height, width) if use_sigmoid=False
    """
    # Get model output (logits)
    output = model(x)['out']  # Shape: (batch_size, 2, height, width)
    
    if use_sigmoid:
        # For binary classification: apply sigmoid to get probabilities
        # Take the road class (index 1) and apply sigmoid
        binary_mask = torch.sigmoid(output[:, 1:2, :, :])  # Shape: (batch_size, 1, height, width)
        return binary_mask
    else:
        # For 2-class classification: apply softmax across classes
        probs = F.softmax(output, dim=1)  # Shape: (batch_size, 2, height, width)
        return probs
