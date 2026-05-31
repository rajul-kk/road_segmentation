import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101


def build_model(backbone: str = "resnet50") -> nn.Module:
    """
    Build a DeepLabV3 model configured for binary road segmentation (2 classes).

    Args:
        backbone: One of:
            "resnet50"  — lighter, faster (default)
            "resnet101" — deeper, better accuracy, ~2x parameters

    Returns:
        DeepLabV3 model with ImageNet-pretrained backbone and a 2-class head.
        Weights are downloaded on first call.
    """
    if backbone == "resnet101":
        model = deeplabv3_resnet101(weights="DEFAULT")
    elif backbone == "resnet50":
        model = deeplabv3_resnet50(weights="DEFAULT")
    else:
        raise ValueError(f"Unsupported backbone: {backbone!r}. Choose 'resnet50' or 'resnet101'.")

    model.classifier[4]     = nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    return model
