import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101


def build_model(backbone: str = "resnet50") -> nn.Module:
    """
    Build a DeepLabV3 model configured for binary road segmentation (2 classes).

    Args:
        backbone: "resnet50" (default) or "resnet101".
                  mobilenet_v3_large support is reserved for Plan 2 (different head structure).

    Returns:
        DeepLabV3 model with ImageNet-pretrained backbone and a 2-class head.
        Weights are downloaded on first call.
    """
    if backbone == "resnet101":
        model = deeplabv3_resnet101(weights="DEFAULT")
    else:
        model = deeplabv3_resnet50(weights="DEFAULT")

    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    return model
