import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101


# ── DeepLabV3 ─────────────────────────────────────────────────────────────────

def build_model(backbone: str = "resnet50") -> nn.Module:
    """
    DeepLabV3 for binary road segmentation (2 classes).

    Args:
        backbone: "resnet50" (default) or "resnet101" (higher accuracy, ~2× params).

    Returns:
        Model whose forward() returns {'out': (B, 2, H, W)} logits — same resolution
        as input, no need for external upsampling.
    """
    if backbone == "resnet101":
        model = deeplabv3_resnet101(weights="DEFAULT")
    elif backbone == "resnet50":
        model = deeplabv3_resnet50(weights="DEFAULT")
    else:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose 'resnet50' or 'resnet101'.")

    model.classifier[4]     = nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    return model


# ── SegFormer ─────────────────────────────────────────────────────────────────

class _SegFormerWrapper(nn.Module):
    """
    Thin wrapper that makes HuggingFace SegFormer look identical to DeepLabV3
    from the outside:  model(x) → {'out': (B, num_classes, H, W)}

    SegFormer's decoder head outputs at H/4 × W/4.  We bilinearly upsample
    back to the original input resolution here so the loss and metrics code
    needs no changes.
    """

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        h, w   = x.shape[-2:]
        logits = self.model(pixel_values=x).logits          # (B, C, H/4, W/4)
        logits = F.interpolate(logits, size=(h, w),
                               mode="bilinear", align_corners=False)
        return {"out": logits}


def build_segformer(variant: str = "b2", num_classes: int = 2) -> nn.Module:
    """
    SegFormer for semantic segmentation, wrapped to match the DeepLabV3 interface.

    Variants and trade-offs (all pretrained on ImageNet-1k):
        b0  —  3.7 M params  — fastest, smallest, weakest
        b1  —  13.7 M params — light option
        b2  —  25.4 M params — best accuracy/size trade-off  ← recommended
        b3  —  44.6 M params — stronger encoder
        b4  —  64.1 M params — near-B5 accuracy, slightly cheaper
        b5  —  82.0 M params — highest accuracy, slowest

    Args:
        variant:     One of "b0" – "b5".
        num_classes: Output classes (default 2: road / background).

    Returns:
        _SegFormerWrapper whose forward() returns {'out': (B, C, H, W)}.

    Requires:
        pip install transformers
    """
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError:
        raise ImportError(
            "SegFormer requires the `transformers` package.\n"
            "Install with:  pip install transformers"
        )

    model_id = f"nvidia/mit-{variant}"
    hf_model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,   # replaces pretrained head with new one
    )
    return _SegFormerWrapper(hf_model)
