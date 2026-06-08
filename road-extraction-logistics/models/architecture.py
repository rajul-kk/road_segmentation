import torch
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

    Variants (all pretrained on ImageNet-1k):
        b0  —  3.7 M   — fastest, smallest
        b2  —  25.4 M  — best size/accuracy trade-off  ← recommended
        b5  —  82.0 M  — highest accuracy

    Returns:
        _SegFormerWrapper whose forward() returns {'out': (B, C, H, W)}.
    """
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError:
        raise ImportError("Install with:  pip install transformers")

    hf_model = SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/mit-{variant}",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return _SegFormerWrapper(hf_model)


# ── Mask2Former ───────────────────────────────────────────────────────────────

class _Mask2FormerInferenceWrapper(nn.Module):
    """
    Inference-only wrapper that aggregates Mask2Former's query-based outputs
    into a standard semantic logit map: model(x) → {'out': (B, C, H, W)}.

    Mask2Former produces N learnable query predictions, each with:
      - class_queries_logits: (B, Q, num_labels + 1)   (+1 = no-object class)
      - masks_queries_logits: (B, Q, H/4, W/4)

    We upsample the masks, convert to probabilities, then aggregate:
        semantic_logit[b, c, h, w] = Σ_q  class_prob[b,q,c] × mask_prob[b,q,h,w]

    Used by run_inference.py and app.py.
    For training use the raw model returned by build_mask2former() directly.
    """

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        h, w    = x.shape[-2:]
        outputs = self.model(pixel_values=x)

        mask_logits = F.interpolate(
            outputs.masks_queries_logits,
            size=(h, w), mode="bilinear", align_corners=False,
        )  # (B, Q, H, W)

        # Exclude the no-object class (last entry)
        class_probs = torch.softmax(outputs.class_queries_logits, dim=-1)[..., :-1]  # (B, Q, C)
        mask_probs  = torch.sigmoid(mask_logits)                                       # (B, Q, H, W)

        semantic_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        return {"out": semantic_logits}


def build_mask2former(backbone: str = "swin-t", num_classes: int = 2) -> nn.Module:
    """
    Mask2Former for semantic road segmentation.

    Returns the raw HuggingFace model so the training loop can pass
    class_labels + mask_labels and use the built-in Hungarian-matched loss.
    Wrap with _Mask2FormerInferenceWrapper for inference.

    Backbone options and P100 16 GB guidance at 512px:
        "swin-t"  — ~47 M params, batch=4  ← recommended starting point
        "swin-s"  — ~69 M params, batch=4
        "swin-b"  — ~107 M params, batch=2
    """
    try:
        from transformers import Mask2FormerForUniversalSegmentation
    except ImportError:
        raise ImportError("Install with:  pip install transformers")

    model_ids = {
        "swin-t": "facebook/mask2former-swin-tiny-ade-semantic",
        "swin-s": "facebook/mask2former-swin-small-ade-semantic",
        "swin-b": "facebook/mask2former-swin-base-ade-semantic",
    }
    if backbone not in model_ids:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose: {list(model_ids)}")

    return Mask2FormerForUniversalSegmentation.from_pretrained(
        model_ids[backbone],
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )


# ── Unified model loader ───────────────────────────────────────────────────────

def load_model(
    model_path: str = None,
    model_type: str = None,
    backbone: str = None,
    variant: str = None,
    device=None,
) -> nn.Module:
    """
    Build and load a trained segmentation model from a checkpoint.

    Reads MODEL_TYPE from config by default; any arg can be overridden by callers.

    Args:
        model_path: Path to .pth checkpoint. Defaults to cfg path for the chosen type.
        model_type: "segformer" or "deeplabv3". Defaults to cfg.MODEL_TYPE.
        backbone:   DeepLabV3 backbone ("resnet50" / "resnet101"). Ignored for SegFormer.
        variant:    SegFormer variant ("b0"–"b5"). Ignored for DeepLabV3.
        device:     torch.device. Defaults to cfg.DEVICE.

    Returns:
        Model in eval() mode on the requested device.
    """
    import os
    import config as cfg

    model_type = model_type or cfg.MODEL_TYPE
    device     = device     or cfg.DEVICE

    if model_type == "segformer":
        variant    = variant    or cfg.SEGFORMER_VARIANT
        model_path = model_path or cfg.SEGFORMER_MODEL_PATH
        model      = build_segformer(variant=variant).to(device)
    else:
        backbone   = backbone   or cfg.BACKBONE
        model_path = model_path or cfg.FINAL_MODEL_PATH
        model      = build_model(backbone=backbone).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt  = torch.load(model_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    iou   = ckpt.get("val_iou")
    tag   = f"epoch={epoch}" + (f", val_iou={iou:.4f}" if iou else "")
    print(f"Loaded {model_type} ({tag}) from {model_path}")
    return model
