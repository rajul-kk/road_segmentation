import torch
import numpy as np


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Binary segmentation metrics for a single (H, W) prediction/target pair.

    Args:
        pred:   Class tensor (H, W) with values 0 or 1.
        target: Ground-truth class tensor (H, W) with values 0 or 1.

    Returns:
        dict with keys: iou, accuracy, precision, recall, f1
    """
    pred   = pred.view(-1)
    target = target.view(-1)

    tp = torch.sum((pred == 1) & (target == 1)).item()
    tn = torch.sum((pred == 0) & (target == 0)).item()
    fp = torch.sum((pred == 1) & (target == 0)).item()
    fn = torch.sum((pred == 0) & (target == 1)).item()

    eps = 1e-7
    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)

    return {"iou": iou, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def average_metrics(metrics_list: list) -> dict:
    """Average a list of per-image metric dicts into one summary dict."""
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
