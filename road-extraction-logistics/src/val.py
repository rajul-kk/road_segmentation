import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import config as cfg
from src.dataset import RoadSegmentationDataset
from models.architecture import build_model


def calculate_metrics(pred, target):
    """
    Calculate binary segmentation metrics.

    Args:
        pred:   Predicted class tensor (H, W) with values 0/1.
        target: Ground-truth class tensor (H, W) with values 0/1.

    Returns:
        dict with keys: iou, accuracy, precision, recall, f1
    """
    pred = pred.view(-1)
    target = target.view(-1)

    tp = torch.sum((pred == 1) & (target == 1)).item()
    tn = torch.sum((pred == 0) & (target == 0)).item()
    fp = torch.sum((pred == 1) & (target == 0)).item()
    fn = torch.sum((pred == 0) & (target == 1)).item()

    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou       = tp / (tp + fp + fn + 1e-7)

    return {"iou": iou, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def validate():
    device = cfg.DEVICE
    print(f"Using device: {device}")

    # Try canonical path then known fallbacks (different locations used across training runs)
    fallback_paths = [cfg.FINAL_MODEL_PATH, "models/DeeplabsV3.pth", "models/DeeplabsV3_road_final.pth"]
    model_path = next((p for p in fallback_paths if os.path.exists(p)), None)
    if model_path is None:
        print(f"❌ Model not found at any known path: {fallback_paths}")
        return
    if model_path != cfg.FINAL_MODEL_PATH:
        print(f"📂 Using fallback model: {model_path}")

    print(f"Loading weights from {model_path}...")
    model = build_model(backbone=cfg.BACKBONE).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    dataset = RoadSegmentationDataset(data_dir=cfg.DATA_DIR, augment=False)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    all_metrics = []
    print(f"\nEvaluating on {len(dataset)} images...")

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            for i in range(images.size(0)):
                all_metrics.append(calculate_metrics(preds[i], masks[i]))

    print("\n" + "=" * 30)
    print("      VALIDATION RESULTS")
    print("=" * 30)
    for key, label in [("iou", "Mean IoU"), ("accuracy", "Pixel Accuracy"),
                        ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 Score")]:
        print(f"{label:<16} {np.mean([m[key] for m in all_metrics]):.4f}")
    print("=" * 30)


if __name__ == "__main__":
    validate()
