import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import config as cfg
from src.dataset import RoadSegmentationDataset
from src.metrics import calculate_metrics, average_metrics
from models.architecture import build_model


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

    _, val_files = cfg.get_train_val_split()
    dataset = RoadSegmentationDataset(data_dir=cfg.DATA_DIR, augment=False, file_list=val_files, img_size=cfg.IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    all_metrics = []
    print(f"\nEvaluating on {len(dataset)} held-out val images ({cfg.VAL_SPLIT*100:.0f}% split)...")

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            for i in range(images.size(0)):
                all_metrics.append(calculate_metrics(preds[i], masks[i]))

    avg = average_metrics(all_metrics)
    print("\n" + "=" * 30)
    print("      VALIDATION RESULTS")
    print("=" * 30)
    for key, label in [("iou", "Mean IoU"), ("accuracy", "Pixel Accuracy"),
                        ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 Score")]:
        print(f"{label:<16} {avg[key]:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    validate()
