import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.dataset import RoadSegmentationDataset
from models.architecture import model

# ============== CONFIGURATION ==============
MODEL_PATH = "models/checkpoints/DeeplabsV3_road_final.pth"
DATA_DIR = "data/raw/train"  # Using train dir but we will split or look for a val split if available
BATCH_SIZE = 4
# ===========================================

def calculate_metrics(pred, target):
    """
    Calculate segmentation metrics.
    
    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)
    
    Returns:
        dict: Metrics containing IoU, Pixel Accuracy, Precision, Recall, F1
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    tp = torch.sum((pred == 1) & (target == 1)).item()
    tn = torch.sum((pred == 0) & (target == 0)).item()
    fp = torch.sum((pred == 1) & (target == 0)).item()
    fn = torch.sum((pred == 0) & (target == 1)).item()
    
    # Pixel Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    
    # precision, recall, f1
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    # Intersection over Union (IoU)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-7)
    
    return {
        "iou": iou,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def validate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    # Try primary path then the flat models/ fallback (where train.py originally saved)
    fallback_paths = [MODEL_PATH, "models/DeeplabsV3.pth", "models/DeeplabsV3_road_final.pth"]
    current_model_path = next((p for p in fallback_paths if os.path.exists(p)), None)
    if current_model_path is None:
        print(f"❌ Model not found at any known path: {fallback_paths}")
        return
    if current_model_path != MODEL_PATH:
        print(f"📂 Primary path not found, using fallback: {current_model_path}")
        
    print(f"Loading weights from {current_model_path}...")
    checkpoint = torch.load(current_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Dataset
    dataset = RoadSegmentationDataset(data_dir=DATA_DIR)
    # For validation, we could use a slice or a separate val dir
    # Here we just use the whole dir for demonstration if no explicit val dir exists
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_metrics = []
    
    print(f"\nEvaluating on {len(dataset)} images...")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            # DeepLabV3 output is dict with 'out'
            outputs = model(images)['out']
            
            # Binary predictions
            # Softmax or Argmax over classes (2 classes: background, road)
            preds = torch.argmax(outputs, dim=1)  # Shape: (B, H, W)
            
            for i in range(images.size(0)):
                metrics = calculate_metrics(preds[i], masks[i])
                all_metrics.append(metrics)
    
    # Average metrics
    avg_iou = np.mean([m['iou'] for m in all_metrics])
    avg_acc = np.mean([m['accuracy'] for m in all_metrics])
    avg_prec = np.mean([m['precision'] for m in all_metrics])
    avg_rec = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    
    print("\n" + "="*30)
    print("      VALIDATION RESULTS")
    print("="*30)
    print(f"Mean IoU:       {avg_iou:.4f}")
    print(f"Pixel Accuracy: {avg_acc:.4f}")
    print(f"Precision:      {avg_prec:.4f}")
    print(f"Recall:         {avg_rec:.4f}")
    print(f"F1 Score:       {avg_f1:.4f}")
    print("="*30)

if __name__ == "__main__":
    validate()
