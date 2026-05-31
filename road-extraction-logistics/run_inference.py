"""
Road Segmentation Inference Script

This script loads a trained DeepLabV3 model and performs road segmentation
on satellite images. It processes images from the test directory and saves
predicted road masks.

Usage:
    python run_inference.py
"""

import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

from models.architecture import model as base_model
from src.post_process import clean_mask, get_skeleton
from src.preprocessing import apply_clahe

# ============== CONFIGURATION ==============
MODEL_PATH = "models/DeeplabsV3_road_final.pth"  # Trained model path
INPUT_DIR = "data/raw/test"                       # Directory with input images (*_sat.jpg or any RGB images)
OUTPUT_DIR = "data/masks/predicted"              # Directory to save predicted masks
CLEAN_DIR = "data/masks/cleaned"                 # Directory for cleaned masks
SKEL_DIR = "data/masks/skeletons"                # Directory for road centerlines
THRESHOLD = 0.5                                   # Probability threshold for road vs background (0.0-1.0)
USE_CLAHE = True                                   # Apply CLAHE contrast enhancement
# ============================================

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(SKEL_DIR, exist_ok=True)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Build model exactly like during training
print("Loading model architecture...")
model = base_model.to(device)
model.eval()

# Load checkpoint
print(f"Loading model weights from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)
# Handle both checkpoint format (with 'model_state_dict') and direct state_dict
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
else:
    model.load_state_dict(checkpoint)
    print("✅ Loaded model state dict")

print("Model loaded successfully!")

# Preprocessing: must match training (same as RoadSegmentationDataset)
to_tensor = T.ToTensor()
normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

def preprocess_image(img: Image.Image) -> torch.Tensor:
    """
    Preprocess image for DeepLabV3 model.
    
    Args:
        img: PIL Image (RGB)
    
    Returns:
        Preprocessed tensor with batch dimension (1, 3, H, W)
    """
    x = to_tensor(img.convert("RGB"))
    x = normalize(x)
    return x.unsqueeze(0)  # Add batch dimension

@torch.no_grad()
def predict_mask(img: Image.Image) -> Image.Image:
    """
    Predict road mask for an input image.
    
    Args:
        img: PIL Image (RGB)
    
    Returns:
        PIL Image with binary road mask (0 = background, 255 = road)
    """
    x = preprocess_image(img).to(device)
    
    # Forward pass
    output = model(x)['out']  # Shape: (1, 2, H, W) - 2 classes: background, road
    
    # Use road class (index 1) and apply sigmoid for binary classification
    road_logits = output[:, 1:2, :, :]  # Shape: (1, 1, H, W)
    road_probs = torch.sigmoid(road_logits)  # Shape: (1, 1, H, W)
    
    # Threshold to get binary mask
    mask = (road_probs > THRESHOLD).float().cpu().numpy()[0, 0]  # Shape: (H, W)
    
    # Convert to uint8 image (0 or 255)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_img

def is_image_file(fname: str) -> bool:
    """Check if file is a supported image format."""
    exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    return any(fname.lower().endswith(e) for e in exts)

def main():
    """Main inference function."""
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
    
    # Get list of image files
    files = [f for f in os.listdir(INPUT_DIR) if is_image_file(f)]
    
    if not files:
        print(f"⚠️  No images found in {INPUT_DIR}")
        return
    
    print(f"\n📁 Found {len(files)} images in {INPUT_DIR}")
    print(f"💾 Saving predictions to {OUTPUT_DIR}")
    print(f"🎯 Threshold: {THRESHOLD}")
    print("=" * 60)
    
    # Process and save each image one by one
    processed_count = 0
    skipped_count = 0
    
    for idx, fname in enumerate(files, 1):
        in_path = os.path.join(INPUT_DIR, fname)
        
        # Generate output filename first to check if already exists
        # For DeepGlobe format: if input is *_sat.jpg, output *_roadmask.png
        # Otherwise, add _roadmask suffix
        if fname.endswith("_sat.jpg"):
            mask_name = fname.replace("_sat.jpg", "_roadmask.png")
        else:
            name, ext = os.path.splitext(fname)
            mask_name = f"{name}_roadmask.png"
        
        out_path = os.path.join(OUTPUT_DIR, mask_name)
        clean_path = os.path.join(CLEAN_DIR, mask_name)
        skel_path = os.path.join(SKEL_DIR, mask_name)
        
        # Skip if mask already exists (resume functionality)
        if os.path.exists(out_path):
            skipped_count += 1
            print(f"[{idx}/{len(files)}] Skipping (already exists): {fname}", flush=True)
            continue
        
        try:
            # Load image
            img = Image.open(in_path).convert("RGB")
            
            # Apply CLAHE if enabled
            if USE_CLAHE:
                img = apply_clahe(img)
                
            original_size = img.size
            print(f"[{idx}/{len(files)}] Processing: {fname} ({original_size[0]}x{original_size[1]})", flush=True)
            
            # Predict mask
            mask_img = predict_mask(img)
            
            # Save raw predicted mask
            mask_img.save(out_path)
            
            # Post-processing
            mask_array = np.array(mask_img)
            
            # Cleaning
            cleaned_array = clean_mask(mask_array)
            Image.fromarray(cleaned_array).save(clean_path)
            
            # Skeletonization (Centerlines)
            skel_array = get_skeleton(cleaned_array)
            Image.fromarray(skel_array).save(skel_path)
            
            processed_count += 1
            print(f"   ✅ Saved: {out_path}")
            print(f"   ✨ Post-processed: Cleaned and Skeletonized saved.")
            
        except Exception as e:
            print(f"   ❌ Error processing {fname}: {e}", flush=True)
            continue
    
    print("=" * 60)
    print(f"✅ Inference complete!")
    print(f"   Processed: {processed_count} images")
    if skipped_count > 0:
        print(f"   Skipped (already existed): {skipped_count} images")
    print(f"📂 Masks saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

