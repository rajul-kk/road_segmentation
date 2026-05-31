"""
Road Segmentation Inference Script

Loads a trained DeepLabV3 model and predicts road masks for satellite images.
For each image: raw mask → morphological clean → skeletonized centerline.

Usage:
    python run_inference.py
"""

import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

import config as cfg
from models.architecture import build_model
from src.post_process import clean_mask, get_skeleton
from src.preprocessing import apply_clahe

# ── Preprocessing (must match training) ──────────────────────────────────────
_to_tensor = T.ToTensor()
_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _preprocess(img: Image.Image) -> torch.Tensor:
    x = _to_tensor(img.convert("RGB"))
    return _normalize(x).unsqueeze(0)


def is_image_file(fname: str) -> bool:
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"))


def main():
    # ── Load model ────────────────────────────────────────────────────────────
    device = cfg.DEVICE
    print(f"Using device: {device}")

    if not os.path.exists(cfg.FINAL_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {cfg.FINAL_MODEL_PATH}")

    print(f"Loading model from {cfg.FINAL_MODEL_PATH}...")
    model = build_model(backbone=cfg.BACKBONE).to(device)
    model.eval()

    ckpt = torch.load(cfg.FINAL_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    if 'epoch' in ckpt:
        print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']}")
    else:
        print("✅ Loaded model weights")

    @torch.no_grad()
    def predict_mask(img: Image.Image) -> Image.Image:
        x = _preprocess(img).to(device)
        logits = model(x)['out']                        # (1, 2, H, W)
        probs = torch.softmax(logits, dim=1)[:, 1:2]   # road class probability
        mask = (probs > cfg.THRESHOLD).float().cpu().numpy()[0, 0]
        return Image.fromarray((mask * 255).astype(np.uint8))

    # ── I/O setup ─────────────────────────────────────────────────────────────
    if not os.path.exists(cfg.INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {cfg.INPUT_DIR}")

    for d in (cfg.OUTPUT_DIR, cfg.CLEAN_DIR, cfg.SKEL_DIR):
        os.makedirs(d, exist_ok=True)

    files = [f for f in os.listdir(cfg.INPUT_DIR) if is_image_file(f)]
    if not files:
        print(f"No images found in {cfg.INPUT_DIR}")
        return

    print(f"\nFound {len(files)} images in {cfg.INPUT_DIR}")
    print(f"Threshold: {cfg.THRESHOLD}  |  CLAHE: {cfg.USE_CLAHE}")
    print("=" * 60)

    processed = skipped = 0

    for idx, fname in enumerate(files, 1):
        mask_name = fname.replace("_sat.jpg", "_roadmask.png") if fname.endswith("_sat.jpg") \
                    else f"{os.path.splitext(fname)[0]}_roadmask.png"

        out_path   = os.path.join(cfg.OUTPUT_DIR, mask_name)
        clean_path = os.path.join(cfg.CLEAN_DIR,  mask_name)
        skel_path  = os.path.join(cfg.SKEL_DIR,   mask_name)

        if os.path.exists(out_path):
            skipped += 1
            print(f"[{idx}/{len(files)}] Skip (exists): {fname}")
            continue

        try:
            img = Image.open(os.path.join(cfg.INPUT_DIR, fname)).convert("RGB")
            if cfg.USE_CLAHE:
                img = apply_clahe(img)

            print(f"[{idx}/{len(files)}] Processing: {fname} ({img.size[0]}x{img.size[1]})", flush=True)

            mask_img = predict_mask(img)
            mask_img.save(out_path)

            mask_arr = np.array(mask_img)
            Image.fromarray(clean_mask(mask_arr)).save(clean_path)
            Image.fromarray(get_skeleton(mask_arr)).save(skel_path)

            processed += 1
            print(f"   ✅ Saved mask + cleaned + skeleton")

        except Exception as e:
            print(f"   ❌ Error: {e}", flush=True)

    print("=" * 60)
    print(f"✅ Done — processed: {processed}, skipped: {skipped}")
    print(f"Masks → {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
