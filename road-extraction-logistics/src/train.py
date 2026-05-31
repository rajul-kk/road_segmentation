import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import signal

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import config as cfg
from src.dataset import RoadSegmentationDataset
from src.metrics import calculate_metrics, average_metrics
from models.architecture import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepLabV3 road segmentation model")
    p.add_argument("--data-dir",      default=cfg.DATA_DIR,          help="Training image directory")
    p.add_argument("--epochs",        type=int,   default=cfg.NUM_EPOCHS)
    p.add_argument("--batch-size",    type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--lr",            type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--backbone",      default=cfg.BACKBONE,           choices=["resnet50", "resnet101"])
    p.add_argument("--val-split",     type=float, default=cfg.VAL_SPLIT)
    p.add_argument("--checkpoint",    default=cfg.CHECKPOINT_PATH)
    p.add_argument("--output-model",  default=cfg.FINAL_MODEL_PATH)
    p.add_argument("--num-workers",   type=int,   default=cfg.NUM_WORKERS)
    p.add_argument("--no-clahe",      action="store_true",            help="Disable CLAHE preprocessing")
    p.add_argument("--no-amp",        action="store_true",            help="Disable mixed-precision training")
    return p.parse_args()


args = parse_args()

# ── Device + directories ──────────────────────────────────────────────────────
device = cfg.DEVICE
print(f"Using device: {device}")
use_amp = device.type == "cuda" and not args.no_amp

os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

if not os.path.exists(args.data_dir):
    raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

use_clahe = cfg.USE_CLAHE and not args.no_clahe

# ── Train / val split ─────────────────────────────────────────────────────────
train_files, val_files = cfg.get_train_val_split(data_dir=args.data_dir, val_fraction=args.val_split)
train_dataset = RoadSegmentationDataset(args.data_dir, augment=True,  use_clahe=use_clahe, file_list=train_files)
val_dataset   = RoadSegmentationDataset(args.data_dir, augment=False, use_clahe=use_clahe, file_list=val_files)

if len(train_dataset) == 0:
    raise ValueError("Training split is empty — check images in DATA_DIR.")

print(f"Train: {len(train_dataset)} images  |  Val: {len(val_dataset)} images")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=cfg.PIN_MEMORY)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=cfg.PIN_MEMORY)

# ── Model ─────────────────────────────────────────────────────────────────────
model = build_model(backbone=args.backbone).to(device)


# ── Loss ──────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = torch.softmax(predict, dim=1)[:, 1, :, :]
        target  = target.float()
        inter   = (predict * target).sum(dim=(1, 2))
        union   = predict.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        return 1 - ((2. * inter + self.smooth) / (union + self.smooth)).mean()


class_weights = torch.tensor([1.0, 10.0]).to(device)
ce_criterion  = nn.CrossEntropyLoss(weight=class_weights)
dice_criterion = DiceLoss()


def criterion(outputs, targets):
    return ce_criterion(outputs, targets) + dice_criterion(outputs, targets)


optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)  # tracks val IoU
scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, scaler, val_iou, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_iou': val_iou,
        'backbone': args.backbone,
        'num_classes': cfg.NUM_CLASSES,
    }, path)
    print(f"💾 Checkpoint saved: {path} (Epoch {epoch + 1})")


def load_checkpoint(path, model, optimizer, scaler):
    if not os.path.exists(path):
        return 0
    print(f"📂 Found checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    start = ckpt['epoch'] + 1
    print(f"✅ Resuming from epoch {start + 1}")
    return start


# ── Val helper ────────────────────────────────────────────────────────────────
def validate_epoch(model, loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1).cpu()
            for i in range(preds.size(0)):
                all_metrics.append(calculate_metrics(preds[i], masks[i]))
    model.train()
    return average_metrics(all_metrics)


start_epoch = load_checkpoint(args.checkpoint, model, optimizer, scaler)

if start_epoch >= args.epochs:
    print(f"Training already complete ({start_epoch} epochs). Delete checkpoint to retrain.")
    sys.exit(0)


# ── Interrupt handler ─────────────────────────────────────────────────────────
interrupted = False


def signal_handler(sig, frame):
    global interrupted
    print("\n⚠️  Interrupt received — saving checkpoint before exit...")
    interrupted = True


signal.signal(signal.SIGINT, signal_handler)


# ── Training loop ─────────────────────────────────────────────────────────────
model.train()
current_val_iou = 0.0

print(f"\n🚀 Starting training — epochs {start_epoch + 1} to {args.epochs}  |  AMP: {use_amp}  |  backbone: {args.backbone}")
print("=" * 60)

try:
    for epoch in range(start_epoch, args.epochs):
        if interrupted:
            break

        epoch_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")

        for batch_idx, (images, masks) in enumerate(bar):
            if interrupted:
                break

            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(images)['out']
                loss   = criterion(output, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

        # Per-epoch validation
        val_metrics = validate_epoch(model, val_loader)
        current_val_iou = val_metrics['iou']
        scheduler.step(current_val_iou)  # plateau on val IoU, not train loss

        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} — "
              f"train loss: {avg_loss:.4f}  |  "
              f"val IoU: {val_metrics['iou']:.4f}  F1: {val_metrics['f1']:.4f}")

        if (epoch + 1) % cfg.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(epoch, model, optimizer, scaler, current_val_iou, args.checkpoint)

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    save_checkpoint(epoch, model, optimizer, scaler, current_val_iou, args.checkpoint)
    raise

# ── Save final model ──────────────────────────────────────────────────────────
if not interrupted:
    print("\n✅ Training complete!")
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'backbone': args.backbone,
        'num_classes': cfg.NUM_CLASSES,
        'val_iou': current_val_iou,
    }, args.output_model)
    print(f"💾 Final model saved to {args.output_model}")
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print("🗑️  Checkpoint removed")
else:
    save_checkpoint(epoch, model, optimizer, scaler, current_val_iou, args.checkpoint)
    print(f"\n⏸️  Training paused at epoch {epoch + 1}. Run again to resume.")
