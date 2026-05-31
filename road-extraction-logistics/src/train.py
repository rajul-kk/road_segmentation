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
from models.architecture import build_model

# ── Setup ─────────────────────────────────────────────────────────────────────
device = cfg.DEVICE
print(f"Using device: {device}")

os.makedirs(os.path.dirname(cfg.FINAL_MODEL_PATH), exist_ok=True)

if not os.path.exists(cfg.DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {cfg.DATA_DIR}")

dataset = RoadSegmentationDataset(data_dir=cfg.DATA_DIR, augment=True, use_clahe=cfg.USE_CLAHE)
if len(dataset) == 0:
    raise ValueError("Dataset is empty — check that images and masks are in the correct directory.")

print(f"Dataset size: {len(dataset)} images")
dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

model = build_model(backbone=cfg.BACKBONE).to(device)


# ── Loss ──────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = torch.softmax(predict, dim=1)[:, 1, :, :]
        target = target.float()
        intersection = (predict * target).sum(dim=(1, 2))
        union = predict.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        return 1 - ((2. * intersection + self.smooth) / (union + self.smooth)).mean()


class_weights = torch.tensor([1.0, 10.0]).to(device)
ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
dice_criterion = DiceLoss()


def criterion(outputs, targets):
    return ce_criterion(outputs, targets) + dice_criterion(outputs, targets)


optimizer = Adam(model.parameters(), lr=cfg.LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'backbone': cfg.BACKBONE,
        'num_classes': cfg.NUM_CLASSES,
    }, path)
    print(f"💾 Checkpoint saved: {path} (Epoch {epoch + 1})")


def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        print(f"📂 Found checkpoint: {path}")
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start = ckpt['epoch'] + 1
        print(f"✅ Resuming from epoch {start + 1}")
        return start
    return 0


start_epoch = load_checkpoint(cfg.CHECKPOINT_PATH, model, optimizer)

if start_epoch >= cfg.NUM_EPOCHS:
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
current_loss = 0.0

print(f"\n🚀 Starting training from epoch {start_epoch + 1} to {cfg.NUM_EPOCHS}")
print("=" * 50)

try:
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        if interrupted:
            break

        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}", unit="batch")

        for batch_idx, (images, masks) in enumerate(progress_bar):
            if interrupted:
                break

            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            output = model(images)['out']
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

        current_loss = avg_loss
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} — avg loss: {avg_loss:.4f}")
        scheduler.step(current_loss)

        if (epoch + 1) % cfg.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(epoch, model, optimizer, current_loss, cfg.CHECKPOINT_PATH)

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    save_checkpoint(epoch, model, optimizer, current_loss, cfg.CHECKPOINT_PATH)
    raise

# ── Save final model ──────────────────────────────────────────────────────────
if not interrupted:
    print("\n✅ Training complete!")
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': cfg.NUM_EPOCHS,
        'backbone': cfg.BACKBONE,
        'num_classes': cfg.NUM_CLASSES,
    }, cfg.FINAL_MODEL_PATH)
    print(f"💾 Final model saved to {cfg.FINAL_MODEL_PATH}")
    if os.path.exists(cfg.CHECKPOINT_PATH):
        os.remove(cfg.CHECKPOINT_PATH)
        print("🗑️  Checkpoint removed (training complete)")
else:
    save_checkpoint(epoch, model, optimizer, current_loss, cfg.CHECKPOINT_PATH)
    print(f"\n⏸️  Training paused at epoch {epoch + 1}. Run again to resume.")
