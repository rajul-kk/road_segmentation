import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import signal

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.dataset import RoadSegmentationDataset
from models.architecture import model

# ============== CONFIGURATION ==============
CHECKPOINT_PATH = "models/checkpoint.pth"
FINAL_MODEL_PATH = "models/DeeplabsV3.pth"
DATA_DIR = "data/raw/train"
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
SAVE_EVERY_N_EPOCHS = 1  # Save checkpoint every N epochs
# ===========================================

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Verify data directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

# Create dataset and dataloader
dataset = RoadSegmentationDataset(data_dir=DATA_DIR)

if len(dataset) == 0:
    raise ValueError("Dataset is empty! Check that images and masks are in the correct directories.")

print(f"Dataset size: {len(dataset)} images")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Move model to device
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# ============== CHECKPOINT HANDLING ==============
start_epoch = 0

def save_checkpoint(epoch, model, optimizer, loss, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_type': 'deeplabv3_resnet50',
        'num_classes': 2
    }, path)
    print(f"💾 Checkpoint saved: {path} (Epoch {epoch + 1})")

def load_checkpoint(path, model, optimizer):
    """Load training checkpoint and return starting epoch."""
    if os.path.exists(path):
        print(f"📂 Found checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resuming from epoch {start_epoch + 1}")
        return start_epoch
    return 0

# Check for existing checkpoint
start_epoch = load_checkpoint(CHECKPOINT_PATH, model, optimizer)

if start_epoch >= NUM_EPOCHS:
    print(f"Training already complete ({start_epoch} epochs). Delete checkpoint to retrain.")
    sys.exit(0)

# ============== GRACEFUL INTERRUPT HANDLING ==============
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    print("\n⚠️ Interrupt received! Saving checkpoint before exit...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

# ============== TRAINING LOOP ==============
model.train()
current_loss = 0.0

print(f"\n🚀 Starting training from epoch {start_epoch + 1} to {NUM_EPOCHS}")
print("=" * 50)

try:
    for epoch in range(start_epoch, NUM_EPOCHS):
        if interrupted:
            break
            
        epoch_loss = 0.0
        
        # Progress bar for each epoch
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            if interrupted:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(images)['out']
            
            # Calculate loss
            loss = criterion(output, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")
        
        current_loss = avg_loss
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch (or every N epochs)
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(epoch, model, optimizer, current_loss, CHECKPOINT_PATH)

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    save_checkpoint(epoch, model, optimizer, current_loss, CHECKPOINT_PATH)
    raise

# ============== SAVE FINAL MODEL ==============
if not interrupted:
    print("\n✅ Training complete!")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': NUM_EPOCHS,
        'model_type': 'deeplabv3_resnet50',
        'num_classes': 2
    }, FINAL_MODEL_PATH)
    print(f"💾 Final model saved to {FINAL_MODEL_PATH}")
    
    # Optionally remove checkpoint after successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"🗑️ Checkpoint removed (training complete)")
else:
    # Save checkpoint on interrupt
    save_checkpoint(epoch, model, optimizer, current_loss, CHECKPOINT_PATH)
    print(f"\n⏸️ Training paused at epoch {epoch + 1}. Run again to resume.")
