import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.dataset import RoadSegmentationDataset
from models.architecture import model

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Verify data directory exists (DeepGlobe format: images and masks in same folder)
data_dir = "data/raw/train"

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

# Create dataset and dataloader
dataset = RoadSegmentationDataset(data_dir=data_dir)

if len(dataset) == 0:
    raise ValueError("Dataset is empty! Check that images and masks are in the correct directories.")

print(f"Dataset size: {len(dataset)} images")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Move model to device
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    
    # Progress bar for each epoch
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)  # Shape: (batch, height, width) with class indices 0 or 1
        
        # Forward pass
        optimizer.zero_grad()
        output = model(images)['out']  # Shape: (batch, 2, height, width)
        
        # Calculate loss
        loss = criterion(output, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update progress bar with current loss
        epoch_loss += loss.item()
        avg_loss = epoch_loss / (batch_idx + 1)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")
    
    print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

print("Training complete!")

# Save the trained model
model_save_path = "models/DeeplabsV3.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': num_epochs,
    'model_type': 'deeplabv3_resnet50',
    'num_classes': 2
}, model_save_path)
print(f"Model saved to {model_save_path}")

