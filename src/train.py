import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import json
import glob

class LandslideDataset(Dataset):
    """Landslide Dataset for training."""

    def __init__(self, tiles_dir, labels_dir, split='train'):
        self.tiles_dir = os.path.join(tiles_dir, split)
        self.labels_dir = os.path.join(labels_dir, split)
        self.tile_files = glob.glob(os.path.join(self.tiles_dir, '*.npy'))

    def __len__(self):
        return len(self.tile_files)

    def __getitem__(self, idx):
        tile_path = self.tile_files[idx]
        label_path = os.path.join(self.labels_dir, os.path.basename(tile_path))

        tile = np.load(tile_path)
        label = np.load(label_path)

        # Convert to tensor and float
        tile = torch.from_numpy(tile).float()
        label = torch.from_numpy(label).float().unsqueeze(0) # Add channel dimension

        return tile, label


class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(DiceBCELoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.weight_bce * bce + self.weight_dice * dice

def train_model(config):
    """Trains the U-Net model."""
    print("--- Starting Model Training ---")

    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tiles_dir = config['project_structure']['tiles_dir']
    labels_dir = config['project_structure']['labels_dir']
    experiments_dir = config['project_structure']['experiments_dir']
    os.makedirs(experiments_dir, exist_ok=True)

    # --- 2. DataLoaders ---
    train_dataset = LandslideDataset(tiles_dir, labels_dir, split='train')
    val_dataset = LandslideDataset(tiles_dir, labels_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # --- 3. Model ---
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights='imagenet',
        in_channels=config['model']['in_channels'],
        classes=config['model']['out_classes'],
    ).to(device)

    # --- 4. Loss, Optimizer, Scheduler ---
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(config['training']['epochs']):
        print(f"--- Epoch {epoch+1}/{config['training']['epochs']} ---")

        # Training
        model.train()
        train_loss = 0
        for tiles, labels in train_loader:
            tiles, labels = tiles.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(tiles)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for tiles, labels in val_loader:
                tiles, labels = tiles.to(device), labels.to(device)
                outputs = model(tiles)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(experiments_dir, 'best_model.pth'))
            print("Model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == config['training']['early_stopping_patience']:
            print("Early stopping.")
            break

        scheduler.step()

    print("--- Model Training Complete ---")

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_model(config)
