# AHI Regression Training

# This script trains a deep learning model (ResNet-18) to predict per-subject Apnea-Hypopnea Index (AHI)
# from windowed EEG spectrogram images. It loads precomputed spectrogram metadata, builds subject-stratified
# training and validation splits, and implements a robust PyTorch training pipeline with:
#   - Class-balance monitoring
#   - Learning rate scheduling and warmup
#   - Early stopping
#   - Data augmentation
#   - Subject-level aggregation for evaluation
# The workflow enables reproducible, leakage-free training and validation for AHI regression from EEG,
# supporting robust model development for sleep apnea severity prediction.

# 1. Imports & Configuration
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_DIR     = Path("/Volumes/T9/projects/sleep-apnea-classification-using-eeg-spectrograms/data/spectrograms")
METADATA_CSV = BASE_DIR / 'window_metadata.csv'    # output from Data Preparation
SPLITS_CSV   = BASE_DIR / 'spectrogram_splits.csv' # output from Data Preparation
OUTPUT_MODEL = BASE_DIR / 'resnet18_ahi_regressor.pth'
PRED_CSV     = BASE_DIR / 'validation_predictions.csv'

# Training hyperparameters
BATCH_SIZE           = 16
LR                   = 1e-4
EPOCHS               = 30
RANDOM_STATE         = 26
EARLY_STOP_PATIENCE  = 10
WARMUP_EPOCHS        = 5
GRADIENT_CLIP        = 1.0

# Set seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(RANDOM_STATE)

# 2. Dataset Definition
class SpectrogramWindowDataset(Dataset):
    """
    Returns (spectrogram_tensor, subject_AHI) per window.
    The regression target is the subject's overall overnight AHI.
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        # map each subject to its overall AHI for window-level regression
        self.ahi_map = self.df.groupby('subject_id')['ahi'].first().to_dict()
        self.df['ahi_target'] = self.df['subject_id'].map(self.ahi_map)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),   # match ResNet input size
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),           # convert to [C,H,W] tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['spectrogram_path']).convert('RGB')
        x = self.transform(img)
        y = torch.tensor(row['ahi_target'], dtype=torch.float32)
        return x, y

# 3. Load Metadata and Build DataLoaders
window_meta = pd.read_csv(METADATA_CSV)
# compute severity bins for class-balance monitoring
severity_cut = pd.cut(
    window_meta['ahi'], bins=[-1,5,15,30,np.inf],
    labels=['healthy','mild','moderate','severe']
)
window_meta['severity'] = pd.Series(severity_cut, index=window_meta.index)
# log severity distribution
print("[INFO] Severity distribution:")
print(window_meta['severity'].value_counts(), "\n")

splits    = pd.read_csv(SPLITS_CSV)[['subject_id','fold']]
full_df   = window_meta.merge(splits, on='subject_id', how='inner')
# split into training (folds 1-4) and validation (fold 0)
train_df  = full_df[full_df['fold'] != 0].reset_index(drop=True)
val_df    = full_df[full_df['fold'] == 0].reset_index(drop=True)
# monitor class balance per split
print("[INFO] Train severity distribution:")
print(pd.Series(train_df['severity']).value_counts(), "\n")
print("[INFO] Val severity distribution:")
print(pd.Series(val_df['severity']).value_counts(), "\n")

def make_loader(df: pd.DataFrame, shuffle: bool) -> DataLoader:
    ds = SpectrogramWindowDataset(df)
    return DataLoader(ds,
                      batch_size=BATCH_SIZE,
                      shuffle=shuffle,
                      num_workers=0,
                      pin_memory=(device.type=='cuda'))

train_loader = make_loader(pd.DataFrame(train_df), shuffle=True)
val_loader   = make_loader(pd.DataFrame(val_df), shuffle=False)

# Visualize AHI Distribution in Train and Validation Sets
plt.figure(figsize=(10,5))
plt.hist(train_df['ahi'], bins=50, alpha=0.5, label='Train')
plt.hist(val_df['ahi'], bins=50, alpha=0.5, label='Val')
plt.xlabel('AHI')
plt.ylabel('Count')
plt.legend()
plt.title('AHI Distribution in Train and Validation Sets')
plt.show()

# 4. Model Setup & Scheduler
model       = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # explicit weights
in_features = model.fc.in_features
model.fc    = nn.Linear(in_features, 1)                  # single-value regression head

# Unfreeze all layers but use different learning rates
for param in model.parameters():
    param.requires_grad = True

# Use different learning rates for backbone and head
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if 'fc' in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

model = model.to(device)

criterion = nn.SmoothL1Loss()  # More robust than L1Loss
optimizer = optim.Adam([
    {'params': backbone_params, 'lr': LR * 0.1},  # Lower LR for pretrained backbone
    {'params': head_params, 'lr': LR}             # Higher LR for new head
])
# reduce learning rate if validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# 5. Training & Validation Loop with Early Stopping
best_mse          = float('inf')
no_improve_epochs = 0
train_losses_history = []
val_losses_history = []

for epoch in range(1, EPOCHS+1):
    # Learning rate warmup
    if epoch <= WARMUP_EPOCHS:
        warmup_factor = epoch / WARMUP_EPOCHS
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR * warmup_factor
    
    # training phase
    model.train()
    train_losses = []
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        preds = model(x_batch.to(device)).squeeze()
        loss  = criterion(preds, y_batch.to(device))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        
        optimizer.step()
        train_losses.append(loss.item())
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # validation phase
    model.eval()
    all_preds, all_targets = [], []
    val_ids = val_loader.dataset.df['subject_id'].values
    val_losses = []
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            preds = model(x_batch.to(device)).squeeze()
            loss = criterion(preds, y_batch.to(device))
            val_losses.append(loss.item())
            
            batch_pred = preds.cpu().numpy()
            all_preds.append(batch_pred)
            all_targets.append(y_batch.numpy())
    
    val_preds = np.concatenate(all_preds)
    val_targs = np.concatenate(all_targets)
    df_val    = pd.DataFrame({
        'subject_id': val_ids[:len(val_preds)],
        'pred_ahi':   val_preds,
        'true_ahi':   val_targs
    })
    subj_agg  = df_val.groupby('subject_id').mean()
    val_mse   = ((subj_agg['pred_ahi'] - subj_agg['true_ahi'])**2).mean()
    val_mae   = np.abs(subj_agg['pred_ahi'] - subj_agg['true_ahi']).mean()

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    train_losses_history.append(avg_train_loss)
    val_losses_history.append(avg_val_loss)

    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"  Val MSE (subject): {val_mse:.4f} | Val MAE (subject): {val_mae:.4f}")
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

    # scheduler step
    scheduler.step(val_mse)

    # early stopping check
    if val_mse < best_mse:
        best_mse          = val_mse
        no_improve_epochs = 0
        torch.save(model.state_dict(), OUTPUT_MODEL)  # save best
        print(f"  *** New best model saved! Val MSE: {val_mse:.4f} ***")
    else:
        no_improve_epochs += 1
        print(f"  No improvement for {no_improve_epochs} epochs")
        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"Early stopping: no improvement in {EARLY_STOP_PATIENCE} epochs")
            break
    
    print("-" * 50)

# 6. Final Save & Predictions
# ensure best model saved
torch.save(model.state_dict(), OUTPUT_MODEL)
# save subject-level validation predictions
subj_agg.reset_index().to_csv(PRED_CSV, index=False)
print("Training complete. Model and validation predictions saved.")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses_history, label='Train Loss')
plt.plot(val_losses_history, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.scatter(subj_agg['true_ahi'], subj_agg['pred_ahi'], alpha=0.6)
max_ahi = float(subj_agg['true_ahi'].max())
plt.plot([0, max_ahi], [0, max_ahi], 'r--', label='Perfect Prediction')
plt.xlabel('True AHI')
plt.ylabel('Predicted AHI')
plt.legend()
plt.title('Predicted vs True AHI (Subject Level)')
plt.tight_layout()
plt.show() 