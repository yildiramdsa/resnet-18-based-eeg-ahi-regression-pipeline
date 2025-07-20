# Inference and Evaluation

# This script loads the trained AHI regression model,
# runs per-window inference on held-out data or new EDF recordings,
# aggregates predictions to the subject level, and evaluates performance.

# 1. Imports & Configuration
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_DIR      = Path("/Volumes/T9/projects/sleep-apnea-classification-using-eeg-spectrograms/data/spectrograms")
METADATA_CSV  = BASE_DIR / 'window_metadata.csv'
SPLITS_CSV    = BASE_DIR / 'spectrogram_splits.csv'
MODEL_PATH    = BASE_DIR / 'resnet18_ahi_regressor.pth'

# 2. Define Inference Dataset
class InferenceDataset(Dataset):
    """
    Dataset for loading spectrogram windows for inference.
    Returns (image_tensor, subject_id) for each window.
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['spectrogram_path']).convert('RGB')
        x   = self.transform(img)
        sid = row['subject_id']  # string
        return x, sid

# 3. Load Model for Inference
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# 4. Run Inference on Validation Fold
window_meta = pd.read_csv(METADATA_CSV)
folds       = pd.read_csv(SPLITS_CSV)[['subject_id','fold']]
val_df      = window_meta.merge(folds, on='subject_id')
val_df      = val_df[val_df['fold']==0].reset_index(drop=True)

infer_dataset = InferenceDataset(val_df)
infer_loader  = DataLoader(infer_dataset, batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=False)

all_preds = []
all_ids   = []
with torch.no_grad():
    for x_batch, sid_batch in infer_loader:
        x = x_batch.to(device)
        preds = model(x).squeeze().cpu().numpy()
        all_preds.extend(preds.tolist())
        # Convert subject IDs to strings to avoid tensor issues
        if isinstance(sid_batch, torch.Tensor):
            all_ids.extend([str(sid.item()) for sid in sid_batch])
        else:
            all_ids.extend([str(sid) for sid in sid_batch])

df_pred = pd.DataFrame({'subject_id': all_ids, 'pred_ahi': all_preds})
subj_pred = df_pred.groupby('subject_id').mean().reset_index()

# 5. Evaluate Performance
true_ahi = window_meta[['subject_id','ahi']].drop_duplicates()

# Debug the merge issue
print("subj_pred shape:", subj_pred.shape)
print("true_ahi shape:", true_ahi.shape)
print("subj_pred subject_id dtype:", subj_pred['subject_id'].dtype)
print("true_ahi subject_id dtype:", true_ahi['subject_id'].dtype)
print("subj_pred subject_ids (first 10):", subj_pred['subject_id'].tolist()[:10])
print("true_ahi subject_ids (first 10):", true_ahi['subject_id'].tolist()[:10])

# Ensure both have same data type
subj_pred['subject_id'] = subj_pred['subject_id'].astype(str)
true_ahi['subject_id'] = true_ahi['subject_id'].astype(str)

results = subj_pred.merge(true_ahi, on='subject_id')

rmse = np.sqrt(((results['pred_ahi'] - results['ahi'])**2).mean())
pearson = results[['pred_ahi','ahi']].corr().iloc[0,1]
print(f"Validation RMSE: {rmse:.2f}, Pearson r: {pearson:.3f}")

# 6. Plot Results
plt.figure(figsize=(6,6))
plt.scatter(results['ahi'], results['pred_ahi'], alpha=0.6)
plt.plot([0,60],[0,60], 'r--')
plt.xlabel('True AHI')
plt.ylabel('Predicted AHI')
plt.title('AHI Regression: True vs Predicted')
plt.show()