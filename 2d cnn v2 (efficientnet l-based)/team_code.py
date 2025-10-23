#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import joblib
from helper_code import *

# V1: subset leads
#  CHANGE THIS TO TEST ALL 3-SETS OF LEADS 
# either 0&3, 3&6, 6&9 or 9&12
# first_lead = 9
# last_lead = 12
# V2: all 12 leads used

# set threshold to [0,1]
threshold = 0.5 


# Path to wavelet-transformed .npy files and training label CSV
TRAIN_DATA_DIR = "/gpfs/home5/dnowee/data/data_processed/hpass_znorm_morletwavelet_code15_4096/train"
TRAIN_LABEL_CSV = "/gpfs/home5/dnowee/data/training_labels.csv"
SUBSET_TRAIN_LABEL_CSV="/gpfs/home5/dnowee/data/train_subset_2.csv"
TEST_DATA_DIR = "/gpfs/scratch1/shared/dnowee/data/test"
CALIBRATION_DATA_DIR = "/gpfs/scratch1/shared/dnowee/data/calibration"
VALID_DATA_DIR = "/gpfs/scratch1/shared/dnowee/data/validation"

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # load train set
    dataset = Code15WaveletDataset(TRAIN_DATA_DIR, TRAIN_LABEL_CSV)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    if len(dataset) == 0:
        print("Dataset is empty. Check your CSV and .npy files.")
        return
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = create_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # training model
    model.train()
    for epoch in range(10):
        total_loss = 0
        if verbose:
            print(f"\nStarting epoch {epoch+1}")
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log progress every 10 batches
            if verbose and (i + 1) % 10 == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(outputs).flatten()
                    preds = (probs > 0.5).int()
                    acc = (preds == labels.int().flatten()).float().mean().item()
                print(f"  Batch {i+1:03d}: loss = {loss.item():.4f}, acc = {acc:.2%}, avg loss = {total_loss / (i+1):.4f}")

        if verbose:
            print(f"Epoch {epoch+1} average loss: {total_loss / len(dataloader):.4f}")

    os.makedirs(model_folder, exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(model_folder, f"model_epoch{epoch+1}.pt"))

    torch.save(model.state_dict(), os.path.join(model_folder, "model.pt"))
    
    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
# def load_model(model_folder, verbose):
#     model = create_model()
#     model.load_state_dict(torch.load(os.path.join(model_folder, "model.pt"), map_location=torch.device('cpu')))
#     model.eval()
#     return model
def load_model(model_folder, verbose, epoch=None):
    model = create_model()
    filename = f"model_epoch{epoch}.pt" if epoch else "model.pt"
    model.load_state_dict(torch.load(os.path.join(model_folder, filename), map_location=torch.device('cpu')))
    model.eval()
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Inferring from .npy files instead of WFDB
    record_base = os.path.basename(record)
    npy_path = os.path.join(TEST_DATA_DIR, f"{record_base}_wavelet.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing wavelet file: {npy_path}")

    x = np.load(npy_path).astype(np.float32)
    x = torch.tensor(x).unsqueeze(0)  # (1, 12, freq, time)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()
        label = int(prob > threshold)

    return label, prob

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# dataset class to load .npy wavelet files & labels
class Code15WaveletDataset(Dataset):
    def __init__(self, data_dir, label_csv):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(label_csv)
        self.labels_df['exam_id'] = self.labels_df['exam_id'].astype(str)        
        
        # Filter only records for which the .npy file exists
        self.labels_df = self.labels_df[
            self.labels_df['exam_id'].apply(lambda x: os.path.exists(os.path.join(data_dir, f"{x}_wavelet.npy")))
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        exam_id = str(row['exam_id'])
        label = int(row['chagas'])  # binary label: 0 or 1

        # Load wavelet-transformed ECG from .npy
        x = np.load(os.path.join(self.data_dir, f"{exam_id}_wavelet.npy")).astype(np.float32)

        # Use all 12 leads directly
        # x shape: (12, freq, time)
        return torch.tensor(x), torch.tensor(label, dtype=torch.float32)


# Create EfficientNet model for binary classification
# Replaces final classification layer with a linear output for sigmoid use
def create_model():
    # model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)

    # Replace first conv layer: Conv2d(3, 32, ...)
    model.features[0][0] = nn.Conv2d(12, 32, kernel_size=3, stride=2, padding=1, bias=False)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    return model
    

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

def calibrate_model(record, model, verbose):
    # Inferring from .npy files instead of WFDB
    record_base = os.path.basename(record)
    npy_path = os.path.join(CALIBRATION_DATA_DIR, f"{record_base}_wavelet.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing wavelet file: {npy_path}")

    x = np.load(npy_path).astype(np.float32)
    x = torch.tensor(x).unsqueeze(0)  # (1, 12, freq, time)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()
        label = int(prob > 0.5)

    return label, prob