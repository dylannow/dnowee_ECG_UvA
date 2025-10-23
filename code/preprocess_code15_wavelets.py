import os
import argparse
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from neurodsp.timefrequency.wavelets import compute_wavelet_transform
from pathlib import Path
from tqdm import tqdm
import sys


# ------------- Command-line Arguments -------------
parser = argparse.ArgumentParser(description="Preprocess CODE-15 ECGs with high pass filter, z-score normalization, zero padding and a morlet wavelet transform.")
parser.add_argument('--input_base', type=str, required=True, help="Base folder containing CODE-15 batches")
parser.add_argument('--output_base', type=str, required=True, help="Output folder for wavelet-transformed .npy files")
parser.add_argument('--labels_dir', type=str, required=True, help="Directory containing train/test/validation CSV files")
args = parser.parse_args()

input_base = Path(args.input_base)
output_base = Path(args.output_base)
labels_dir = Path(args.labels_dir)

# ------------- File Paths -------------
label_files = {
    "train": labels_dir / "training_labels.csv",
    "test": labels_dir / "test_labels.csv",
    "validation": labels_dir / "validation_labels.csv",
    "calibration": labels_dir / "calibration_labels.csv"
}

# ------------- Settings -------------
fs = 400
pad_length = 4096
wavelet_freqs = np.linspace(1, 50, 50)
num_leads = 12

# ------------- Signal Processing Functions -------------
def highpass_filter(signal, fs, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return filtfilt(b, a, signal, axis=0)

def zscore_normalize(signal):
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    std[std == 0] = 1
    return (signal - mean) / std

def pad_signal(signal, target_length):
    n = signal.shape[0]
    if n >= target_length:
        return signal[:target_length]
    else:
        pad_width = target_length - n
        return np.pad(signal, ((0, pad_width), (0, 0)), mode='constant')

def transform_one_lead(lead_signal, freqs):
    tfr = compute_wavelet_transform(lead_signal, fs=fs, freqs=freqs)
    return np.abs(tfr)

def transform_record(input_dir, record_name, output_dir):
    try:
        record_path = os.path.join(input_dir, record_name)
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # shape: (time, 12)

        # Check for missing or empty signal
        if signal is None or signal.shape[0] == 0 or not np.isfinite(signal).all():
            print(f"Skipping {record_name}: invalid signal")
            return

        # Pad and preprocess
        signal = pad_signal(signal, pad_length)
        signal = highpass_filter(signal, fs=fs)
        signal = zscore_normalize(signal)

        transformed = []
        for lead_idx in range(num_leads):
            lead_sig = signal[:, lead_idx]
            wave = transform_one_lead(lead_sig, wavelet_freqs)
            if wave.shape[1] == 0:
                print(f"Wavelet failed on {record_name}, lead {lead_idx}")
                return
            transformed.append(wave)

        out_array = np.stack(transformed, axis=0)  # (12, 50, time)

        # Check final shape before saving
        if out_array.shape[0] != 12 or out_array.shape[1] != 50 or out_array.shape[2] == 0:
            print(f"Skipping {record_name}: invalid output shape {out_array.shape}")
            return

        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{record_name}_wavelet.npy"), out_array)

    except Exception as e:
        print(f"Exception in {record_name}: {e}")

# ------------- Main Loop -------------
for split, label_path in label_files.items():
    label_df = pd.read_csv(label_path)
    split_output_dir = output_base / split

    for _, row in tqdm(label_df.iterrows(), total=len(label_df), desc=f"Processing {split}", file=sys.stdout, ncols=100, mininterval=1):
        exam_id = str(row["exam_id"])
        source_dir = input_base / row["source_part"]
        try:
            transform_record(str(source_dir), exam_id, str(split_output_dir))
        except Exception as e:
            print(f"Failed on {exam_id} in {source_dir}: {e}")
