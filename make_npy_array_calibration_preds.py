import os
import numpy as np
from helper_code import *

# Paths to predictions and label headers for calibration set
prediction_dir = "/gpfs/home5/dnowee/efficientnet_1/full_test_pred/balanced_12l"
label_dir = "/scratch-shared/dnowee/data/physionet_format/test_labels" # .../physionet_format/calibration_labels/

records = [f[:-4] for f in os.listdir(prediction_dir) if f.endswith(".txt")]

probs = []
labels = []

for rec in records:
    pred_file = os.path.join(prediction_dir, rec + ".txt")
    hea_file = os.path.join(label_dir, rec + ".hea")

    try:
        pred_text = load_text(pred_file)
        label = load_label(hea_file)
        
        # Prob and label extraction
        prob = get_probability(pred_text, allow_missing=False)

        probs.append(prob)
        print(f"{rec} → label = {label}, type = {type(label)}")
        labels.append(label)
    except Exception as e:
        print(f"Skipping {rec}: {e}")

probs = np.array(probs)
labels = np.array(labels)

# Save for calibration
np.save("test_probs.npy", probs)
np.save("test_labels.npy", labels)
