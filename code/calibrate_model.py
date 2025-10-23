from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, precision_recall_curve, f1_score, brier_score_loss
import matplotlib.pyplot as plt
import numpy as np

# Load y_true and y_probs from npy files
y_true = np.load("calibration_labels.npy")
y_probs = np.load("calibration_probs.npy")

# Fit calibration models
platt_model = LogisticRegression()
platt_model.fit(y_probs.reshape(-1, 1), y_true)

iso_model = IsotonicRegression(out_of_bounds='clip')
iso_model.fit(y_probs, y_true)

# Load your test or validation raw probabilities and labels
probs_test = np.load("test_probs.npy")
labels_test = np.load("test_labels.npy")


# Calibrate test outputs
platt_calibrated = platt_model.predict_proba(probs_test.reshape(-1, 1))[:, 1]
iso_calibrated = iso_model.predict(probs_test)


# Evaluate
# to find AUROC, AUPRC, and optimal threshold
for name, calibrated in zip(['Platt', 'Isotonic'], [platt_calibrated, iso_calibrated]):
    print(f"=== {name} ===")
    print("AUROC:", roc_auc_score(labels_test, calibrated))
    print("AUPRC:", average_precision_score(labels_test, calibrated))
    print("Brier score:", brier_score_loss(labels_test, calibrated))

    precisions, recalls, thresholds = precision_recall_curve(labels_test, calibrated)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    print("Best F1:", f1s[best_idx], "at threshold", thresholds[best_idx])


# Visualize PR curve
plt.figure()
plt.plot(recalls, precisions, label='Calibrated PR curve (Isotonic)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve after Calibration')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("pr_curve_calibrated.png")