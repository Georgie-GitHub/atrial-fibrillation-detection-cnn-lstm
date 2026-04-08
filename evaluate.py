"""
=============================================================
 Evaluation Script — AF Detection (BINARY)
 Classes : Normal (0) | AF (1)
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import scipy.signal
import os
import csv

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR       = "data/training2017"
REFERENCE_FILE = "D:/MLproject/AfibDetection/data/training2017/REFERENCE.csv"
MODEL_PATH     = "af_cnn_lstm_model.keras"

FS             = 300
SEGMENT_LEN    = FS * 10
LOWCUT         = 0.5
HIGHCUT        = 40.0
FILTER_ORDER   = 4
RANDOM_SEED    = 42

CLASS_NAMES    = ["Normal", "AF"]
COLORS         = ["#4CAF50", "#F44336"]

THRESHOLD      = 0.5   # you can tune this later


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def bandpass_filter(signal, lowcut=LOWCUT, highcut=HIGHCUT,
                    fs=FS, order=FILTER_ORDER):
    nyq  = fs / 2.0
    b, a = scipy.signal.butter(order,
                               [lowcut / nyq, highcut / nyq],
                               btype="band")
    return scipy.signal.filtfilt(b, a, signal)


def normalize(signal):
    std = signal.std()
    return (signal - signal.mean()) / (std if std > 1e-6 else 1.0)


def segment_signal(signal, seg_len=SEGMENT_LEN, step=None):
    if step is None:
        step = seg_len // 2
    starts = range(0, len(signal) - seg_len + 1, step)
    return np.array([signal[s:s + seg_len] for s in starts], dtype=np.float32)


# ─────────────────────────────────────────────
# 1. LOAD DATA (ONLY NORMAL + AF)
# ─────────────────────────────────────────────
print("\n🔹 Loading data (Normal + AF only)...")

label_map = {"N": 0, "A": 1}

X_all, y_all = [], []

with open(REFERENCE_FILE, "r") as f:
    for row in csv.reader(f):
        if len(row) < 2:
            continue

        name, label = row[0].strip(), row[1].strip()

        if label not in label_map:
            continue  # skip "Other"

        mat_path = os.path.join(DATA_DIR, f"{name}.mat")
        if not os.path.exists(mat_path):
            continue

        signal = scipy.io.loadmat(mat_path)["val"].squeeze().astype(np.float32)

        filtered = bandpass_filter(signal)
        segs = segment_signal(filtered)

        if len(segs) == 0:
            continue

        segs = np.array([normalize(s) for s in segs])

        X_all.append(segs)
        y_all.extend([label_map[label]] * len(segs))

X = np.concatenate(X_all, axis=0)[..., np.newaxis]
y = np.array(y_all, dtype=np.int32)

print(f"Total segments : {len(X)}")

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
)

print(f"Test segments  : {len(X_test)}")


# ─────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────
print(f"\n🔹 Loading model from {MODEL_PATH} ...")

model = load_model(MODEL_PATH, compile=False)
print("Model loaded ✓")


# ─────────────────────────────────────────────
# 3. PREDICT
# ─────────────────────────────────────────────
print("\n🔹 Running predictions...")

y_prob_raw = model.predict(X_test, verbose=1)

# Handle BOTH cases (binary or 3-class model)
if y_prob_raw.shape[1] == 1:
    # Binary model
    y_prob = y_prob_raw.flatten()
else:
    # 3-class model → take AF probability
    y_prob = y_prob_raw[:, 1]

y_pred = (y_prob >= THRESHOLD).astype(int)


# ─────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  OVERALL ACCURACY")
print("=" * 60)
print(f"  {accuracy_score(y_test, y_pred)*100:.2f} %")

print("\n" + "=" * 60)
print("  CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred,
                             target_names=CLASS_NAMES, digits=4))


# ─────────────────────────────────────────────
# 5. CONFUSION MATRIX + MANUAL METRICS
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

precision = TP / (TP + FP + 1e-8)
recall    = TP / (TP + FN + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

print("=" * 60)
print("  BINARY METRICS (AF as Positive)")
print("=" * 60)
print(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ─────────────────────────────────────────────
# 6. ROC CURVE
# ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"\nROC-AUC : {roc_auc:.4f}")


# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("AF Detection — Binary Evaluation", fontsize=14, fontweight="bold")

# Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# ROC curve
axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("FPR")
axes[1].set_ylabel("TPR")
axes[1].legend()

# Probability distribution
axes[2].hist(y_prob[y_test == 0], bins=50, alpha=0.5, label="Normal")
axes[2].hist(y_prob[y_test == 1], bins=50, alpha=0.5, label="AF")
axes[2].axvline(THRESHOLD, linestyle="--", color="black")
axes[2].set_title("Prediction Probability")
axes[2].legend()

plt.tight_layout()
plt.savefig("binary_af_evaluation.png", dpi=150)
plt.show()

print("\n✅ Saved → binary_af_evaluation.png")