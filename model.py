"""
=============================================================
 Atrial Fibrillation Detection — CNN + LSTM (Keras/TensorFlow)
=============================================================
Dataset  : PhysioNet CinC Challenge 2017
           https://physionet.org/content/challenge-2017/1.0.0/

Install dependencies:
    pip install tensorflow wfdb neurokit2 scikit-learn imbalanced-learn matplotlib seaborn numpy scipy

Directory structure expected:
    data/
      training2017/
        A00001.mat
        A00001.hea
        REFERENCE.csv
        ...
=============================================================
"""

import os
import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    DATA_DIR        = "data/training2017"       # Path to CinC 2017 data folder
    REFERENCE_FILE  = "data/training2017/REFERENCE.csv"

    FS              = 300       # Sampling frequency (Hz) — CinC 2017 is 300 Hz
    SEGMENT_LEN_SEC = 10        # Each CNN input window (seconds)
    SEGMENT_LEN     = FS * SEGMENT_LEN_SEC   # = 3000 samples

    # Bandpass filter
    LOWCUT          = 0.5       # Hz
    HIGHCUT         = 40.0      # Hz
    FILTER_ORDER    = 4

    # Model
    BATCH_SIZE      = 32
    EPOCHS          = 50
    LEARNING_RATE   = 1e-3
    DROPOUT_RATE    = 0.3
    L2_REG          = 1e-4

    # Binary classification: AF vs Non-AF
    # Set to True for binary, False for 4-class (N, A, O, ~)
    BINARY          = True

    RANDOM_SEED     = 42
    MODEL_SAVE_PATH = "af_cnn_lstm_model.keras"

cfg = Config()
tf.random.set_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)


# ─────────────────────────────────────────────
#  1. DATA LOADING
# ─────────────────────────────────────────────
def load_cinc2017(data_dir: str, reference_file: str):
    """
    Load CinC 2017 .mat ECG files and their labels.
    Returns:
        records : list of 1-D numpy arrays (raw ECG signals, variable length)
        labels  : list of str labels ('N', 'A', 'O', '~')
    """
    import csv
    records, labels = [], []

    with open(reference_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            record_name, label = row[0].strip(), row[1].strip()
            mat_path = os.path.join(data_dir, f"{record_name}.mat")
            if not os.path.exists(mat_path):
                continue
            mat = scipy.io.loadmat(mat_path)
            # CinC 2017 stores the signal under key 'val'
            signal = mat["val"].squeeze().astype(np.float32)
            records.append(signal)
            labels.append(label)

    print(f"Loaded {len(records)} records.")
    return records, labels


# ─────────────────────────────────────────────
#  2. PREPROCESSING
# ─────────────────────────────────────────────
def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float,
                    fs: int, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low, high = lowcut / nyq, highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    return scipy.signal.filtfilt(b, a, signal)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalization per segment."""
    std = signal.std()
    if std < 1e-6:
        return signal - signal.mean()
    return (signal - signal.mean()) / std


def segment_signal(signal: np.ndarray, seg_len: int, step: int = None) -> np.ndarray:
    """
    Slice a 1-D signal into overlapping windows.
    step defaults to seg_len // 2  (50% overlap).
    Returns array of shape (N_segments, seg_len).
    """
    if step is None:
        step = seg_len // 2
    starts = range(0, len(signal) - seg_len + 1, step)
    return np.array([signal[s:s + seg_len] for s in starts], dtype=np.float32)


def preprocess_dataset(records, labels, cfg: Config):
    """
    Filter → normalize → segment every record.
    Returns:
        X : np.ndarray  (total_segments, seg_len, 1)
        y : np.ndarray  (total_segments,)  int labels
    """
    X_all, y_all = [], []
    label_map = {"N": 0, "A": 1, "O": 2, "~": 3}

    for signal, label in zip(records, labels):
        # --- filter ---
        filtered = bandpass_filter(signal, cfg.LOWCUT, cfg.HIGHCUT,
                                   cfg.FS, cfg.FILTER_ORDER)
        # --- segment ---
        segs = segment_signal(filtered, cfg.SEGMENT_LEN)
        if len(segs) == 0:
            continue
        # --- normalize each segment ---
        segs = np.array([normalize(s) for s in segs])

        X_all.append(segs)
        y_all.extend([label_map[label]] * len(segs))

    X = np.concatenate(X_all, axis=0)          # (N, 3000)
    X = X[..., np.newaxis]                      # (N, 3000, 1) — channel dim for Conv1D
    y = np.array(y_all, dtype=np.int32)

    if cfg.BINARY:
        # AF (class 1) vs everything else (class 0)
        y = (y == 1).astype(np.int32)

    print(f"Segments: {X.shape[0]}  |  Shape: {X.shape}  |  "
          f"AF ratio: {y.mean():.2%}")
    return X, y


# ─────────────────────────────────────────────
#  3. CNN + LSTM MODEL
# ─────────────────────────────────────────────
def build_cnn_lstm(input_shape: tuple, num_classes: int,
                   dropout: float, l2: float) -> tf.keras.Model:
    """
    Architecture:
        Input → [Conv1D → BN → ReLU → MaxPool] x3
               → [Conv1D → BN → ReLU] x1
               → BiLSTM → Dropout
               → Dense(64) → Dropout
               → Output (sigmoid / softmax)
    """
    inp = layers.Input(shape=input_shape, name="ecg_input")

    # ── CNN BLOCK 1 ──────────────────────────
    x = layers.Conv1D(32, kernel_size=7, padding="same",
                      kernel_regularizer=regularizers.l2(l2))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)          # 3000 → 1500

    # ── CNN BLOCK 2 ──────────────────────────
    x = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)          # 1500 → 750

    # ── CNN BLOCK 3 ──────────────────────────
    x = layers.Conv1D(128, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)          # 750 → 375

    # ── CNN BLOCK 4 (deeper features) ────────
    x = layers.Conv1D(256, kernel_size=3, padding="same",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)          # 375 → 187

    # ── BIDIRECTIONAL LSTM ───────────────────
    # BiLSTM captures forward and backward temporal dependencies
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True,
                    dropout=dropout, recurrent_dropout=0.1)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False,
                    dropout=dropout)
    )(x)

    # ── DENSE HEAD ───────────────────────────
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)

    # ── OUTPUT ───────────────────────────────
    if num_classes == 2:
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inp, out, name="ECG_CNN_BiLSTM")
    return model


# ─────────────────────────────────────────────
#  4. TRAINING
# ─────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val, cfg: Config):
    num_classes = 2 if cfg.BINARY else 4
    input_shape = (cfg.SEGMENT_LEN, 1)

    model = build_cnn_lstm(input_shape, num_classes,
                           cfg.DROPOUT_RATE, cfg.L2_REG)
    model.summary()

    # ── Loss & optimizer ─────────────────────
    loss = "binary_crossentropy" if cfg.BINARY else "sparse_categorical_crossentropy"
    metrics = ["accuracy",
               tf.keras.metrics.AUC(name="auc"),
               tf.keras.metrics.Precision(name="precision"),
               tf.keras.metrics.Recall(name="recall")]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=loss,
        metrics=metrics
    )

    # ── Class weights (handles imbalance) ────
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")

    # ── Callbacks ────────────────────────────
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_auc", patience=8,
            restore_best_weights=True, mode="max", verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            cfg.MODEL_SAVE_PATH, monitor="val_auc",
            save_best_only=True, mode="max", verbose=1
        ),
    ]

    # ── Train ────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=cb_list,
        verbose=1
    )
    return model, history


# ─────────────────────────────────────────────
#  5. EVALUATION & PLOTS
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, history, cfg: Config):
    """Run evaluation and plot results."""
    y_prob = model.predict(X_test, verbose=0)

    if cfg.BINARY:
        y_pred = (y_prob.squeeze() >= 0.5).astype(int)
        auc    = roc_auc_score(y_test, y_prob.squeeze())
        target_names = ["Non-AF", "AF"]
    else:
        y_pred = np.argmax(y_prob, axis=1)
        auc    = roc_auc_score(y_test, y_prob, multi_class="ovr")
        target_names = ["Normal", "AF", "Other", "Noisy"]

    print("\n" + "=" * 50)
    print(" CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"ROC-AUC Score : {auc:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("AF Detection — CNN + BiLSTM Results", fontsize=15, fontweight="bold")

    # ── Training curves ───────────────────────
    axes[0, 0].plot(history.history["loss"], label="Train Loss")
    axes[0, 0].plot(history.history["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Epoch")

    axes[0, 1].plot(history.history["auc"], label="Train AUC")
    axes[0, 1].plot(history.history["val_auc"], label="Val AUC")
    axes[0, 1].set_title("AUC")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Epoch")

    axes[0, 2].plot(history.history["accuracy"], label="Train Acc")
    axes[0, 2].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0, 2].set_title("Accuracy")
    axes[0, 2].legend()
    axes[0, 2].set_xlabel("Epoch")

    # ── Confusion Matrix ──────────────────────
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=axes[1, 0])
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_ylabel("True Label")
    axes[1, 0].set_xlabel("Predicted Label")

    # ── ROC Curve (binary only) ───────────────
    if cfg.BINARY:
        fpr, tpr, _ = roc_curve(y_test, y_prob.squeeze())
        axes[1, 1].plot(fpr, tpr, color="darkorange",
                        label=f"ROC (AUC = {auc:.3f})")
        axes[1, 1].plot([0, 1], [0, 1], color="navy", linestyle="--")
        axes[1, 1].set_title("ROC Curve")
        axes[1, 1].set_xlabel("False Positive Rate")
        axes[1, 1].set_ylabel("True Positive Rate")
        axes[1, 1].legend()

    # ── Class distribution ────────────────────
    unique, counts = np.unique(y_test, return_counts=True)
    axes[1, 2].bar([target_names[i] for i in unique], counts,
                   color=["steelblue", "tomato"])
    axes[1, 2].set_title("Test Set Class Distribution")
    axes[1, 2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("af_results.png", dpi=150)
    plt.show()
    print("\nResults saved to af_results.png")


# ─────────────────────────────────────────────
#  6. INFERENCE — single ECG
# ─────────────────────────────────────────────
def predict_single(model, raw_signal: np.ndarray, cfg: Config) -> dict:
    """
    Run AF prediction on a single raw ECG signal.
    Returns: { 'label': str, 'af_probability': float }
    """
    filtered = bandpass_filter(raw_signal, cfg.LOWCUT, cfg.HIGHCUT, cfg.FS)
    segs     = segment_signal(filtered, cfg.SEGMENT_LEN)

    if len(segs) == 0:
        return {"label": "Signal too short", "af_probability": None}

    segs = np.array([normalize(s) for s in segs])[..., np.newaxis]
    probs = model.predict(segs, verbose=0).squeeze()

    if cfg.BINARY:
        af_prob = float(np.mean(probs))   # average probability across segments
        label   = "AF" if af_prob >= 0.5 else "Non-AF"
        return {"label": label, "af_probability": round(af_prob, 4)}
    else:
        avg_probs = np.mean(probs, axis=0)
        class_names = ["Normal", "AF", "Other", "Noisy"]
        return {
            "label": class_names[np.argmax(avg_probs)],
            "probabilities": dict(zip(class_names, avg_probs.tolist()))
        }


# ─────────────────────────────────────────────
#  7. MAIN
# ─────────────────────────────────────────────
def main():
    print("\n🔹 Loading data...")
    records, labels = load_cinc2017(cfg.DATA_DIR, cfg.REFERENCE_FILE)

    print("\n🔹 Preprocessing...")
    X, y = preprocess_dataset(records, labels, cfg)

    # ── Train / Val / Test split: 70 / 15 / 15 ──
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=cfg.RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176,     # 0.176 × 0.85 ≈ 0.15 of total
        stratify=y_temp, random_state=cfg.RANDOM_SEED
    )
    print(f"\nTrain: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    print("\n🔹 Training model...")
    model, history = train_model(X_train, y_train, X_val, y_val, cfg)

    print("\n🔹 Evaluating model...")
    evaluate_model(model, X_test, y_test, history, cfg)

    # ── Quick inference demo ──────────────────
    print("\n🔹 Demo inference on first test sample...")
    sample = X_test[0].squeeze()   # (3000,)
    result = predict_single(model, sample, cfg)
    print(f"   Prediction: {result}")

    print(f"\n✅ Model saved to: {cfg.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()