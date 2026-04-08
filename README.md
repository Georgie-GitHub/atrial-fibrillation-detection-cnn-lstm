# 🫀 Atrial Fibrillation Detection using CNN + BiLSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-Educational-green)

## 📌 Overview

This project implements a deep learning model to automatically detect **Atrial Fibrillation (AF)** from raw ECG signals using a hybrid **CNN + Bidirectional LSTM** architecture.

- **CNN** → extracts local morphological features from raw ECG waveforms (QRS shape, absence of P waves, irregular baseline)
- **Bidirectional LSTM** → captures temporal dependencies in both forward and backward directions across the 10-second signal window
- **Goal** → binary classification: **AF vs Normal** with high recall to minimise missed diagnoses

> AFib is the most common serious cardiac arrhythmia, affecting over 37 million people globally and increasing stroke risk by 5×. Early automated detection is clinically critical.

---

## 📊 Dataset

This project uses the **PhysioNet/CinC Challenge 2017** dataset:

🔗 https://physionet.org/content/challenge-2017/1.0.0/

| Property | Value |
|----------|-------|
| Sampling rate | 300 Hz |
| Recording length | 9 – 60 seconds |
| Total recordings | ~8,528 |
| Labels | N (Normal), A (AFib), O (Other), ~ (Noisy) |
| Task | Binary: AF vs Normal |

### 📥 Download Instructions

The recommended method is via `wget` directly from the terminal:

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2017/1.0.0/training2017/
```

This recursively downloads all `.mat`, `.hea`, and `REFERENCE.csv` files. Alternatively:

1. Visit the dataset link above
2. Download all files manually
3. Extract them into a folder named `data/training2017/` inside the project directory

### 📁 Expected Directory Structure

```
project-folder/
│── data/
│   └── training2017/
│       ├── REFERENCE.csv       ← maps recording names to labels
│       ├── A00001.mat          ← ECG signal (binary MATLAB format)
│       ├── A00001.hea          ← header metadata
│       └── ...
│── af_cnn_lstm_model.keras     ← saved trained model
│── train.py                    ← model training script
│── evaluate.py                 ← evaluation & metrics script
│── requirements.txt
└── README.md
```

> ⚠️ **Note:** The dataset is not included in this repository due to size and licensing constraints. You must download it separately from PhysioNet.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install tensorflow numpy scipy scikit-learn imbalanced-learn matplotlib seaborn
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

**1. Ensure the dataset is placed in the correct folder** (see structure above)

**2. Run the training script:**

```bash
python train.py
```

This will:
- Load and preprocess all ECG recordings
- Apply bandpass filtering (0.5–40 Hz) and Z-score normalisation
- Segment signals into 10-second windows with 50% overlap
- Train the CNN + BiLSTM model
- Save the best model to `af_cnn_lstm_model.keras`

**3. Run evaluation on the test set:**

```bash
python evaluate.py
```

This will print the full classification report, confusion matrix, and ROC-AUC, and save a plot to `binary_af_evaluation.png`.

---

## 🧠 Model Architecture

The model is a hybrid **CNN + Bidirectional LSTM** that operates on raw ECG waveforms.

```
Input: (3000, 1)  ← 10 seconds × 300 Hz, 1 channel
    │
    ├── Conv1D(32, kernel=7) → BatchNorm → ReLU → MaxPool  → (1500, 32)
    ├── Conv1D(64, kernel=5) → BatchNorm → ReLU → MaxPool  → (750, 64)
    ├── Conv1D(128, kernel=5) → BatchNorm → ReLU → MaxPool → (375, 128)
    ├── Conv1D(256, kernel=3) → BatchNorm → ReLU → MaxPool → (187, 256)
    │
    ├── BiLSTM(128 units, return_sequences=True)   ← forward + backward pass
    ├── BiLSTM(64 units, return_sequences=False)   ← collapses to context vector
    │
    ├── Dense(64, ReLU) → Dropout(0.3)
    │
    └── Dense(1, Sigmoid)  ← output: probability of AFib (0.0 to 1.0)
```

### Preprocessing Pipeline

| Step | Details |
|------|---------|
| Bandpass filter | Butterworth order 4, 0.5–40 Hz (`filtfilt` — zero phase) |
| Segmentation | 10-second windows (3000 samples), 50% overlap |
| Normalisation | Z-score per segment: `(x − mean) / std` |
| Input shape | `(N, 3000, 1)` — N segments × 3000 samples × 1 channel |
| Train/Val/Test | 70% / 15% / 15%, stratified by class |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss | Binary Crossentropy |
| Optimiser | Adam (lr = 1e-3) |
| Batch size | 32 |
| Max epochs | 50 (early stopping, patience = 8) |
| Regularisation | L2 (λ = 1e-4), Dropout (0.3) |
| Class imbalance | `compute_class_weight('balanced')` |
| LR schedule | ReduceLROnPlateau (factor = 0.5, patience = 4) |

---

## 📈 Results

Evaluated on the held-out test set (15% of data, never seen during training):

### Overall Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **97.84%** |
| **ROC-AUC** | **0.9964** |

### Per-Class Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.9955 | 0.9797 | 0.9875 | 4034 |
| **AF** | **0.8746** | **0.9695** | **0.9196** | **590** |
| Weighted avg | 0.9800 | 0.9784 | 0.9788 | 4624 |

### Confusion Matrix (AF as Positive class)

```
                  Predicted Normal    Predicted AF
Actual Normal           3952               82       ← 82 false alarms
Actual AF                 18              572       ← only 18 missed!
```

| | Value |
|--|-------|
| True Positives (TP) | 572 |
| False Positives (FP) | 82 |
| False Negatives (FN) | 18 |
| True Negatives (TN) | 3952 |
| **Precision** | 0.8746 |
| **Recall** | 0.9695 |
| **F1 Score** | 0.9196 |

> The high recall (96.95%) is the most clinically important result — the model misses only 18 out of 590 real AF cases, prioritising patient safety over false alarm reduction.

---

## 📌 Key Features

- End-to-end deep learning on **raw ECG waveforms** — no manual feature engineering
- **Hybrid CNN + BiLSTM** architecture combining spatial and temporal learning
- Zero-phase Butterworth bandpass filtering to remove noise without distorting signal shape
- **50% overlapping segmentation** to maximise data from variable-length recordings
- Class imbalance handling via balanced class weights
- Early stopping and learning rate scheduling to prevent overfitting
- Full evaluation suite: confusion matrix, ROC curve, probability distribution plots

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3.8+ | Core language |
| TensorFlow / Keras | Model definition, training, inference |
| NumPy | Array operations and signal processing |
| SciPy | Butterworth filter (`scipy.signal`) and `.mat` file loading |
| Scikit-learn | Train/test split, metrics, class weights |
| Matplotlib / Seaborn | Visualisation of results |

---

## 📚 Citation

If you use this dataset, please cite:

```
Clifford, G., Liu, C., Moody, B., Lehman, L., Silva, I., Li, Q., Johnson, A., & Mark, R. (2017).
AF Classification from a Short Single Lead ECG Recording: the PhysioNet/Computing in Cardiology Challenge 2017.
Computing in Cardiology, 44. https://doi.org/10.22489/CinC.2017.065-469
```

Dataset available at: https://physionet.org/content/challenge-2017/1.0.0/

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for:
- Improvements to the model architecture
- Additional preprocessing techniques
- Extension to multi-class classification (Normal / AF / Other / Noisy)

---

## 📄 License

This project is for **educational and research purposes only**. The PhysioNet dataset is subject to its own licensing terms — please review them at the dataset link above before use.

---

## 👤 Author

**Your Name**
- GitHub: https://github.com/Georgie-GitHub
- LinkedIn: https://www.linkedin.com/in/georgieberalie/
