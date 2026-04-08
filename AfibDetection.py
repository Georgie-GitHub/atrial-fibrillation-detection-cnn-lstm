"""
=============================================================
 Real-Time AF Detection — ESP32 + AD8232 ECG Sensor
=============================================================
 Reads raw ECG samples from a COM port (sent by ESP32),
 buffers 10 seconds of data, runs the CNN+BiLSTM model
 every second, and prints the prediction live.

 Prerequisites:
   pip install tensorflow pyserial numpy scipy matplotlib

 ESP32 Arduino sketch should:
   - Sample the AD8232 at 300 Hz (one analogRead every ~3.33 ms)
   - Send each sample as a plain integer over Serial, one per line
   - Example Serial output line: "512"

 Usage:
   python realtime_af_detection.py --port COM3 --model af_cnn_lstm_model.keras

 Arguments:
   --port    COM port your ESP32 is on  (e.g. COM3  or  /dev/ttyUSB0)
   --model   Path to the saved .keras model file
   --baud    Baud rate (default 115200 — must match your Arduino sketch)
   --fs      Sampling frequency in Hz (default 300)
=============================================================
"""

import argparse
import time
import threading
import collections
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.signal
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Real-time AF detection from ESP32 ECG")
parser.add_argument("--port",  type=str,   default="COM5",                  help="COM port (e.g. COM3 or /dev/ttyUSB0)")
parser.add_argument("--model", type=str,   default="af_cnn_lstm_model.keras",help="Path to trained .keras model")
parser.add_argument("--baud",  type=int,   default=115200,                   help="Serial baud rate")
parser.add_argument("--fs",    type=int,   default=300,                      help="Sampling frequency in Hz")
args = parser.parse_args()

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
FS              = args.fs
SEGMENT_LEN     = FS * 10          # 10-second window = 3000 samples
PREDICT_EVERY   = FS * 1           # run model every 1 second = 300 new samples
DISPLAY_SAMPLES = FS * 5           # show 5 seconds on the live plot
LOWCUT          = 0.5
HIGHCUT         = 40.0
FILTER_ORDER    = 4
AF_THRESHOLD    = 0.5              # binary model — 0.5 is standard

# ─────────────────────────────────────────────
#  PREPROCESSING HELPERS
# ─────────────────────────────────────────────
def bandpass_filter(signal):
    nyq  = FS / 2.0
    b, a = scipy.signal.butter(FILTER_ORDER,
                               [LOWCUT / nyq, HIGHCUT / nyq],
                               btype="band")
    return scipy.signal.filtfilt(b, a, signal)


def normalize(signal):
    std = signal.std()
    return (signal - signal.mean()) / (std if std > 1e-6 else 1.0)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
print(f"\n🔹 Loading model from: {args.model}")
model = load_model(args.model)
print("✅ Model loaded")
print(f"   Input shape  : {model.input_shape}")
print(f"   Output shape : {model.output_shape}")

# ─────────────────────────────────────────────
#  SERIAL PORT
# ─────────────────────────────────────────────
print(f"\n🔹 Opening serial port {args.port} at {args.baud} baud...")
try:
    ser = serial.Serial(args.port, args.baud, timeout=2)
    time.sleep(2)           # wait for ESP32 to reset after serial open
    ser.flushInput()
    print(f"✅ Serial port open")
except serial.SerialException as e:
    print(f"❌ Could not open {args.port}: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────
# Ring buffer holding the last SEGMENT_LEN raw samples
ring_buffer    = collections.deque(maxlen=SEGMENT_LEN)

# Display buffer for the live plot (last 5 seconds)
display_buffer = collections.deque(maxlen=DISPLAY_SAMPLES)

# Thread-safe prediction result
prediction_lock   = threading.Lock()
current_label     = "Waiting..."
current_prob      = 0.0
sample_count      = 0          # counts new samples since last prediction
prediction_count  = 0          # total predictions made


# ─────────────────────────────────────────────
#  SERIAL READER THREAD
#  Reads samples from ESP32, fills the ring buffer.
#  Triggers a prediction every PREDICT_EVERY new samples.
# ─────────────────────────────────────────────
def read_serial():
    global sample_count, current_label, current_prob, prediction_count

    print("\n🔹 Serial reader thread started. Collecting data...\n")

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # ── Parse sample ─────────────────
            try:
                raw_value = float(line)
            except ValueError:
                # Skip non-numeric lines (e.g. ESP32 boot messages)
                continue

            ring_buffer.append(raw_value)
            display_buffer.append(raw_value)
            sample_count += 1

            # ── Run prediction every PREDICT_EVERY samples ──
            if sample_count >= PREDICT_EVERY and len(ring_buffer) == SEGMENT_LEN:
                sample_count = 0
                run_prediction()

        except serial.SerialException as e:
            print(f"\n⚠️  Serial error: {e}")
            break
        except Exception as e:
            print(f"\n⚠️  Unexpected error: {e}")
            continue


def run_prediction():
    global current_label, current_prob, prediction_count

    # Get current 10-second window
    segment = np.array(ring_buffer, dtype=np.float32)

    # Preprocess
    try:
        filtered = bandpass_filter(segment)
    except Exception:
        # Not enough data for filter yet
        return
    normed   = normalize(filtered)

    # Shape: (1, 3000, 1)
    X = normed[np.newaxis, :, np.newaxis]

    # Predict
    prob  = float(model.predict(X, verbose=0)[0][0])
    label = "⚠️  AFIB DETECTED" if prob >= AF_THRESHOLD else "✅ Normal"

    prediction_count += 1
    timestamp = time.strftime("%H:%M:%S")

    with prediction_lock:
        current_label = label
        current_prob  = prob

    # Print to terminal
    bar_len  = 30
    filled   = int(bar_len * prob)
    bar      = "█" * filled + "░" * (bar_len - filled)
    af_pct   = prob * 100
    norm_pct = (1 - prob) * 100

    print(f"[{timestamp}]  #{prediction_count:04d}  |  "
          f"P(AF)={af_pct:5.1f}%  P(Normal)={norm_pct:5.1f}%  "
          f"[{bar}]  →  {label}")


# ─────────────────────────────────────────────
#  LIVE PLOT
# ─────────────────────────────────────────────
fig, (ax_ecg, ax_pred) = plt.subplots(
    2, 1, figsize=(12, 6),
    gridspec_kw={"height_ratios": [3, 1]}
)
fig.suptitle("Real-Time ECG — AF Detection", fontsize=13, fontweight="bold")
fig.patch.set_facecolor("#1a1a2e")
for ax in (ax_ecg, ax_pred):
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["top"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["right"].set_color("#444")

# ECG waveform
x_ecg   = np.arange(DISPLAY_SAMPLES) / FS
ecg_line, = ax_ecg.plot(x_ecg, np.zeros(DISPLAY_SAMPLES),
                         color="#00d4ff", linewidth=0.9)
ax_ecg.set_xlim(0, DISPLAY_SAMPLES / FS)
ax_ecg.set_ylabel("Amplitude", color="white")
ax_ecg.set_xlabel("Time (s)", color="white")
ax_ecg.set_title("Live ECG Signal (last 5 s)", color="white", fontsize=10)

# Prediction bar
prob_bar = ax_pred.barh(
    ["AF Probability"], [0.0],
    color="#F44336", height=0.4, edgecolor="none"
)[0]
ax_pred.set_xlim(0, 1)
ax_pred.axvline(AF_THRESHOLD, color="white", linestyle="--",
                linewidth=1.2, label=f"Threshold ({AF_THRESHOLD})")
ax_pred.set_xlabel("P(AF)", color="white")
ax_pred.tick_params(axis="y", colors="white")
ax_pred.tick_params(axis="x", colors="white")
ax_pred.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

# Label text
label_text = ax_ecg.text(
    0.01, 0.95, "Collecting data...",
    transform=ax_ecg.transAxes,
    fontsize=12, fontweight="bold",
    color="white", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f3460", alpha=0.8)
)

plt.tight_layout()


def update_plot(frame):
    with prediction_lock:
        label = current_label
        prob  = current_prob

    # Update ECG waveform
    if len(display_buffer) == DISPLAY_SAMPLES:
        data = np.array(display_buffer, dtype=np.float32)
        # Normalise for display only
        d_std = data.std()
        data_norm = (data - data.mean()) / (d_std if d_std > 1e-6 else 1.0)
        ecg_line.set_ydata(data_norm)
        ax_ecg.set_ylim(data_norm.min() - 0.5, data_norm.max() + 0.5)
    elif len(display_buffer) > 0:
        pad  = DISPLAY_SAMPLES - len(display_buffer)
        data = np.array(display_buffer, dtype=np.float32)
        d_std = data.std()
        data_norm = (data - data.mean()) / (d_std if d_std > 1e-6 else 1.0)
        ecg_line.set_ydata(np.concatenate([np.zeros(pad), data_norm]))

    # Update AF probability bar
    prob_bar.set_width(prob)
    prob_bar.set_color("#F44336" if prob >= AF_THRESHOLD else "#4CAF50")

    # Update label
    label_text.set_text(label if label != "Waiting..." else "⏳ Collecting 10 s of data...")
    label_text.set_color("#FF5252" if "AFIB" in label else
                         "#69F0AE" if "Normal" in label else "white")

    return ecg_line, prob_bar, label_text


# ─────────────────────────────────────────────
#  START
# ─────────────────────────────────────────────
reader_thread = threading.Thread(target=read_serial, daemon=True)
reader_thread.start()

ani = animation.FuncAnimation(
    fig, update_plot,
    interval=200,          # refresh plot every 200 ms
    blit=False,
    cache_frame_data=False
)

print("\n📺 Live plot window opened.")
print("   Close the plot window or press Ctrl+C to stop.\n")

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    print("\n🔹 Closing serial port...")
    ser.close()
    print("✅ Done.")