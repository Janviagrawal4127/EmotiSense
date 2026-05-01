# 🎤 EmotiSense — Complete Project Summary

## 🧭 Project Overview

**EmotiSense** is an end-to-end **Speech Emotion Recognition (SER)** system that detects human emotions from audio or video input in real time. It uses a custom **Transformer-based deep learning model** trained on the RAVDESS dataset and is served via a polished **Streamlit web application**.

---

## 📁 Final Project Structure

```
SpeechEmotion/
├── app.py                        ← Streamlit frontend (main entry point)
├── train.py                      ← Standalone local training script
├── requirements.txt              ← Python dependencies
├── README.md                     ← Project documentation
├── audio_speech_actors_01-24/    ← RAVDESS dataset (1440 WAV files, 24 actors)
└── content/
    └── ser_model/                ← Trained model artifacts (ready to use)
        ├── best_model.pt         ← Model weights (~3.5 MB)
        ├── scaler.pkl            ← StandardScaler for feature normalization
        ├── label_encoder.pkl     ← LabelEncoder (8 emotion classes)
        ├── config.json           ← Model + training hyperparameters
        ├── training_curves.png   ← Loss & accuracy history plots
        ├── confusion_matrix.png  ← Per-class test results
        └── class_distribution.png ← Dataset balance chart
```

---

## 🎯 Task & Dataset

| Property | Detail |
|----------|--------|
| **Task** | 8-class Speech Emotion Classification |
| **Dataset** | RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) |
| **Files** | 1440 WAV speech files |
| **Actors** | 24 (Actor_01 to Actor_24) |
| **Modality** | Speech only (modality code `03-01-*`) |
| **Sample Rate** | 16,000 Hz |

### 8 Emotion Classes

| Code | Emotion | Emoji |
|------|---------|-------|
| 01 | Neutral | 😐 |
| 02 | Calm | 😌 |
| 03 | Happy | 😊 |
| 04 | Sad | 😢 |
| 05 | Angry | 😠 |
| 06 | Fearful | 😨 |
| 07 | Disgust | 🤢 |
| 08 | Surprised | 😲 |

---

## 🔊 Audio Preprocessing Pipeline

Each audio file goes through this pipeline before feature extraction:

```
WAV File
   ↓
librosa.load()           — decode at 16 kHz mono
   ↓
librosa.effects.trim()   — remove leading/trailing silence
   ↓
Butterworth Low-Pass Filter  ← (NEW)
   cutoff = 4000 Hz, order = 5
   Implementation: scipy.signal.butter + sosfilt (SOS form)
   Removes high-freq noise, preserves speech fundamentals + formants F1–F4
   ↓
Feature Extraction
```

---

## 🎵 Feature Extraction

262-dimensional feature vector per frame (20ms frames, 50% overlap):

| Feature | Dims | Description |
|---------|------|-------------|
| MFCC | 40 | Mel-frequency cepstral coefficients |
| Δ MFCC | 40 | First-order delta (velocity) |
| ΔΔ MFCC | 40 | Second-order delta (acceleration) |
| Chroma | 12 | Pitch class energy distribution |
| Mel-Spectrogram | 128 | Log-power mel filterbank |
| ZCR | 1 | Zero-crossing rate |
| RMS Energy | 1 | Root mean square energy |
| **Total** | **262** | **Per frame** |

**Sequence chunking**: Frames are grouped into chunks of `seq_len=50` with 50% overlap → generates multiple training samples per audio file.

---

## 🧠 Model Architecture — SpeechEmotionTransformer

```
Input: (B, 50, 262)
   ↓
Input Projection: Linear(262→128) → LayerNorm → ReLU → Dropout(0.15)
   ↓
Positional Encoding: Learnable (1, 60, 128)
   ↓
Transformer Encoder (4 layers)
   Each layer: Pre-LN, 8 attention heads, FFN dim=512, Dropout(0.3)
   ↓
Global Pooling: AvgPool + MaxPool → concat → (B, 256)
   ↓
Classifier MLP:
   Linear(256→128) → GELU → Dropout(0.3)
   Linear(128→64)  → GELU → Dropout(0.15)
   Linear(64→8)
   ↓
Output: (B, 8) logits
```

**Total trainable parameters**: ~1.2M

---

## 🏋️ Training Strategy

### Hyperparameters (`config.json`)

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Max epochs | 150 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Label smoothing | 0.1 |
| Early stop patience | 20 |
| Min delta | 1e-4 |

### Optimizer & Scheduler
- **Optimizer**: AdamW
- **LR Scheduler**: CosineAnnealingWarmRestarts (`T_0=20, T_mult=2, eta_min=1e-6`)

### Data Split (stratified)
| Split | Ratio |
|-------|-------|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

### Anti-Overfitting Techniques
| Technique | Setting |
|-----------|---------|
| Early stopping | patience = 20 |
| Dropout | 0.3 (encoder), 0.15 (head) |
| Label smoothing | 0.1 |
| Weight decay (L2) | 1e-4 |
| Gradient clipping | max_norm = 1.0 |
| Time masking (aug) | random 1/8 of frames zeroed |
| Feature masking (aug) | random 1/8 of feature dims zeroed |
| Weighted random sampler | inverse class-frequency weights |
| Feature normalization | StandardScaler (fit on train) |
| **Butterworth low-pass** | cutoff=4000 Hz, order=5 |

---

## 📊 Model Performance

### Test Results

| Metric | Value |
|--------|-------|
| 🏆 **Test Accuracy** | **93.75%** |
| 📈 Peak Val Accuracy | ~92–93% |
| 🚂 Train Accuracy | ~96–97% |
| ⏹️ Epochs run | ~70 (early stopped) |

### Per-Class Accuracy (from Confusion Matrix)

| Emotion | Correct/Total | Accuracy |
|---------|--------------|----------|
| Angry | 38/38 | **100%** |
| Calm | 38/38 | **100%** |
| Fearful | 39/39 | **100%** |
| Neutral | 19/19 | **100%** |
| Surprised | 36/38 | **94.7%** |
| Happy | 35/39 | **89.7%** |
| Sad | 33/39 | **84.6%** |
| Disgust | 32/38 | **84.2%** |

> Disgust and Sad are the hardest — acoustically similar to Calm and other emotions.

---

## 🖥️ Streamlit Application (`app.py`)

### 4 Tabs

| Tab | Feature |
|-----|---------|
| 🎬 **Upload File** | WAV / MP3 / OGG / FLAC / M4A / **MP4 video** |
| 🎙️ **Record Live** | Browser microphone with auto-analysis |
| 📊 **Emotion Guide** | Reference cards for all 8 emotions + radar chart history |
| ℹ️ **How It Works** | Architecture overview + training strategy |

### Key UI Features
- **Dark theme** with glassmorphism and gradient hero banner
- **Plotly charts**: horizontal bar chart (confidence per emotion), radar chart (history)
- **Waveform visualizer** for live recordings
- **Segment tool** for MP4 videos (pick start/length)
- **Session history** — last 10 predictions tracked
- **Inference time** displayed per prediction
- GPU/CPU auto-detection shown in sidebar

### Inference Flow
```
Audio Input (upload / microphone / video)
   ↓
Butterworth Low-Pass Filter (4 kHz)
   ↓
Feature Extraction → chunks of (50, 262)
   ↓
StandardScaler.transform()
   ↓
SpeechEmotionTransformer.forward()  [batch of chunks]
   ↓
Softmax → average probabilities across all chunks
   ↓
argmax → predicted emotion + confidence scores
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | **PyTorch** ≥ 2.0 |
| Audio Processing | **librosa** ≥ 0.10 |
| Signal Filtering | **scipy** ≥ 1.11 (Butterworth filter) |
| ML Utilities | **scikit-learn** (StandardScaler, LabelEncoder, metrics) |
| Frontend | **Streamlit** ≥ 1.32 |
| Charts | **Plotly** ≥ 5.18 |
| Video Support | **moviepy** (MP4 audio extraction) |
| Live Recording | **audio-recorder-streamlit** |
| Audio I/O | **soundfile** |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app (model already trained)
streamlit run app.py
# → Open http://localhost:8501

# 3. Optional: Re-train the model locally
python train.py
```

---

## 🔄 Recent Changes Made

| Change | Files Affected |
|--------|---------------|
| ✅ Cleaned up extra files (notebooks, zip, cache) | — |
| ✅ Added Butterworth low-pass filter (4kHz, order=5) | `train.py`, `app.py` |
| ✅ Added `scipy>=1.11.0` dependency | `requirements.txt` |
| ✅ Updated README to reflect clean structure | `README.md` |
