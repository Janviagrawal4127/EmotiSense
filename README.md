
# 🎤 EmotiSense — Speech Emotion Recognition

A professional Speech Emotion Recognition (SER) system using a custom **Transformer** model trained on the full RAVDESS dataset.

## 📁 Project Structure

```
SpeechEmotion/
├── app.py                             # Streamlit frontend (main entry point)
├── train.py                           # Standalone training script (local GPU/CPU)
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── audio_speech_actors_01-24/         # RAVDESS dataset (WAV files)
└── content/
    └── ser_model/                     # Trained model artifacts
        ├── best_model.pt              # Model weights
        ├── scaler.pkl                 # StandardScaler for feature normalization
        ├── label_encoder.pkl          # LabelEncoder for emotion classes
        ├── config.json                # Model & training config
        ├── training_curves.png        # Loss/accuracy curves
        ├── confusion_matrix.png       # Per-class confusion matrix
        └── class_distribution.png    # Dataset class distribution
```

## 🚀 Quick Start

### 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### 2 — Run the Frontend
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

The model artifacts are already trained and saved in `content/ser_model/`.

---

## 🔁 Re-train the Model (optional)

If you want to re-train from scratch using the local RAVDESS dataset:

```bash
python train.py
```

This will:
1. Scan `./audio_speech_actors_01-24` for WAV files
2. Extract multi-modal audio features
3. Train the Transformer model
4. Save artifacts to `./content/ser_model/`

> 💡 Training runs on GPU if available, otherwise CPU (slower).

---

## 🎯 8 Emotion Classes

| Emotion   | Code | Emoji |
|-----------|------|-------|
| Neutral   | 01   | 😐    |
| Calm      | 02   | 😌    |
| Happy     | 03   | 😊    |
| Sad       | 04   | 😢    |
| Angry     | 05   | 😠    |
| Fearful   | 06   | 😨    |
| Disgust   | 07   | 🤢    |
| Surprised | 08   | 😲    |

---

## 🧠 Model Architecture

- **Input**: Multi-modal features — MFCC×3 + Chroma + Mel-Spec + ZCR + RMS = **262 dims** per frame
- **Sequence**: 50 frames per chunk (20ms frames, 50% overlap)
- **Encoder**: 4-layer Transformer (8 heads, 128d model dim, pre-norm)
- **Pooling**: Avg-pool + Max-pool → concatenated (256d)
- **Head**: 3-layer MLP with GELU activation → 8 emotion classes

## 🛡️ Anti-Overfitting Techniques

| Technique                  | Setting          |
|---------------------------|------------------|
| Early stopping            | patience = 20    |
| Dropout                   | 0.3              |
| Label smoothing           | 0.1              |
| Weight decay              | 1e-4             |
| Cosine Annealing LR       | warm restarts    |
| Gradient clipping         | max_norm = 1.0   |
| Data augmentation         | time + feat mask |
| Weighted random sampler   | class balance    |
| Feature normalization     | StandardScaler   |

## 📊 Results

- Validation accuracy: **~95%**
- Test accuracy: **~95%**

---

## 🎙️ Using the App

The Streamlit app supports three input modes:

| Tab | Method | Description |
|-----|--------|-------------|
| 🎬 Upload File | WAV / MP3 / OGG / FLAC / M4A / **MP4** | Upload any audio or video file |
| 🎙️ Record Live | Browser microphone | Real-time recording with auto-analysis |
| 📊 Emotion Guide | — | Reference for all 8 emotion classes |
| ℹ️ How It Works | — | Architecture and training details |

For video (MP4), audio is extracted automatically via `moviepy`.
