"""
🎤 EmotiSense — Speech Emotion Recognition App
Professional Streamlit frontend powered by the trained Transformer model.

Run: streamlit run app.py
"""

import os
import io
import json
import pickle
import time
import tempfile
import warnings
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ── Optional moviepy for MP4 support ──────────────────────────
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# ── Optional audio recorder ─────────────────────────────────────
try:
    from audio_recorder_streamlit import audio_recorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

# ── soundfile for WAV decode ────────────────────────────────────
import soundfile as sf
from scipy.signal import butter, sosfilt

# ── SpeechRecognition for transcription ─────────────────────────
try:
    import speech_recognition as sr_lib
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ───────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotiSense — Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────
# CSS — Premium Dark Theme
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:   #0a0e1a;
    --bg-card:      #111827;
    --bg-glass:     rgba(255,255,255,0.04);
    --border:       rgba(255,255,255,0.08);
    --accent-glow:  #6366f1;
    --accent-light: #818cf8;
    --accent-2:     #06b6d4;
    --text-primary: #f1f5f9;
    --text-muted:   #94a3b8;
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
    border-right: 1px solid var(--border);
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0a0e1a 0%, #1e1b4b 50%, #0a0e1a 100%);
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 32px;
    border: 1px solid rgba(99,102,241,0.3);
    box-shadow: 0 0 60px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.07);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 30%, rgba(99,102,241,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero p { color: var(--text-muted); font-size: 1.05rem; margin: 0; }
.hero .badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
    background: rgba(99,102,241,0.15); color: var(--accent-light);
    border: 1px solid rgba(99,102,241,0.3); margin-top: 12px;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(99,102,241,0.3); }

/* ── Emotion result ── */
.emotion-result {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(6,182,212,0.1));
    border-radius: 20px;
    border: 1px solid rgba(99,102,241,0.25);
    margin: 20px 0;
    position: relative;
    overflow: hidden;
}
.emotion-result::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 50%, rgba(99,102,241,0.05) 0%, transparent 70%);
}
.emotion-emoji { font-size: 4.5rem; display: block; margin-bottom: 12px; }
.emotion-label {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.emotion-conf { color: var(--text-muted); font-size: 0.95rem; margin-top: 6px; }

/* ── Metric chips ── */
.metric-chip {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.05); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px 16px; margin: 4px;
    font-size: 0.875rem; color: var(--text-muted);
}
.metric-chip strong { color: var(--text-primary); }

/* ── Sidebar ── */
.sidebar-section { margin-bottom: 24px; }
.sidebar-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-muted);
    margin-bottom: 12px;
}

/* ── Status pills ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
}
.status-ok   { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.status-warn { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.status-err  { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-glow), var(--accent-2)) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    font-size: 0.95rem !important; padding: 12px 28px !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(99,102,241,0.6) !important;
    transform: translateY(-1px) !important;
}
.stFileUploader {
    border: 2px dashed rgba(99,102,241,0.4) !important;
    border-radius: 16px !important;
    background: rgba(99,102,241,0.04) !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: rgba(6,182,212,0.06);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 16px;
    padding: 22px 26px;
    margin-top: 16px;
    position: relative;
}
.transcript-box .tr-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #06b6d4;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.transcript-box .tr-text {
    font-size: 1.05rem;
    line-height: 1.7;
    color: #e2e8f0;
    font-style: italic;
    word-break: break-word;
}
.transcript-box .tr-empty {
    color: #475569;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# EMOTION CONFIG
# ───────────────────────────────────────────────────────────────
EMOTION_INFO = {
    "neutral":   {"emoji": "😐", "color": "#94a3b8", "description": "Calm and composed tone"},
    "calm":      {"emoji": "😌", "color": "#3b82f6", "description": "Relaxed and peaceful"},
    "happy":     {"emoji": "😊", "color": "#f59e0b", "description": "Joyful and positive tone"},
    "sad":       {"emoji": "😢", "color": "#60a5fa", "description": "Melancholic or sorrowful"},
    "angry":     {"emoji": "😠", "color": "#ef4444", "description": "Intense and aggressive"},
    "fearful":   {"emoji": "😨", "color": "#a78bfa", "description": "Anxious or frightened"},
    "disgust":   {"emoji": "🤢", "color": "#10b981", "description": "Repulsed or disapproving"},
    "surprised": {"emoji": "😲", "color": "#f97316", "description": "Startled or astonished"},
}


# ───────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (must match training)
# ───────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SpeechEmotionTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"] * 0.5)
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=cfg["seq_len"] + 10,
                                          dropout=cfg["dropout"] * 0.5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"], batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg["num_layers"],
            norm=nn.LayerNorm(d_model)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(cfg["dropout"] * 0.5),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        return self.classifier(torch.cat([avg_pool, max_pool], dim=-1))


# ───────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ───────────────────────────────────────────────────────────────
def butterworth_lowpass(y, sr, cutoff=4000, order=5):
    """Apply a Butterworth low-pass filter to the audio signal.

    Removes high-frequency noise above `cutoff` Hz while preserving
    speech fundamentals and the first 3–4 formants (most emotion-relevant).

    Args:
        y      : audio signal (1-D numpy float32 array)
        sr     : sample rate (Hz)
        cutoff : cutoff frequency in Hz (default 4000 Hz)
        order  : Butterworth filter order (default 5)

    Returns:
        Filtered audio array (float32, same shape as y)
    """
    nyquist = sr / 2.0
    normal_cutoff = cutoff / nyquist        # normalise to [0, 1]
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sosfilt(sos, y).astype(np.float32)


def extract_sequences(audio_data, sr, cfg):
    """Extract frames (20ms, 50% overlap) and chunk into seq_len blocks.
    Applies a Butterworth low-pass filter (4 kHz cutoff) before
    feature extraction to suppress high-frequency noise.
    """
    seq_len = cfg.get("seq_len", 50)
    hop = cfg["hop_length"]
    n_fft = cfg.get("n_fft", 320)

    # Resample if needed
    if sr != cfg["sample_rate"]:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=cfg["sample_rate"])
    sr = cfg["sample_rate"]

    y = audio_data.astype(np.float32)
    y = butterworth_lowpass(y, sr)          # 🔽 Low-pass filter (4 kHz cutoff, order 5)

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg["n_mfcc"], n_fft=n_fft, hop_length=hop)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_chroma=cfg["n_chroma"])
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=cfg["n_mels"])
    mel_db = librosa.power_to_db(mel, ref=np.max)
    zcr    = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    rms    = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)

    features = np.vstack([mfcc, delta1, delta2, chroma, mel_db, zcr, rms]).T  # (T, 262)

    chunks = []
    step = max(1, seq_len // 2)
    for start in range(0, features.shape[0] - seq_len + 1, step):
        chunks.append(features[start:start+seq_len, :])

    if len(chunks) == 0:
        if features.shape[0] < seq_len:
            pad_w = seq_len - features.shape[0]
            padded = np.pad(features, ((0, pad_w), (0, 0)), mode="edge")
            chunks.append(padded)
            
    return chunks


# ───────────────────────────────────────────────────────────────
# MODEL LOADER
# ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_artifacts(model_dir: str):
    """Load model, scaler, label encoder and config."""
    model_dir = Path(model_dir)
    errors = []

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        errors.append("config.json not found")
        return None, None, None, None, errors

    with open(config_path) as f:
        cfg = json.load(f)

    # Load label encoder
    le_path = model_dir / "label_encoder.pkl"
    if not le_path.exists():
        errors.append("label_encoder.pkl not found")
        return None, None, None, cfg, errors

    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # Load scaler
    sc_path = model_dir / "scaler.pkl"
    if not sc_path.exists():
        errors.append("scaler.pkl not found")
        return None, None, le, cfg, errors

    with open(sc_path, "rb") as f:
        scaler = pickle.load(f)

    # Load model weights
    weights_path = model_dir / "best_model.pt"
    if not weights_path.exists():
        errors.append("best_model.pt not found")
        return None, scaler, le, cfg, errors

    input_dim = 40 * 3 + 12 + 128 + 1 + 1  # 262
    num_classes = len(le.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpeechEmotionTransformer(input_dim, num_classes, cfg)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, scaler, le, cfg, errors


def predict(audio_data, sr, model, scaler, le, cfg):
    """Run inference on raw audio across all sequence chunks and average."""
    device = next(model.parameters()).device
    
    chunks = extract_sequences(audio_data, sr, cfg)
    if not chunks:
        return le.classes_[0], np.zeros(len(le.classes_))
        
    scaled_chunks = []
    for chunk in chunks:
        scaled = scaler.transform(chunk).astype(np.float32)
        scaled_chunks.append(scaled)
        
    feat_t = torch.tensor(np.array(scaled_chunks)).to(device)  # (Num_Chunks, seq_len, F)

    with torch.no_grad():
        logits = model(feat_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()  # (Num_Chunks, num_classes)
        
    # Average the probabilities across all chunks in the file
    avg_probs = probs.mean(axis=0)
    pred_idx = avg_probs.argmax()
    
    return le.classes_[pred_idx], avg_probs


# ───────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("## 🎤 EmotiSense")
    st.markdown('<p style="color:#64748b; font-size:0.8rem;">Speech Emotion Recognition</p>',
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">MODEL DIRECTORY</div>', unsafe_allow_html=True)
    model_dir = st.text_input(
        "Path to model artifacts",
        value="./content/ser_model",
        help="Folder containing best_model.pt, scaler.pkl, label_encoder.pkl, config.json",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-title">ABOUT</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="color:#64748b; font-size:0.82rem; line-height:1.6;">
<b style="color:#94a3b8;">Architecture:</b> Transformer Encoder<br>
<b style="color:#94a3b8;">Dataset:</b> RAVDESS (1440 files)<br>
<b style="color:#94a3b8;">Emotions:</b> 8 classes<br>
<b style="color:#94a3b8;">Features:</b> MFCC + Delta + Chroma + Mel + ZCR + RMS<br>
<b style="color:#94a3b8;">Framework:</b> PyTorch + Streamlit
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    device_str = "🚀 GPU" if torch.cuda.is_available() else "💻 CPU"
    st.markdown(f'<span class="status-pill status-ok">{device_str}</span>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# LOAD MODEL
# ───────────────────────────────────────────────────────────────
model_obj, scaler_obj, le_obj, cfg_obj, load_errors = load_model_artifacts(model_dir)
model_loaded = model_obj is not None

# ───────────────────────────────────────────────────────────────
# HERO
# ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎤 EmotiSense</h1>
    <p>Real-time Speech Emotion Recognition powered by a Transformer model<br>
    trained on the RAVDESS dataset — upload audio, video, or any speech file to detect emotions.</p>
    <span class="badge">8 EMOTION CLASSES · TRANSFORMER · PYTORCH · MP4 VIDEO SUPPORT</span>
</div>
""", unsafe_allow_html=True)

# Model status
if not model_loaded:
    st.markdown(f"""
<div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
     border-radius:12px; padding:16px 20px; margin-bottom:20px;">
    <b style="color:#ef4444;">⚠️  Model not loaded</b><br>
    <span style="color:#94a3b8; font-size:0.9rem;">
    {', '.join(load_errors) if load_errors else 'Set the model directory in the sidebar.'}<br><br>
    <b>To get started:</b><br>
    1. Run the Colab notebook to train the model<br>
    2. Download <code>ser_artifacts.zip</code><br>
    3. Extract next to <code>app.py</code> as <code>ser_model/</code><br>
    4. Set the path in the sidebar
    </span>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown(f"""
<div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25);
     border-radius:12px; padding:12px 18px; margin-bottom:20px; display:inline-flex; align-items:center; gap:10px;">
    <span style="color:#10b981; font-size:1.2rem;">✅</span>
    <span style="color:#94a3b8; font-size:0.9rem;">
    Model loaded — {len(le_obj.classes_)} emotions · {cfg_obj.get('d_model',128)}d Transformer
    </span>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# TRANSCRIPTION HELPER
# ───────────────────────────────────────────────────────────────
def transcribe_audio(audio_array: np.ndarray, sample_rate: int) -> str:
    """Convert audio numpy array to text using Google Web Speech API.

    Returns the transcribed string, or an informative message on failure.
    """
    if not SR_AVAILABLE:
        return "⚠️ Install 'SpeechRecognition' to enable transcription: pip install SpeechRecognition"

    recognizer = sr_lib.Recognizer()

    # Write float audio to WAV bytes in memory
    wav_buf = io.BytesIO()
    import wave, struct
    pcm = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(wav_buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    wav_buf.seek(0)

    try:
        with sr_lib.AudioFile(wav_buf) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr_lib.UnknownValueError:
        return "🔇 Speech not recognized — try speaking more clearly"
    except sr_lib.RequestError as e:
        return f"🌐 Network error: {e}"
    except Exception as e:
        return f"❌ Transcription error: {e}"


def render_transcript(text: str):
    """Render a styled transcript card."""
    if text.startswith(("⚠️", "🔇", "🌐", "❌")):
        body = f'<div class="tr-empty">{text}</div>'
    else:
        body = f'<div class="tr-text">" {text} "</div>'

    st.markdown(f"""
<div class="transcript-box">
  <div class="tr-label">🗣️ &nbsp; Speech Transcript</div>
  {body}
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# AUDIO EXTRACTION HELPER
# ───────────────────────────────────────────────────────────────
def extract_audio_from_video(video_bytes: bytes, ext: str = ".mp4"):
    """Extract mono audio array from video bytes using moviepy."""
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid:
        tmp_vid.write(video_bytes)
        tmp_vid_path = tmp_vid.name

    tmp_wav_path = tmp_vid_path.replace(ext, "_audio.wav")
    try:
        clip = VideoFileClip(tmp_vid_path)
        clip.audio.write_audiofile(tmp_wav_path, fps=16000, nbytes=2,
                                   codec='pcm_s16le', logger=None)
        clip.close()
        y, sr = librosa.load(tmp_wav_path, sr=16000, mono=True)
    finally:
        if os.path.exists(tmp_vid_path):  os.unlink(tmp_vid_path)
        if os.path.exists(tmp_wav_path):  os.unlink(tmp_wav_path)
    return y, sr


# ───────────────────────────────────────────────────────────────
# MAIN CONTENT
# ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎬 Upload File", "🎙️ Record Live", "📊 Emotion Guide", "ℹ️ How It Works"])

# ── TAB 1: Upload & Predict ────────────────────────────────────
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📂 Upload Audio or Video File")

        # Format badges
        is_mp4_ok = MOVIEPY_AVAILABLE
        mp4_badge = (
            '<span style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.3);'
            'border-radius:6px;padding:2px 8px;font-size:0.72rem;font-weight:600;">MP4 ✓</span>'
            if is_mp4_ok else
            '<span style="background:rgba(239,68,68,0.12);color:#ef4444;border:1px solid rgba(239,68,68,0.3);'
            'border-radius:6px;padding:2px 8px;font-size:0.72rem;font-weight:600;" '
            'title="Run: pip install moviepy">MP4 (install moviepy)</span>'
        )
        st.markdown(
            f'<p style="color:#64748b;font-size:0.88rem;margin-bottom:14px;">'
            f'Audio: WAV · MP3 · OGG · FLAC · M4A &nbsp;|&nbsp; Video: {mp4_badge}</p>',
            unsafe_allow_html=True
        )

        accepted_types = ["wav", "mp3", "ogg", "flac", "m4a"]
        if is_mp4_ok:
            accepted_types.append("mp4")

        uploaded_file = st.file_uploader(
            "Drop your file here",
            type=accepted_types,
            label_visibility="collapsed"
        )

        is_video = uploaded_file is not None and uploaded_file.name.lower().endswith(".mp4")

        if uploaded_file:
            if is_video:
                st.video(uploaded_file)
                st.markdown(f"""
<div style="margin-top:10px;">
<span class="metric-chip">🎬 <strong>{uploaded_file.name}</strong></span>
<span class="metric-chip">Size: <strong>{uploaded_file.size/1024/1024:.2f} MB</strong></span>
<span style="background:rgba(99,102,241,0.15);color:#818cf8;border:1px solid rgba(99,102,241,0.3);
border-radius:8px;padding:4px 10px;font-size:0.78rem;margin:4px;">🔊 Audio will be extracted automatically</span>
</div>
""", unsafe_allow_html=True)
            else:
                st.audio(uploaded_file)
                st.markdown(f"""
<div style="margin-top:10px;">
<span class="metric-chip">🎵 <strong>{uploaded_file.name}</strong></span>
<span class="metric-chip">Size: <strong>{uploaded_file.size/1024:.1f} KB</strong></span>
</div>
""", unsafe_allow_html=True)

        # Segment option for long videos
        segment_start = 0.0
        if is_video:
            st.markdown("---")
            st.markdown('<p style="color:#94a3b8;font-size:0.85rem;font-weight:600;">⏱️ Analyze Segment (optional)</p>',
                        unsafe_allow_html=True)
            seg_col1, seg_col2 = st.columns(2)
            with seg_col1:
                segment_start = st.number_input("Start (sec)", min_value=0.0, value=0.0, step=0.5)
            with seg_col2:
                segment_len = st.number_input("Length (sec)", min_value=1.0, value=5.0, step=0.5)

        analyze_btn = st.button(
            "🔍 Analyze Emotion",
            disabled=not (model_loaded and uploaded_file),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Tips card
        st.markdown("""
<div style="background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.2);
     border-radius:12px; padding:16px; margin-top:16px;">
<p style="color:#818cf8; font-weight:600; margin-bottom:8px;">💡 Tips for Best Results</p>
<ul style="color:#64748b; font-size:0.84rem; margin:0; padding-left:18px; line-height:1.9;">
  <li>Audio: 2–4 seconds of clear speech works best</li>
  <li>Video: Use the segment tool to pick the most emotional moment</li>
  <li>Quiet environment = higher accuracy</li>
  <li>Supported video: MP4 · H.264 · any resolution</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col_result:
        result_placeholder = st.empty()

        if analyze_btn and uploaded_file and model_loaded:
            with st.spinner("🧠 Extracting audio & analyzing emotion..."):
                try:
                    file_bytes = uploaded_file.read()

                    if is_video:
                        if not MOVIEPY_AVAILABLE:
                            result_placeholder.error("❌ moviepy not installed. Run: pip install moviepy")
                            st.stop()
                        # Write video to temp, extract audio
                        y_full, sr = extract_audio_from_video(file_bytes, ext=".mp4")
                        # Crop to chosen segment
                        start_sample = int(segment_start * sr)
                        end_sample   = int((segment_start + segment_len) * sr)
                        y = y_full[start_sample:end_sample]
                        if len(y) == 0:
                            y = y_full  # fallback to full
                    else:
                        ext = "." + uploaded_file.name.rsplit(".", 1)[-1]
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name
                        y, sr = librosa.load(tmp_path, sr=None, mono=True)
                        os.unlink(tmp_path)

                    duration  = len(y) / sr
                    start_t   = time.time()
                    emotion, probs = predict(y, sr, model_obj, scaler_obj, le_obj, cfg_obj)
                    infer_ms  = (time.time() - start_t) * 1000

                    info  = EMOTION_INFO.get(emotion, {})
                    emoji = info.get("emoji", "🎭")
                    color = info.get("color", "#6366f1")
                    desc  = info.get("description", "")
                    conf  = probs.max() * 100

                    # History
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "file": uploaded_file.name,
                        "emotion": emotion,
                        "confidence": conf,
                        "probs": probs.tolist(),
                        "classes": le_obj.classes_.tolist(),
                        "type": "video" if is_video else "audio",
                    })

                    with result_placeholder.container():
                        # Video segment info
                        if is_video:
                            st.markdown(f"""
<div style="background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.25);
border-radius:10px;padding:10px 16px;margin-bottom:12px;font-size:0.83rem;color:#94a3b8;">
🎬 Analyzing segment <b style="color:#06b6d4;">{segment_start:.1f}s → {segment_start+segment_len:.1f}s</b>
&nbsp;·&nbsp; Audio extracted: <b style="color:#e2e8f0;">{duration:.1f}s</b>
</div>
""", unsafe_allow_html=True)

                        st.markdown(f"""
<div class="emotion-result">
    <span class="emotion-emoji">{emoji}</span>
    <div class="emotion-label">{emotion.upper()}</div>
    <div class="emotion-conf">{desc}</div>
    <div style="margin-top:14px; display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
        <span class="metric-chip">Confidence: <strong style="color:{color};">{conf:.1f}%</strong></span>
        <span class="metric-chip">Duration: <strong>{duration:.1f}s</strong></span>
        <span class="metric-chip">Inference: <strong>{infer_ms:.0f}ms</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

                        # Probability bar chart
                        sorted_idx    = np.argsort(probs)[::-1]
                        sorted_labels = [le_obj.classes_[i] for i in sorted_idx]
                        sorted_probs  = [probs[i] * 100 for i in sorted_idx]
                        bar_colors    = [EMOTION_INFO.get(e, {}).get("color", "#6366f1") for e in sorted_labels]

                        fig = go.Figure(go.Bar(
                            x=sorted_probs, y=sorted_labels,
                            orientation='h',
                            marker=dict(color=bar_colors, opacity=0.85,
                                        line=dict(color="rgba(255,255,255,0.1)", width=1)),
                            text=[f"{p:.1f}%" for p in sorted_probs],
                            textposition='inside',
                            textfont=dict(color='white', size=12, family='Inter'),
                        ))
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#94a3b8', family='Inter'),
                            xaxis=dict(gridcolor='rgba(255,255,255,0.06)', range=[0, 105]),
                            yaxis=dict(gridcolor='rgba(0,0,0,0)'),
                            margin=dict(l=0, r=20, t=30, b=0),
                            height=300,
                            title=dict(text="Confidence per Emotion",
                                       font=dict(size=13, color='#cbd5e1')),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # ── Speech-to-Text Transcript ──────────────────
                        with st.spinner("📝 Transcribing speech..."):
                            transcript = transcribe_audio(y, sr)
                        render_transcript(transcript)

                except Exception as e:
                    result_placeholder.error(f"❌ Error during analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

        elif not analyze_btn:
            with result_placeholder.container():
                st.markdown("""
<div class="card" style="text-align:center; padding:60px 20px; color:#475569;">
    <div style="font-size:3rem; margin-bottom:16px;">🎯</div>
    <div style="font-size:1.1rem; color:#64748b;">Upload an audio or video file<br>and click <b style="color:#818cf8;">Analyze Emotion</b> to get started</div>
    <div style="margin-top:16px; font-size:0.82rem; color:#334155;">Supports WAV · MP3 · OGG · FLAC · M4A · <b style="color:#818cf8;">MP4 VIDEO</b></div>
</div>
""", unsafe_allow_html=True)


# ── TAB 2: Live Recorder ────────────────────────────────────────
with tab2:

    rec_col, res_col = st.columns([1, 1], gap="large")

    with rec_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎙️ Live Microphone Recorder")
        st.markdown("""
<p style="color:#64748b; font-size:0.88rem; margin-bottom:20px;">
Click the mic button below, speak your emotion, then click again to stop.<br>
The app will <b style="color:#818cf8;">automatically analyze</b> your recording.
</p>
""", unsafe_allow_html=True)

        if not RECORDER_AVAILABLE:
            st.markdown("""
<div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
     border-radius:12px; padding:16px;">
<b style="color:#ef4444;">⚠️ Recorder not installed</b><br>
<span style="color:#94a3b8; font-size:0.88rem;">Run: <code>pip install audio-recorder-streamlit</code></span>
</div>
""", unsafe_allow_html=True)
            audio_bytes_recorded = None
        else:
            # Recorder widget — pulse animation
            st.markdown("""
<div style="display:flex;flex-direction:column;align-items:center;gap:12px;padding:16px 0;">
    <div style="font-size:0.82rem;color:#64748b;letter-spacing:0.05em;">PRESS TO START RECORDING</div>
</div>
""", unsafe_allow_html=True)

            audio_bytes_recorded = audio_recorder(
                text="",
                recording_color="#ef4444",
                neutral_color="#6366f1",
                icon_name="microphone",
                icon_size="3x",
                pause_threshold=3.0,   # auto-stop after 3s silence
                sample_rate=16000,
            )

            # Status hint
            st.markdown("""
<div style="text-align:center;margin-top:8px;">
  <span style="font-size:0.80rem;color:#475569;">
    🔴 Click mic to record &nbsp;·&nbsp; Click again to stop
  </span>
</div>
""", unsafe_allow_html=True)

        # Recording tips
        st.markdown("---")
        st.markdown("""
<div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.2);
     border-radius:12px;padding:16px;margin-top:4px;">
<p style="color:#818cf8;font-weight:600;margin-bottom:8px;">💡 Recording Tips</p>
<ul style="color:#64748b;font-size:0.83rem;margin:0;padding-left:18px;line-height:1.9;">
  <li>Speak <b>2–5 seconds</b> for best results</li>
  <li>Use a <b>quiet environment</b> — no background music</li>
  <li>Speak naturally with <b>clear emotion</b></li>
  <li>Browser will ask for <b>microphone permission</b> on first use</li>
  <li>Auto-stops after <b>3 seconds of silence</b></li>
</ul>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recording History in sidebar-like panel
        if "rec_history" in st.session_state and st.session_state.rec_history:
            st.markdown("---")
            st.markdown("#### 🕐 Recent Recordings")
            for i, rec in enumerate(reversed(st.session_state.rec_history[-5:])):
                info = EMOTION_INFO.get(rec["emotion"], {})
                em   = info.get("emoji", "🎭")
                col  = info.get("color", "#6366f1")
                st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
     background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
     border-radius:10px;margin-bottom:6px;">
  <span style="font-size:1.5rem;">{em}</span>
  <div>
    <div style="color:{col};font-weight:700;font-size:0.9rem;">{rec['emotion'].upper()}</div>
    <div style="color:#475569;font-size:0.75rem;">{rec['conf']:.1f}% confidence · {rec['duration']:.1f}s</div>
  </div>
</div>
""", unsafe_allow_html=True)

    with res_col:
        rec_result_ph = st.empty()

        # ── Auto-analyze on new recording ──────────────────────────
        if audio_bytes_recorded and model_loaded:
            # Detect if this is a new recording (bytes changed)
            prev_bytes = st.session_state.get("last_rec_bytes", b"")

            if audio_bytes_recorded != prev_bytes and len(audio_bytes_recorded) > 2000:
                st.session_state["last_rec_bytes"] = audio_bytes_recorded

                with rec_result_ph.container():
                    with st.spinner("🧠 Analyzing your voice..."):
                        try:
                            # Decode WAV bytes from recorder
                            buf = io.BytesIO(audio_bytes_recorded)
                            y_rec, sr_rec = sf.read(buf, dtype="float32")

                            # Mono if stereo
                            if y_rec.ndim > 1:
                                y_rec = y_rec.mean(axis=1)

                            duration_rec = len(y_rec) / sr_rec

                            # Waveform chart
                            t_axis = np.linspace(0, duration_rec, len(y_rec))
                            downsample = max(1, len(y_rec) // 1000)
                            fig_wave = go.Figure(go.Scatter(
                                x=t_axis[::downsample],
                                y=y_rec[::downsample],
                                mode='lines',
                                line=dict(color='#6366f1', width=1.2),
                                fill='tozeroy',
                                fillcolor='rgba(99,102,241,0.15)',
                            ))
                            fig_wave.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=120,
                                margin=dict(l=0, r=0, t=20, b=0),
                                xaxis=dict(gridcolor='rgba(255,255,255,0.05)',
                                           tickfont=dict(color='#475569'),
                                           title=dict(text='Time (s)',
                                                      font=dict(color='#475569'))),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.05)',
                                           showticklabels=False),
                                title=dict(text=f'Recorded Audio — {duration_rec:.2f}s',
                                           font=dict(color='#94a3b8', size=12)),
                            )
                            st.plotly_chart(fig_wave, use_container_width=True)

                            # Playback
                            st.audio(audio_bytes_recorded, format="audio/wav")

                            # Predict
                            start_t = time.time()
                            emotion_r, probs_r = predict(y_rec, sr_rec, model_obj,
                                                         scaler_obj, le_obj, cfg_obj)
                            infer_ms_r = (time.time() - start_t) * 1000

                            info_r  = EMOTION_INFO.get(emotion_r, {})
                            emoji_r = info_r.get("emoji", "🎭")
                            color_r = info_r.get("color", "#6366f1")
                            desc_r  = info_r.get("description", "")
                            conf_r  = probs_r.max() * 100

                            # Save to rec history
                            if "rec_history" not in st.session_state:
                                st.session_state.rec_history = []
                            st.session_state.rec_history.append({
                                "emotion": emotion_r,
                                "conf": conf_r,
                                "duration": duration_rec,
                            })

                            # Result card
                            st.markdown(f"""
<div class="emotion-result">
    <span class="emotion-emoji">{emoji_r}</span>
    <div class="emotion-label">{emotion_r.upper()}</div>
    <div class="emotion-conf">{desc_r}</div>
    <div style="margin-top:14px;display:flex;justify-content:center;gap:12px;flex-wrap:wrap;">
        <span class="metric-chip">Confidence: <strong style="color:{color_r};">{conf_r:.1f}%</strong></span>
        <span class="metric-chip">Duration: <strong>{duration_rec:.1f}s</strong></span>
        <span class="metric-chip">Inference: <strong>{infer_ms_r:.0f}ms</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

                            # Probability bars
                            s_idx  = np.argsort(probs_r)[::-1]
                            s_lbls = [le_obj.classes_[i] for i in s_idx]
                            s_prbs = [probs_r[i] * 100 for i in s_idx]
                            s_cols = [EMOTION_INFO.get(e, {}).get("color", "#6366f1")
                                      for e in s_lbls]

                            fig_bar = go.Figure(go.Bar(
                                x=s_prbs, y=s_lbls, orientation='h',
                                marker=dict(color=s_cols, opacity=0.85,
                                            line=dict(color="rgba(255,255,255,0.1)", width=1)),
                                text=[f"{p:.1f}%" for p in s_prbs],
                                textposition='inside',
                                textfont=dict(color='white', size=12, family='Inter'),
                            ))
                            fig_bar.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#94a3b8', family='Inter'),
                                xaxis=dict(gridcolor='rgba(255,255,255,0.06)',
                                           range=[0, 105]),
                                yaxis=dict(gridcolor='rgba(0,0,0,0)'),
                                margin=dict(l=0, r=20, t=30, b=0),
                                height=300,
                                title=dict(text="Confidence per Emotion",
                                           font=dict(size=13, color='#cbd5e1')),
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                            # ── Speech-to-Text Transcript ───────────────
                            with st.spinner("📝 Transcribing speech..."):
                                transcript_r = transcribe_audio(y_rec, sr_rec)
                            render_transcript(transcript_r)

                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")

            else:
                with rec_result_ph.container():
                    # Show last result if same recording re-rendered
                    if st.session_state.get("rec_history"):
                        last = st.session_state.rec_history[-1]
                        info_l = EMOTION_INFO.get(last["emotion"], {})
                        st.markdown(f"""
<div class="emotion-result">
    <span class="emotion-emoji">{info_l.get('emoji','🎭')}</span>
    <div class="emotion-label">{last['emotion'].upper()}</div>
    <div class="emotion-conf">Last recording result</div>
    <div style="margin-top:10px;">
        <span class="metric-chip">Confidence: <strong style="color:{info_l.get('color','#6366f1')};">{last['conf']:.1f}%</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)
        else:
            with rec_result_ph.container():
                not_ready_msg = (
                    "⚠️ Load the model first (set path in sidebar)"
                    if not model_loaded else
                    "Press the 🎙️ mic button to start recording"
                )
                st.markdown(f"""
<div class="card" style="text-align:center;padding:60px 20px;">
    <div style="font-size:3.5rem;margin-bottom:16px;">🎙️</div>
    <div style="font-size:1.1rem;color:#64748b;">{not_ready_msg}</div>
    <div style="margin-top:16px;font-size:0.82rem;color:#334155;">
        Recording auto-analyzes instantly after you stop
    </div>
</div>
""", unsafe_allow_html=True)


# ── TAB 3: Emotion Guide ───────────────────────────────────────
with tab3:
    st.markdown("### 🎭 Emotion Classes")
    st.markdown('<p style="color:#64748b;">The model recognizes 8 distinct emotions from the RAVDESS dataset.</p>',
                unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (emotion, info) in enumerate(EMOTION_INFO.items()):
        with cols[i % 4]:
            st.markdown(f"""
<div style="background:{info['color']}18; border:1px solid {info['color']}44;
     border-radius:14px; padding:20px; text-align:center; margin-bottom:16px;
     transition:all 0.2s;">
    <div style="font-size:2.5rem; margin-bottom:8px;">{info['emoji']}</div>
    <div style="font-weight:700; font-size:1rem; color:{info['color']}; margin-bottom:4px;">{emotion.upper()}</div>
    <div style="color:#64748b; font-size:0.78rem;">{info['description']}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Prediction History")

    if "history" in st.session_state and st.session_state.history:
        hist = st.session_state.history[-10:]  # Last 10

        # Radar chart of last prediction
        if hist:
            last = hist[-1]
            labels = last["classes"]
            vals   = last["probs"]
            colors_radar = [EMOTION_INFO.get(l, {}).get("color", "#6366f1") for l in labels]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[v * 100 for v in vals] + [vals[0] * 100],
                theta=labels + [labels[0]],
                fill='toself',
                fillcolor='rgba(99,102,241,0.15)',
                line=dict(color='#6366f1', width=2),
                name="Emotion Probabilities"
            ))
            fig_radar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(gridcolor='rgba(255,255,255,0.1)', visible=True,
                                    range=[0, 100], color='#94a3b8'),
                    angularaxis=dict(color='#94a3b8', gridcolor='rgba(255,255,255,0.1)')
                ),
                font=dict(color='#94a3b8', family='Inter'),
                height=350,
                title=dict(text=f"Last Prediction: {last['emotion'].upper()} ({last['confidence']:.1f}%)",
                           font=dict(color='#cbd5e1', size=13)),
                margin=dict(t=60, b=0)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # History table
        tbl_data = {
            "File": [h["file"] for h in hist],
            "Emotion": [f"{EMOTION_INFO.get(h['emotion'],{}).get('emoji','🎭')} {h['emotion']}" for h in hist],
            "Confidence": [f"{h['confidence']:.1f}%" for h in hist],
        }
        st.dataframe(tbl_data, use_container_width=True)
    else:
        st.info("No predictions yet — analyze some audio files to see your history.")


# ── TAB 4: How It Works ────────────────────────────────────────
with tab4:
    st.markdown("### 🔬 Architecture Overview")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
<div class="card">
<h4 style="color:#818cf8; margin-top:0;">🎵 Feature Extraction Pipeline</h4>
<ol style="color:#94a3b8; line-height:2;">
    <li>Load audio at <b style="color:#e2e8f0;">16 kHz</b></li>
    <li>Pad/trim to <b style="color:#e2e8f0;">3 seconds</b></li>
    <li>Extract <b style="color:#e2e8f0;">MFCC (40)</b> + Delta + Delta²</li>
    <li>Extract <b style="color:#e2e8f0;">Chroma (12)</b> features</li>
    <li>Extract <b style="color:#e2e8f0;">Mel-Spectrogram (128)</b></li>
    <li>Extract <b style="color:#e2e8f0;">ZCR + RMS Energy</b></li>
    <li>Stack → <b style="color:#e2e8f0;">262-dim per frame</b></li>
    <li>Resize to <b style="color:#e2e8f0;">128 frames</b> → (128 × 262)</li>
</ol>
</div>
""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
<div class="card">
<h4 style="color:#06b6d4; margin-top:0;">🧠 Transformer Model</h4>
<ul style="color:#94a3b8; line-height:2; list-style:none; padding:0;">
    <li>⚡ <b style="color:#e2e8f0;">Input Projection</b>: Linear → LayerNorm → ReLU</li>
    <li>📍 <b style="color:#e2e8f0;">Positional Encoding</b>: Learnable embeddings</li>
    <li>🔄 <b style="color:#e2e8f0;">4× Transformer Layers</b>: Pre-LN, 8 heads</li>
    <li>🏊 <b style="color:#e2e8f0;">Pooling</b>: Avg + Max → concat</li>
    <li>🎯 <b style="color:#e2e8f0;">Classifier</b>: 3-layer MLP + GELU</li>
    <li>🛡️ <b style="color:#e2e8f0;">Anti-overfitting</b>: Dropout (0.3), Label Smoothing</li>
    <li>⏹️ <b style="color:#e2e8f0;">Early Stopping</b>: patience=20</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card" style="margin-top:0;">
<h4 style="color:#f59e0b; margin-top:0;">📊 Training Strategy</h4>
<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; color:#94a3b8;">
    <div>
        <div style="color:#f59e0b; font-weight:700; margin-bottom:8px;">Data</div>
        <ul style="font-size:0.85rem; line-height:1.8; padding-left:16px;">
            <li>1440 RAVDESS files</li>
            <li>80/10/10 stratified split</li>
            <li>Weighted random sampler</li>
            <li>Time & feature masking</li>
        </ul>
    </div>
    <div>
        <div style="color:#f59e0b; font-weight:700; margin-bottom:8px;">Optimization</div>
        <ul style="font-size:0.85rem; line-height:1.8; padding-left:16px;">
            <li>AdamW optimizer</li>
            <li>Cosine Annealing LR</li>
            <li>Gradient clipping</li>
            <li>Label smoothing 0.1</li>
        </ul>
    </div>
    <div>
        <div style="color:#f59e0b; font-weight:700; margin-bottom:8px;">Regularization</div>
        <ul style="font-size:0.85rem; line-height:1.8; padding-left:16px;">
            <li>Dropout (0.3)</li>
            <li>Weight decay 1e-4</li>
            <li>Early stopping (p=20)</li>
            <li>Feature normalization</li>
        </ul>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# FOOTER
# ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:40px 0 20px; color:#334155; font-size:0.8rem;">
    EmotiSense · Speech Emotion Recognition · Built with PyTorch & Streamlit<br>
    RAVDESS Dataset · Transformer Architecture
</div>
""", unsafe_allow_html=True)
