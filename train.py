# ─────────────────────────────────────────────
# CELL 1: Install Dependencies
# ─────────────────────────────────────────────
print('✅ Dependencies installed')

# ─────────────────────────────────────────────
# CELL 2: Kaggle Auth + Dataset Download
# ─────────────────────────────────────────────
# from google.colab import files
# import os

# print('📂 Upload your kaggle.json')
# uploaded = files.upload()

# os.makedirs('/root/.kaggle', exist_ok=True)
# print('⬇️ Downloading RAVDESS full dataset...')
# print('✅ Dataset ready')

# ─────────────────────────────────────────────
# CELL 3: Imports
# ─────────────────────────────────────────────
import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import warnings
import pickle
from scipy.signal import butter, sosfilt

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🖥️  Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')

# ─────────────────────────────────────────────
# CELL 4: Configuration
# ─────────────────────────────────────────────
CFG = {
    # Audio
    'sample_rate': 16000,
    'n_fft': 320,             # 20ms frame at 16kHz
    'hop_length': 160,        # 50% overlap
    'n_mfcc': 40,
    'n_mels': 128,
    'n_chroma': 12,
    
    # Sequence Grouping
    # We will group frames into blocks of 50 to maintain Transformer structure
    'seq_len': 50,
    
    # Training
    'batch_size': 64,
    'epochs': 150,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 20,
    'min_delta': 1e-4,
    'label_smoothing': 0.1,
    
    # Model
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'dim_feedforward': 512,
    'dropout': 0.3,
    
    # Paths
    'data_dir': './audio_speech_actors_01-24',
    'save_dir': './content/ser_model',
}

import os
os.makedirs(CFG['save_dir'], exist_ok=True)

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

EMOTION_COLORS = {
    'neutral': '#95a5a6', 'calm': '#3498db', 'happy': '#f1c40f',
    'sad': '#2980b9', 'angry': '#e74c3c', 'fearful': '#9b59b6',
    'disgust': '#27ae60', 'surprised': '#e67e22'
}

print('✅ Config ready')
print(f'   Frames per group: {CFG["seq_len"]}')
print(f'   Emotions: {list(EMOTION_MAP.values())}')


# ─────────────────────────────────────────────
# CELL 5: Feature Extraction
# ─────────────────────────────────────────────
def butterworth_lowpass(y, sr, cutoff=4000, order=5):
    """Apply a Butterworth low-pass filter to remove high-frequency noise.

    Args:
        y      : audio signal (1-D numpy array)
        sr     : sample rate (Hz)
        cutoff : low-pass cutoff frequency in Hz (default 4000 Hz)
                 Keeps speech fundamentals + first 3-4 formants.
        order  : Butterworth filter order (default 5)

    Returns:
        Filtered audio array (same shape as y)
    """
    nyquist = sr / 2.0
    normal_cutoff = cutoff / nyquist        # normalised to [0, 1]
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sosfilt(sos, y).astype(np.float32)


def extract_sequences(file_path, cfg):
    """Extract multi-modal audio features from a WAV file and split into sequences.
    Applies a Butterworth low-pass filter (cutoff=4000 Hz, order=5) before
    feature extraction to suppress high-frequency noise.

    Each file is split into frames of 20ms with 50% overlap, then grouped into
    sequences of `seq_len` (e.g., 50 frames).

    Returns a list of (seq_len, feature_dim) arrays.
    """
    sr = cfg['sample_rate']
    hop = cfg['hop_length']
    n_fft = cfg.get('n_fft', 320)
    seq_len = cfg['seq_len']

    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        y, _ = librosa.effects.trim(y)      # Remove pure silence
        y = butterworth_lowpass(y, sr)       # 🔽 Low-pass filter (4 kHz cutoff)
    except Exception:
        return None

    # Extract frame-level features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg['n_mfcc'], n_fft=n_fft, hop_length=hop)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_chroma=cfg['n_chroma'])
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=cfg['n_mels'])
    mel_db = librosa.power_to_db(mel, ref=np.max)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)

    # features shape: (T, 262) where T is total frames in the audio
    features = np.vstack([mfcc, delta1, delta2, chroma, mel_db, zcr, rms]).T
    
    # Chunk into sequences of seq_len
    # We use 50% overlap between chunks sequentially to generate more samples!
    chunks = []
    step = max(1, seq_len // 2)
    for start in range(0, features.shape[0] - seq_len + 1, step):
        chunks.append(features[start:start+seq_len, :])
    
    # If file was shorter than 1 sequence, pad and add
    if len(chunks) == 0:
        if features.shape[0] < seq_len:
            pad_w = seq_len - features.shape[0]
            padded = np.pad(features, ((0, pad_w), (0, 0)), mode='edge')
            chunks.append(padded)
            
    return chunks


def get_label(file_path):
    """Parse emotion label from RAVDESS filename."""
    name = os.path.basename(file_path)
    parts = name.replace('.wav', '').split('-')
    emotion_code = parts[2]  # 3rd segment is emotion
    return EMOTION_MAP.get(emotion_code, None)


print('✅ Feature extraction functions defined')
print(f'   Feature dim per frame: 262')


# ─────────────────────────────────────────────
# CELL 6: Build Dataset
# ─────────────────────────────────────────────
print('🔍 Scanning RAVDESS directory...')

import os
import glob
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import Counter

audio_files = glob.glob(os.path.join(CFG['data_dir'], '**', '*.wav'), recursive=True)
# Filter to speech-only (modality 03)
audio_files = [f for f in audio_files if os.path.basename(f).startswith('03-01-')]

print(f'   Found {len(audio_files)} speech audio files')

X_list, y_list = [], []
failed = 0

for fp in tqdm(audio_files, desc='Extracting sequences'):
    label = get_label(fp)
    if label is None:
        failed += 1
        continue
    
    # Extract multiple frame chunks for this single audio
    chunks = extract_sequences(fp, CFG)
    if chunks is None:
        failed += 1
        continue
    
    # Label each sequence chunk according to the original audio!
    for chunk in chunks:
        X_list.append(chunk)
        y_list.append(label)

X = np.array(X_list, dtype=np.float32)  # (N_sequences, seq_len, 262)
y_raw = np.array(y_list)

le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f'\n✅ Dataset built: {X.shape[0]} sequence samples')
print(f'   X shape: {X.shape}  |  Classes: {le.classes_}')
print(f'   Failed files: {failed}')

dist = Counter(y_raw)
fig, ax = plt.subplots(figsize=(10, 4))
emotions_ = [e for e in le.classes_]
counts_ = [dist[e] for e in emotions_]
colors_ = [EMOTION_COLORS[e] for e in emotions_]
bars = ax.bar(emotions_, counts_, color=colors_, edgecolor='black', linewidth=0.8)
ax.bar_label(bars, fontsize=10, fontweight='bold')
ax.set_title('Expanded Sequence Dataset Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Sequence Count')
ax.set_xlabel('Emotion')
import matplotlib.pyplot as plt
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(CFG['save_dir'], 'class_distribution.png'), dpi=120)
plt.show()


# ─────────────────────────────────────────────
# CELL 7: Normalize + Train/Val/Test Split
# ─────────────────────────────────────────────
N, T, F = X.shape

# Normalize per-feature across all samples
X_flat = X.reshape(-1, F)  # (N*T, F)
scaler = StandardScaler()
X_flat_scaled = scaler.fit_transform(X_flat).astype(np.float32)
X = X_flat_scaled.reshape(N, T, F)

# Save scaler for inference
with open(os.path.join(CFG['save_dir'], 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(CFG['save_dir'], 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

# 80/10/10 stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

print(f'Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}')

# Weighted sampler to handle slight imbalance
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ─────────────────────────────────────────────
# CELL 8: Dataset & DataLoader
# ─────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            # Time masking
            if np.random.rand() < 0.5:
                t_start = np.random.randint(0, x.shape[0] // 2)
                t_mask = np.random.randint(1, x.shape[0] // 8 + 1)
                x = x.clone()
                x[t_start:t_start+t_mask, :] = 0
            # Feature masking
            if np.random.rand() < 0.5:
                f_start = np.random.randint(0, x.shape[1] // 2)
                f_mask = np.random.randint(1, x.shape[1] // 8 + 1)
                x = x.clone()
                x[:, f_start:f_start+f_mask] = 0
        return x, self.y[idx]


train_ds = EmotionDataset(X_train, y_train, augment=True)
val_ds   = EmotionDataset(X_val,   y_val,   augment=False)
test_ds  = EmotionDataset(X_test,  y_test,  augment=False)

train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                          sampler=sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'],
                          shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CFG['batch_size'],
                          shuffle=False, num_workers=0, pin_memory=True)

print(f'✅ DataLoaders ready')
print(f'   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

# ─────────────────────────────────────────────
# CELL 9: Transformer Model
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SpeechEmotionTransformer(nn.Module):
    """Transformer encoder for Speech Emotion Recognition."""
    def __init__(self, input_dim, num_classes, cfg):
        super().__init__()
        d_model = cfg['d_model']

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(cfg['dropout'] * 0.5)
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=cfg['seq_len'] + 10,
                                          dropout=cfg['dropout'] * 0.5)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg['nhead'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout'],
            batch_first=True,
            norm_first=True   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg['num_layers'],
            norm=nn.LayerNorm(d_model)
        )

        # Classification head with global average + max pooling
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg['dropout'] * 0.5),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.pos_enc(x)             # (B, T, d_model)
        x = self.transformer(x)         # (B, T, d_model)

        # Aggregate temporal dimension
        avg_pool = x.mean(dim=1)        # (B, d_model)
        max_pool = x.max(dim=1).values  # (B, d_model)
        pooled = torch.cat([avg_pool, max_pool], dim=-1)  # (B, d_model*2)

        return self.classifier(pooled)  # (B, num_classes)


num_classes = len(le.classes_)
input_dim = X.shape[-1]  # 262

model = SpeechEmotionTransformer(input_dim, num_classes, CFG).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'✅ Model ready')
print(f'   Input dim: {input_dim}  |  Classes: {num_classes}')
print(f'   Trainable parameters: {total_params:,}')
print(model)

# ─────────────────────────────────────────────
# CELL 10: Training Utilities
# ─────────────────────────────────────────────
class EarlyStopping:
    """Stops training when validation loss doesn't improve."""
    def __init__(self, patience=20, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.stop = True


def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE, phase='train'):
    is_train = (phase == 'train')
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return total_loss / total, correct / total

print('✅ Training utilities ready')

# ─────────────────────────────────────────────
# CELL 11: Training Loop
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])
optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'],
                        weight_decay=CFG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6)

early_stopping = EarlyStopping(patience=CFG['patience'],
                               min_delta=CFG['min_delta'], verbose=False)

history = {'train_loss': [], 'val_loss': [],
           'train_acc': [],  'val_acc': [], 'lr': []}

best_val_acc = 0.0
print('🚀 Training started...\n')

for epoch in range(1, CFG['epochs'] + 1):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, 'train')
    va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      DEVICE, 'val')
    scheduler.step()

    lr_now = optimizer.param_groups[0]['lr']
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(va_acc)
    history['lr'].append(lr_now)

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), os.path.join(CFG['save_dir'], 'best_model.pt'))

    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d}/{CFG["epochs"]} | '
              f'Loss: {tr_loss:.4f}/{va_loss:.4f} | '
              f'Acc: {tr_acc*100:.2f}%/{va_acc*100:.2f}% | '
              f'LR: {lr_now:.2e}')

    early_stopping(va_loss, model)
    if early_stopping.stop:
        print(f'\n🛑 Early stopping at epoch {epoch}')
        print(f'   Best validation loss: {early_stopping.best_loss:.4f}')
        model.load_state_dict(early_stopping.best_state)
        break

print(f'\n✅ Training done. Best Val Acc: {best_val_acc*100:.2f}%')

# ─────────────────────────────────────────────
# CELL 12: Training Curves
# ─────────────────────────────────────────────
epochs_ran = len(history['train_loss'])
ep_axis = range(1, epochs_ran + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Training History', fontsize=16, fontweight='bold')

# Loss
axes[0].plot(ep_axis, history['train_loss'], label='Train', color='#3498db', linewidth=2)
axes[0].plot(ep_axis, history['val_loss'],   label='Val',   color='#e74c3c', linewidth=2)
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy
axes[1].plot(ep_axis, [a*100 for a in history['train_acc']], label='Train', color='#3498db', linewidth=2)
axes[1].plot(ep_axis, [a*100 for a in history['val_acc']],   label='Val',   color='#e74c3c', linewidth=2)
axes[1].set_title('Accuracy (%)'); axes[1].set_xlabel('Epoch'); axes[1].legend()
axes[1].grid(alpha=0.3)

# Learning rate
axes[2].plot(ep_axis, history['lr'], color='#2ecc71', linewidth=2)
axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch')
axes[2].set_yscale('log'); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CFG['save_dir'], 'training_curves.png'), dpi=120)
plt.show()

# ─────────────────────────────────────────────
# CELL 13: Test Evaluation + Metrics
# ─────────────────────────────────────────────
# Load best model
model.load_state_dict(torch.load(os.path.join(CFG['save_dir'], 'best_model.pt'),
                                 map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = (all_preds == all_labels).mean() * 100
print(f'\n🏆 Test Accuracy: {test_acc:.2f}%')
print('\n📊 Classification Report:')
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, ax=ax)
ax.set_title(f'Confusion Matrix — Test Acc: {test_acc:.2f}%', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(CFG['save_dir'], 'confusion_matrix.png'), dpi=120)
plt.show()

# ─────────────────────────────────────────────
# CELL 14: Save All Artifacts
# ─────────────────────────────────────────────
import json

# Save config alongside model
with open(os.path.join(CFG['save_dir'], 'config.json'), 'w') as f:
    serializable_cfg = {k: v for k, v in CFG.items()}
    json.dump(serializable_cfg, f, indent=2)

# Zip everything for download

print('📦 Artifacts saved to /content/ser_artifacts.zip')
print('   Contents:')

# Optional: download
# from google.colab import files
# files.download('/content/ser_artifacts.zip')
# print('\n✅ Download started!')

# ─────────────────────────────────────────────
# CELL 15: Quick Inference Test (optional)
# ─────────────────────────────────────────────
# def predict_emotion(file_path, model, scaler, le, cfg, device):
#     \"\"\"Predict emotion from a single audio file.\"\"\"
#     feat = extract_features(file_path, cfg)  # (T, F)
#     if feat is None:
#         return None, None
# 
#     # Scale
#     feat = scaler.transform(feat).astype(np.float32)
#     feat_t = torch.tensor(feat).unsqueeze(0).to(device)  # (1, T, F)
# 
#     model.eval()
#     with torch.no_grad():
#         logits = model(feat_t)
#         probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
#         pred_idx = probs.argmax()
# 
#     emotion = le.classes_[pred_idx]
#     return emotion, probs
# 
# 
# # Test on a random file
# test_file = audio_files[0]
# emotion, probs = predict_emotion(test_file, model, scaler, le, CFG, DEVICE)
# true_label = get_label(test_file)
# 
# print(f'File: {os.path.basename(test_file)}')
# print(f'True emotion  : {true_label}')
# print(f'Predicted     : {emotion}')
# print(f'\nProbabilities:')
# for em, p in sorted(zip(le.classes_, probs), key=lambda x: -x[1]):
#     bar = '█' * int(p * 30)
#     print(f'  {em:10s}: {p*100:5.1f}%  {bar}')

