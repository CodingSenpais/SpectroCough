"""
features.py
-----------
Hybrid feature extraction for SpectroCough (Microphone Version)

Includes:
1. Log-Mel Spectrogram (CNN branch)
2. Enhanced Acoustic Features (Dense branch)

Upgraded for:
- Noisy microphone signals
- Better class separation
"""

import numpy as np
import librosa

import tensorflow_hub as hub
import tensorflow as tf

from ml_pipeline.panel2_microphone.config import (
    TARGET_SAMPLE_RATE,
    N_MELS,
    FFT_WINDOW_SIZE,
    HOP_LENGTH,
    FFT_SIZE,
    MEL_FMIN,
    MEL_FMAX,
    USE_LOG_MEL,
    N_MFCC,
    USE_MFCC_DELTA,
    USE_MFCC_DELTA_DELTA,
)


# ============================================================
#  LOAD YAMNET EMBEDDING MODEL
# ============================================================

print(" Loading YAMNet embedding model...")

yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

print(" YAMNet loaded successfully!")

# ============================================================
# 🎼 MEL-SPECTROGRAM FEATURES (CNN BRANCH)
# ============================================================

def extract_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Extract Log-Mel Spectrogram
    """

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH,
        win_length=FFT_WINDOW_SIZE,
        n_mels=N_MELS,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
        power=2.0
    )

    if USE_LOG_MEL:
        mel = librosa.power_to_db(mel, ref=np.max)

    return mel.astype(np.float32)


# ============================================================
# 🎧 ENHANCED ACOUSTIC FEATURES (Dense Branch)
# ============================================================

def extract_acoustic_features(y: np.ndarray) -> np.ndarray:
    """
    Enhanced acoustic feature extraction for mic audio
    """

    features = []

    # ========================================================
    # MFCC FEATURES
    # ========================================================
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=TARGET_SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    if USE_MFCC_DELTA:
        delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(delta, axis=1))

    if USE_MFCC_DELTA_DELTA:
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(delta2, axis=1))

    # ========================================================
    # ENERGY FEATURES
    # ========================================================
    rms = librosa.feature.rms(y=y, frame_length=FFT_SIZE, hop_length=HOP_LENGTH)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    features.append(np.max(rms))

    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=FFT_SIZE, hop_length=HOP_LENGTH
    )
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # ========================================================
    # SPECTRAL FEATURES
    # ========================================================
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=TARGET_SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH
    )
    features.append(np.mean(spectral_centroid))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=TARGET_SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH
    )
    features.append(np.mean(spectral_bandwidth))

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=TARGET_SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH
    )
    features.append(np.mean(spectral_rolloff))
    features.append(np.var(spectral_rolloff))  # NEW

    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=TARGET_SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH
    )
    features.extend(np.mean(spectral_contrast, axis=1))

    # ========================================================
    # 🔥 MICROPHONE-SPECIFIC FEATURES (NEW)
    # ========================================================

    # Harmonic vs noise separation
    harmonic, percussive = librosa.effects.hpss(y)
    features.append(np.mean(harmonic))
    features.append(np.mean(percussive))

    # Spectral flatness (noise indicator)
    flatness = librosa.feature.spectral_flatness(y=y)
    features.append(np.mean(flatness))
    features.append(np.std(flatness))

    # Spectral entropy (complexity)
    S = np.abs(librosa.stft(y))
    S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-8)
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-8), axis=0)
    features.append(np.mean(entropy))

    # ========================================================
    return np.array(features, dtype=np.float32)



# ============================================================
# 🔥 YAMNET EMBEDDING EXTRACTION
# ============================================================

def extract_yamnet_embedding(
    y: np.ndarray
) -> np.ndarray:
    """
    Extract robust audio embeddings using YAMNet.

    Purpose:
    - Improve microphone-domain robustness
    - Learn richer temporal/audio representations
    - Preserve environmental invariance

    Returns:
    - 1024-dim embedding vector
    """

    # --------------------------------------------------------
    # YAMNet expects:
    # - mono float32
    # - 16kHz audio
    # - range [-1, 1]
    # --------------------------------------------------------

    waveform = tf.convert_to_tensor(
        y,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Run YAMNet
    # --------------------------------------------------------

    scores, embeddings, spectrogram = yamnet_model(
        waveform
    )

    # --------------------------------------------------------
    # Temporal pooling
    # Convert frame embeddings → single embedding
    # --------------------------------------------------------

    embedding = tf.reduce_mean(
        embeddings,
        axis=0
    )

    return embedding.numpy().astype(np.float32)

# ============================================================
# 🔗 HYBRID INTERFACE
# ============================================================


def extract_hybrid_features(
    y: np.ndarray
):
    """
    Hybrid feature extraction pipeline.

    Returns:
    ------------------------------------------------
    mel:
        CNN spectrogram branch

    acoustic:
        handcrafted respiratory acoustic features

    embedding:
        YAMNet pretrained audio embeddings
    """

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------
    mel = extract_mel_spectrogram(y)

    # --------------------------------------------------------
    # Handcrafted acoustic features
    # --------------------------------------------------------
    acoustic = extract_acoustic_features(y)

    # --------------------------------------------------------
    # Pretrained audio embeddings
    # --------------------------------------------------------
    embedding = extract_yamnet_embedding(y)

    if embedding.shape[0] != 1024:
        raise ValueError(
            f"Expected 1024-dim embedding, got {embedding.shape}"
        )

    return (
        mel,
        acoustic,
        embedding
    )