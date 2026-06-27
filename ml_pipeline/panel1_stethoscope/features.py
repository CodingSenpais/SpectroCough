"""
features.py
-----------
Feature extraction utilities for SpectroCough v1.

This module extracts:
1. Log-Mel Spectrograms (CNN branch)
2. Acoustic Statistical Features (Dense branch)

IMPORTANT RULES:
- Input audio MUST already be standardized
- NO augmentation here
- NO disk writes
- Same logic used for training, validation, and inference
"""

import numpy as np
import librosa

from ml_pipeline.panel1_stethoscope.config import (
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
# MEL-SPECTROGRAM FEATURES (CNN BRANCH)
# ============================================================

def extract_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Extract Log-Mel Spectrogram from standardized audio.

    Parameters
    ----------
    y : np.ndarray
        Standardized mono audio waveform (16 kHz, 6 seconds)

    Returns
    -------
    np.ndarray
        Log-Mel spectrogram (shape: [n_mels, time_frames])
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
# ACOUSTIC STATISTICAL FEATURES (DENSE BRANCH)
# ============================================================

def extract_acoustic_features(y: np.ndarray) -> np.ndarray:
    """
    Extract acoustic statistical features from standardized audio.

    Features include:
    - MFCC statistics
    - Delta & Delta-Delta MFCC statistics
    - RMS Energy statistics
    - Zero Crossing Rate
    - Spectral statistics

    Parameters
    ----------
    y : np.ndarray
        Standardized mono audio waveform

    Returns
    -------
    np.ndarray
        1D acoustic feature vector
    """

    features = []

    # -------------------------------
    # MFCC FEATURES
    # -------------------------------
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=TARGET_SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH
    )

    # MFCC mean & std
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Delta MFCC
    if USE_MFCC_DELTA:
        delta_mfcc = librosa.feature.delta(mfcc)
        features.extend(np.mean(delta_mfcc, axis=1))

    # Delta-Delta MFCC
    if USE_MFCC_DELTA_DELTA:
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(delta2_mfcc, axis=1))

    # -------------------------------
    # ENERGY & TEMPORAL FEATURES
    # -------------------------------
    rms = librosa.feature.rms(y=y, frame_length=FFT_SIZE, hop_length=HOP_LENGTH)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=FFT_SIZE, hop_length=HOP_LENGTH
    )
    features.append(np.mean(zcr))

    # -------------------------------
    # SPECTRAL FEATURES
    # -------------------------------
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

    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=TARGET_SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH
    )
    features.extend(np.mean(spectral_contrast, axis=1))

    return np.array(features, dtype=np.float32)


# ============================================================
# HYBRID FEATURE INTERFACE
# ============================================================

def extract_hybrid_features(y: np.ndarray):
    """
    Extract both Mel-spectrogram and acoustic features
    from the same standardized audio sample.

    Parameters
    ----------
    y : np.ndarray
        Standardized audio waveform

    Returns
    -------
    mel : np.ndarray
        Log-Mel spectrogram (CNN input)
    acoustic : np.ndarray
        Acoustic feature vector (Dense input)
    """
    mel = extract_mel_spectrogram(y)
    acoustic = extract_acoustic_features(y)

    return mel, acoustic
