"""
audio_standardize.py
--------------------
Light audio standardization utilities for SpectroCough v1.

IMPORTANT DESIGN PRINCIPLES:
- Dataset audio is already clinically preprocessed
- NO denoising, NO VAD, NO aggressive filtering
- ONLY light standardization for consistency
- Must be used identically in training and inference

This module performs:
1. Audio loading
2. Resampling to 16 kHz
3. Mono conversion
4. Fixed-length enforcement (6 seconds)
5. RMS-based loudness normalization
"""

import numpy as np
import librosa

from ml_pipeline.panel1_stethoscope.config import (
    TARGET_SAMPLE_RATE,
    TARGET_NUM_SAMPLES,
    FORCE_MONO,
    USE_RMS_NORMALIZATION,
    TARGET_RMS,
)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(file_path: str):
    """
    Load an audio file without altering its original signal content.

    Parameters
    ----------
    file_path : str
        Path to the .wav file

    Returns
    -------
    y : np.ndarray
        Audio waveform (float32)
    sr : int
        Original sample rate
    """
    y, sr = librosa.load(
        file_path,
        sr=None,          # Preserve original sample rate
        mono=False        # Preserve channels initially
    )
    return y.astype(np.float32), sr


# ============================================================
# MONO CONVERSION
# ============================================================

def convert_to_mono(y: np.ndarray) -> np.ndarray:
    """
    Convert stereo/multi-channel audio to mono by channel averaging.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (shape: [n_channels, n_samples] or [n_samples])

    Returns
    -------
    np.ndarray
        Mono audio signal
    """
    if y.ndim == 1:
        return y

    # Average channels
    return np.mean(y, axis=0)


# ============================================================
# RESAMPLING
# ============================================================

def resample_audio(y: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    Resample audio to the target sample rate (16 kHz).

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal
    orig_sr : int
        Original sample rate

    Returns
    -------
    np.ndarray
        Resampled audio signal
    """
    if orig_sr == TARGET_SAMPLE_RATE:
        return y

    return librosa.resample(
        y=y,
        orig_sr=orig_sr,
        target_sr=TARGET_SAMPLE_RATE
    )


# ============================================================
# FIXED-LENGTH ENFORCEMENT
# ============================================================

def enforce_fixed_length(y: np.ndarray) -> np.ndarray:
    """
    Pad or crop audio to exactly TARGET_NUM_SAMPLES.

    Padding:
    - Uses zero-padding (safe and non-invasive)

    Cropping:
    - Center crop if audio is longer

    Parameters
    ----------
    y : np.ndarray
        Audio signal (mono)

    Returns
    -------
    np.ndarray
        Fixed-length audio signal
    """
    num_samples = len(y)

    if num_samples == TARGET_NUM_SAMPLES:
        return y

    # If too short → pad
    if num_samples < TARGET_NUM_SAMPLES:
        pad_width = TARGET_NUM_SAMPLES - num_samples
        return np.pad(y, (0, pad_width), mode="constant")

    # If too long → center crop
    start = (num_samples - TARGET_NUM_SAMPLES) // 2
    end = start + TARGET_NUM_SAMPLES
    return y[start:end]


# ============================================================
# RMS LOUDNESS NORMALIZATION
# ============================================================

def rms_normalize(y: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """
    Apply RMS-based loudness normalization.

    IMPORTANT:
    - Preserves relative disease energy differences
    - Uses a GLOBAL target RMS
    - Does NOT equalize per class

    Parameters
    ----------
    y : np.ndarray
        Audio signal
    target_rms : float
        Desired RMS level

    Returns
    -------
    np.ndarray
        RMS-normalized audio
    """
    rms = np.sqrt(np.mean(y ** 2))

    if rms < 1e-8:
        # Avoid division by zero for silent segments
        return y

    gain = target_rms / rms
    return y * gain


# ============================================================
# FULL STANDARDIZATION PIPELINE
# ============================================================

def standardize_audio(file_path: str) -> np.ndarray:
    """
    Complete audio standardization pipeline.

    Steps (STRICT ORDER):
    1. Load audio
    2. Convert to mono
    3. Resample to 16 kHz
    4. Enforce fixed length (6 seconds)
    5. Apply RMS normalization

    Parameters
    ----------
    file_path : str
        Path to audio file

    Returns
    -------
    np.ndarray
        Standardized audio waveform
    """
    # Load
    y, sr = load_audio(file_path)

    # Mono conversion
    if FORCE_MONO:
        y = convert_to_mono(y)

    # Resample
    y = resample_audio(y, sr)

    # Fixed length
    y = enforce_fixed_length(y)

    # Loudness normalization
    if USE_RMS_NORMALIZATION:
        y = rms_normalize(y)

    return y.astype(np.float32)
