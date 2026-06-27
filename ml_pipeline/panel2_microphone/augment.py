"""
augment.py
----------
Lightweight ONLINE augmentation module for SpectroCough (Microphone Version)

Design:
- Class-aware augmentation
- Single augmentation per sample (no stacking)
- Only meaningful augmentations (no distortion-heavy ops)
"""

import numpy as np
import librosa
import random

from ml_pipeline.panel2_microphone.config import (
    USE_AUGMENTATION,
    TARGET_SAMPLE_RATE,
    TIME_STRETCH_RANGE,
    PITCH_SHIFT_RANGE,
    NOISE_SNR_RANGE,
    GAIN_DB_RANGE,
    MAX_TIME_SHIFT_SAMPLES,
    get_augmentation_count,
    INDEX_TO_CLASS
)

# ============================================================
# 🔊 NOISE INJECTION (MOST IMPORTANT)
# ============================================================

def add_noise(y):
    """Add Gaussian noise based on SNR"""
    snr_db = random.uniform(*NOISE_SNR_RANGE)

    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise


# ============================================================
# 🎚️ GAIN ADJUSTMENT
# ============================================================

def apply_gain(y):
    gain_db = random.uniform(*GAIN_DB_RANGE)
    gain = 10 ** (gain_db / 20)
    return y * gain


# ============================================================
# 🎵 PITCH SHIFT (LIGHT)
# ============================================================

def pitch_shift(y):
    steps = random.uniform(*PITCH_SHIFT_RANGE)
    return librosa.effects.pitch_shift(y, sr=TARGET_SAMPLE_RATE, n_steps=steps)


# ============================================================
# ⏱️ TIME STRETCH (LIGHT)
# ============================================================

def time_stretch(y):
    rate = random.uniform(*TIME_STRETCH_RANGE)
    y_stretched = librosa.effects.time_stretch(y, rate)

    # Fix length
    if len(y_stretched) > len(y):
        return y_stretched[:len(y)]
    else:
        return np.pad(y_stretched, (0, len(y) - len(y_stretched)))


# ============================================================
# 🔄 TIME SHIFT (VERY LIGHT)
# ============================================================

def time_shift(y):
    shift = random.randint(-MAX_TIME_SHIFT_SAMPLES, MAX_TIME_SHIFT_SAMPLES)
    return np.roll(y, shift)


# ============================================================
# 🎯 RANDOM AUGMENTATION SELECTOR
# ============================================================

def apply_random_augmentation(y):
    """
    Select ONE augmentation randomly
    """

    augmentations = [
        add_noise,      # highest priority
        apply_gain,
        pitch_shift,
        time_stretch,
        time_shift
    ]

    aug_func = random.choice(augmentations)

    try:
        return aug_func(y)
    except Exception:
        return y


# ============================================================
# 🚀 MAIN FUNCTION
# ============================================================

def maybe_augment(y, label_idx):
    """
    Apply ONLINE augmentation (single-step, class-aware)
    """

    if not USE_AUGMENTATION:
        return y

    label_name = INDEX_TO_CLASS[label_idx]

    # Get augmentation count (0 or 1)
    num_aug = get_augmentation_count(label_name)

    # 🔥 IMPORTANT: NO STACKING
    if num_aug > 0:
        y = apply_random_augmentation(y)

    return y.astype(np.float32)