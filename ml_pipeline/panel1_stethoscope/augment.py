"""
augment.py
----------
Online raw-audio augmentation utilities for SpectroCough v1.

IMPORTANT RULES (STRICT):
- Augment RAW AUDIO ONLY
- Apply ONLY during training
- NEVER save augmented audio to disk
- NEVER augment Mel or acoustic features directly
- Validation/Test/Inference must remain untouched

All augmentations are physiologically safe for cough sounds.
"""

import numpy as np
import librosa
import random

from ml_pipeline.panel1_stethoscope.config import (
    USE_AUGMENTATION,
    MAX_AUGMENTATIONS_PER_SAMPLE,
    TIME_STRETCH_RANGE,
    PITCH_SHIFT_RANGE,
    NOISE_SNR_RANGE,
    GAIN_DB_RANGE,
    MAX_TIME_SHIFT_SAMPLES,
    TARGET_SAMPLE_RATE,
)

# ============================================================
# BASIC AUGMENTATION OPERATIONS
# ============================================================

def time_stretch(y: np.ndarray) -> np.ndarray:
    """Apply random time stretching."""
    rate = random.uniform(*TIME_STRETCH_RANGE)
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray) -> np.ndarray:
    """Apply small pitch shift (±1 semitone)."""
    n_steps = random.uniform(*PITCH_SHIFT_RANGE)
    return librosa.effects.pitch_shift(
        y=y,
        sr=TARGET_SAMPLE_RATE,
        n_steps=n_steps
    )


def add_noise(y: np.ndarray) -> np.ndarray:
    """Add low-level white noise based on SNR."""
    snr_db = random.uniform(*NOISE_SNR_RANGE)

    signal_power = np.mean(y ** 2)
    if signal_power < 1e-8:
        return y

    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=y.shape)

    return y + noise


def gain_adjust(y: np.ndarray) -> np.ndarray:
    """Apply random gain adjustment in dB."""
    gain_db = random.uniform(*GAIN_DB_RANGE)
    gain = 10 ** (gain_db / 20)
    return y * gain


def time_shift(y: np.ndarray) -> np.ndarray:
    """Apply small circular time shift."""
    shift = random.randint(-MAX_TIME_SHIFT_SAMPLES, MAX_TIME_SHIFT_SAMPLES)
    return np.roll(y, shift)


# ============================================================
# COMPOSITE AUGMENTATION
# ============================================================

def random_augmentation(y: np.ndarray) -> np.ndarray:
    """
    Apply a random combination of augmentations.

    Strategy:
    - Randomly choose 1 to 3 augmentations
    - Order is randomized
    - Keeps signal physiologically realistic
    """
    augmentations = [
        time_stretch,
        pitch_shift,
        add_noise,
        gain_adjust,
        time_shift,
    ]

    num_ops = random.randint(1, min(3, len(augmentations)))
    ops = random.sample(augmentations, num_ops)

    y_aug = y.copy()
    for op in ops:
        y_aug = op(y_aug)

    return y_aug


# ============================================================
# TRAIN-TIME AUGMENTATION INTERFACE
# ============================================================

def maybe_augment(y: np.ndarray) -> np.ndarray:
    """
    Conditionally apply augmentation.

    This function should be called ONLY during training.

    Logic:
    - If augmentation disabled → return original
    - Else randomly decide to augment or not
    """
    if not USE_AUGMENTATION:
        return y

    # 50% chance to keep original (stability)
    if random.random() < 0.5:
        return y

    return random_augmentation(y)


def generate_augmented_batch(y: np.ndarray) -> list:
    """
    Generate multiple augmented versions of a single audio sample.

    Used when explicitly balancing minority classes.

    Returns a list of augmented waveforms.
    """
    augmented_samples = []

    for _ in range(MAX_AUGMENTATIONS_PER_SAMPLE):
        augmented_samples.append(random_augmentation(y))

    return augmented_samples
