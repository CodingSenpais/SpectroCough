"""
config.py
----------
Central configuration file for SpectroCough v1
(Disease-Only Hybrid Mel + Acoustic Classifier)

This file defines:
- Dataset paths and class labels
- Audio standardization parameters
- Augmentation ranges
- Feature extraction settings
- Model hyperparameters
- Training configuration
- Evaluation settings

IMPORTANT:
- Dataset is clinically preprocessed
- Only light standardization is applied
- Augmentation is ONLINE and TRAIN-ONLY
"""

from pathlib import Path

from runtime.base_paths import (
    PANEL1_DATASET_DIR,
    PANEL1_MODEL_PATH,
    PANEL1_SCALER_PATH
)

# ============================================================
# PROJECT PATHS
# ============================================================

DATASET_ROOT = PANEL1_DATASET_DIR


# ============================================================
# MODEL SAVE PATHS
# ============================================================

MODEL_SAVE_PATH = PANEL1_MODEL_PATH

SCALER_SAVE_PATH = PANEL1_SCALER_PATH

# ============================================================
# DATASET & CLASS CONFIGURATION
# ============================================================

# Only 4 disease classes for Version-1
CLASSES = [
    "asthma",
    "bronchial",
    "copd",
    "pneumonia",
    "healthy",
]

NUM_CLASSES = len(CLASSES)

# Map class name -> index
CLASS_TO_INDEX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
INDEX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_INDEX.items()}

# ============================================================
# AUDIO STANDARDIZATION PARAMETERS
# ============================================================

# Mandatory unified sample rate
TARGET_SAMPLE_RATE = 16_000  # Hz

# Fixed audio duration (seconds)
TARGET_DURATION_SEC = 6.0
TARGET_NUM_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION_SEC)

# Mono conversion
FORCE_MONO = True

# Loudness normalization
USE_RMS_NORMALIZATION = True
TARGET_RMS = 0.1  # Global target, preserve relative disease energy

# ============================================================
# DATA AUGMENTATION (TRAIN-ONLY, ONLINE)
# ============================================================

USE_AUGMENTATION = True

# Each original sample can generate up to N augmented versions
MAX_AUGMENTATIONS_PER_SAMPLE = 3

# Time-stretch range (physiologically safe)
TIME_STRETCH_RANGE = (0.95, 1.08)

# Pitch shift range (semitones)
PITCH_SHIFT_RANGE = (-1, 1)

# Additive noise (Signal-to-Noise Ratio in dB)
NOISE_SNR_RANGE = (20, 30)

# Gain variation (dB)
GAIN_DB_RANGE = (-3.0, 3.0)

# Small temporal shift (samples)
MAX_TIME_SHIFT_SAMPLES = int(0.05 * TARGET_SAMPLE_RATE)  # 50 ms

# ============================================================
# MEL-SPECTROGRAM FEATURE PARAMETERS
# ============================================================

N_MELS = 128
FFT_WINDOW_SIZE = int(0.025 * TARGET_SAMPLE_RATE)  # 25 ms
HOP_LENGTH = int(0.010 * TARGET_SAMPLE_RATE)       # 10 ms
FFT_SIZE = 512
MEL_FMIN = 20
MEL_FMAX = TARGET_SAMPLE_RATE // 2

USE_LOG_MEL = True

# ============================================================
# ACOUSTIC FEATURE PARAMETERS
# ============================================================

# MFCC configuration
N_MFCC = 20
USE_MFCC_DELTA = True
USE_MFCC_DELTA_DELTA = True

# ============================================================
# DATA SPLITTING
# ============================================================

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42
SHUFFLE_DATASET = True

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================

# CNN (Mel branch)
CNN_FILTERS = [32, 64, 128]
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
CNN_DROPOUT = 0.3

# Dense (Acoustic branch)
ACOUSTIC_DENSE_UNITS = [128, 64]
ACOUSTIC_DROPOUT = 0.3

# Fusion & Classifier
FUSION_DENSE_UNITS = [128]
CLASSIFIER_DROPOUT = 0.4

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3

USE_CLASS_WEIGHTS = True

# Loss function
LOSS_FUNCTION = "categorical_crossentropy"  # or "focal"

# ============================================================
# EVALUATION & METRICS
# ============================================================

METRICS = [
    "accuracy",
]

COMPUTE_MACRO_F1 = True
COMPUTE_PER_CLASS_RECALL = True

# ============================================================
# RUNTIME SETTINGS
# ============================================================

NUM_WORKERS = 4
PIN_MEMORY = True

# ============================================================
# SANITY CHECKS
# ============================================================

assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, \
    "Train/Val/Test splits must sum to 1.0"

assert NUM_CLASSES == 5, \
    "SpectroCough v1 is configured for exactly 5 disease classes"
