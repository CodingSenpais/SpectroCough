"""
config.py
---------
Single-stage configuration for SpectroCough (Microphone Version)

Setup:
- Single model (3 classes)
- Balanced dataset
- 5-second fixed audio
- Light online augmentation
"""

from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

from runtime.base_paths import (
    PANEL2_DATASET_DIR,
    PANEL2_MODEL_PATH,
    PANEL2_SCALER_PATH
)

DATASET_ROOT = PANEL2_DATASET_DIR

# ============================================================
# MODEL SAVE PATHS
# ============================================================

MODEL_SAVE_PATH = PANEL2_MODEL_PATH

SCALER_SAVE_PATH = PANEL2_SCALER_PATH

# ============================================================
# 🎯 CLASS DEFINITIONS (FINAL)
# ============================================================

CLASSES = ["covid19", "healthy_cough", "sneezing"]

NUM_CLASSES = len(CLASSES)

CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(CLASSES)}
INDEX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_INDEX.items()}

# ============================================================
# 🎧 AUDIO STANDARDIZATION
# ============================================================

TARGET_SAMPLE_RATE = 16000

# 🔥 UPDATED: 5 SECONDS
TARGET_DURATION_SEC = 5.0
TARGET_NUM_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION_SEC)

FORCE_MONO = True

USE_RMS_NORMALIZATION = True
TARGET_RMS = 0.1

# ============================================================
# 🔥 AUGMENTATION SETTINGS (ONLINE ONLY)
# ============================================================

USE_AUGMENTATION = True

def get_augmentation_count(label_name: str):
    """
    Light augmentation strategy (balanced dataset)
    """

    if label_name == "covid19":
        return 1

    elif label_name == "healthy_cough":
        return 1

    elif label_name == "sneezing":
        return 0  # already large & diverse

    return 1


# -------- AUGMENTATION RANGES (LIGHT) -------- #

TIME_STRETCH_RANGE = (0.95, 1.05)   # reduced
PITCH_SHIFT_RANGE = (-1, 1)         # reduced
NOISE_SNR_RANGE = (20, 30)          # mild noise
GAIN_DB_RANGE = (-3, 3)             # smaller gain

MAX_TIME_SHIFT_SAMPLES = int(0.05 * TARGET_SAMPLE_RATE)  # ~50ms

# ============================================================
# 🎼 MEL-SPECTROGRAM PARAMETERS
# ============================================================

N_MELS = 128
FFT_WINDOW_SIZE = int(0.025 * TARGET_SAMPLE_RATE)
HOP_LENGTH = int(0.010 * TARGET_SAMPLE_RATE)
FFT_SIZE = 512

MEL_FMIN = 20
MEL_FMAX = TARGET_SAMPLE_RATE // 2

USE_LOG_MEL = True

# ============================================================
# 🎵 ACOUSTIC FEATURES
# ============================================================

N_MFCC = 20
USE_MFCC_DELTA = True
USE_MFCC_DELTA_DELTA = True

# ============================================================
# 📊 DATA SPLIT
# ============================================================

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42
SHUFFLE_DATASET = True

# ============================================================
# 🧠 MODEL HYPERPARAMETERS
# ============================================================

CNN_FILTERS = [32, 64, 128, 256]
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
CNN_DROPOUT = 0.3

ACOUSTIC_DENSE_UNITS = [128, 64]
ACOUSTIC_DROPOUT = 0.3

FUSION_DENSE_UNITS = [128]
CLASSIFIER_DROPOUT = 0.4


# ============================================================
# 🚀 TRAINING
# ============================================================

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 3e-4

USE_CLASS_WEIGHTS = True

LOSS_FUNCTION = "categorical_crossentropy"



# ============================================================
# 📈 EVALUATION
# ============================================================

METRICS = ["accuracy"]

COMPUTE_MACRO_F1 = True
COMPUTE_PER_CLASS_RECALL = True

# ============================================================
# ⚙️ RUNTIME
# ============================================================

NUM_WORKERS = 4
PIN_MEMORY = True

# ============================================================
# ✅ SANITY CHECK
# ============================================================

assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6


# ============================================================
# YAMNET EMBEDDING SETTINGS
# ============================================================

USE_YAMNET_EMBEDDINGS = True

YAMNET_EMBEDDING_DIM = 1024

YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"

FREEZE_YAMNET = True

# ============================================================
# FUSION SETTINGS
# ============================================================

EMBEDDING_DENSE_UNITS = [256, 128]

USE_HYBRID_FUSION = True

# ============================================================
# 🔥 TRAINING STABILITY
# ============================================================

USE_MIXED_PRECISION = False

GRADIENT_CLIP_NORM = 1.0