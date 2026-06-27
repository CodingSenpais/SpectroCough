"""
dataset.py
----------
Single-model dataset pipeline for SpectroCough (Microphone Version)

Features:
- 3-class classification (covid19 / healthy_cough / sneezing)
- Balanced dataset (sneezing sampling)
- Online augmentation (light)
- Hybrid feature extraction (Mel + Acoustic)
"""

import random
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_pipeline.panel2_microphone.config import (
    DATASET_ROOT,
    CLASSES,
    CLASS_TO_INDEX,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    SHUFFLE_DATASET,
)

from ml_pipeline.panel2_microphone.audio_standardize import standardize_audio
from ml_pipeline.panel2_microphone.augment import maybe_augment
from ml_pipeline.panel2_microphone.features import extract_hybrid_features


# ============================================================
# 🔍 DATASET SCANNING (SIMPLIFIED)
# ============================================================

def scan_dataset() -> List[Tuple[str, int]]:
    """
    Scan dataset and build balanced sample list
    """

    samples = []

    for cls in CLASSES:
        class_dir = DATASET_ROOT / cls

        if not class_dir.exists():
            raise ValueError(f"Missing folder: {class_dir}")

        files = list(class_dir.glob("*.wav"))

        # 🔥 Balance sneezing class
        if cls == "sneezing":
            if len(files) > 450:
                files = random.sample(files, 450)

        label_idx = CLASS_TO_INDEX[cls]

        for file in files:
            samples.append((str(file), label_idx))

    if SHUFFLE_DATASET:
        random.seed(RANDOM_SEED)
        random.shuffle(samples)

    return samples


# ============================================================
# ✂️ DATA SPLITTING
# ============================================================

def split_dataset(samples: List[Tuple[str, int]]):

    labels = [label for _, label in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(1.0 - TRAIN_SPLIT),
        stratify=labels,
        random_state=RANDOM_SEED
    )

    temp_labels = [label for _, label in temp_samples]

    val_ratio_adjusted = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(1.0 - val_ratio_adjusted),
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )

    return train_samples, val_samples, test_samples


# ============================================================
# 📦 DATASET CLASS
# ============================================================

class SpectroCoughDataset:

    def __init__(self, samples, training: bool, scaler: StandardScaler = None):
        self.samples = samples
        self.training = training
        self.scaler = scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        file_path, label = self.samples[idx]

        # 1. Standardize audio (5 sec, mic optimized)
        y = standardize_audio(file_path)

        # 2. Apply augmentation ONLY during training
        if self.training:
            y = maybe_augment(y, label)

        # 3. Extract features
        mel, acoustic, embedding = extract_hybrid_features(y)

        # 4. Scale acoustic features
        if self.scaler is not None:
            acoustic = self.scaler.transform(acoustic.reshape(1, -1)).squeeze()

        # 5. One-hot encoding
        label_onehot = np.zeros(len(CLASSES), dtype=np.float32)
        label_onehot[label] = 1.0

        return mel, acoustic, embedding, label_onehot


# ============================================================
# 📊 SCALER FITTING
# ============================================================

def fit_acoustic_scaler(train_samples):

    features = []

    for file_path, _ in train_samples:

        y = standardize_audio(file_path)

        # ----------------------------------------------------
        # Extract hybrid features
        # ----------------------------------------------------
        _, acoustic, _ = extract_hybrid_features(y)

        features.append(acoustic)

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    features = np.array(features)

    # --------------------------------------------------------
    # Fit scaler
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(features)

    return scaler


# ============================================================
# 🏗️ BUILD DATASETS
# ============================================================

def build_datasets():

    samples = scan_dataset()

    if len(samples) == 0:
        raise ValueError("No data found. Check dataset structure.")

    train_samples, val_samples, test_samples = split_dataset(samples)

    scaler = fit_acoustic_scaler(train_samples)

    train_ds = SpectroCoughDataset(
        train_samples, training=True, scaler=scaler
    )

    val_ds = SpectroCoughDataset(
        val_samples, training=False, scaler=scaler
    )

    test_ds = SpectroCoughDataset(
        test_samples, training=False, scaler=scaler
    )

    return train_ds, val_ds, test_ds