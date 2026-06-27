"""
dataset.py
----------
Dataset utilities for SpectroCough v1.

Responsibilities:
- Scan dataset directory with class-wise folders
- Infer labels from folder names
- Perform train/val/test split
- Apply audio standardization
- Apply augmentation ONLY during training
- Extract hybrid Mel + acoustic features on-the-fly
- Yield data in model-ready format

STRICT RULES:
- No feature saving
- No augmented audio saving
- No augmentation for validation/test
"""

import os
import random
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_pipeline.panel1_stethoscope.config import (
    DATASET_ROOT,
    CLASSES,
    CLASS_TO_INDEX,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    SHUFFLE_DATASET,
)

from ml_pipeline.panel1_stethoscope.audio_standardize import standardize_audio
from ml_pipeline.panel1_stethoscope.augment import maybe_augment
from ml_pipeline.panel1_stethoscope.features import extract_hybrid_features


# ============================================================
# DATASET INDEXING
# ============================================================

def scan_dataset() -> List[Tuple[str, int]]:
    """
    Scan dataset directory and collect file paths with labels.

    Returns
    -------
    samples : list of (file_path, label_index)
    """
    samples = []

    for class_name in CLASSES:
        class_dir = DATASET_ROOT / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for file in class_dir.glob("*.wav"):
            samples.append((str(file), CLASS_TO_INDEX[class_name]))

    if SHUFFLE_DATASET:
        random.seed(RANDOM_SEED)
        random.shuffle(samples)

    return samples


# ============================================================
# DATASET SPLITTING
# ============================================================

def split_dataset(samples: List[Tuple[str, int]]):
    """
    Split dataset into train, validation, and test sets.

    Returns
    -------
    train_samples, val_samples, test_samples
    """
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
# DATASET GENERATOR
# ============================================================

class SpectroCoughDataset:
    """
    Dataset generator for SpectroCough v1.

    Each __getitem__ returns:
    - Mel-spectrogram (CNN input)
    - Acoustic feature vector (Dense input)
    - One-hot encoded label
    """

    def __init__(self, samples, training: bool, scaler: StandardScaler = None):
        self.samples = samples
        self.training = training
        self.scaler = scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        # 1. Standardize audio
        y = standardize_audio(file_path)

        # 2. Apply augmentation ONLY if training
        if self.training:
            y = maybe_augment(y)

        # 3. Extract hybrid features
        mel, acoustic = extract_hybrid_features(y)

        # 4. Scale acoustic features if scaler is provided
        if self.scaler is not None:
            acoustic = self.scaler.transform(acoustic.reshape(1, -1)).squeeze()

        # 5. One-hot encode label
        label_onehot = np.zeros(len(CLASSES), dtype=np.float32)
        label_onehot[label] = 1.0

        return mel, acoustic, label_onehot


# ============================================================
# SCALER FITTING
# ============================================================

def fit_acoustic_scaler(train_samples) -> StandardScaler:
    """
    Fit StandardScaler on acoustic features from training set ONLY.

    Returns
    -------
    scaler : StandardScaler
    """
    acoustic_features = []

    for file_path, _ in train_samples:
        y = standardize_audio(file_path)
        _, acoustic = extract_hybrid_features(y)
        acoustic_features.append(acoustic)

    acoustic_features = np.array(acoustic_features)
    scaler = StandardScaler()
    scaler.fit(acoustic_features)

    return scaler


# ============================================================
# DATASET BUILD INTERFACE
# ============================================================

def build_datasets():
    """
    Build train, validation, and test dataset objects.

    Returns
    -------
    train_ds, val_ds, test_ds
    """
    samples = scan_dataset()
    train_samples, val_samples, test_samples = split_dataset(samples)

    # Fit scaler on training data ONLY
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
