"""
model.py
--------
Hybrid neural network architecture for SpectroCough v1.

Architecture:
- CNN branch for Log-Mel Spectrograms
- Dense branch for Acoustic Statistical Features
- Late fusion of both embeddings
- Softmax disease probability output

STRICT RULES:
- No fake / non-cough logic
- No diagnostic claims
- Probability output only
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from ml_pipeline.panel1_stethoscope.config import (
    NUM_CLASSES,
    N_MELS,
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    CNN_POOL_SIZE,
    CNN_DROPOUT,
    ACOUSTIC_DENSE_UNITS,
    ACOUSTIC_DROPOUT,
    FUSION_DENSE_UNITS,
    CLASSIFIER_DROPOUT,
    LEARNING_RATE,
)


# ============================================================
# MEL-SPECTROGRAM CNN BRANCH
# ============================================================

def build_mel_cnn_branch(input_shape):
    """
    Build CNN branch for Mel-spectrogram input.

    Parameters
    ----------
    input_shape : tuple
        Shape of Mel-spectrogram (n_mels, time, 1)

    Returns
    -------
    model : tf.keras.Model
        CNN encoder model
    """
    mel_input = layers.Input(shape=input_shape, name="mel_input")
    x = mel_input

    for idx, filters in enumerate(CNN_FILTERS):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=CNN_KERNEL_SIZE,
            padding="same",
            name=f"mel_conv_{idx}"
        )(x)
        x = layers.BatchNormalization(name=f"mel_bn_{idx}")(x)
        x = layers.Activation("relu", name=f"mel_relu_{idx}")(x)
        x = layers.MaxPooling2D(
            pool_size=CNN_POOL_SIZE,
            name=f"mel_pool_{idx}"
        )(x)
        x = layers.Dropout(CNN_DROPOUT, name=f"mel_dropout_{idx}")(x)

    x = layers.GlobalAveragePooling2D(name="mel_gap")(x)

    return models.Model(mel_input, x, name="Mel_CNN_Encoder")


# ============================================================
# ACOUSTIC FEATURE DENSE BRANCH
# ============================================================

def build_acoustic_dense_branch(input_dim):
    """
    Build Dense branch for acoustic statistical features.

    Parameters
    ----------
    input_dim : int
        Number of acoustic features

    Returns
    -------
    model : tf.keras.Model
        Dense encoder model
    """
    acoustic_input = layers.Input(
        shape=(input_dim,),
        name="acoustic_input"
    )
    x = acoustic_input

    for idx, units in enumerate(ACOUSTIC_DENSE_UNITS):
        x = layers.Dense(
            units,
            activation="relu",
            name=f"acoustic_dense_{idx}"
        )(x)
        x = layers.BatchNormalization(name=f"acoustic_bn_{idx}")(x)
        x = layers.Dropout(ACOUSTIC_DROPOUT, name=f"acoustic_dropout_{idx}")(x)

    return models.Model(acoustic_input, x, name="Acoustic_Dense_Encoder")


# ============================================================
# FULL HYBRID MODEL
# ============================================================

def build_spectrocough_model(
    mel_input_shape,
    acoustic_input_dim
):
    """
    Build full SpectroCough v1 hybrid model.

    Parameters
    ----------
    mel_input_shape : tuple
        (n_mels, time_frames, 1)
    acoustic_input_dim : int
        Dimension of acoustic feature vector

    Returns
    -------
    model : tf.keras.Model
        Compiled hybrid model
    """

    # Build branches
    mel_encoder = build_mel_cnn_branch(mel_input_shape)
    acoustic_encoder = build_acoustic_dense_branch(acoustic_input_dim)

    # Inputs
    mel_input = mel_encoder.input
    acoustic_input = acoustic_encoder.input

    # Encoded features
    mel_embedding = mel_encoder(mel_input)
    acoustic_embedding = acoustic_encoder(acoustic_input)

    # Late fusion
    fused = layers.Concatenate(name="late_fusion")(
        [mel_embedding, acoustic_embedding]
    )

    # Fusion dense layers
    x = fused
    for idx, units in enumerate(FUSION_DENSE_UNITS):
        x = layers.Dense(
            units,
            activation="relu",
            name=f"fusion_dense_{idx}"
        )(x)
        x = layers.BatchNormalization(name=f"fusion_bn_{idx}")(x)
        x = layers.Dropout(
            CLASSIFIER_DROPOUT,
            name=f"fusion_dropout_{idx}"
        )(x)

    # Output layer
    output = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="disease_probability"
    )(x)

    # Model
    model = models.Model(
        inputs=[mel_input, acoustic_input],
        outputs=output,
        name="SpectroCough_v1"
    )

    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
