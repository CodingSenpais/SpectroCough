"""
model.py
--------
Next-Generation Hybrid Respiratory Audio Model
for SpectroCough (Microphone Version)

Architecture:
------------------------------------------------
1. Mel Spectrogram CNN Branch
2. Handcrafted Acoustic Feature Branch
3. YAMNet Embedding Branch
4. Hybrid Fusion
5. Softmax Classification

Design Philosophy:
------------------------------------------------
- Preserve explainability ecosystem
- Use pretrained embeddings ONLY as feature extractors
- Keep handcrafted respiratory acoustic reasoning
- Reduce overfitting from complex temporal stacks
- Improve microphone-domain robustness

Compatible:
------------------------------------------------
- Python 3.13.5
- TensorFlow 2.x
- tensorflow-hub
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from ml_pipeline.panel2_microphone.config import (
    NUM_CLASSES,

    # CNN
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    CNN_POOL_SIZE,
    CNN_DROPOUT,

    # Acoustic branch
    ACOUSTIC_DENSE_UNITS,
    ACOUSTIC_DROPOUT,

    # Embedding branch
    EMBEDDING_DENSE_UNITS,

    # Fusion
    FUSION_DENSE_UNITS,
    CLASSIFIER_DROPOUT,

    # Training
    LEARNING_RATE
)

# ============================================================
# 🎼 MEL CNN BRANCH
# ============================================================

def build_mel_cnn_branch(input_shape):

    mel_input = layers.Input(
        shape=input_shape,
        name="mel_input"
    )

    x = mel_input

    # --------------------------------------------------------
    # CNN Blocks
    # --------------------------------------------------------
    for filters in CNN_FILTERS:

        x = layers.Conv2D(
            filters,
            CNN_KERNEL_SIZE,
            padding="same"
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(
            filters,
            CNN_KERNEL_SIZE,
            padding="same"
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(
            CNN_POOL_SIZE
        )(x)

        x = layers.Dropout(
            CNN_DROPOUT
        )(x)

    # --------------------------------------------------------
    # Global Feature Pooling
    # --------------------------------------------------------
    x = layers.GlobalAveragePooling2D()(x)

    # --------------------------------------------------------
    return models.Model(
        mel_input,
        x,
        name="Mel_CNN_Encoder"
    )


# ============================================================
# 🎵 HANDCRAFTED ACOUSTIC FEATURE BRANCH
# ============================================================

def build_acoustic_dense_branch(input_dim):

    acoustic_input = layers.Input(
        shape=(input_dim,),
        name="acoustic_input"
    )

    x = acoustic_input

    for units in ACOUSTIC_DENSE_UNITS:

        x = layers.Dense(units)(x)

        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)

        x = layers.Dropout(
            ACOUSTIC_DROPOUT
        )(x)

    return models.Model(
        acoustic_input,
        x,
        name="Acoustic_Encoder"
    )


# ============================================================
# 🔥 YAMNET EMBEDDING BRANCH
# ============================================================

def build_embedding_branch(embedding_dim):

    embedding_input = layers.Input(
        shape=(embedding_dim,),
        name="embedding_input"
    )

    x = embedding_input

    for units in EMBEDDING_DENSE_UNITS:

        x = layers.Dense(units)(x)

        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)

        x = layers.Dropout(0.3)(x)

    return models.Model(
        embedding_input,
        x,
        name="YAMNet_Embedding_Encoder"
    )


# ============================================================
# 🔗 HYBRID FUSION
# ============================================================

def fusion_layer(
    mel_feat,
    acoustic_feat,
    embedding_feat
):
    """
    Hybrid attention fusion.

    Goal:
    - Let model dynamically weight:
        - CNN respiratory patterns
        - handcrafted acoustic stats
        - pretrained audio embeddings
    """

    combined = layers.Concatenate(
        name="feature_fusion"
    )([
        mel_feat,
        acoustic_feat,
        embedding_feat
    ])

    # --------------------------------------------------------
    # Lightweight attention weighting
    # --------------------------------------------------------
    attention = layers.Dense(
        combined.shape[-1],
        activation="sigmoid",
        name="fusion_attention"
    )(combined)

    weighted = layers.Multiply(
        name="attention_weighted_features"
    )([
        combined,
        attention
    ])

    return weighted


# ============================================================
# 🚀 FULL MODEL
# ============================================================

def build_spectrocough_model(
    mel_input_shape,
    acoustic_input_dim,
    embedding_dim=1024
):

    # ========================================================
    # BRANCHES
    # ========================================================

    mel_encoder = build_mel_cnn_branch(
        mel_input_shape
    )

    acoustic_encoder = build_acoustic_dense_branch(
        acoustic_input_dim
    )

    embedding_encoder = build_embedding_branch(
        embedding_dim
    )

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    mel_input = mel_encoder.input
    acoustic_input = acoustic_encoder.input
    embedding_input = embedding_encoder.input

    # --------------------------------------------------------
    # Encoded Features
    # --------------------------------------------------------

    mel_feat = mel_encoder(mel_input)

    acoustic_feat = acoustic_encoder(acoustic_input)

    embedding_feat = embedding_encoder(embedding_input)

    # ========================================================
    # HYBRID FUSION
    # ========================================================

    fused = fusion_layer(
        mel_feat,
        acoustic_feat,
        embedding_feat
    )

    x = fused

    # ========================================================
    # CLASSIFIER HEAD
    # ========================================================

    for units in FUSION_DENSE_UNITS:

        x = layers.Dense(units)(x)

        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)

        x = layers.Dropout(
            CLASSIFIER_DROPOUT
        )(x)

    # --------------------------------------------------------
    # Final Output
    # --------------------------------------------------------

    output = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="disease_classifier"
    )(x)

    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = models.Model(
        inputs=[
            mel_input,
            acoustic_input,
            embedding_input
        ],
        outputs=output,
        name="SpectroCough_Hybrid_YAMNet_Fusion"
    )

    # ========================================================
    # COMPILE
    # ========================================================

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# 🧪 QUICK SANITY TEST
# ============================================================

if __name__ == "__main__":

    mel_shape = (128, 500, 1)

    acoustic_dim = 93

    embedding_dim = 1024

    model = build_spectrocough_model(
        mel_shape,
        acoustic_dim,
        embedding_dim
    )

    model.summary()