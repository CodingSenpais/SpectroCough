"""
embedding_utils.py
------------------
Centralized YAMNet embedding utilities
for SpectroCough (Microphone Version)

Purpose:
------------------------------------------------
- Load YAMNet once globally
- Extract robust audio embeddings
- Perform temporal pooling
- Validate embedding consistency

Design Philosophy:
------------------------------------------------
- Lightweight
- Production-friendly
- Reusable across:
    - training
    - inference
    - benchmarking
    - future Panel 1 integration

Compatible:
------------------------------------------------
- Python 3.13.5
- TensorFlow 2.x
- tensorflow-hub
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from ml_pipeline.panel2_microphone.config import (
    YAMNET_EMBEDDING_DIM
)

# ============================================================
# 🚀 LOAD YAMNET MODEL
# ============================================================

print("\n🚀 Loading YAMNet embedding model...")

yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

print("✅ YAMNet loaded successfully!\n")


# ============================================================
# 🔍 VALIDATE EMBEDDING
# ============================================================

def validate_embedding(
    embedding: np.ndarray
):
    """
    Validate embedding dimensions and integrity.
    """

    # --------------------------------------------------------
    # Check dimensionality
    # --------------------------------------------------------
    if embedding.shape[0] != YAMNET_EMBEDDING_DIM:

        raise ValueError(
            f"Expected embedding dimension "
            f"{YAMNET_EMBEDDING_DIM}, "
            f"got {embedding.shape}"
        )

    # --------------------------------------------------------
    # Check NaNs
    # --------------------------------------------------------
    if np.isnan(embedding).any():

        raise ValueError(
            "Embedding contains NaN values."
        )

    # --------------------------------------------------------
    # Check Inf
    # --------------------------------------------------------
    if np.isinf(embedding).any():

        raise ValueError(
            "Embedding contains Inf values."
        )


# ============================================================
# 🔥 TEMPORAL POOLING
# ============================================================

def pool_embeddings(
    embeddings: tf.Tensor
) -> np.ndarray:
    """
    Convert frame-wise embeddings
    into a single global embedding.

    Input:
    ------------------------------------------------
    Shape:
        (num_frames, 1024)

    Output:
    ------------------------------------------------
    Shape:
        (1024,)
    """

    pooled = tf.reduce_mean(
        embeddings,
        axis=0
    )

    return pooled.numpy().astype(np.float32)


# ============================================================
# 🎧 EXTRACT YAMNET EMBEDDING
# ============================================================

def extract_yamnet_embedding(
    y: np.ndarray
) -> np.ndarray:
    """
    Extract robust audio embeddings using YAMNet.

    Input:
    ------------------------------------------------
    y:
        mono waveform
        float32
        16kHz

    Returns:
    ------------------------------------------------
    embedding:
        shape = (1024,)
    """

    # --------------------------------------------------------
    # Safety conversion
    # --------------------------------------------------------

    y = y.astype(np.float32)

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    waveform = tf.convert_to_tensor(
        y,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Run YAMNet
    # --------------------------------------------------------

    scores, embeddings, spectrogram = yamnet_model(
        waveform
    )

    # --------------------------------------------------------
    # Temporal pooling
    # --------------------------------------------------------

    embedding = pool_embeddings(
        embeddings
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_embedding(
        embedding
    )

    return embedding


# ============================================================
# 🧪 QUICK SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("✅ Embedding utility module loaded successfully.")

    dummy_audio = np.random.randn(
        16000 * 5
    ).astype(np.float32)

    embedding = extract_yamnet_embedding(
        dummy_audio
    )

    print(
        f"Embedding shape: {embedding.shape}"
    )

    print(
        f"Embedding dtype: {embedding.dtype}"
    )

    print("\n✅ Sanity check passed.")