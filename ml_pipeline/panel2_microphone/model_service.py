"""
model_service.py
----------------
Single-model inference pipeline for SpectroCough

Classes:
- covid19
- healthy_cough
- sneezing
"""

import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import time

from ml_pipeline.panel2_microphone.audio_standardize import (
    standardize_audio,
    extract_multi_windows
)
from ml_pipeline.panel2_microphone.features import extract_hybrid_features
from ml_pipeline.panel2_microphone.config import INDEX_TO_CLASS

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "spectrocough_yamnet_fusion.keras"
SCALER_PATH = BASE_DIR / "scalers" / "scaler_yamnet.pkl"

# ============================================================
# 🔄 LOAD MODEL & SCALER
# ============================================================

print("🚀 Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("✅ Model loaded successfully!")

# ============================================================
# 🔧 PREPROCESSING
# ============================================================

def prepare_input(audio_path: str):
    """
    Standardize + feature extraction + scaling
    """

    # 1. Standardize audio (5 sec)
    y = standardize_audio(audio_path)

    # 2. Extract hybrid features
    mel, acoustic, embedding = extract_hybrid_features(y)

    # --------------------------------------------------------
    # Scale handcrafted acoustic features
    # --------------------------------------------------------
    acoustic = scaler.transform(
        acoustic.reshape(1, -1)
    )

    # --------------------------------------------------------
    # Expand Mel dimensions
    # --------------------------------------------------------
    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)

    # --------------------------------------------------------
    # Expand embedding dimensions
    # --------------------------------------------------------
    embedding = np.expand_dims(
        embedding,
        axis=0
    )

    return mel, acoustic, embedding


# ============================================================
# 🔮 PREDICTION FUNCTION
# ============================================================

def predict_audio(audio_path: str):
    """
    Multi-window respiratory inference.

    Pipeline:
    ------------------------------------------------
    audio
      ↓
    multiple cough windows
      ↓
    predict each window
      ↓
    aggregate probabilities
      ↓
    final prediction
    """

    start_time = time.time()

    # ========================================================
    # LOAD + PREPROCESS FULL AUDIO
    # ========================================================

    y = standardize_audio(audio_path)

    # ========================================================
    # MULTI-WINDOW EXTRACTION
    # ========================================================

    windows = extract_multi_windows(
        y,
        num_windows=3
    )

    # ========================================================
    # STORE WINDOW PREDICTIONS
    # ========================================================

    all_probs = []

    # ========================================================
    # PREDICT EACH WINDOW
    # ========================================================

    for segment in windows:

        # ----------------------------------------------------
        # Extract hybrid features
        # ----------------------------------------------------

        mel, acoustic, embedding = extract_hybrid_features(
            segment
        )

        # ----------------------------------------------------
        # Scale acoustic features
        # ----------------------------------------------------

        acoustic = scaler.transform(
            acoustic.reshape(1, -1)
        )

        # ----------------------------------------------------
        # Expand dimensions
        # ----------------------------------------------------

        mel = np.expand_dims(
            mel,
            axis=-1
        )

        mel = np.expand_dims(
            mel,
            axis=0
        )

        embedding = np.expand_dims(
            embedding,
            axis=0
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        probs = model.predict(
            (
                mel,
                acoustic,
                embedding
            ),
            verbose=0
        )[0]

        all_probs.append(probs)

    # ========================================================
    # AGGREGATE PREDICTIONS
    # ========================================================

    all_probs = np.array(all_probs)

    # Mean probability aggregation
    final_probs = np.mean(
        all_probs,
        axis=0
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    pred_idx = int(
        np.argmax(final_probs)
    )

    pred_class = INDEX_TO_CLASS[
        pred_idx
    ]

    confidence = float(
        final_probs[pred_idx]
    )

    inference_time = (
        time.time() - start_time
    )


    # ========================================================
    # EXTRACT REPRESENTATIVE ACOUSTIC VECTOR
    # ========================================================

    # Use strongest window for explainability/runtime engines

    representative_segment = windows[0]

    _, acoustic_vector, _ = extract_hybrid_features(
        representative_segment
    )

    # ========================================================
    # RETURN PANEL-1-COMPATIBLE SESSION
    # ========================================================

    return {

        # ----------------------------------------------------
        # Unified prediction fields
        # ----------------------------------------------------

        "predicted_class": pred_class,

        "confidence": confidence,

        "probabilities": {

            INDEX_TO_CLASS[i]: float(p)

            for i, p in enumerate(final_probs)
        },

        # ----------------------------------------------------
        # Runtime compatibility fields
        # ----------------------------------------------------

        "acoustic_vector": acoustic_vector.tolist(),

        "audio": representative_segment,

        "sr": 16000,

        # ----------------------------------------------------
        # Additional metadata
        # ----------------------------------------------------

        "audio_type": "microphone",

        "inference_time_sec": float(
            inference_time
        ),

        "num_windows_used": len(
            windows
        )
    }


# ============================================================
# 🧪 TEST RUN
# ============================================================

if __name__ == "__main__":

    test_audio ="8Vz9jwmadkRfjbwrHwGZ2MsUtcz1_cough-heavy.wav"  # replace with your file

    result = predict_audio(test_audio)

    print("\n🎯 FINAL RESULT:")
    print(result)