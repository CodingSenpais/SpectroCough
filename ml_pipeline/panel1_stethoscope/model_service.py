"""
model_service.py
----------------
Central model loading + inference service for SpectroCough.

Responsibilities:
- Load trained model once at startup
- Load acoustic scaler
- Run audio standardization
- Extract hybrid features
- Run prediction
- Return structured prediction dictionary

Used by:
api_server.py
"""

import numpy as np
import tensorflow as tf
import joblib

from ml_pipeline.panel1_stethoscope.config import CLASSES
from ml_pipeline.panel1_stethoscope.audio_standardize import standardize_audio
from ml_pipeline.panel1_stethoscope.features import extract_hybrid_features

from runtime.explanation_engine import generate_explanation
from runtime.fingerprint_engine import generate_fingerprint
from runtime.spectrogram_engine import build_spectrogram_analysis

from runtime.base_paths import PANEL1_MODEL_PATH, PANEL1_SCALER_PATH


# ============================================================
# LOAD MODEL + SCALER ONCE (GLOBAL SINGLETON)
# ============================================================

print("Loading SpectroCough model...")


model = tf.keras.models.load_model(PANEL1_MODEL_PATH)
scaler = joblib.load(PANEL1_SCALER_PATH)

print("Model and scaler loaded successfully.")


# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def predict_audio(audio_path: str):
    """
    Runs full inference pipeline on a single audio file.

    Parameters
    ----------
    audio_path : str
        Path to uploaded audio file

    Returns
    -------
    dict
        Structured prediction output
    """

    # --------------------------------------------------------
    # 1. Standardize audio
    # --------------------------------------------------------
    y = standardize_audio(audio_path)

    # --- NEW: Preserve raw audio for spectrogram comparison ---
    raw_audio_signal = y.copy()

    # --------------------------------------------------------
    # 2. Extract hybrid features
    # --------------------------------------------------------
    mel, acoustic = extract_hybrid_features(y)

    # --- NEW: Preserve raw acoustic vector (for explainability) ---
    raw_acoustic_vector = acoustic.copy()

    # --------------------------------------------------------
    # 3. Scale acoustic features
    # --------------------------------------------------------
    acoustic = scaler.transform(acoustic.reshape(1, -1))

    # --------------------------------------------------------
    # 4. Prepare model inputs
    # --------------------------------------------------------
    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)

    # --------------------------------------------------------
    # 5. Predict probabilities
    # --------------------------------------------------------
    probs = model.predict((mel, acoustic), verbose=0)[0]

    predicted_index = int(np.argmax(probs))
    predicted_class = CLASSES[predicted_index]
    confidence = float(probs[predicted_index])

    # --------------------------------------------------------
    # NEW: Explainability Package
    # --------------------------------------------------------
    explanation_package = generate_explanation(
        acoustic_vector=raw_acoustic_vector,
        predicted_class=predicted_class,
        confidence=confidence
    )

    # --------------------------------------------------------
    # NEW: Acoustic Fingerprint (for visual comparison)
    # --------------------------------------------------------
    fingerprint_data = generate_fingerprint(
        acoustic_vector=raw_acoustic_vector,
        predicted_class=predicted_class
    )

    # --------------------------------------------------------
    # NEW: Spectrogram Comparison Package
    # --------------------------------------------------------
    spectrogram_package = build_spectrogram_analysis(
        predicted_class,
        raw_audio_signal,
        22050
    )

    # --------------------------------------------------------
    # 6. Build probability dictionary
    # --------------------------------------------------------
    probability_dict = {
        cls: float(prob)
        for cls, prob in zip(CLASSES, probs)
    }

    # --------------------------------------------------------
    # 7. Return structured response
    # --------------------------------------------------------
    return {

        # --------------------------------------------------------
        # Core prediction
        # --------------------------------------------------------

        "predicted_class": predicted_class,

        "confidence": confidence,

        "probabilities": probability_dict,

        # --------------------------------------------------------
        # Runtime compatibility
        # --------------------------------------------------------

        "acoustic_vector": raw_acoustic_vector.tolist(),

        "audio": raw_audio_signal,

        "sr": 22050,

        "audio_type": "stethoscope",

        # --------------------------------------------------------
        # Explainability payloads
        # --------------------------------------------------------

        "explanation": explanation_package,

        "fingerprint": fingerprint_data,

        "spectrogram_analysis": spectrogram_package
    }


# ============================================================
# UNIFIED ROUTER COMPATIBILITY
# ============================================================

def predict_cough(audio_path: str):
    """
    Unified inference alias for API router.
    """

    return predict_audio(audio_path)