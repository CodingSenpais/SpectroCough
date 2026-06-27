"""
prediction_session_manager.py
------------------------------

Central Prediction Session Manager for SpectroCough.

Purpose:
- Create unified prediction session object
- Merge outputs from all runtime engines
- Provide single source of truth for:
    - UI pages
    - Chatbot RAG
    - API responses
    - Explainability layer

STRICT DESIGN:
- No memory persistence
- Single prediction session only
- Deterministic behavior
- No model loading
- No dataset scanning

Session Contains:
- Prediction metadata
- Feature explanation
- Acoustic fingerprint
- Counterfactual reasoning
- Class comparison
- Decision boundary analysis
"""

from typing import Dict, Any
import numpy as np
from datetime import datetime, UTC

# Runtime engines
from runtime.explanation_engine import generate_explanation
from runtime.fingerprint_engine import generate_fingerprint
from runtime.counterfactual_engine import compute_counterfactual
from runtime.class_comparison_engine import compute_class_comparison
from runtime.decision_boundary_engine import compute_decision_boundaries
from runtime.spectrogram_engine import build_spectrogram_analysis


# ==========================================================
# SESSION METADATA
# ==========================================================

SESSION_VERSION = "SC_SESSION_V1"

# ==========================================================
# SAFE ENGINE EXECUTION
# ==========================================================

def safe_runtime_call(
    fn,
    default=None,
    **kwargs
):
    """
    Prevent runtime engines from crashing
    the entire prediction session.
    """

    try:
        return fn(**kwargs)

    except Exception as e:

        print(
            f"[Runtime Warning] "
            f"{fn.__name__} failed: {e}"
        )

        return default


# ==========================================================
# MAIN SESSION BUILDER
# ==========================================================

def build_prediction_session(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    confidence: float,
    probabilities: Dict[str, float] = None,
    analysis_type: str = "AI Analysis",
    audio_type: str = "stethoscope",
    audio=None,
    sr=None,
) -> Dict[str, Any]:
    """
    Construct full prediction session.

    Parameters
    ----------
    acoustic_vector : np.ndarray
        93-dim acoustic feature vector
    predicted_class : str
    confidence : float
    probabilities : dict (optional)
    analysis_type : str

    Returns
    -------
    dict : unified session object
    """

    if acoustic_vector is None:
        raise ValueError("acoustic_vector required")

    # ------------------------------------------------------
    # 1. Explanation package
    # ------------------------------------------------------
    explanation_data = safe_runtime_call(
        generate_explanation,
        default={},
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        confidence=confidence,
        audio_type=audio_type
    )

    # ------------------------------------------------------
    # 2. Fingerprint (visual comparison)
    # ------------------------------------------------------
    fingerprint_data = safe_runtime_call(
        generate_fingerprint,
        default={},
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        audio_type=audio_type
    )

    # ------------------------------------------------------
    # 3. Counterfactual reasoning
    # ------------------------------------------------------
    counterfactual_data = safe_runtime_call(
        compute_counterfactual,
        default={},
        acoustic_vector=acoustic_vector,
        current_class=predicted_class,
        audio_type=audio_type
    )

    # ------------------------------------------------------
    # 4. Class similarity comparison
    # ------------------------------------------------------
    class_comparison = safe_runtime_call(
        compute_class_comparison,
        default={},
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        audio_type=audio_type
    )

    # ------------------------------------------------------
    # 5. Decision boundary analysis
    # ------------------------------------------------------
    boundary_analysis = safe_runtime_call(
        compute_decision_boundaries,
        default={},
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        audio_type=audio_type
    )

    # ------------------------------------------------------
    # 6. Spectrogram analysis (visual lab)
    # ------------------------------------------------------
    spectrogram_analysis = None

    if audio is not None and sr is not None:

        spectrogram_analysis = safe_runtime_call(
            build_spectrogram_analysis,
            default={},
            predicted_class=predicted_class,
            audio=audio,
            sr=sr,
            audio_type=audio_type
        )

    # ------------------------------------------------------
    # 7. Session object
    # ------------------------------------------------------
    session_object = {
        "session_metadata": {
            "session_version": SESSION_VERSION,
            "timestamp": datetime.now(UTC).isoformat(),
            "analysis_type": analysis_type,
            "audio_type": audio_type
        },

        "prediction": {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "probabilities": probabilities or {}
        },

        "spectrogram_analysis": spectrogram_analysis,

        "acoustic_vector": acoustic_vector.tolist(),

        "explanation": explanation_data,
        "fingerprint": fingerprint_data,
        "counterfactual": counterfactual_data,
        "class_comparison": class_comparison,
        "decision_boundaries": boundary_analysis,

        "session_type": "single_prediction_session",
        "memory": None
    }

    return session_object


# ==========================================================
# LIGHTWEIGHT SESSION SUMMARY (For UI / Chatbot)
# ==========================================================

def build_session_summary(session_object: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create compact session summary.

    Used for:
    - Chatbot prompt injection
    - UI quick display
    """

    prediction = session_object["prediction"]
    explanation = session_object["explanation"]

    return {
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "closest_class": explanation.get("closest_alternate_class"),
        "top_deviations": explanation.get("top_deviating_features", []),
        "layman_summary": explanation.get("layman_summary")
    }


# ==========================================================
# SESSION VALIDATION
# ==========================================================

def validate_prediction_session(session_object: Dict[str, Any]) -> bool:
    """
    Validate session integrity.
    """

    required_keys = [
        "session_metadata",
        "prediction",
        "acoustic_vector",
        "explanation",
        "fingerprint",
        "counterfactual",
        "class_comparison",
        "decision_boundaries"
    ]

    return all(key in session_object for key in required_keys)


# ==========================================================
# SAFE SESSION ACCESS HELPERS
# ==========================================================

def get_prediction(session_object: Dict[str, Any]) -> Dict[str, Any]:
    return session_object.get("prediction", {})


def get_explanation(session_object: Dict[str, Any]) -> Dict[str, Any]:
    return session_object.get("explanation", {})


def get_fingerprint(session_object: Dict[str, Any]) -> Dict[str, Any]:
    return session_object.get("fingerprint", {})


def get_counterfactual(session_object: Dict[str, Any]) -> Dict[str, Any]:
    return session_object.get("counterfactual", {})


def get_class_comparison(session_object: Dict[str, Any]) -> Dict[str, Any]:
    return session_object.get("class_comparison", {})


def get_decision_boundaries(session_object: Dict[str, Any]) -> Dict[str, Any]:
    return session_object.get("decision_boundaries", {})