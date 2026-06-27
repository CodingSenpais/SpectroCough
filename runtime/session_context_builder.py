"""
session_context_builder.py
----------------------------
Builds structured session context for SpectroCough RAG system.

Responsibilities:
- Accept model prediction output
- Accept acoustic feature vector
- Call explanation_engine
- Build clean session JSON
- No memory persistence
- No history retention
- Single-session only (STRICT)

Designed for:
- Hybrid RAG + Dynamic Injection
- Chatbot prompt building
- Explainability UI layer
"""

from typing import Dict, Any
import numpy as np

# Local runtime import
from runtime.explanation_engine import generate_explanation

from runtime.fingerprint_engine import generate_fingerprint


# ==========================================================
# SESSION CONTEXT BUILDER
# ==========================================================

def build_session_context(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    confidence: float,
    audio_type: str = "stethoscope"
) -> Dict[str, Any]:
    """
    Construct full session-aware context for RAG injection.

    Parameters
    ----------
    acoustic_vector : np.ndarray
        93-dim acoustic feature vector (same used for model)
    predicted_class : str
        Model predicted class label
    confidence : float
        Model confidence score

    Returns
    -------
    dict:
        Fully structured session context
    """

    # Step 1: Generate explainability package
    explanation_data = generate_explanation(
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        confidence=confidence,
        audio_type=audio_type
    )

    # Step 2: Extract top deviating features
    top_features = explanation_data["top_deviating_features"]

    # ----------------------------------------------------------
    # NEW: Generate fingerprint data (visual comparison)
    # ----------------------------------------------------------
    fingerprint_data = generate_fingerprint(
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        audio_type=audio_type
    )

    # Step 3: Build session object
    session_context = {
        # Prediction summary
        "prediction_summary": {
            "predicted_class": predicted_class,
            "confidence_score": float(confidence),
            "audio_type": audio_type
        },

        # Full explanation package
        "explanation_summary": {
            "layman_summary": explanation_data.get("layman_summary"),
            "scientific_summary": explanation_data.get("scientific_summary"),
            "closest_alternate_class": explanation_data.get("closest_alternate_class")
        },

        # Detailed feature analysis
        "acoustic_feature_analysis": explanation_data["feature_analysis"],

        # Top deviations
        "top_deviations": top_features,

        # Visual fingerprint data
        "fingerprint_data": fingerprint_data,

        # Metadata
        "explainability_ready": True,
        "session_type": "single_prediction_session",
        "memory": None  # STRICT: no persistence
    }

    return session_context


# ==========================================================
# LIGHTWEIGHT PROMPT-INJECTION FORMATTER
# ==========================================================

def format_session_for_prompt(session_context: Dict[str, Any]) -> str:
    """
    Convert structured session context into compact
    injection string for LLM prompt.

    Designed for:
    - RAG system prompt augmentation
    - Minimal token overhead
    """

    summary = session_context["prediction_summary"]
    deviations = session_context["top_deviations"]

    lines = []
    lines.append("CURRENT SESSION PREDICTION CONTEXT:")
    lines.append(f"- Predicted Class: {summary['predicted_class']}")
    lines.append(f"- Audio Type: {summary.get('audio_type', 'unknown')}")
    lines.append(f"- Confidence: {summary['confidence_score']:.4f}")
    lines.append("")

    lines.append("TOP ACOUSTIC DEVIATIONS:")
    for item in deviations:
        lines.append(
            f"- {item['feature']}: "
            f"z={item['z_score']:.3f}, "
            f"Δ={item['percent_difference']:.2f}%"
        )

    return "\n".join(lines)


# ==========================================================
# OPTIONAL SAFE VALIDATION
# ==========================================================

def validate_session_context(session_context: Dict[str, Any]) -> bool:
    """
    Ensure session context integrity before RAG injection.
    """
    required_keys = [
        "prediction_summary",
        "explanation_summary",
        "acoustic_feature_analysis",
        "top_deviations",
        "fingerprint_data",
        "explainability_ready"
    ]

    for key in required_keys:
        if key not in session_context:
            return False

    return True