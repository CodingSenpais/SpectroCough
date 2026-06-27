"""
rag_runtime_orchestrator.py
---------------------------

SpectroCough Runtime Orchestrator.

This is the central runtime controller connecting:

ML Pipeline → Explainability → Visualization → Counterfactual →
Class Comparison → Decision Boundary → Prediction Session → RAG

Purpose:
- Provide single unified runtime entry
- Coordinate all runtime engines
- Build full prediction session
- Provide UI-ready outputs
- Provide chatbot session context
- Ensure deterministic pipeline flow

STRICT DESIGN:
- Stateless execution
- No memory persistence
- No model loading here
- No dataset scanning
- Orchestrates runtime modules only
"""

from typing import Dict, Any
import numpy as np

# Runtime modules
from runtime.prediction_session_manager import (
    build_prediction_session,
    build_session_summary,
    validate_prediction_session
)

from runtime.visualization_engine import generate_visualization_payload
from runtime.class_comparison_engine import compute_class_comparison
from runtime.decision_boundary_engine import compute_decision_boundaries
from runtime.counterfactual_engine import compute_counterfactual
from runtime.explanation_engine import generate_explanation
from runtime.chatbot_core import call_groq_llm

# ==========================================================
# MAIN RUNTIME PIPELINE
# ==========================================================

def run_full_runtime_pipeline(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    confidence: float,
    probabilities: Dict[str, float] = None,
    analysis_type: str = "AI Analysis",
    audio_type: str = "stethoscope"
) -> Dict[str, Any]:
    """
    Execute full SpectroCough runtime pipeline.

    This builds everything needed by:
    - UI pages
    - RAG chatbot
    - Explainability system
    - Visual aids
    - What-if logic page
    """

    # ------------------------------------------------------
    # Step 1: Build unified prediction session
    # ------------------------------------------------------
    session_object = build_prediction_session(
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        analysis_type=analysis_type,
        audio_type=audio_type
    )

    if not validate_prediction_session(session_object):
        raise RuntimeError("Invalid prediction session generated")

    # ------------------------------------------------------
    # Step 2: Visualization payload
    # ------------------------------------------------------
    visualization_payload = generate_visualization_payload(
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        audio_type=audio_type
    )

    # ------------------------------------------------------
    # Step 3: Class comparison (explicit output)
    # ------------------------------------------------------
    class_comparison = compute_class_comparison(
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        audio_type=audio_type
    )

    # ------------------------------------------------------
    # Step 4: Decision boundary analysis
    # ------------------------------------------------------
    boundary_analysis = compute_decision_boundaries(
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        audio_type=audio_type
    )

    # ------------------------------------------------------
    # Step 5: Counterfactual reasoning
    # ------------------------------------------------------
    counterfactual_data = compute_counterfactual(
        acoustic_vector=acoustic_vector,
        current_class=predicted_class,
        audio_type=audio_type
    )

    # ------------------------------------------------------
    # NEW: LLM explanation for EACH target class
    # ------------------------------------------------------

    for comp in counterfactual_data.get("comparisons", []):

        target = comp["target_class"]
        changes = comp.get("top_changes", [])

        lines = []

        for c in changes:
            lines.append(
                f"{c['feature']}: "
                f"{c['current_value']:.4f} → {c['target_value']:.4f} "
                f"({c['direction']} {abs(c['difference']):.4f})"
            )

        counterfactual_text = "\n".join(lines)

        cf_prompt = f"""
        The model predicted the cough as: {predicted_class}

        Now we are analyzing how this prediction could change
        if the cough characteristics moved toward: {target}

        Required feature adjustments:

        {counterfactual_text}

        Explain this in a natural and intuitive way.

        Describe:

        - how the respiratory sound characteristics would change
        - how energy, frequency, or airflow patterns shift
        - why the acoustic profile would resemble {target} more than {predicted_class}

        If audio_type is microphone:
        focus on respiratory sound patterns.

        If audio_type is stethoscope:
        focus on cough acoustics.

        Rules:
        - Use the numbers provided
        - Do NOT invent new values
        - Explain acoustics in human terms
        """

        llm_output = call_groq_llm(
            system_prompt="You explain counterfactual cough acoustics clearly.",
            user_prompt=cf_prompt
        )

        comp["llm_explanation"] = llm_output

    # ------------------------------------------------------
    # Step 6: Explanation engine (NEW)
    # ------------------------------------------------------

    explanation_payload = generate_explanation(
        acoustic_vector=acoustic_vector,
        predicted_class=predicted_class,
        confidence=confidence,
        audio_type=audio_type
    )

    # ------------------------------------------------------
    # Healthy baseline statistics for comparison
    # ------------------------------------------------------

    healthy_class = (
        "healthy_cough"
        if audio_type == "microphone"
        else "healthy"
    )

    healthy_stats_text = ""

    try:

        healthy_payload = generate_explanation(
            acoustic_vector=acoustic_vector,
            predicted_class=healthy_class,
            confidence=confidence,
            audio_type=audio_type
        )

        healthy_analysis = healthy_payload.get(
            "feature_analysis",
            {}
        )

        healthy_lines = []

        for feature, metrics in healthy_analysis.items():

            healthy_lines.append(
                f"{feature}: "
                f"healthy_mean={metrics['class_mean']:.4f}, "
                f"healthy_std={metrics['class_std']:.4f}"
            )

        healthy_stats_text = "\n".join(
            healthy_lines
        )

    except Exception:

        healthy_stats_text = (
            "Healthy baseline unavailable."
        )

    # ------------------------------------------------------
    # Step 6.5: LLM explanation generation
    # ------------------------------------------------------

    # ------------------------------------------------------
    # Build advanced LLM prompt with KB context
    # ------------------------------------------------------

    # distinctive disease features
    distinctive = explanation_payload.get("distinctive_features", [])
    feature_list = (
        ", ".join(distinctive)
        if distinctive
        else "No distinctive features available"
    )

    # ------------------------------------------------------
    # Semantic KB integration not enabled yet
    # ------------------------------------------------------

    semantic_text = ""

    # ------------------------------------------------------
    # Build statistics block from explanation engine
    # ------------------------------------------------------

    feature_analysis = explanation_payload.get("feature_analysis", {})

    stats_lines = []

    for feature, metrics in feature_analysis.items():

        stats_lines.append(
            f"{feature}: "
            f"user={metrics['user_value']:.4f}, "
            f"mean={metrics['class_mean']:.4f}, "
            f"std={metrics['class_std']:.4f}, "
            f"z={metrics['z_score']:.3f}, "
            f"Δ={metrics['percent_difference']:.2f}%"
        )

    stats_text = "\n".join(stats_lines)



    # ------------------------------------------------------
    # Extract raw acoustic feature values (for LLM prompt)
    # ------------------------------------------------------

    # rms = float(acoustic_vector[0])
    # rms_std = float(acoustic_vector[1])
    # zcr = float(acoustic_vector[2])
    # spectral_centroid = float(acoustic_vector[3])
    # spectral_bandwidth = float(acoustic_vector[4])
    # spectral_rolloff = float(acoustic_vector[5])
    # spectral_contrast = float(acoustic_vector[6])

    # Extract same values used by flash cards
    user_features = explanation_payload["user_feature_values"]

    rms = float(user_features["rms"])
    rms_std = float(user_features["rms_std"])
    zcr = float(user_features["zcr"])
    spectral_centroid = float(user_features["spectral_centroid"])
    spectral_bandwidth = float(user_features["spectral_bandwidth"])
    spectral_rolloff = float(user_features["spectral_rolloff"])
    spectral_contrast = float(user_features["spectral_contrast"])

    # ------------------------------------------------------
    # LLM PROMPT
    # ------------------------------------------------------
    llm_prompt = f"""
    You are a respiratory acoustic analysis expert.

    A respiratory sound classifier produced the following prediction:

    PREDICTED CLASS: {predicted_class}

    AUDIO TYPE: {audio_type}

    Prediction confidence: {confidence}

    ---------------------------------

    User cough acoustic feature values:

    RMS = {rms:.4f}
    RMS_STD = {rms_std:.4f}
    ZCR = {zcr:.4f}
    Spectral Centroid = {spectral_centroid:.4f}
    Spectral Bandwidth = {spectral_bandwidth:.4f}
    Spectral Rolloff = {spectral_rolloff:.4f}
    Spectral Contrast = {spectral_contrast:.4f}

    ---------------------------------

    Dataset baseline statistics for {predicted_class}:

    {stats_text}

    ---------------------------------

    Healthy cough baseline statistics:

    {healthy_stats_text}

    ---------------------------------

    Most distinctive acoustic features for this disease:

    {feature_list}

    ---------------------------------

    Acoustic feature interpretations:

    {semantic_text}

    ---------------------------------

    Write two sections:

    LAYMAN:

    Describe how the cough likely sounds to a listener using natural human language.

    Do NOT repeat the same standard phrases each time.

    Instead, dynamically describe:

    • loudness of the cough
    • sharpness or softness of the bursts
    • dryness or moisture in the sound
    • presence of wheezing or rough airflow
    • rhythm or irregular breathing patterns

    Make the description feel natural and varied.

    Examples of possible descriptions (do NOT copy them directly):

    - short explosive bursts of air
    - strained airflow during coughing
    - rough rasping cough sound
    - heavy chest-driven cough
    - shallow repeated cough bursts
    - weak breathy cough

    Always compare the sound to a normal healthy cough and explain how it differs.

    Rules:
    - Do NOT include numeric values
    - Do NOT mention acoustic feature names
    - Do NOT repeat identical wording across responses
    - Each explanation must sound slightly different

    SCIENTIFIC:

    Use the acoustic statistics provided.

    Explain:
    1. Describe the predicted cough audio of user mentioning their acoustic metric values too
    2. Which acoustic features deviate most from the predicted-class baseline
    3. Compare those values against the healthy baseline
    4. Explain why those deviations match the predicted disease

    Rules:

    - Always reference numeric values
    - Compare against both:
      - predicted-class baseline
      - healthy baseline
    - Use ONLY the numeric values provided in the "User cough acoustic feature values" section.
    - Do NOT generate new numbers.  
    """

    llm_text = call_groq_llm(
        system_prompt="You explain cough predictions using acoustic analysis.",
        user_prompt=llm_prompt
    )

    # Split sections
    # ------------------------------------------------------
    # Parse LLM output into LAYMAN and SCIENTIFIC sections
    # ------------------------------------------------------

    import re

    layman_text = ""
    scientific_text = ""

    parts = re.split(
        r"scientific\s*:?",
        llm_text,
        maxsplit=1,
        flags=re.IGNORECASE
    )

    if len(parts) == 2:

        layman_text = (
            parts[0]
            .replace("LAYMAN", "")
            .replace("Layman", "")
            .replace("##", "")
            .strip()
        )

        scientific_text = parts[1].strip()

    else:

        layman_text = llm_text.strip()
        scientific_text = ""

    explanation_payload["layman_summary"] = layman_text
    explanation_payload["scientific_summary"] = scientific_text

    # ------------------------------------------------------
    # Step 7: Compact session summary
    # ------------------------------------------------------
    session_summary = build_session_summary(session_object)

    # ------------------------------------------------------
    # Final response structure
    # ------------------------------------------------------
    return {
        "prediction_session": session_object,
        "session_summary": session_summary,

        "visualization": visualization_payload,

        "class_comparison": class_comparison,

        "decision_boundaries": boundary_analysis,

        "counterfactual": counterfactual_data,

        # NEW: Explanation payload
        "explanation": explanation_payload,

        "runtime_status": "success"
    }


# ==========================================================
# LIGHTWEIGHT PIPELINES (For API Endpoints)
# ==========================================================

def run_visualization_only(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    audio_type: str = "stethoscope"
):
    return generate_visualization_payload(
        acoustic_vector,
        predicted_class,
        audio_type
    )


def run_counterfactual_only(
    acoustic_vector: np.ndarray,
    current_class: str,
    target_class: str = None,
    audio_type: str = "stethoscope"
):
    return compute_counterfactual(
        acoustic_vector,
        current_class,
        target_class,
        audio_type
    )


def run_boundary_analysis_only(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    audio_type: str = "stethoscope"
):
    return compute_decision_boundaries(
        acoustic_vector,
        predicted_class,
        audio_type
    )


def run_class_comparison_only(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    audio_type: str = "stethoscope"
):
    return compute_class_comparison(
        acoustic_vector,
        predicted_class,
        audio_type
    )


# ==========================================================
# SAFE VALIDATION HELPERS
# ==========================================================

def validate_runtime_inputs(
    acoustic_vector,
    predicted_class,
    confidence
):
    """
    Validate runtime inputs before pipeline execution.
    """

    if acoustic_vector is None:
        raise ValueError("acoustic_vector required")

    if not isinstance(predicted_class, str):
        raise ValueError("predicted_class must be string")

    if confidence is None:
        raise ValueError("confidence required")

    return True