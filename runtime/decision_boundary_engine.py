"""
decision_boundary_engine.py
---------------------------

Decision boundary and threshold analysis engine for SpectroCough.

Purpose:
- Estimate statistical decision boundaries
- Compute feature threshold ranges
- Determine class switch conditions
- Support "What-If Changed" logic page
- Provide prediction sensitivity analysis

STRICT DESIGN:
- Uses class_statistics.json only
- No model loading
- No probability recomputation
- Statistical reasoning only
- Deterministic
"""

import json
from typing import Dict, Any
import numpy as np

from runtime.feature_parser import parse_major_features

# ==========================================================
# LOAD CLASS STATISTICS
# ==========================================================

def load_class_statistics(
    audio_type="stethoscope"
) -> Dict[str, Any]:

    from runtime.base_paths import WEB_KB_DIR

    if audio_type == "microphone":

        kb_path = (
            WEB_KB_DIR /
            "microphone_profiles" /
            "class_statistics.json"
        )

    else:

        kb_path = (
            WEB_KB_DIR /
            "stethoscope_profiles" /
            "class_statistics.json"
        )

    # ------------------------------------------------------
    # Safe fallback
    # ------------------------------------------------------

    if not kb_path.exists():

        print(
            f"[Runtime Warning] "
            f"Missing class statistics: {kb_path}"
        )

        return {}

    with open(kb_path, "r") as f:
        return json.load(f)


# ==========================================================
# COMPUTE STATISTICAL RANGE
# ==========================================================

def compute_statistical_range(mean: float, std: float, k: float = 2.0):
    """
    Compute statistical boundary range.

    mean ± k * std
    k=2 → ~95% statistical region
    """

    lower = float(mean - k * std)
    upper = float(mean + k * std)

    return lower, upper


# ==========================================================
# FEATURE SENSITIVITY ANALYSIS
# ==========================================================

def compute_feature_sensitivity(
    user_features: Dict[str, float],
    class_means: Dict[str, float],
    class_stds: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute normalized distance from class baseline.
    Higher value → more sensitive feature.
    """

    sensitivity = {}

    for feature_name, user_value in user_features.items():

        mean_val = class_means.get(feature_name, 0.0)
        std_val = class_stds.get(feature_name, 1e-6)

        if std_val < 1e-8:
            score = 0.0
        else:
            score = abs(user_value - mean_val) / std_val

        sensitivity[feature_name] = float(score)

    return sensitivity


# ==========================================================
# CLASS SWITCH ANALYSIS
# ==========================================================

def compute_class_switch_conditions(
    user_features: Dict[str, float],
    class_stats: Dict[str, Any],
    predicted_class: str
):
    """
    Determine which classes are reachable by feature shifts.
    """

    switch_map = {}

    for cls_name, stats in class_stats.items():

        if cls_name == predicted_class:
            continue

        means = stats["feature_means"]
        stds = stats["feature_stds"]

        feature_requirements = {}

        for feature_name, user_value in user_features.items():

            mean_val = means.get(feature_name, 0.0)
            std_val = stds.get(feature_name, 1e-6)

            lower, upper = compute_statistical_range(mean_val, std_val)

            within_range = lower <= user_value <= upper

            required_shift = mean_val - user_value

            feature_requirements[feature_name] = {
                "target_mean": float(mean_val),
                "lower_threshold": lower,
                "upper_threshold": upper,
                "within_target_range": bool(within_range),
                "required_shift": float(required_shift),
                "required_shift_z": float(required_shift / max(std_val, 1e-6))
            }

        switch_map[cls_name] = feature_requirements

    return switch_map


# ==========================================================
# MAIN DECISION BOUNDARY API
# ==========================================================

def compute_decision_boundaries(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    audio_type: str = "stethoscope"
) -> Dict[str, Any]:
    """
    Generate full decision boundary analysis.
    """

    class_stats = load_class_statistics(audio_type)

    # ------------------------------------------------------
    # KB safety
    # ------------------------------------------------------

    if not class_stats:

        return {

            "audio_type": audio_type,

            "predicted_class": predicted_class,

            "user_features": {},

            "predicted_class_feature_ranges": {},

            "feature_sensitivity_scores": {},

            "top_sensitive_features": [],

            "class_switch_conditions": {},

            "boundary_summary":
                "Decision boundary KB unavailable."
        }

    if predicted_class not in class_stats:
        raise ValueError(f"{predicted_class} not found in class statistics")

    user_features = parse_major_features(acoustic_vector, audio_type)

    class_means = class_stats[predicted_class]["feature_means"]
    class_stds = class_stats[predicted_class]["feature_stds"]

    # ------------------------------------------------------
    # Statistical boundaries for predicted class
    # ------------------------------------------------------
    predicted_class_ranges = {}

    for feature_name, mean_val in class_means.items():

        std_val = class_stds.get(feature_name, 1e-6)
        lower, upper = compute_statistical_range(mean_val, std_val)

        predicted_class_ranges[feature_name] = {
            "mean": float(mean_val),
            "lower_threshold": lower,
            "upper_threshold": upper
        }

    # ------------------------------------------------------
    # Feature sensitivity ranking
    # ------------------------------------------------------
    sensitivity_scores = compute_feature_sensitivity(
        user_features,
        class_means,
        class_stds
    )

    sorted_sensitivity = sorted(
        sensitivity_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_sensitive_features = [
        {"feature": name, "sensitivity_score": score}
        for name, score in sorted_sensitivity[:3]
    ]

    # ------------------------------------------------------
    # Class switching analysis
    # ------------------------------------------------------
    switch_conditions = compute_class_switch_conditions(
        user_features,
        class_stats,
        predicted_class
    )

    entity_name = (
        "respiratory sound profile"
        if audio_type == "microphone"
        else "cough profile"
    )

    # ------------------------------------------------------
    # Human-readable explanation
    # ------------------------------------------------------
    if top_sensitive_features:

        summary_text = (
            f"The predicted {entity_name} for {predicted_class} "
            f"is most sensitive to "
            f"{', '.join([f['feature'] for f in top_sensitive_features])}. "
            f"Shifting these features beyond statistical thresholds may "
            f"lead to a different classification."
        )

    else:

        summary_text = (
            "Insufficient data for sensitivity analysis."
        )

    # ------------------------------------------------------
    # Final structure
    # ------------------------------------------------------
    return {
        "predicted_class": predicted_class,

        "audio_type": audio_type,

        # User feature values
        "user_features": user_features,

        # Statistical ranges for predicted class
        "predicted_class_feature_ranges": predicted_class_ranges,

        # Sensitivity analysis
        "feature_sensitivity_scores": sensitivity_scores,
        "top_sensitive_features": top_sensitive_features,

        # Possible class switching conditions
        "class_switch_conditions": switch_conditions,

        # Human explanation
        "boundary_summary": summary_text
    }