"""
visualization_engine.py
-----------------------

Visualization data preparation engine for SpectroCough.

Purpose:
- Prepare visualization-ready acoustic comparison data
- Support visual aids page
- Generate user vs class comparison structures
- Provide radar/waveform comparison logic
- Create UI-ready visualization payload

STRICT DESIGN:
- Uses class_statistics.json only
- No model loading
- No dataset scanning
- No image generation
- Deterministic computation only
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
# NORMALIZATION FOR VISUAL SCALE
# ==========================================================

def normalize_value(value, mean, std, clip_range=3.0):
    """
    Convert feature value into normalized visual scale [0,1].
    Based on z-score normalization.
    """

    if std < 1e-8:
        z = 0.0
    else:
        z = (value - mean) / std

    z = max(min(z, clip_range), -clip_range)
    scaled = (z + clip_range) / (2 * clip_range)

    return float(scaled)


# ==========================================================
# BUILD VISUAL PROFILE
# ==========================================================

def build_visual_profile(user_features, class_means, class_stds):
    """
    Create normalized feature profile for visualization.
    """

    profile = {}

    for feature_name, user_value in user_features.items():
        mean_val = class_means.get(feature_name, 0.0)
        std_val = class_stds.get(feature_name, 1e-6)

        profile[feature_name] = normalize_value(
            user_value,
            mean_val,
            std_val
        )

    return profile


# ==========================================================
# BUILD CLASS BASELINE PROFILE
# ==========================================================

def build_class_baseline_profile(class_means):
    """
    Baseline always centered in visualization space.
    """

    return {feature: 0.5 for feature in class_means.keys()}


# ==========================================================
# BUILD ALL CLASS COMPARISON
# ==========================================================

def build_all_class_comparisons(
    user_features,
    class_stats
):
    """
    Build comparison profiles for all classes.
    """

    all_profiles = {}

    for cls_name, stats in class_stats.items():

        class_means = stats["feature_means"]
        class_stds = stats["feature_stds"]

        user_profile = build_visual_profile(
            user_features,
            class_means,
            class_stds
        )

        baseline_profile = build_class_baseline_profile(class_means)

        all_profiles[cls_name] = {
            "user_profile": user_profile,
            "class_baseline": baseline_profile
        }

    return all_profiles


# ==========================================================
# MAIN VISUALIZATION API
# ==========================================================

def generate_visualization_payload(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    audio_type: str = "stethoscope"
) -> Dict[str, Any]:
    """
    Generate full visualization structure for UI.
    """

    class_stats = load_class_statistics(audio_type)

    # ------------------------------------------------------
    # KB safety
    # ------------------------------------------------------

    if not class_stats:

        return {

            "audio_type": audio_type,

            "predicted_class": predicted_class,

            "feature_axes": [],

            "user_vs_predicted": {},

            "user_vs_healthy": {},

            "user_vs_all_classes": {},

            "visualization_summary":
                "Visualization KB unavailable."
        }

    if predicted_class not in class_stats:
        raise ValueError(f"{predicted_class} not found in class statistics")

    user_features = parse_major_features(acoustic_vector, audio_type)

    predicted_stats = class_stats[predicted_class]
    predicted_means = predicted_stats["feature_means"]
    predicted_stds = predicted_stats["feature_stds"]

    # ------------------------------------------------------
    # User vs predicted class
    # ------------------------------------------------------
    user_vs_predicted = {
        "user_profile": build_visual_profile(
            user_features,
            predicted_means,
            predicted_stds
        ),
        "class_baseline": build_class_baseline_profile(predicted_means)
    }

    # ------------------------------------------------------
    # User vs healthy reference (if exists)
    # ------------------------------------------------------
    # ------------------------------------------------------
    # Healthy reference mapping
    # ------------------------------------------------------

    healthy_reference = None

    healthy_class = (
        "healthy_cough"
        if audio_type == "microphone"
        else "healthy"
    )

    if healthy_class in class_stats:

        healthy_stats = class_stats[
            healthy_class
        ]

        healthy_reference = {

            "user_profile":
            build_visual_profile(
                user_features,
                healthy_stats["feature_means"],
                healthy_stats["feature_stds"]
            ),

            "class_baseline":
            build_class_baseline_profile(
                healthy_stats["feature_means"]
            )
        }

    # ------------------------------------------------------
    # User vs all classes
    # ------------------------------------------------------
    all_class_profiles = build_all_class_comparisons(
        user_features,
        class_stats
    )

    # ------------------------------------------------------
    # Feature axis labels for UI
    # ------------------------------------------------------
    feature_axes = list(user_features.keys())

    # ------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------
    entity_name = (
        "respiratory sound profile"
        if audio_type == "microphone"
        else "cough acoustic profile"
    )

    summary_text = (
        f"Visualization compares your {entity_name} against "
        f"{predicted_class} and other reference classes "
        f"to highlight similarities and differences in "
        f"acoustic behavior."
    )


    # ------------------------------------------------------
    # Final structure
    # ------------------------------------------------------
    return {
        "predicted_class": predicted_class,

        "audio_type": audio_type,

        "feature_axes": feature_axes,

        "user_vs_predicted": user_vs_predicted,
        "user_vs_healthy": healthy_reference,
        "user_vs_all_classes": all_class_profiles,

        "visualization_summary": summary_text
    }