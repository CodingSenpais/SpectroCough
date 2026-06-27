"""
fingerprint_engine.py
----------------------
Generates radar-ready acoustic fingerprint
for SpectroCough explainability UI.

STRICT DESIGN:
- Uses class_statistics.json (KB-1)
- Uses same feature parsing logic
- No model loading
- No dataset scanning
- No medical assumptions
- Fully deterministic

Output:
{
    "predicted_class": str,
    "user_fingerprint": {...},
    "class_baseline": {...},
    "z_scores": {...}
}
"""

import json
from typing import Dict, Any
import numpy as np


from runtime.feature_parser import parse_major_features

def load_class_statistics(audio_type="stethoscope"):

    from runtime.base_paths import WEB_KB_DIR

    if audio_type == "microphone":
        kb_path = (
            WEB_KB_DIR
            / "microphone_profiles"
            / "class_statistics.json"
        )
    else:
        kb_path = (
            WEB_KB_DIR
            / "stethoscope_profiles"
            / "class_statistics.json"
        )

    if not kb_path.exists():
        raise FileNotFoundError(
            f"class_statistics.json not found: {kb_path}"
        )

    with open(kb_path, "r") as f:
        return json.load(f)


# ==========================================================
# BUILD MULTI-CLASS BASELINES (Visual Comparison)
# ==========================================================

def build_all_class_baselines(class_stats, user_features):
    """
    Build radar baseline for all classes for visual comparison.
    """

    all_baselines = {}

    for cls, stats in class_stats.items():
        means = stats["feature_means"]
        stds = stats["feature_stds"]

        baseline = {}

        for feature_name, user_value in user_features.items():
            mean_val = means.get(feature_name, 0.0)
            std_val = stds.get(feature_name, 1e-6)

            if std_val < 1e-8:
                z = 0.0
            else:
                z = (user_value - mean_val) / std_val

            baseline[feature_name] = normalize_to_radar_range(z)

        all_baselines[cls] = baseline

    return all_baselines


# ==========================================================
# FIND CLOSEST CLASS (for visual comparison)
# ==========================================================

def find_closest_class(user_features, class_stats, predicted_class):
    min_distance = float("inf")
    closest = None

    for cls, stats in class_stats.items():
        if cls == predicted_class:
            continue

        means = stats["feature_means"]

        distance = 0.0
        for feature, value in user_features.items():
            mean_val = means.get(feature, 0.0)
            std_val = stats["feature_stds"].get(
                feature,
                1.0
            )

            std_val = max(
                std_val,
                1e-6
            )

            distance += (
                (value - mean_val) / std_val
            ) ** 2

        distance = float(np.sqrt(distance))

        if distance < min_distance:
            min_distance = distance
            closest = cls

    return closest


# ==========================================================
# NORMALIZATION FOR RADAR
# ==========================================================

def normalize_to_radar_range(z_score: float, clip_range: float = 3.0) -> float:
    """
    Convert z-score to bounded radar scale [0, 1].

    - Clip z-score to [-clip_range, clip_range]
    - Scale to 0–1
    """

    z_clipped = max(min(z_score, clip_range), -clip_range)
    scaled = (z_clipped + clip_range) / (2 * clip_range)

    return float(scaled)


# ==========================================================
# MAIN FINGERPRINT GENERATOR
# ==========================================================

def generate_fingerprint(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    audio_type: str = "stethoscope"
) -> Dict[str, Any]:

    class_stats = load_class_statistics(audio_type)

    if predicted_class not in class_stats:
        raise ValueError(f"{predicted_class} not found in class statistics")

    user_features = parse_major_features(
        acoustic_vector,
        audio_type
    )

    # ----------------------------------------------------------
    # NEW: Closest competing class
    # ----------------------------------------------------------
    closest_class = find_closest_class(
        user_features,
        class_stats,
        predicted_class
    )

    class_means = class_stats[predicted_class]["feature_means"]
    class_stds = class_stats[predicted_class]["feature_stds"]

    z_scores = {}
    radar_values = {}
    baseline_values = {}

    for feature_name, user_value in user_features.items():

        mean_val = class_means.get(feature_name, 0.0)
        std_val = class_stds.get(feature_name, 1e-6)

        if std_val < 1e-8:
            z = 0.0
        else:
            z = (user_value - mean_val) / std_val

        z_scores[feature_name] = float(z)

        radar_values[feature_name] = normalize_to_radar_range(z)

        # Baseline always center (0.5 in radar scale)
        baseline_values[feature_name] = 0.5

    # ----------------------------------------------------------
    # NEW: Multi-class baseline comparison
    # ----------------------------------------------------------
    all_class_baselines = build_all_class_baselines(
        class_stats,
        user_features
    )

    # ----------------------------------------------------------
    # NEW: Feature axis metadata (for UI rendering)
    # ----------------------------------------------------------
    feature_axes = list(user_features.keys())

    return {
        "predicted_class": predicted_class,
        
        "audio_type": audio_type,

        # User radar values
        "user_fingerprint": radar_values,

        # Baseline of predicted class
        "class_baseline": baseline_values,

        # NEW: All class comparison data
        "all_class_baselines": all_class_baselines,

        # NEW: Closest competing class
        "closest_class": closest_class,

        # Feature axis labels
        "feature_axes": feature_axes,

        # Raw z scores
        "z_scores": z_scores
    }