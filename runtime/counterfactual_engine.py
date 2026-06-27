"""
counterfactual_engine.py
-------------------------
Generates counterfactual explanations for SpectroCough.

Purpose:
- Determine minimal feature shifts required
  to move prediction from current_class → target_class.

STRICT DESIGN:
- Uses KB-1 only
- No model retraining
- No probability recomputation
- Statistical directional reasoning only
- Fully deterministic
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
):

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
# FIND CLOSEST ALTERNATE CLASS
# ==========================================================

def find_closest_class(
    user_features,
    class_stats,
    current_class
):

    min_distance = float("inf")
    closest_class = None

    for cls, stats in class_stats.items():

        if cls == current_class:
            continue

        means = stats["feature_means"]
        stds = stats["feature_stds"]

        distance = 0.0

        for feature, value in user_features.items():

            mean_val = means.get(feature, 0.0)

            std_val = max(
                stds.get(feature, 1.0),
                1e-6
            )

            z = (
                value - mean_val
            ) / std_val

            distance += z ** 2

        distance = float(np.sqrt(distance))

        if distance < min_distance:
            min_distance = distance
            closest_class = cls

    return closest_class


# ==========================================================
# COUNTERFACTUAL CORE LOGIC
# ==========================================================

def compute_counterfactual(
    acoustic_vector: np.ndarray,
    current_class: str,
    target_class: str = None,
    audio_type: str = "stethoscope"
) -> Dict[str, Any]:

    class_stats = load_class_statistics(audio_type)

    # ----------------------------------------------------------
    # KB safety
    # ----------------------------------------------------------

    if not class_stats:

        return {

            "audio_type": audio_type,

            "current_class": current_class,

            "comparisons": []
        }

    if current_class not in class_stats:
        raise ValueError(f"{current_class} not found in class stats")

    # Only validate target if provided
    if target_class is not None and target_class not in class_stats:
        raise ValueError(f"{target_class} not found in class stats")

    user_features = parse_major_features(acoustic_vector, audio_type)

    # ----------------------------------------------------------
    # NEW: Multi-class counterfactual (ALL other classes)
    # ----------------------------------------------------------

    if target_class is not None:

        all_classes = [target_class]

    else:

        all_classes = list(class_stats.keys())

    comparisons = []

    for target_class in all_classes:

        if target_class == current_class:
            continue

        target_means = class_stats[target_class]["feature_means"]

        feature_shifts = []

        for feature_name, current_value in user_features.items():

            target_mean = target_means.get(feature_name, 0.0)

            delta = target_mean - current_value
            direction = "increase" if delta > 0 else "decrease"

            percent_change = (
                (delta / current_value) * 100
                if current_value != 0 else 0
            )

            feature_shifts.append({
                "feature": feature_name,
                "current_value": float(current_value),
                "target_value": float(target_mean),
                "difference": float(delta),
                "percent_change": float(percent_change),
                "direction": direction
            })

        # Sort by smallest shift → most realistic change
        feature_shifts = sorted(
            feature_shifts,
            key=lambda x: abs(x["difference"])
        )

        comparisons.append({
            "target_class": target_class,
            "top_changes": feature_shifts[:3],
            "all_changes": feature_shifts
        })

    return {

        "audio_type": audio_type,

        "current_class": current_class,

        "comparisons": comparisons
    }