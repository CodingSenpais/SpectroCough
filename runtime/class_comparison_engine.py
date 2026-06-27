"""
class_comparison_engine.py
--------------------------

Class similarity and comparison engine for SpectroCough.

Purpose:
- Compare user acoustic features against all disease class profiles
- Compute distance to class centroids
- Rank closest classes
- Provide feature-wise comparison
- Support explainability and UI visualization

STRICT DESIGN:
- Uses class_statistics.json only
- No model loading
- No dataset scanning
- Fully deterministic
- Stateless computation
"""

import json
from typing import Dict, Any, List
import numpy as np


from runtime.feature_parser import parse_major_features


# ==========================================================
# LOAD CLASS STATISTICS
# ==========================================================

def load_class_statistics(
    audio_type="stethoscope"
) -> Dict[str, Any]:
    """
    Load modality-aware class statistics KB.
    """

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
# DISTANCE COMPUTATION
# ==========================================================

def compute_feature_distance(
    user_features,
    class_means,
    class_stds
):

    distance = 0.0

    for feature_name, user_value in user_features.items():

        mean_val = class_means.get(feature_name, 0.0)

        std_val = max(
            class_stds.get(feature_name, 1.0),
            1e-6
        )

        z = (
            user_value - mean_val
        ) / std_val

        distance += z ** 2

    return float(np.sqrt(distance))


# ==========================================================
# FEATURE-WISE COMPARISON
# ==========================================================

def compute_feature_difference_map(
    user_features: Dict[str, float],
    class_means: Dict[str, float],
    class_stds: Dict[str, float]
) -> Dict[str, float]:
    """
    Feature-wise z-score difference relative to class baseline.
    """

    diff_map = {}

    for feature_name, user_value in user_features.items():

        mean_val = class_means.get(feature_name, 0.0)

        std_val = max(
            class_stds.get(feature_name, 1.0),
            1e-6
        )

        diff_map[feature_name] = float(
            (user_value - mean_val) / std_val
        )

    return diff_map


# ==========================================================
# RANK CLASSES BY SIMILARITY
# ==========================================================

def rank_classes_by_similarity(
    user_features: Dict[str, float],
    class_stats: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Compute similarity ranking across all classes.
    """

    ranking = []

    for cls_name, stats in class_stats.items():
        class_means = stats["feature_means"]

        distance = compute_feature_distance(
            user_features,
            class_means,
            stats["feature_stds"]
        )

        ranking.append({
            "class_name": cls_name,
            "distance": distance
        })

    ranking.sort(key=lambda x: x["distance"])
    return ranking


# ==========================================================
# MAIN COMPARISON API
# ==========================================================

def compute_class_comparison(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    audio_type: str = "stethoscope"
) -> Dict[str, Any]:
    """
    Compare user acoustic features against all disease classes.

    Returns structured comparison object.
    """

    class_stats = load_class_statistics(audio_type)

    # ------------------------------------------------------
    # KB safety
    # ------------------------------------------------------

    if not class_stats:

        return {

            "audio_type": audio_type,

            "predicted_class": predicted_class,

            "class_similarity_ranking": [],

            "closest_competing_class": None,

            "feature_difference_vs_predicted": {},

            "top_feature_differences": [],

            "user_features": {}
        }

    if predicted_class not in class_stats:
        raise ValueError(f"{predicted_class} not found in class statistics")

    user_features = parse_major_features(acoustic_vector, audio_type)

    # ------------------------------------------------------
    # Similarity ranking
    # ------------------------------------------------------
    similarity_ranking = rank_classes_by_similarity(
        user_features,
        class_stats
    )

    # ------------------------------------------------------
    # Closest competing class
    # ------------------------------------------------------
    closest_competitor = None
    for entry in similarity_ranking:
        if entry["class_name"] != predicted_class:
            closest_competitor = entry
            break

    # ------------------------------------------------------
    # Feature comparison vs predicted class
    # ------------------------------------------------------
    predicted_means = class_stats[predicted_class]["feature_means"]

    feature_difference = compute_feature_difference_map(
        user_features,
        predicted_means,
        class_stats[predicted_class]["feature_stds"]
    )

    # ------------------------------------------------------
    # Top differing features
    # ------------------------------------------------------
    sorted_diffs = sorted(
        feature_difference.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_feature_differences = [
        {"feature": name, "difference": value}
        for name, value in sorted_diffs[:3]
    ]

    # ------------------------------------------------------
    # Output structure
    # ------------------------------------------------------
    return {
        "predicted_class": predicted_class,

        "audio_type": audio_type,

        # Ranking of all classes by similarity
        "class_similarity_ranking": similarity_ranking,

        # Closest alternate class
        "closest_competing_class": closest_competitor,

        # Feature comparison
        "feature_difference_vs_predicted": feature_difference,

        # Top features driving differences
        "top_feature_differences": top_feature_differences,

        # Raw user feature values
        "user_features": user_features
    }