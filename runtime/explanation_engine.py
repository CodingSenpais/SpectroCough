"""
explanation_engine.py
----------------------
SpectroCough Explainability Engine (KB-2 Runtime Layer)

Responsibilities:
- Load class_statistics.json (KB-1)
- Parse major acoustic features from 93-dim vector
- Compute deviation vs predicted class
- Compute Z-score
- Compute percentage difference
- Rank most deviating features
- Return structured explanation JSON

STRICT DESIGN:
- No model loading
- No dataset scanning
- No training logic
- Uses same feature ordering as features.py

"""

import json
from typing import Dict, Any

import numpy as np

from runtime.feature_parser import parse_major_features

# ==========================================================
# CLASS STATISTICS LOADER
# ==========================================================

def load_class_statistics(
    audio_type="stethoscope"
) -> Dict[str, Any]:
    """
    Load modality-aware class statistics.
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
            f"Missing KB file: {kb_path}"
        )

        return {}

    with open(
        kb_path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# ==========================================================
# DISEASE PROFILE LOADER
# ==========================================================

def load_disease_profiles(
    audio_type="stethoscope"
):
    """
    Load modality-aware disease profiles.
    """

    from runtime.base_paths import WEB_KB_DIR

    if audio_type == "microphone":

        kb_path = (
            WEB_KB_DIR /
            "microphone_profiles" /
            "disease_profiles" /
            "microphone_acoustic_profiles.json"
        )

    else:

        kb_path = (
            WEB_KB_DIR /
            "stethoscope_profiles" /
            "disease_profiles" /
            "disease_acoustic_profiles.json"
        )

    # ------------------------------------------------------
    # Safe fallback
    # ------------------------------------------------------

    if not kb_path.exists():

        print(
            f"[Runtime Warning] "
            f"Missing disease profiles: {kb_path}"
        )

        return {}

    with open(
        kb_path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)

# ==========================================================
# CORE EXPLANATION LOGIC
# ==========================================================

def compute_feature_deviation(
    user_features: Dict[str, float],
    class_means: Dict[str, float],
    class_stds: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Compute deviation metrics for each feature.

    Returns:
    {
        feature_name: {
            user_value,
            class_mean,
            class_std,
            z_score,
            percent_difference
        }
    }
    """

    results = {}

    for feature_name, user_value in user_features.items():

        mean_val = class_means.get(feature_name, 0.0)
        std_val = max(class_stds.get(feature_name, 0.0), 1e-3)

        # Avoid division by zero
        if std_val < 1e-8:
            z_score = 0.0
        else:
            z_score = (user_value - mean_val) / std_val
            z_score = max(min(z_score, 5.0), -5.0)

        # Percentage difference relative to class mean
        if abs(mean_val) < 1e-8:
            percent_diff = 0.0
        else:
            percent_diff = ((user_value - mean_val) / abs(mean_val)) * 100.0

        results[feature_name] = {
            "user_value": float(user_value),
            "class_mean": float(mean_val),
            "class_std": float(std_val),
            "z_score": float(z_score),
            "percent_difference": float(percent_diff)
        }

    return results

# ==========================================================
# STATISTICAL THRESHOLD RANGE COMPUTATION
# ==========================================================

def compute_threshold_ranges(
    class_means: Dict[str, float],
    class_stds: Dict[str, float],
    k: float = 2.0
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical threshold ranges using mean ± k*std.

    Default k=2 (~95% dataset range).
    """

    thresholds = {}

    for feature, mean_val in class_means.items():

        std_val = class_stds.get(feature, 1e-6)

        lower = float(mean_val - k * std_val)
        upper = float(mean_val + k * std_val)

        thresholds[feature] = {
            "mean": float(mean_val),
            "lower_threshold": lower,
            "upper_threshold": upper
        }

    return thresholds

def rank_top_deviations(
    deviation_dict: Dict[str, Dict[str, float]],
    top_k: int = 3
):
    """
    Rank features by absolute Z-score.
    """

    sorted_items = sorted(
        deviation_dict.items(),
        key=lambda item: abs(item[1]["z_score"]),
        reverse=True
    )

    return sorted_items[:top_k]


# ==========================================================
# CLOSEST CLASS COMPUTATION
# ==========================================================

def find_closest_class(user_features, class_stats, predicted_class):
    """
    Find closest alternate class based on Euclidean distance
    between user features and class means.
    """

    min_distance = float("inf")
    closest_class = None

    for cls, stats in class_stats.items():
        if cls == predicted_class:
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
# RAG CONTEXT BUILDER
# ==========================================================

def build_rag_feature_context(
    deviation_results: Dict[str, Dict[str, float]],
    top_features
) -> Dict[str, Any]:
    """
    Prepare structured feature context for LLM prompt injection.
    """

    context = []

    for feature_name, metrics in top_features:

        context.append({
            "feature": feature_name,
            "user_value": metrics["user_value"],
            "class_mean": metrics["class_mean"],
            "class_std": metrics["class_std"],
            "z_score": metrics["z_score"],
            "percent_difference": metrics["percent_difference"]
        })

    return context

# ==========================================================
# PUBLIC EXPLANATION API
# ==========================================================

def generate_explanation(
    acoustic_vector: np.ndarray,
    predicted_class: str,
    confidence: float,
    audio_type: str = "stethoscope"
) -> Dict[str, Any]:
    """
    Generate full explanation package for a prediction.

    Parameters
    ----------
    acoustic_vector : np.ndarray
        93-dim acoustic feature vector (scaled or unscaled allowed)
    predicted_class : str
        Model predicted class name
    confidence : float
        Model prediction confidence

    Returns
    -------
    Dict containing structured explanation data.
    """

    class_stats = load_class_statistics(audio_type)
    # Load disease acoustic profiles
    disease_profiles = (load_disease_profiles(audio_type) or {})

    # ----------------------------------------------------------
    # KB safety
    # ----------------------------------------------------------

    if not class_stats:

        return {

            "prediction": predicted_class,

            "confidence": float(confidence),

            "audio_type": audio_type,

            "feature_analysis": {},

            "user_feature_values": {},

            "feature_threshold_ranges": {},

            "rag_feature_context": [],

            "top_deviating_features": [],

            "closest_alternate_class": None,

            "explanation_context": {},

            "layman_summary":
                "Knowledge base unavailable.",

            "scientific_summary": None
        }

    if predicted_class not in class_stats:
        raise ValueError(
            f"Class '{predicted_class}' not found in class_statistics.json"
        )

    user_features = parse_major_features(acoustic_vector, audio_type)

    # ----------------------------------------------------------
    # Extract disease distinctive features
    # ----------------------------------------------------------

    distinctive_features = []

    if predicted_class in disease_profiles:
        distinctive_features = disease_profiles[predicted_class].get(
            "most_distinct_features",
            []
        )

    class_means = class_stats[predicted_class]["feature_means"]
    class_stds = class_stats[predicted_class]["feature_stds"]

    deviation_results = compute_feature_deviation(
        user_features,
        class_means,
        class_stds
    )

    top_features = rank_top_deviations(deviation_results, top_k=3)

    # ----------------------------------------------------------
    # NEW: Statistical threshold ranges
    # ----------------------------------------------------------

    threshold_ranges = compute_threshold_ranges(
        class_means,
        class_stds
    )

    # ----------------------------------------------------------
    # NEW: Structured feature context for RAG
    # ----------------------------------------------------------

    rag_feature_context = build_rag_feature_context(
        deviation_results,
        top_features
    )

    # ----------------------------------------------------------
    # NEW: Closest alternate class
    # ----------------------------------------------------------
    closest_class = find_closest_class(
        user_features,
        class_stats,
        predicted_class
    )

    # ----------------------------------------------------------
    # NEW: Build human-readable summaries
    # ----------------------------------------------------------
    layman_summary = None
    scientific_summary = None

    explanation_output = {
        "prediction": predicted_class,

        "confidence": float(confidence),
        
        "audio_type": audio_type,

        "distinctive_features": distinctive_features,

        # Detailed per-feature analysis
        "feature_analysis": deviation_results,

        # NEW: Feature values for UI metrics
        "user_feature_values": user_features,

        # NEW: Statistical thresholds
        "feature_threshold_ranges": threshold_ranges,

        # NEW: RAG structured context
        "rag_feature_context": rag_feature_context,

        # Ranked deviations
        "top_deviating_features": [
            {
                "feature": name,
                "z_score": metrics["z_score"],
                "percent_difference": metrics["percent_difference"]
            }
            for name, metrics in top_features
        ],

        # NEW: Closest competing class
        "closest_alternate_class": closest_class,
        
        # Context for LLM explanation generation
        "explanation_context": {
            "predicted_class": predicted_class,
            "closest_class": closest_class,
            "top_features": [name for name, _ in top_features],
            "confidence": float(confidence)
        },

        # NEW: Human-readable summaries
        "layman_summary": layman_summary,
        "scientific_summary": scientific_summary
    }

    return explanation_output