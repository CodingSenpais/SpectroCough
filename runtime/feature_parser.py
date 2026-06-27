"""
feature_parser.py
-----------------
Unified acoustic feature parsing for SpectroCough.

Purpose
-------
Provides canonical interpretable feature extraction
from handcrafted acoustic vectors for:

- explainability
- fingerprinting
- visualization
- counterfactual analysis
- class comparison

Supports:
- Panel 1 (stethoscope)
- Panel 2 (microphone)

STRICT DESIGN
-------------
- No model loading
- No dataset scanning
- Deterministic only
"""

from typing import Dict
import numpy as np


# ==========================================================
# PANEL 1 — STETHOSCOPE PARSER
# ==========================================================

def parse_major_features_stethoscope(
    acoustic_vector: np.ndarray
) -> Dict[str, float]:

    idx = 80

    rms_mean = float(acoustic_vector[idx])
    rms_std = float(acoustic_vector[idx + 1])
    idx += 2

    zcr_mean = float(acoustic_vector[idx])
    idx += 1

    spectral_centroid = float(acoustic_vector[idx])
    idx += 1

    spectral_bandwidth = float(acoustic_vector[idx])
    idx += 1

    spectral_rolloff = float(acoustic_vector[idx])
    idx += 1

    spectral_contrast_vals = acoustic_vector[idx: idx + 7]

    spectral_contrast_mean = float(
        np.mean(spectral_contrast_vals)
    )

    return {
        "rms": rms_mean,
        "rms_std": rms_std,
        "zcr": zcr_mean,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "spectral_contrast": spectral_contrast_mean
    }


# ==========================================================
# PANEL 2 — MICROPHONE PARSER
# ==========================================================

def parse_major_features_microphone(
    acoustic_vector: np.ndarray
) -> Dict[str, float]:

    idx = 80

    # RMS
    rms_mean = float(acoustic_vector[idx])
    rms_std = float(acoustic_vector[idx + 1])

    idx += 3  # skip rms max

    # ZCR
    zcr_mean = float(acoustic_vector[idx])

    idx += 2  # skip zcr std

    # Spectral centroid
    spectral_centroid = float(acoustic_vector[idx])
    idx += 1

    # Spectral bandwidth
    spectral_bandwidth = float(acoustic_vector[idx])
    idx += 1

    # Spectral rolloff
    spectral_rolloff = float(acoustic_vector[idx])

    idx += 2  # skip rolloff variance

    # Spectral contrast
    spectral_contrast_vals = acoustic_vector[idx: idx + 7]

    spectral_contrast_mean = float(
        np.mean(spectral_contrast_vals)
    )

    return {
        "rms": rms_mean,
        "rms_std": rms_std,
        "zcr": zcr_mean,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "spectral_contrast": spectral_contrast_mean
    }


# ==========================================================
# UNIFIED INTERFACE
# ==========================================================

def parse_major_features(
    acoustic_vector: np.ndarray,
    audio_type: str = "stethoscope"
) -> Dict[str, float]:

    if audio_type == "microphone":

        return parse_major_features_microphone(
            acoustic_vector
        )

    return parse_major_features_stethoscope(
        acoustic_vector
    )