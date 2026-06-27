"""
infer.py
--------
Production-ready inference module for SpectroCough.

Features:
- Load trained hybrid model
- Load pre-fitted acoustic scaler (scaler.pkl)
- Standardize single cough audio
- Extract Mel + Acoustic features
- Predict probability distribution
- Return structured dictionary (for API)
- Still supports CLI execution

Deployment-safe:
- No dataset scanning
- No scaler refitting
- No training dependencies
"""

import sys
from pathlib import Path

# ============================================================
# ROOT PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml_pipeline.panel1_stethoscope.model_service import (
    predict_cough as predict_stethoscope
)

from ml_pipeline.panel2_microphone.model_service import (
    predict_audio as predict_microphone
)




# ============================================================
# UNIFIED INFERENCE ROUTER
# ============================================================

def run_inference(
    audio_path: str,
    audio_type: str = "stethoscope"
):
    """
    Unified inference router for SpectroCough.

    Routes audio to:
    - Panel 1 (stethoscope)
    - Panel 2 (microphone)
    """

    audio_path = Path(audio_path)

    audio_path = audio_path.resolve()

    audio_type = str(
        audio_type
    ).strip().lower()

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if audio_path.suffix.lower() not in {
        ".wav",
        ".mp3",
        ".ogg",
        ".m4a"
    }:

        raise ValueError(
            "Unsupported audio format."
        )
    # ========================================================
    # STETHOSCOPE PIPELINE
    # ========================================================

    if audio_type == "stethoscope":

        result = predict_stethoscope(
            audio_path
        )

        result["audio_type"] = "stethoscope"

        return result

    # ========================================================
    # MICROPHONE PIPELINE
    # ========================================================

    elif audio_type == "microphone":

        result = predict_microphone(
            audio_path
        )

        result["audio_type"] = "microphone"

        return result

    # ========================================================
    # INVALID TYPE
    # ========================================================

    else:

        raise ValueError(
            f"Unsupported audio_type: {audio_type}"
        )


# ============================================================
# CLI INFERENCE (FOR TERMINAL USE)
# ============================================================

def infer_single_audio(
    audio_path: str,
    audio_type: str = "stethoscope"
):
    """
    CLI-compatible wrapper that prints results nicely.
    """

    print(f"\nRunning inference on: {audio_path}")

    result = run_inference(
        audio_path,
        audio_type=audio_type
    )
    
    print("\n===== SpectroCough – Probability Output =====")
    for cls, prob in result["probabilities"].items():
        print(f"{cls:<12}: {prob:.4f}")

    print(f"\nPredicted Class : {result['predicted_class']}")
    print(f"Confidence      : {result['confidence']:.4f}")

    print("\nNOTE:")
    print("This output represents probabilistic pre-screening results only.")
    print("It is NOT a medical diagnosis.")



# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python infer.py <audio.wav> "
            "[stethoscope|microphone]"
        )

        sys.exit(1)

    audio_file = sys.argv[1]

    audio_type = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "stethoscope"
    )

    infer_single_audio(
        audio_file,
        audio_type
    )