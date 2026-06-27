import matplotlib
matplotlib.use("Agg")

import numpy as np
import base64
import io
import librosa
import matplotlib.pyplot as plt

from runtime.base_paths import (
    PANEL1_REFERENCE_SPEC_DIR,
    PANEL2_REFERENCE_SPEC_DIR
)

# ==========================================================
# MODALITY CLASS MAPS
# ==========================================================

PANEL1_CLASSES = [
    "asthma",
    "bronchial",
    "copd",
    "pneumonia",
    "healthy"
]

PANEL2_CLASSES = [
    "covid19",
    "healthy_cough",
    "sneezing"
]

# ==========================================================
# PATH RESOLUTION
# ==========================================================

def get_reference_dir(audio_type: str):

    if audio_type == "microphone":
        return PANEL2_REFERENCE_SPEC_DIR

    return PANEL1_REFERENCE_SPEC_DIR

# ==========================================================
# CLASS RESOLUTION
# ==========================================================

def get_class_list(audio_type: str):

    if audio_type == "microphone":
        return PANEL2_CLASSES

    return PANEL1_CLASSES

# ==========================================================
# Convert spectrogram array -> base64 image
# ==========================================================

def spectrogram_array_to_base64(spec):

    fig = plt.figure(figsize=(4, 3))

    plt.imshow(
        spec,
        aspect='auto',
        origin='lower',
        cmap="magma"
    )

    plt.axis("off")

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")

# ==========================================================
# Load random reference spectrogram
# ==========================================================

def load_reference_spectrogram(
    class_name,
    audio_type="stethoscope"
):

    reference_dir = get_reference_dir(
        audio_type
    )

    npy_file = (
        reference_dir /
        f"{class_name}.npy"
    )

    if not npy_file.exists():
        return None

    spec = np.load(
        npy_file,
        allow_pickle=False
    )

    if spec.ndim != 2:
        return None

    return spectrogram_array_to_base64(
        spec
    )

# ==========================================================
# Generate spectrogram from user audio
# ==========================================================


def create_user_spectrogram(audio, sr):

    if audio is None:
        return None

    if len(audio) == 0:
        return None

    if sr is None or sr <= 0:
        return None

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return spectrogram_array_to_base64(
        mel_db
    )

# ==========================================================
# Build visual comparison payload
# ==========================================================

def build_spectrogram_analysis(
    predicted_class,
    audio,
    sr,
    audio_type="stethoscope"
):

    user_spec = create_user_spectrogram(
        audio,
        sr
    )

    class_list = get_class_list(
        audio_type
    )

    # ------------------------------------------------------
    # Load all reference spectrograms safely
    # ------------------------------------------------------

    all_refs = {}

    for cls in class_list:

        all_refs[cls] = {

            "reference_image":
            load_reference_spectrogram(
                cls,
                audio_type
            )
        }

    # ------------------------------------------------------
    # Healthy reference selection
    # ------------------------------------------------------

    healthy_class = (
        "healthy_cough"
        if audio_type == "microphone"
        else "healthy"
    )

    return {

        "audio_type": audio_type,

        "user_spectrogram": user_spec,

        "predicted_class": predicted_class,

        "healthy_comparison": {

            "class": healthy_class,

            "reference_image":
            load_reference_spectrogram(
                healthy_class,
                audio_type
            )
        },

        "all_class_comparisons": all_refs
    }