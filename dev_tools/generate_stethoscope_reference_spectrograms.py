"""
generate_reference_spectrogram_pngs.py
--------------------------------------

Offline utility for SpectroCough.

Purpose:
Generate Mel-spectrogram PNGs from the dataset so they can be used
as reference examples in the Visual Lab.

Dataset Structure Expected:

dataset/
    asthma/
        a1.wav
        a2.wav
    bronchial/
    copd/
    pneumonia/
    healthy/

Output:

web_kb/reference_spectrograms/
    asthma/
    bronchial/
    copd/
    pneumonia/
    healthy/
"""

import librosa
import librosa.display
import matplotlib

# Use headless backend (important for servers)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from runtime.base_paths import DATASET_DIR, REFERENCE_SPEC_DIR


# =========================================================
# PATH CONFIGURATION
# =========================================================

OUTPUT_DIR = REFERENCE_SPEC_DIR


# =========================================================
# AUDIO + SPECTROGRAM PARAMETERS
# =========================================================

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

MAX_FILES_PER_CLASS = 20

SUPPORTED_CLASSES = [
    "asthma",
    "bronchial",
    "copd",
    "pneumonia",
    "healthy"
]


# =========================================================
# GENERATE MEL SPECTROGRAM
# =========================================================

def generate_spectrogram(audio_path: Path, output_path: Path):

    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Generate mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    # Convert to log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(4, 3))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        cmap="magma"
    )

    plt.axis("off")
    plt.tight_layout()

    # Save image
    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


# =========================================================
# PROCESS CLASS
# =========================================================

def process_class(class_name: str):

    class_path = DATASET_DIR / class_name
    output_class_path = OUTPUT_DIR / class_name

    if not class_path.exists():
        print(f"Skipping {class_name} (folder missing)")
        return

    output_class_path.mkdir(parents=True, exist_ok=True)

    audio_files = list(class_path.glob("*.wav"))

    if len(audio_files) == 0:
        print(f"No audio files found for {class_name}")
        return

    audio_files = audio_files[:MAX_FILES_PER_CLASS]

    print(f"\nProcessing {class_name} ({len(audio_files)} files)")

    for i, audio_file in enumerate(audio_files):

        output_file = output_class_path / f"{class_name}_{i}.png"

        generate_spectrogram(audio_file, output_file)

        print("Saved:", output_file)


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nGenerating reference spectrogram PNGs...\n")

    for cls in SUPPORTED_CLASSES:
        process_class(cls)

    print("\nAll spectrogram PNGs generated successfully.")


if __name__ == "__main__":
    main()