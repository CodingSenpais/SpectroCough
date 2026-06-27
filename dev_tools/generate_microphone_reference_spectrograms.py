"""
generate_reference_spectrograms.py
----------------------------------

Creates representative spectrograms for:

- covid19
- healthy_cough
- sneezing

Used by:
runtime/spectrogram_engine.py
"""
import librosa
import numpy as np
from pathlib import Path

from ml_pipeline.panel2_microphone.audio_standardize import (
    standardize_audio
)


from ml_pipeline.panel2_microphone.config import (
    DATASET_ROOT,
    CLASSES
)

from runtime.base_paths import (
    PANEL2_REFERENCE_SPEC_DIR
)

# ==========================================================
# SETTINGS
# ==========================================================

NUM_SAMPLES_PER_CLASS = 20

# ==========================================================
# MAIN
# ==========================================================

def create_mel_spectrogram(y):

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=16000,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db.astype(np.float32)

def generate_reference_for_class(class_name):

    class_dir = DATASET_ROOT / class_name

    files = sorted(
        class_dir.glob("*.wav")
    )

    if len(files) == 0:
        print(f"No files found for {class_name}")
        return

    files = files[:NUM_SAMPLES_PER_CLASS]

    specs = []

    for wav_file in files:

        try:

            y = standardize_audio(
                str(wav_file)
            )

            mel = create_mel_spectrogram(y)

            specs.append(mel)

        except Exception as e:

            print(
                f"Skipping {wav_file.name}: {e}"
            )

    if len(specs) == 0:
        return

    avg_spec = np.mean(
        specs,
        axis=0
    )

    save_path = (
        PANEL2_REFERENCE_SPEC_DIR /
        f"{class_name}.npy"
    )

    np.save(
        save_path,
        avg_spec
    )

    print(
        f"Saved: {save_path}"
    )


def main():

    PANEL2_REFERENCE_SPEC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for cls in CLASSES:

        print(
            f"\nGenerating {cls}..."
        )

        generate_reference_for_class(
            cls
        )

    print(
        "\nFinished generating reference spectrograms."
    )


if __name__ == "__main__":
    main()