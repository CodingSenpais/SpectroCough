"""
generate_class_statistics.py
----------------------------
Generate Panel-2 Web KB class_statistics.json

Computes:

- rms
- rms_std
- zcr
- spectral_centroid
- spectral_bandwidth
- spectral_rolloff
- spectral_contrast

for:

- covid19
- healthy_cough
- sneezing

Output:
web_kb/microphone_profiles/class_statistics.json
"""

import json
import numpy as np

from pathlib import Path
from collections import defaultdict

from ml_pipeline.panel2_microphone.config import (
    DATASET_ROOT,
    CLASSES
)

from ml_pipeline.panel2_microphone.audio_standardize import (
    standardize_audio
)

# IMPORTANT:
# Acoustic features only
from ml_pipeline.panel2_microphone.acoustic_features_only import (
    extract_acoustic_features
)

from runtime.feature_parser import (
    parse_major_features_microphone
)

from runtime.base_paths import WEB_KB_DIR


# ==========================================================
# OUTPUT PATH
# ==========================================================

OUTPUT_PATH = (
    WEB_KB_DIR
    / "microphone_profiles"
    / "class_statistics.json"
)


# ==========================================================
# GENERATE STATISTICS
# ==========================================================

def generate_statistics():

    final_output = {}

    print("\n====================================")
    print("Generating Panel-2 Class Statistics")
    print("====================================\n")

    for class_name in CLASSES:

        print(f"\nProcessing class: {class_name}")

        class_dir = DATASET_ROOT / class_name

        files = sorted(
            class_dir.glob("*.wav")
        )

        if len(files) == 0:

            print(
                f"WARNING: No files found in {class_dir}"
            )

            continue

        feature_storage = defaultdict(list)

        # --------------------------------------------------
        # Process each file
        # --------------------------------------------------

        for idx, audio_file in enumerate(files):

            try:

                # ------------------------------------------
                # Standardize audio
                # ------------------------------------------

                y = standardize_audio(
                    str(audio_file)
                )

                # ------------------------------------------
                # Acoustic features only
                # ------------------------------------------

                acoustic = extract_acoustic_features(
                    y
                )

                # ------------------------------------------
                # Extract major interpretable features
                # ------------------------------------------

                major_features = (
                    parse_major_features_microphone(
                        acoustic
                    )
                )

                # ------------------------------------------
                # Store feature values
                # ------------------------------------------

                for feature_name, value in major_features.items():

                    feature_storage[
                        feature_name
                    ].append(float(value))

            except Exception as e:

                print(
                    f"[FAILED] {audio_file.name}"
                )

                print(e)

            if (idx + 1) % 50 == 0:

                print(
                    f"Processed {idx + 1}/{len(files)}"
                )

        # --------------------------------------------------
        # Compute means and stds
        # --------------------------------------------------

        feature_means = {}
        feature_stds = {}

        for feature_name, values in feature_storage.items():

            values = np.array(values)

            feature_means[
                feature_name
            ] = float(
                np.mean(values)
            )

            feature_stds[
                feature_name
            ] = float(
                np.std(values)
            )

        final_output[class_name] = {

            "feature_means":
                feature_means,

            "feature_stds":
                feature_stds,

            "num_samples":
                len(files)
        }

        print(
            f"Completed {class_name} "
            f"({len(files)} samples)"
        )

    # ------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_output,
            f,
            indent=4
        )

    print("\n====================================")
    print("Generation Complete")
    print("====================================")
    print(f"\nSaved to:\n{OUTPUT_PATH}")


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":

    generate_statistics()