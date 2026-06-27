"""
benchmark.py
-------------
Benchmarking & Evaluation Utility
for SpectroCough (Microphone Version)

Purpose:
------------------------------------------------
- Evaluate trained models
- Generate classification metrics
- Save confusion matrix
- Save ROC curves
- Save classification report
- Measure inference latency

Outputs:
------------------------------------------------
benchmark_results/
│
├── classification_report.txt
├── confusion_matrix.png
├── roc_curve.png

Compatible:
------------------------------------------------
- Python 3.13.5
- TensorFlow 2.x
"""

import os
import time
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

from ml_pipeline.panel2_microphone.dataset import build_datasets

from ml_pipeline.panel2_microphone.model_service import (
    model
)

from ml_pipeline.panel2_microphone.config import (
    CLASSES
)

# ============================================================
# 🔍 BENCHMARK SETTINGS
# ============================================================

SHOW_CLASSIFICATION_REPORT = True
SHOW_CONFUSION_MATRIX = True
SHOW_PER_CLASS_RECALL = True

SAVE_RESULTS = True

OUTPUT_DIR = "benchmark_results"

# ============================================================
# 📁 CREATE OUTPUT DIR
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# 💾 SAVE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(report: str):

    report_path = os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    print(
        f"\n💾 Classification report saved:"
        f"\n{report_path}"
    )

# ============================================================
# 💾 SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(cm):

    plt.figure(figsize=(7, 6))

    plt.imshow(
        cm,
        interpolation='nearest',
        cmap=plt.cm.Blues
    )

    plt.title("Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(len(CLASSES))

    plt.xticks(
        tick_marks,
        CLASSES,
        rotation=45
    )

    plt.yticks(
        tick_marks,
        CLASSES
    )

    # --------------------------------------------------------
    # Annotate values
    # --------------------------------------------------------

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white"
                if cm[i, j] > thresh
                else "black"
            )

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n💾 Confusion matrix saved:"
        f"\n{save_path}"
    )

# ============================================================
# 💾 SAVE ROC CURVE
# ============================================================

def save_roc_curve(y_true, y_probs):

    # --------------------------------------------------------
    # Binarize labels
    # --------------------------------------------------------

    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(len(CLASSES))
    )

    plt.figure(figsize=(8, 6))

    # --------------------------------------------------------
    # ROC per class
    # --------------------------------------------------------

    for i, cls in enumerate(CLASSES):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            y_probs[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{cls} (AUC = {roc_auc:.2f})"
        )

    # --------------------------------------------------------
    # Random baseline
    # --------------------------------------------------------

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--'
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend(loc="lower right")

    plt.grid(True)

    save_path = os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n💾 ROC curve saved:"
        f"\n{save_path}"
    )

# ============================================================
# 📊 EVALUATE MODEL
# ============================================================

def evaluate_model():

    print("\n🚀 Starting Benchmark Evaluation...\n")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    _, _, test_ds = build_datasets()

    print(f"✅ Test Samples: {len(test_ds)}\n")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    y_true = []

    y_pred = []

    y_probs_all = []

    inference_times = []

    # ========================================================
    # LOOP THROUGH TEST SET
    # ========================================================

    for idx in range(len(test_ds)):

        mel, acoustic, embedding, label = test_ds[idx]

        # ----------------------------------------------------
        # Expand dims
        # ----------------------------------------------------

        mel = np.expand_dims(
            mel,
            axis=-1
        )

        mel = np.expand_dims(
            mel,
            axis=0
        )

        acoustic = np.expand_dims(
            acoustic,
            axis=0
        )

        embedding = np.expand_dims(
            embedding,
            axis=0
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        true_idx = int(
            np.argmax(label)
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        start_time = time.time()

        probs = model.predict(
            (
                mel,
                acoustic,
                embedding
            ),
            verbose=0
        )[0]

        end_time = time.time()

        inference_time = end_time - start_time

        inference_times.append(
            inference_time
        )

        pred_idx = int(
            np.argmax(probs)
        )

        y_true.append(true_idx)

        y_pred.append(pred_idx)

        y_probs_all.append(probs)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (idx + 1) % 25 == 0:

            print(
                f"Processed "
                f"{idx + 1}/{len(test_ds)} samples..."
            )

    # ========================================================
    # CONVERT ARRAYS
    # ========================================================

    y_true = np.array(y_true)

    y_pred = np.array(y_pred)

    y_probs_all = np.array(y_probs_all)

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    recalls = recall_score(
        y_true,
        y_pred,
        average=None
    )

    avg_inference_time = float(
        np.mean(inference_times)
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 60)

    print("🎯 BENCHMARK RESULTS")

    print("=" * 60)

    print(
        f"\n✅ Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"✅ Macro F1 Score: "
        f"{macro_f1:.4f}"
    )

    print(
        f"✅ Avg Inference Time: "
        f"{avg_inference_time:.4f} sec/sample"
    )

    # ========================================================
    # PER-CLASS RECALL
    # ========================================================

    if SHOW_PER_CLASS_RECALL:

        print("\n📊 Per-Class Recall:\n")

        for idx, cls in enumerate(CLASSES):

            print(
                f"{cls:<20}: "
                f"{recalls[idx] * 100:.2f}%"
            )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES
    )

    if SHOW_CLASSIFICATION_REPORT:

        print("\n📋 Classification Report:\n")

        print(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    if SHOW_CONFUSION_MATRIX:

        print("\n🧩 Confusion Matrix:\n")

        print(cm)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    if SAVE_RESULTS:

        save_classification_report(
            report
        )

        save_confusion_matrix(
            cm
        )

        save_roc_curve(
            y_true,
            y_probs_all
        )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "accuracy": accuracy,

        "macro_f1": macro_f1,

        "avg_inference_time": avg_inference_time,

        "per_class_recall": {

            CLASSES[i]: float(recalls[i])

            for i in range(len(CLASSES))
        }
    }

# ============================================================
# 🚀 ENTRY POINT
# ============================================================

if __name__ == "__main__":

    results = evaluate_model()

    print("\n✅ Benchmark completed successfully.")