"""
evaluate.py
-----------
Evaluation script for SpectroCough v1.

Responsibilities:
- Load trained hybrid model
- Evaluate on test dataset ONLY
- Compute accuracy, macro-F1, per-class recall
- Generate confusion matrix and classification report

STRICT RULES:
- No training here
- No augmentation
- No diagnostic claims
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from sklearn.preprocessing import label_binarize

from ml_pipeline.panel1_stethoscope.config import (
    CLASSES,
    NUM_CLASSES,
    BATCH_SIZE,
)

from ml_pipeline.panel1_stethoscope.dataset import build_datasets


# ============================================================
# DATASET → TF.DATA
# ============================================================

def dataset_to_tf(dataset):
    """
    Convert SpectroCoughDataset to tf.data.Dataset.
    """

    def generator():
        for i in range(len(dataset)):
            mel, acoustic, label = dataset[i]
            mel = np.expand_dims(mel, axis=-1)
            yield (mel, acoustic), label

    mel, acoustic, label = dataset[0]
    mel = np.expand_dims(mel, axis=-1)

    output_signature = (
        (
            tf.TensorSpec(shape=mel.shape, dtype=tf.float32),
            tf.TensorSpec(shape=acoustic.shape, dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32),
    )

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    ).batch(BATCH_SIZE)


# ============================================================
# MAIN EVALUATION LOGIC
# ============================================================

def main():
    print("Loading test dataset...")
    _, _, test_ds = build_datasets()

    test_tf = dataset_to_tf(test_ds)

    print("Loading trained model...")
    model = tf.keras.models.load_model("spectrocough_v1_baseline.keras")

    print("Running inference on test set...")
    y_true = []
    y_pred = []
    y_probs = []

    for (mel, acoustic), labels in test_tf:
        preds = model.predict((mel, acoustic), verbose=0)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        y_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # ========================================================
    # METRICS
    # ========================================================

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_recall = recall_score(
        y_true, y_pred, average=None
    )
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    

    print("\n===== SpectroCough v1 Evaluation =====")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro-F1     : {macro_f1:.4f}\n")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    print("Per-Class Recall:")
    for cls, recall in zip(CLASSES, per_class_recall):
        print(f"  {cls:<10}: {recall:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASSES,
            digits=4
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("SpectroCough v1 – Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ========================================================
    # ROC CURVES (One-vs-Rest)
    # ========================================================

    print("\nComputing ROC curves...")

    # Binarize true labels
    y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(),
        y_probs.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= NUM_CLASSES

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print("\nPer-Class AUC:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<10}: {roc_auc[i]:.4f}")

    print(f"\nMicro AUC: {roc_auc['micro']:.4f}")
    print(f"Macro AUC: {roc_auc['macro']:.4f}")

    # Plot ROC curves
    plt.figure(figsize=(8, 6))

    for i, cls in enumerate(CLASSES):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"{cls} (AUC = {roc_auc[i]:.3f})"
        )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        linestyle="--",
        label=f"Macro-average (AUC = {roc_auc['macro']:.3f})"
    )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SpectroCough v1 – Multi-Class ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
