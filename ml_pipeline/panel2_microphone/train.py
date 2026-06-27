"""
train.py
--------
Next-Generation Training Pipeline for SpectroCough
(Microphone Version)

Architecture:
------------------------------------------------
1. Mel Spectrogram CNN Branch
2. Handcrafted Acoustic Feature Branch
3. YAMNet Embedding Branch
4. Hybrid Fusion Classifier

Compatible:
------------------------------------------------
- Python 3.13.5
- TensorFlow 2.x
- tensorflow-hub
"""

import os
import joblib
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from ml_pipeline.panel2_microphone.config import (
    CLASSES,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    USE_CLASS_WEIGHTS,
)

from ml_pipeline.panel2_microphone.dataset import build_datasets
from ml_pipeline.panel2_microphone.model import build_spectrocough_model


# ============================================================
# 📦 DATASET → TF.DATA
# ============================================================

def dataset_to_tf(dataset):

    def generator():

        for i in range(len(dataset)):

            mel, acoustic, embedding, label = dataset[i]

            # ------------------------------------------------
            # Add channel dimension for CNN
            # ------------------------------------------------
            mel = np.expand_dims(
                mel,
                axis=-1
            )

            yield (
                (
                    mel,
                    acoustic,
                    embedding
                ),
                label
            )

    # --------------------------------------------------------
    # Sample shapes
    # --------------------------------------------------------

    mel, acoustic, embedding, label = dataset[0]

    mel = np.expand_dims(
        mel,
        axis=-1
    )

    # --------------------------------------------------------
    # Output signature
    # --------------------------------------------------------

    output_signature = (
        (
            tf.TensorSpec(
                shape=(mel.shape[0], None, 1),
                dtype=tf.float32
            ),

            tf.TensorSpec(
                shape=acoustic.shape,
                dtype=tf.float32
            ),

            tf.TensorSpec(
                shape=(1024,),
                dtype=tf.float32
            ),
        ),

        tf.TensorSpec(
            shape=(NUM_CLASSES,),
            dtype=tf.float32
        ),
    )

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )


# ============================================================
# ⚖️ CLASS WEIGHTS
# ============================================================

def compute_weights(dataset):

    labels = []

    for _, _, _, label in dataset:
        labels.append(np.argmax(label))

    labels = np.array(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=labels
    )

    weights_dict = {
        i: w
        for i, w in enumerate(weights)
    }

    return weights_dict


# ============================================================
# 💾 PATH HELPERS
# ============================================================

def get_model_save_path():

    os.makedirs("models", exist_ok=True)

    return "models/spectrocough_yamnet_fusion.keras"


def get_scaler_save_path():

    os.makedirs("scalers", exist_ok=True)

    return "scalers/scaler_yamnet.pkl"


# ============================================================
# 🚀 MAIN TRAINING
# ============================================================

def main():

    print("\n🚀 Training Started")
    print(f"Classes: {CLASSES}\n")

    # ========================================================
    # BUILD DATASETS
    # ========================================================

    train_ds, val_ds, test_ds = build_datasets()

    scaler = train_ds.scaler

    # ========================================================
    # CONVERT TO TF.DATA
    # ========================================================

    train_tf = dataset_to_tf(train_ds)

    val_tf = dataset_to_tf(val_ds)

    # ========================================================
    # TRAIN PIPELINE
    # ========================================================

    train_tf = (
        train_tf
        .cache()
        .shuffle(64)
        .repeat()
        .padded_batch(
            BATCH_SIZE,

            padded_shapes=(
                (
                    (None, None, 1),   # mel
                    (None,),           # acoustic
                    (1024,)            # embedding
                ),

                (NUM_CLASSES,)
            )
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    # ========================================================
    # VALIDATION PIPELINE
    # ========================================================

    val_tf = (
        val_tf
        .cache()
        .padded_batch(
            BATCH_SIZE,

            padded_shapes=(
                (
                    (None, None, 1),
                    (None,),
                    (1024,)
                ),

                (NUM_CLASSES,)
            )
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    # ========================================================
    # SAMPLE SHAPES
    # ========================================================

    sample_mel, sample_acoustic, sample_embedding, _ = train_ds[0]

    mel_input_shape = (
        sample_mel.shape[0],
        None,
        1
    )

    acoustic_input_dim = sample_acoustic.shape[0]

    embedding_dim = sample_embedding.shape[0]

    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_spectrocough_model(
        mel_input_shape,
        acoustic_input_dim,
        embedding_dim
    )

    model.summary()

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = None

    if USE_CLASS_WEIGHTS:

        print("\n⚖️ Computing class weights...")

        class_weights = compute_weights(
            train_ds
        )

        print(
            "Class Weights:",
            class_weights
        )

    # ========================================================
    # CALLBACKS
    # ========================================================

    model_path = get_model_save_path()

    callbacks = [

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max"
        ),

        # ----------------------------------------------------
        # LR scheduling
        # ----------------------------------------------------
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode="max"
        ),

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_yamnet_fusion.keras",

            monitor="val_accuracy",

            mode="max",

            save_best_only=True,

            verbose=1
        )
    ]

    # ========================================================
    # TRAINING STEPS
    # ========================================================

    steps_per_epoch = max(
        1,
        len(train_ds) // BATCH_SIZE
    )

    validation_steps = max(
        1,
        len(val_ds) // BATCH_SIZE
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n🔥 Training...\n")

    model.fit(

        train_tf,

        validation_data=val_tf,

        epochs=EPOCHS,

        steps_per_epoch=steps_per_epoch,

        validation_steps=validation_steps,

        class_weight=class_weights,

        callbacks=callbacks
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    model.save(model_path)

    # ========================================================
    # SAVE SCALER
    # ========================================================

    scaler_path = get_scaler_save_path()

    joblib.dump(
        scaler,
        scaler_path
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n✅ Training Completed!")

    print(
        "🔥 Best model saved at:"
        " models/best_yamnet_fusion.keras"
    )

    print(
        f"Model saved at: {model_path}"
    )

    print(
        f"Scaler saved at: {scaler_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()