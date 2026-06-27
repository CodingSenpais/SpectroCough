"""
train.py
--------
Training script for SpectroCough v1.

Responsibilities:
- Build train/val/test datasets
- Compute class weights
- Build hybrid Mel + Acoustic model
- Train the model with proper metrics
- Save trained model

STRICT RULES:
- No fake / non-cough logic
- No feature or audio saving
- No augmentation for val/test
- Academic, non-diagnostic positioning
"""

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from ml_pipeline.panel1_stethoscope.config import (
    CLASSES,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    USE_CLASS_WEIGHTS,
)

from ml_pipeline.panel1_stethoscope.dataset import build_datasets
from ml_pipeline.panel1_stethoscope.model import build_spectrocough_model


# ============================================================
# DATA LOADER HELPERS
# ============================================================

def dataset_to_tf(dataset):
    """
    Convert SpectroCoughDataset to tf.data.Dataset.
    """

    def generator():
        for i in range(len(dataset)):
            mel, acoustic, label = dataset[i]

            # Add channel dimension for CNN
            mel = np.expand_dims(mel, axis=-1)

            yield (mel, acoustic), label

    # Infer shapes from first sample
    mel, acoustic, label = dataset[0]
    mel = np.expand_dims(mel, axis=-1)

    output_signature = (
        (
            tf.TensorSpec(shape=(mel.shape[0], None, 1), dtype=tf.float32),  # N_MELS, variable time, 1
            tf.TensorSpec(shape=acoustic.shape, dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32),
    )

    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    return tf_dataset


# ============================================================
# CLASS WEIGHT COMPUTATION
# ============================================================

def compute_weights(train_dataset):
    """ Compute class weights from training labels. """
    labels = []

    for _, _, label in train_dataset:
        labels.append(np.argmax(label))

    labels = np.array(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=labels
    )

    return {i: w for i, w in enumerate(weights)}



# ============================================================
# MAIN TRAINING LOGIC
# ============================================================

def main():
    print("Building datasets...")
    train_ds, val_ds, test_ds = build_datasets()

    print("Converting datasets to tf.data...")
    train_tf = dataset_to_tf(train_ds)
    val_tf = dataset_to_tf(val_ds)

    # Batch & prefetch (use padded_batch for variable time dimension)
    train_tf = (
        train_tf
        .shuffle(buffer_size=64)
        .padded_batch(
            BATCH_SIZE,
            padded_shapes=(
                ((None, None, 1), (None,)),  # (mel, acoustic)
                (NUM_CLASSES,)               # label
            )
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    val_tf = (
        val_tf
        .padded_batch(
            BATCH_SIZE,
            padded_shapes=(
                ((None, None, 1), (None,)),
                (NUM_CLASSES,)
            )
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    # Determine input shapes
    sample_mel, sample_acoustic, _ = train_ds[0]
    mel_input_shape = (sample_mel.shape[0], None, 1)  # variable time dimension
    acoustic_input_dim = sample_acoustic.shape[0]

    print("Building model...")
    model = build_spectrocough_model(
        mel_input_shape=mel_input_shape,
        acoustic_input_dim=acoustic_input_dim
    )

    model.summary()

    # Class weights
    class_weights = None
    if USE_CLASS_WEIGHTS:
        print("Computing class weights...")
        class_weights = compute_weights(train_ds)
        print("Class weights:", class_weights)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
    ]

    print("Starting training...")
    history = model.fit(
        train_tf,
        validation_data=val_tf,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print("Saving trained model...")
    model.save("spectrocough_v1_model.keras")

    print("Training completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
