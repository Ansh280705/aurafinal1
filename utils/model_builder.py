"""
utils/model_builder.py
======================
Builds the VGG16-based transfer learning model for binary
brain-tumor classification.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam


# ──────────────────────────────────────────────
#  Model factory
# ──────────────────────────────────────────────
def build_vgg16_model(
    img_size: tuple = (224, 224),
    learning_rate: float = 1e-4,
    dense_units: int = 256,
    dropout_rate: float = 0.5,
    fine_tune_at: int = 15,          # freeze layers before this index
) -> Model:
    """
    Returns a compiled VGG16-based binary classifier.

    Architecture:
        VGG16 (ImageNet, no top)
        → GlobalAveragePooling2D
        → Dense(dense_units, relu) + BatchNormalization + Dropout
        → Dense(1, sigmoid)

    Parameters
    ----------
    img_size      : (H, W) input size
    learning_rate : Adam lr
    dense_units   : units in the FC layer
    dropout_rate  : dropout probability
    fine_tune_at  : fine-tune layers from this index onward
    """
    input_shape = (*img_size, 3)

    # ── Base model ──────────────────────────────
    base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze all base layers initially
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"[INFO] VGG16 total layers : {len(base.layers)}")
    print(f"[INFO] Trainable from idx : {fine_tune_at} "
          f"({len(base.layers) - fine_tune_at} layers)")

    # ── Custom head ─────────────────────────────
    inputs  = tf.keras.Input(shape=input_shape, name="mri_input")
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.Dense(dense_units, name="fc1")(x)
    x       = layers.BatchNormalization(name="bn1")(x)
    x       = layers.Activation("relu", name="relu1")(x)
    x       = layers.Dropout(dropout_rate, name="dropout1")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="BrainTumorVGG16")

    # ── Compile ─────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def get_model_summary(model: Model) -> str:
    """Return model summary as a string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
