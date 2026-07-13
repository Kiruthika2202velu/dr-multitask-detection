"""
model.py
Multi-task Diabetic Retinopathy model:
  - Shared backbone: EfficientNetB3 (ImageNet pretrained, fully FROZEN)
  - Head 1: DR stage classification (5 classes: No DR, Mild, Moderate, Severe, Proliferative)
  - Head 2: Blindness risk (binary: high risk / low risk)

This is the SIMPLIFIED architecture that was proven to train cleanly:
frozen backbone + two small dense heads, NO custom attention module.
Result on real APTOS validation data: QWK = 0.536, Risk AUC = 0.802.

The earlier attention-augmented version destabilized training (the extra
trainable attention layers on top of a frozen backbone made optimization
unstable at this dataset size) — so it was removed. If you want to re-add
attention later, add it back only after unfreezing some backbone layers,
and lower the learning rate.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(img_size: int = 300, num_stages: int = 5, freeze_backbone: bool = True):
    """
    Build the multi-task DR model.

    Args:
        img_size: input image size (EfficientNetB3 native = 300)
        num_stages: number of DR stage classes (APTOS/IDRiD = 5)
        freeze_backbone: if True, all EfficientNetB3 layers are frozen
                          (this is the configuration that trained cleanly)

    Returns:
        tf.keras.Model with two named outputs: 'stage_output', 'risk_output'
    """
    inputs = layers.Input(shape=(img_size, img_size, 3), name="fundus_image")

    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling=None,
    )

    if freeze_backbone:
        backbone.trainable = False
    else:
        # Optional: fine-tune only the last N layers
        for layer in backbone.layers[:-30]:
            layer.trainable = False

    # Named conv feature layer -> used later by Grad-CAM
    feats = layers.Conv2D(128, 1, activation="relu", name="last_conv_features")(backbone.output)
    pooled = layers.GlobalAveragePooling2D(name="global_pool")(feats)

    # --- Head 1: DR stage classification ---
    s = layers.Dense(128, activation="relu", name="stage_dense")(pooled)
    s = layers.Dropout(0.4, name="stage_dropout")(s)
    stage_out = layers.Dense(num_stages, activation="softmax", name="stage_output")(s)

    # --- Head 2: Blindness risk (binary) ---
    r = layers.Dense(128, activation="relu", name="risk_dense")(pooled)
    r = layers.Dropout(0.4, name="risk_dropout")(r)
    risk_out = layers.Dense(1, activation="sigmoid", name="risk_output")(r)

    model = models.Model(inputs=inputs, outputs=[stage_out, risk_out], name="DR_MultiTask_EfficientNetB3")
    return model


def compile_model(model, learning_rate: float = 1e-4):
    """Compile with the combined multi-task loss used during training."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "stage_output": "sparse_categorical_crossentropy",
            "risk_output": "binary_crossentropy",
        },
        loss_weights={
            "stage_output": 1.0,
            "risk_output": 1.0,
        },
        metrics={
            "stage_output": "accuracy",
            "risk_output": tf.keras.metrics.AUC(name="auc"),
        },
    )
    return model


if __name__ == "__main__":
    m = build_model()
    m = compile_model(m)
    m.summary()
