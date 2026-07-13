"""
gradcam.py
Grad-CAM explainability for the DR multi-task model.

Generates a heatmap over the input fundus image showing which regions
(lesions, hemorrhages, exudates) most influenced the model's prediction
for a chosen head ('stage_output' or 'risk_output').

Uses the 'last_conv_features' layer named in model.py as the target
convolutional layer.
"""

import cv2
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(model, img_array, output_name: str = "stage_output", class_index: int = None):
    """
    Args:
        model: trained tf.keras.Model with outputs 'stage_output' and 'risk_output'
        img_array: preprocessed image batch, shape (1, H, W, 3)
        output_name: 'stage_output' or 'risk_output'
        class_index: for stage_output, which class to explain (e.g. predicted class).
                     Ignored for risk_output (binary, single logit).

    Returns:
        2D numpy heatmap normalized to [0, 1]
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer("last_conv_features").output,
            model.get_layer("stage_output").output,
            model.get_layer("risk_output").output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_out, stage_pred, risk_pred = grad_model(img_array)
        if output_name == "stage_output":
            if class_index is None:
                class_index = int(tf.argmax(stage_pred[0]))
            target = stage_pred[:, class_index]
        else:
            target = risk_pred[:, 0]

    grads = tape.gradient(target, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(tf.maximum(heatmap, 0))
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Resize heatmap to match original image and blend it as a jet colormap overlay.
    original_img: float32 array in [0,1], shape (H, W, 3)
    heatmap: 2D array in [0,1]
    Returns: uint8 RGB image with heatmap overlay
    """
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    base = original_img.astype(np.float32)
    overlay = (1 - alpha) * base + alpha * heatmap_color
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)
