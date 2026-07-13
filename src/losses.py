"""
losses.py
Combined multi-task loss for the DR stage + blindness risk model.

L_total = alpha * CrossEntropy(stage_pred, stage_true)
        + beta  * BinaryCrossEntropy(risk_pred, risk_true)

In practice this is handled directly by model.compile(loss={...}, loss_weights={...})
in model.py — this file exists so alpha/beta can be tuned or swapped for
custom losses (e.g. focal loss for class imbalance) without touching model.py.
"""

import tensorflow as tf


def weighted_multitask_loss(alpha: float = 1.0, beta: float = 1.0):
    """
    Returns a dict of losses and loss_weights ready to pass into model.compile().

    Example:
        losses, weights = weighted_multitask_loss(alpha=1.0, beta=1.5)
        model.compile(optimizer=..., loss=losses, loss_weights=weights, metrics=...)
    """
    losses = {
        "stage_output": "sparse_categorical_crossentropy",
        "risk_output": "binary_crossentropy",
    }
    loss_weights = {
        "stage_output": alpha,
        "risk_output": beta,
    }
    return losses, loss_weights


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Optional: focal loss for the risk head if class imbalance (few high-risk
    cases) hurts recall on the minority class. Not used by default.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -tf.reduce_mean(alpha_factor * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return loss_fn
