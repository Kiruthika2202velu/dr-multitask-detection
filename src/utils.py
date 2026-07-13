"""
utils.py
Small shared helpers used across train.py / evaluate.py / gradcam.py / app.py.
"""

import numpy as np

STAGE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

RISK_NAMES = {
    0: "Low Risk",
    1: "High Risk (Severe/Proliferative)",
}


def stage_label_to_name(stage_idx: int) -> str:
    return STAGE_NAMES.get(int(stage_idx), f"Unknown ({stage_idx})")


def risk_label_to_name(risk_idx: int) -> str:
    return RISK_NAMES.get(int(risk_idx), f"Unknown ({risk_idx})")


def format_prediction(stage_probs: np.ndarray, risk_prob: float) -> dict:
    """Turn raw model outputs into a readable prediction summary."""
    stage_idx = int(np.argmax(stage_probs))
    return {
        "stage_index": stage_idx,
        "stage_name": stage_label_to_name(stage_idx),
        "stage_confidence": float(stage_probs[stage_idx]),
        "stage_probs": {STAGE_NAMES[i]: float(p) for i, p in enumerate(stage_probs)},
        "risk_probability": float(risk_prob),
        "risk_name": risk_label_to_name(1 if risk_prob >= 0.5 else 0),
    }
