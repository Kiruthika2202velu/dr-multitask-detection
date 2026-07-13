"""
evaluate.py
Evaluation metrics for the DR multi-task model:
  - QWK (Quadratic Weighted Kappa) for DR stage - the standard clinical
    metric for DR grading (accounts for how "far off" a wrong prediction is,
    not just right/wrong)
  - AUC-ROC for blindness risk (binary classification)
  - Confusion matrix for DR stage

Usage:
    python src/evaluate.py --model checkpoints/best_model.keras \
        --csv <val_csv> --img_dir <val_img_dir>
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix, classification_report

from dataset import engineer_risk_label, make_dataset


def evaluate(model, val_ds):
    stage_true, risk_true = [], []
    stage_pred, risk_pred = [], []

    for imgs, labels in val_ds:
        sp, rp = model.predict(imgs, verbose=0)
        stage_pred.extend(np.argmax(sp, axis=1))
        risk_pred.extend(rp.flatten())
        stage_true.extend(labels["stage_output"].numpy())
        risk_true.extend(labels["risk_output"].numpy())

    qwk = cohen_kappa_score(stage_true, stage_pred, weights="quadratic")
    auc = roc_auc_score(risk_true, risk_pred)
    cm = confusion_matrix(stage_true, stage_pred)
    report = classification_report(stage_true, stage_pred)

    print(f"DR Stage QWK: {qwk:.3f}")
    print(f"Blindness Risk AUC: {auc:.3f}")
    print("\nConfusion Matrix (DR stage):")
    print(cm)
    print("\nClassification Report (DR stage):")
    print(report)

    return {"qwk": qwk, "auc": auc, "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved .keras model")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--id_col", default="id_code")
    parser.add_argument("--stage_col", default="diagnosis")
    parser.add_argument("--img_ext", default=".png")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    df = pd.read_csv(args.csv)
    df = engineer_risk_label(df, stage_col=args.stage_col)

    val_ds = make_dataset(df[args.id_col], df[args.stage_col], df["risk_label"],
                           args.img_dir, batch_size=args.batch_size, img_ext=args.img_ext, shuffle=False)

    evaluate(model, val_ds)


if __name__ == "__main__":
    main()
