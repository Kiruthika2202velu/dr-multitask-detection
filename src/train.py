"""
train.py
Training entry point for the DR multi-task model.

Usage (on Kaggle/Colab, after adding the APTOS dataset):

    python src/train.py \
        --csv /kaggle/input/aptos2019-blindness-detection/train.csv \
        --img_dir /kaggle/input/aptos2019-blindness-detection/train_images \
        --id_col id_code --stage_col diagnosis \
        --epochs 30 --batch_size 16

This trains the SIMPLIFIED architecture (frozen EfficientNetB3 backbone,
no attention module) that was validated to train cleanly:
QWK = 0.536, Risk AUC = 0.802 on real APTOS validation data.
"""

import argparse
import tensorflow as tf

from model import build_model, compile_model
from dataset import engineer_risk_label, split_data, make_dataset
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to labels CSV (e.g. APTOS train.csv)")
    parser.add_argument("--img_dir", required=True, help="Path to folder containing training images")
    parser.add_argument("--id_col", default="id_code", help="Column name for image ID")
    parser.add_argument("--stage_col", default="diagnosis", help="Column name for DR stage label")
    parser.add_argument("--img_ext", default=".png", help="Image file extension")
    parser.add_argument("--img_size", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.keras")
    args = parser.parse_args()

    print("GPU available:", tf.config.list_physical_devices("GPU"))

    # 1. Load labels + engineer risk label
    df = pd.read_csv(args.csv)
    df = engineer_risk_label(df, stage_col=args.stage_col)
    print(df[args.stage_col].value_counts())
    print(df["risk_label"].value_counts())

    # 2. Split
    train_ids, val_ids, train_stage, val_stage, train_risk, val_risk = split_data(
        df, id_col=args.id_col, stage_col=args.stage_col, risk_col="risk_label"
    )
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

    # 3. Build tf.data pipelines
    train_ds = make_dataset(train_ids, train_stage, train_risk, args.img_dir,
                             batch_size=args.batch_size, img_ext=args.img_ext, shuffle=True)
    val_ds = make_dataset(val_ids, val_stage, val_risk, args.img_dir,
                           batch_size=args.batch_size, img_ext=args.img_ext, shuffle=False)

    # 4. Build + compile model (frozen backbone version)
    model = build_model(img_size=args.img_size, num_stages=5, freeze_backbone=True)
    model = compile_model(model, learning_rate=args.lr)
    model.summary()

    # 5. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.checkpoint, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    # 6. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("Training complete. Best model saved to:", args.checkpoint)
    return history


if __name__ == "__main__":
    main()
