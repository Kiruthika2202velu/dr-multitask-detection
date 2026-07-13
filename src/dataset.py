"""
dataset.py
Data loading + preprocessing pipeline for APTOS 2019 / IDRiD fundus images.

Handles:
  - CLAHE contrast enhancement (standard preprocessing for fundus images)
  - Resizing to EfficientNetB3's native input size (300x300)
  - Engineering the blindness-risk label from DR stage (no dataset ships this label)
  - A memory-efficient tf.data pipeline (streams images from disk instead of
    loading the whole dataset into RAM)
"""

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = 300  # EfficientNetB3 native input size


def load_and_preprocess(image_path: str) -> np.ndarray:
    """Read a fundus image from disk, apply CLAHE, resize, normalize to [0,1]."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # CLAHE on the L channel (LAB color space) - boosts lesion contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    return img.astype(np.float32) / 255.0


def engineer_risk_label(df: pd.DataFrame, stage_col: str = "diagnosis") -> pd.DataFrame:
    """
    Blindness-risk rule (clinically motivated, documented for the thesis):
      Stage >= 3 (Severe or Proliferative DR) -> high risk (1)
      Stage <  3 (No DR, Mild, Moderate)       -> low risk  (0)

    If you also have IDRiD's DME (macular edema) grade, you can tighten this:
      risk = 1 if (stage >= 3 or dme_grade >= 2) else 0
    """
    df = df.copy()
    df["risk_label"] = df[stage_col].apply(lambda x: 1 if x >= 3 else 0)
    return df


def split_data(df: pd.DataFrame, id_col: str, stage_col: str, risk_col: str, test_size: float = 0.2, seed: int = 42):
    return train_test_split(
        df[id_col], df[stage_col], df[risk_col],
        test_size=test_size, stratify=df[stage_col], random_state=seed,
    )


def make_dataset(image_ids, stages, risks, image_dir: str, batch_size: int = 16,
                  img_ext: str = ".png", shuffle: bool = True) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset that streams and preprocesses images on the fly.
    Avoids loading the entire dataset into memory at once.
    """

    def _load(image_id):
        path = f"{image_dir}/{image_id.numpy().decode()}{img_ext}"
        return load_and_preprocess(path)

    def _map_fn(image_id, stage, risk):
        img = tf.py_function(_load, [image_id], tf.float32)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img, {"stage_output": stage, "risk_output": risk}

    ds = tf.data.Dataset.from_tensor_slices((image_ids.values, stages.values, risks.values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_ids))
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
