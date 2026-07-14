
# Explainable Multi-Task Learning for Diabetic Retinopathy Stage & Blindness Risk

A deep learning model that, given a single retina fundus image, jointly predicts:

1. **DR stage** — No DR / Mild / Moderate / Severe / Proliferative (5-class classification)
2. **Blindness risk** — High vs. Low risk (binary), engineered from DR stage severity
3. **Grad-CAM explanation** — a heatmap showing which regions of the retina (lesions,
   hemorrhages, exudates) drove the prediction, so the output is auditable rather than
   a black-box number.

## Validated Results (real APTOS 2019 data, Kaggle T4 GPU)

| Metric | Score |
|---|---|
| DR Stage QWK (Quadratic Weighted Kappa) | **0.536** |
| Blindness Risk AUC-ROC | **0.802** |

QWK is the standard clinical metric for DR grading — it penalizes predictions
based on how far off they are (e.g. predicting "Mild" when the truth is "No DR"
is a smaller error than predicting "Proliferative").

## Architecture

```
Fundus Image (300x300x3)
        │
        ▼
EfficientNetB3 backbone (ImageNet-pretrained, FROZEN)
        │
        ▼
Conv2D(128, 1x1) "last_conv_features"  ◄── Grad-CAM target layer
        │
        ▼
GlobalAveragePooling2D
        │
   ┌────┴────┐
   ▼         ▼
Dense(128)  Dense(128)
   │         │
Dropout(0.4) Dropout(0.4)
   │         │
   ▼         ▼
stage_output  risk_output
(5-class      (binary
 softmax)      sigmoid)
```

**Why the backbone is frozen:** an earlier version added a channel-attention
module on top of the backbone and trained unstably (validation accuracy
collapsed to a single class — a known small-dataset / BatchNorm pathology).
Removing the attention module and freezing the entire EfficientNetB3 backbone
(training only the two small dense heads) is what produced stable, legitimate
training curves and the QWK/AUC scores above. This is documented here
deliberately — the "what didn't work and why" is good material for a thesis's
methodology/limitations section.

## Project Structure

```
dr_multitask_project/
├── app.py                  # Streamlit demo: upload image -> stage + risk + Grad-CAM
├── requirements.txt
├── README.md
├── checkpoints/            # trained model saved here (best_model.keras)
├── outputs/                # sample prediction images / heatmaps
└── src/
    ├── model.py            # model architecture (build_model, compile_model)
    ├── dataset.py           # data loading, CLAHE preprocessing, risk-label engineering
    ├── losses.py             # combined multi-task loss definitions
    ├── train.py              # training entry point (CLI)
    ├── evaluate.py            # QWK, AUC, confusion matrix
    ├── gradcam.py             # Grad-CAM heatmap generation + overlay
    └── utils.py               # label name mappings, prediction formatting
```

## Datasets

- **APTOS 2019 Blindness Detection** (Kaggle) — ~3,600 labeled fundus images,
  DR stage 0–4. Hosted directly on Kaggle, no manual download needed.
- **IDRiD** (Indian Diabetic Retinopathy Image Dataset) — smaller, higher-quality,
  includes DME (macular edema) grade which can be merged into a stricter risk rule.

Blindness-risk label is engineered (no dataset ships it directly):

```python
risk_label = 1 if diagnosis >= 3 else 0   # Severe or Proliferative DR = high risk
```

## How to Run

### Option A: Kaggle Notebook (recommended — free GPU, datasets pre-hosted)

1. Create a new Kaggle Notebook, add the "APTOS 2019 Blindness Detection" dataset.
2. Turn on GPU (Settings → Accelerator → GPU T4 x2).
3. Upload the `src/` folder or paste its contents into notebook cells.
4. Run training:
   ```bash
   python src/train.py \
       --csv /kaggle/input/aptos2019-blindness-detection/train.csv \
       --img_dir /kaggle/input/aptos2019-blindness-detection/train_images \
       --id_col id_code --stage_col diagnosis \
       --epochs 30 --batch_size 16
   ```
5. Evaluate:
   ```bash
   python src/evaluate.py --model checkpoints/best_model.keras \
       --csv <val_csv> --img_dir <val_img_dir>
   ```

### Option B: Local machine (inference/demo only, no GPU required)

```bash
pip install -r requirements.txt
streamlit run app.py
```

This launches the upload-an-image demo using a model already trained on
Kaggle/Colab and downloaded into `checkpoints/best_model.keras`.

## Honest Limitations

- Blindness-risk label is a rule-based engineering choice, not a clinically
  validated ground-truth label — this should be stated explicitly in any
  thesis writeup.
- The backbone is fully frozen; fine-tuning deeper layers (with a much lower
  learning rate and more data) could likely improve QWK further, but was not
  stable at this dataset size during experimentation.
- Grad-CAM is a qualitative diagnostic aid, not a formally verified
  attribution method — discuss failure cases (noisy heatmaps) honestly.

## Future Work

- Fine-tune deeper EfficientNetB3 layers with a lower learning rate once more
  labeled data is available.
- Merge IDRiD's DME grade into a stricter, multi-factor risk label.
- Deploy as a batch pipeline: GPS-tagged images → route-level risk heatmap.
=======
