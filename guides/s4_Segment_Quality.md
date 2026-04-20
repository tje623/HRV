You are implementing Step 4 of an ECG artifact detection pipeline. Your task is to write one complete, production-ready Python file: `ecgclean/models/segment_quality.py`.

## Context

This is Stage 0 of the modeling pipeline — the segment-level noise gate. It runs before any beat-level model and masks globally bad segments from all downstream training and inference. The feature matrix from `segment_features.py` already exists. This model's output (`segment_quality_pred`) will be joined back onto the beat feature matrix as a contextual feature.

This is a year-long single-lead ECG (Polar H10, ~256Hz) from a POTS patient. Segments are 60-second windows. The three quality labels are `clean`, `noisy_ok`, and `bad`.

## What this file must implement

### 1. Training function
```python
def train(
    segment_features_path: str,
    segments_labels_path: str,
    output_model_path: str,
    val_fraction: float = 0.2,
    random_seed: int = 42
) -> dict:
```

Loads `segment_features.parquet` and `segments.parquet`. Trains a LightGBM multiclass classifier. Returns a dict of evaluation metrics.

**Split strategy:** Split by `segment_idx` order (temporal split — earlier segments train, later segments validate). Never shuffle randomly. This is mandatory because adjacent segments are correlated.

**Label encoding:**
- `clean` → 0
- `noisy_ok` → 1  
- `bad` → 2

**LightGBM parameters to use:**
```python
params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 5,
    "class_weight": "balanced",
    "random_state": random_seed,
    "n_jobs": -1,
    "verbose": -1
}
```

**NaN handling:** Before training, impute NaN values in the feature matrix with the column median computed from the training set only. Store the median values for use at inference time.

**Evaluation metrics to compute on validation set:**
- Per-class precision, recall, F1
- Macro-averaged F1
- Confusion matrix (3x3)
- Feature importances (top 15 by gain)

Save the trained model to `output_model_path` using `joblib.dump` as a dict:
```python
{
    "model": lgbm_classifier,
    "feature_columns": list_of_feature_column_names,
    "label_encoder": {"clean": 0, "noisy_ok": 1, "bad": 2},
    "train_medians": dict_of_column_medians,
    "val_metrics": metrics_dict,
    "trained_at": ISO_timestamp_string
}
```

Print a full training summary including confusion matrix and top features.

### 2. Inference function
```python
def predict(
    segment_features: pd.DataFrame,
    model_path: str
) -> pd.DataFrame:
```

Loads the saved model dict. Applies the stored median imputation. Returns a DataFrame with columns:
- `segment_idx` (int32)
- `quality_pred` (int: 0/1/2)
- `quality_label` (str: "clean"/"noisy_ok"/"bad")
- `p_clean` (float32)
- `p_noisy_ok` (float32)
- `p_bad` (float32)

### 3. Mask function
```python
def get_bad_segment_mask(predictions_df: pd.DataFrame, include_noisy: bool = False) -> set[int]:
```

Returns the set of `segment_idx` values predicted as `bad`. If `include_noisy=True`, also includes `noisy_ok` segments. Used by downstream training scripts to filter the beat feature matrix.

### 4. CLI
```
python ecgclean/models/segment_quality.py train \
  --segment-features data/processed/segment_features.parquet \
  --segments data/processed/segments.parquet \
  --output models/segment_quality_v1.joblib

python ecgclean/models/segment_quality.py predict \
  --segment-features data/processed/segment_features.parquet \
  --model models/segment_quality_v1.joblib \
  --output data/processed/segment_quality_preds.parquet
```

## Requirements
- Type hints and docstrings on all functions
- `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `joblib` only
- Temporal split is mandatory — add an assertion that raises if shuffle=True is passed accidentally
- If the training set has fewer than 10 segments of any class, print a warning but continue
- All saved artifacts must be versioned: include `trained_at` timestamp and `feature_columns` list in the saved dict so that inference failures from feature mismatch are caught at load time with a clear error message

## Files available in working directory
- `data/processed/segment_features.parquet` — output of segment_features.py
- `data/processed/segments.parquet` — output of data_pipeline.py
- `ecgclean/features/segment_features.py`
- `ecg_artifact_pipeline_v2_FINAL.md`

Write the complete file. Do not abbreviate or use placeholder comments. Every function must be fully implemented.
