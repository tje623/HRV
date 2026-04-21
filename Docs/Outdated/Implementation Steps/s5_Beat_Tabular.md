You are implementing Step 5 of an ECG artifact detection pipeline. Your task is to write one complete, production-ready Python file: `ecgclean/models/beat_artifact_tabular.py`.

## Context

This is Stage 1 — the primary beat-level artifact classifier. It is a LightGBM binary classifier trained on the extended beat feature matrix. This model's output (`p_artifact_tabular`) feeds directly into the ensemble and active learning sampler.

This is a year-long single-lead ECG (Polar H10, ~256Hz) from a POTS patient with high vagal tone. Class imbalance is severe: approximately 170k clean beats vs 2.2k artifact beats (~1.3% positive rate). PR-AUC on the artifact class is the primary evaluation metric — not accuracy, not ROC-AUC.

## What this file must implement

### 1. Training function
```python
def train(
    beat_features_path: str,
    labels_path: str,
    segment_quality_preds_path: str,
    output_model_path: str,
    val_fraction: float = 0.2,
    exclude_bad_segments: bool = True,
    random_seed: int = 42
) -> dict:
```

**Data preparation:**
- Load `beat_features.parquet` and `labels.parquet`
- Join segment quality predictions; if `exclude_bad_segments=True`, drop all beats from segments predicted `bad`
- Binary label: `label == "artifact"` → 1, all others → 0
- Also exclude `hard_filtered == True` beats from training (they are already confirmed artifacts; training on them adds noise)
- Keep `hard_filtered` beats in the evaluation set to verify the model agrees with hard filters

**Split strategy:** Split by `segment_idx` — all beats from earlier segments go to train, later segments to val. Never random beat-level splits. Add an assertion to enforce this.

**Class imbalance:** Use `scale_pos_weight = (# negative samples) / (# positive samples)` in LightGBM params.

**LightGBM parameters:**
```python
params = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": computed_from_data,
    "random_state": random_seed,
    "n_jobs": -1,
    "verbose": -1
}
```

Use early stopping with `callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]`.

**NaN handling:** Impute with training-set column medians. Store medians in saved artifact.

**Threshold tuning:** After training, compute the precision-recall curve on the validation set. Find the threshold that maximizes F1 for the artifact class. Store this as `optimal_threshold` in the saved artifact. Also compute and store thresholds for precision=0.90 and recall=0.90 as alternative operating points.

**Evaluation metrics (on validation set):**
- PR-AUC (artifact class) — primary metric
- ROC-AUC
- F1, precision, recall at optimal threshold
- F1, precision, recall at precision=0.90 threshold
- F1, precision, recall at recall=0.90 threshold
- Confusion matrix at optimal threshold
- Top 20 feature importances by gain
- Calibration: mean predicted probability vs actual positive rate in 10 equal-width probability bins

Save model artifact using `joblib.dump`:
```python
{
    "model": lgbm_classifier,
    "feature_columns": list,
    "train_medians": dict,
    "optimal_threshold": float,
    "threshold_at_precision_90": float,
    "threshold_at_recall_90": float,
    "scale_pos_weight": float,
    "val_metrics": dict,
    "trained_at": ISO_timestamp
}
```

### 2. Inference function
```python
def predict(
    beat_features: pd.DataFrame,
    model_path: str,
    threshold: float | None = None
) -> pd.DataFrame:
```

If `threshold` is None, use the stored `optimal_threshold`. Returns DataFrame with:
- `peak_id` (int64)
- `p_artifact_tabular` (float32)
- `predicted_artifact` (bool) — True if `p_artifact_tabular >= threshold`

### 3. Uncertainty scoring
```python
def get_uncertainty_scores(predictions_df: pd.DataFrame) -> pd.DataFrame:
```

Adds column `uncertainty_tabular = 1 - 2 * |p_artifact_tabular - 0.5|` (1.0 = maximally uncertain, 0.0 = maximally confident). Used by the active learning sampler.

### 4. CLI
```
python ecgclean/models/beat_artifact_tabular.py train \
  --beat-features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --output models/beat_tabular_v1.joblib

python ecgclean/models/beat_artifact_tabular.py predict \
  --beat-features data/processed/beat_features.parquet \
  --model models/beat_tabular_v1.joblib \
  --output data/processed/beat_tabular_preds.parquet
```

## Requirements
- Type hints and docstrings on all functions
- `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `joblib` only
- Temporal split by segment_idx is mandatory — assert this
- PR-AUC must be printed prominently in training summary — not buried
- Print a warning if the artifact class has fewer than 100 training examples
- The calibration output should print as a table: `[bin_range]: predicted=X.XX, actual=X.XX, n=NNNN`
- Feature importance printout must show the top 20 features with their gain scores

## Files available in working directory
- `data/processed/beat_features.parquet` — output of beat_features.py
- `data/processed/labels.parquet` — output of physio_constraints.py
- `data/processed/segment_quality_preds.parquet` — output of segment_quality.py predict
- `ecgclean/models/segment_quality.py`
- `ecg_artifact_pipeline_v2_FINAL.md`

Write the complete file. Do not abbreviate or use placeholder comments. Every function must be fully implemented.
