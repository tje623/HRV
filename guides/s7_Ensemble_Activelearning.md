You are implementing Step 7 of an ECG artifact detection pipeline. Your task is to write three complete, production-ready Python files:
- `ecgclean/models/ensemble.py`
- `ecgclean/active_learning/sampler.py`
- `ecgclean/active_learning/annotation_queue.py`

## Context

At this point the pipeline has:
- Stage 0: segment quality predictions (`segment_quality_preds.parquet`)
- Stage 1: tabular GBM beat predictions (`beat_tabular_preds.parquet`)
- Stage 2a: CNN beat predictions (`beat_cnn_preds.parquet`)

This step fuses those predictions into a final ensemble score and then uses that score — plus disagreement between models — to drive an active learning loop that maximizes the value of future annotation effort.

---

## FILE 1: `ecgclean/models/ensemble.py`

### Primary function
```python
def fuse(
    tabular_preds: pd.DataFrame,
    cnn_preds: pd.DataFrame,
    alpha: float = 0.5
) -> pd.DataFrame:
```

Both input DataFrames have `peak_id` and their respective probability columns (`p_artifact_tabular`, `p_artifact_cnn`). Returns a DataFrame with:
- `peak_id` (int64)
- `p_artifact_tabular` (float32)
- `p_artifact_cnn` (float32)
- `p_artifact_ensemble` (float32): `alpha * p_artifact_tabular + (1 - alpha) * p_artifact_cnn`
- `disagreement` (float32): `|p_artifact_tabular - p_artifact_cnn|`
- `uncertainty_ensemble` (float32): `1 - 2 * |p_artifact_ensemble - 0.5|`
- `predicted_artifact` (bool): `p_artifact_ensemble >= threshold` where threshold defaults to 0.5

### Threshold tuning function
```python
def tune_alpha(
    tabular_preds: pd.DataFrame,
    cnn_preds: pd.DataFrame,
    labels_df: pd.DataFrame,
    alpha_grid: list[float] | None = None
) -> dict:
```

Searches `alpha_grid` (default: `[0.1, 0.2, ..., 0.9]`) for the alpha that maximizes PR-AUC of the artifact class on the provided labels. Returns dict with `best_alpha`, `best_pr_auc`, and a table of all alpha values and their PR-AUC scores. Print the full table.

### Meta-classifier (logistic regression on model outputs)
```python
def train_meta_classifier(
    tabular_preds: pd.DataFrame,
    cnn_preds: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_path: str
) -> dict:
```

Trains a logistic regression on `[p_artifact_tabular, p_artifact_cnn, disagreement]` as features and `label == "artifact"` as target. Uses `class_weight="balanced"`. Saves with joblib. Returns PR-AUC and F1 metrics. This is an alternative to the linear blend — use whichever scores higher on the validation set.

### CLI
```
python ecgclean/models/ensemble.py fuse \
  --tabular-preds data/processed/beat_tabular_preds.parquet \
  --cnn-preds data/processed/beat_cnn_preds.parquet \
  --output data/processed/ensemble_preds.parquet \
  --alpha 0.5

python ecgclean/models/ensemble.py tune-alpha \
  --tabular-preds data/processed/beat_tabular_preds.parquet \
  --cnn-preds data/processed/beat_cnn_preds.parquet \
  --labels data/processed/labels.parquet
```

---

## FILE 2: `ecgclean/active_learning/sampler.py`

This module selects which unlabeled beats to send for manual annotation next. It implements three complementary selection strategies that can be combined.

### Core function
```python
def select_annotation_candidates(
    ensemble_preds: pd.DataFrame,
    labels_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    segment_quality_preds: pd.DataFrame,
    n_candidates: int = 500,
    strategy: str = "combined",
    uncertainty_band: tuple[float, float] = (0.3, 0.7),
    disagreement_threshold: float = 0.3,
    al_iteration: int = 1
) -> pd.DataFrame:
```

`ensemble_preds` has columns from `ensemble.py fuse` output. `labels_df` has existing labels — beats already labeled should be excluded from candidates.

**`strategy` options:**

1. **`"margin"`**: Select beats where `p_artifact_ensemble` is within `uncertainty_band`. Pure uncertainty sampling.

2. **`"committee"`**: Select beats where `disagreement >= disagreement_threshold`. Model committee disagrees — these are the most informative.

3. **`"priority"`**: Select beats with highest `review_priority_score` from `physio_constraints.py` (already in labels_df). These are physiologically suspicious beats flagged by domain knowledge.

4. **`"combined"` (default)**: Merge all three pools, deduplicate by `peak_id`, then sort by a composite score:
   ```python
   composite = (
       0.4 * uncertainty_ensemble +
       0.4 * disagreement +
       0.2 * (review_priority_score / review_priority_score.max())
   )
   ```
   Return top `n_candidates` by composite score.

**Prioritization within any strategy:**
- Beats in segments predicted `noisy_ok` (not `bad`, not `clean`) get a 20% composite score bonus — they are the most borderline
- Beats with `pots_transition_candidate == True` get a 10% composite score bonus
- Beats from `bad` segments are **always excluded** regardless of strategy

**Output DataFrame columns:**
- `peak_id`, `segment_idx`, `p_artifact_ensemble`, `p_artifact_tabular`, `p_artifact_cnn`, `disagreement`, `uncertainty_ensemble`, `review_priority_score`, `composite_score`, `selection_strategy`, `al_iteration`

### Iteration tracking
```python
def record_labels(
    annotation_results: pd.DataFrame,
    labels_path: str,
    al_iteration: int
) -> None:
```

`annotation_results` has `peak_id` and `new_label` (str). Updates `labels.parquet` in-place: sets `label = new_label`, `al_iteration = al_iteration`, and `uncertainty_score` / `disagreement_score` from the candidate selection round. Appends; never drops existing rows.

### Iteration summary
```python
def get_iteration_summary(labels_df: pd.DataFrame) -> pd.DataFrame:
```

Returns a DataFrame summarizing each `al_iteration`:
- Count of labels added
- Label distribution (clean/artifact/etc.)
- Mean uncertainty_score and disagreement_score of selected beats

### CLI
```
python ecgclean/active_learning/sampler.py select \
  --ensemble-preds data/processed/ensemble_preds.parquet \
  --labels data/processed/labels.parquet \
  --segments data/processed/segments.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --n-candidates 500 \
  --strategy combined \
  --al-iteration 1 \
  --output data/processed/al_queue_iteration_1.parquet
```

---

## FILE 3: `ecgclean/active_learning/annotation_queue.py`

This module serializes the annotation candidate queue into a format that the existing annotation UI can consume, and reads back completed annotations.

### Export function
```python
def export_queue(
    candidates_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ecg_samples_df: pd.DataFrame,
    output_path: str,
    context_window_sec: float = 5.0
) -> None:
```

For each candidate beat, export a JSON file (one record per beat) with:
```json
{
  "peak_id": int,
  "timestamp_ns": int,
  "segment_idx": int,
  "current_label": str,
  "p_artifact_ensemble": float,
  "disagreement": float,
  "composite_score": float,
  "rr_prev_ms": float,
  "rr_next_ms": float,
  "context_ecg": [float, ...],
  "context_timestamps_ns": [int, ...],
  "r_peak_index_in_context": int
}
```

`context_ecg` is the raw ECG samples from `context_window_sec / 2` before to `context_window_sec / 2` after the R-peak timestamp. `r_peak_index_in_context` is the index into `context_ecg` where the R-peak falls.

Also export a summary CSV with one row per candidate and the same fields except `context_ecg`.

### Import function
```python
def import_completed_annotations(
    completed_csv_path: str,
    expected_peak_ids: list[int]
) -> pd.DataFrame:
```

Reads a CSV with columns `peak_id` and `label`. Validates that all `peak_id` values are in `expected_peak_ids`. Returns a DataFrame ready for `sampler.record_labels`. Raises with a clear error message if any `peak_id` is not in the expected set or if any `label` value is not in the valid label set: `{"clean", "artifact", "interpolated", "phys_event", "missed_original"}`.

### CLI
```
python ecgclean/active_learning/annotation_queue.py export \
  --candidates data/processed/al_queue_iteration_1.parquet \
  --peaks data/processed/peaks.parquet \
  --labels data/processed/labels.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --output data/annotation_queues/iteration_1/

python ecgclean/active_learning/annotation_queue.py import \
  --completed data/annotation_queues/iteration_1/completed.csv \
  --expected-ids data/processed/al_queue_iteration_1.parquet \
  --labels data/processed/labels.parquet \
  --al-iteration 1
```

---

## Requirements (all three files)
- Type hints and docstrings on all functions
- `pandas`, `numpy`, `scikit-learn`, `joblib`, `json` only (no torch in ensemble.py or sampler.py)
- `annotation_queue.py` must create output directories if they don't exist
- All three files must have a `__init__.py` in their respective directories that exports the primary functions
- The composite score formula in sampler.py must be documented with a comment explaining the weight rationale
- `record_labels` must be atomic: write to a temp file then rename, never partially overwrite the labels parquet

## Files available in working directory
- `data/processed/beat_tabular_preds.parquet` — from beat_artifact_tabular.py
- `data/processed/beat_cnn_preds.parquet` — from beat_artifact_cnn.py
- `data/processed/ensemble_preds.parquet` — from ensemble.py (if already run)
- `data/processed/labels.parquet`, `peaks.parquet`, `segments.parquet`, `ecg_samples.parquet`
- `data/processed/segment_quality_preds.parquet`
- `ecg_artifact_pipeline_v2_FINAL.md`

Write all three files completely. Do not abbreviate or use placeholder comments.
