# Pipeline Recovery — 2026-04-27

## What Changed

### Features Dropped (leakage / data quality)
| Feature | Reason |
|---|---|
| `artifact_fraction` | Counted `label=="artifact"` beats — unannotated beats default to "clean", so this was label-leaky |
| `f_imf_entropy`, `f_imf_mean`, `f_imf_variance` | EMD-derived features had numerical instability; dropped permanently |
| `segment_artifact_fraction`, `segment_clean_beat_count` | Same leakage as `artifact_fraction` |
| `segment_quality_pred` | Stage 0 output used as Stage 1 input — circular dependency |
| `sqi_qrs` filter | Previously filtered on `label=="clean"` — changed to `~hard_filtered` only |

### Features Added
| Feature | Description |
|---|---|
| `segment_zcr` | Zero-crossing rate of bandpass-filtered ECG — measures noise/irregularity |
| `segment_spectral_entropy` | Shannon entropy of Welch PSD — high = broadband noise |
| `segment_qrs_density` | Detected peaks vs expected at `HR_MODAL_LOW_BPM` — low = missed beats |
| `segment_flatline_fraction` | Fraction of 1-second windows with std < threshold — detects disconnection |
| `segment_amplitude_range` | p99 − p1 ADU — robust amplitude measure |

### Sample Rate Lock
All comments, docstrings, and help strings updated from 125 Hz → 130 Hz. `WINDOW_SIZE_SAMPLES=130`, `QRS_WINDOW_SAMPLES=65` derived from `SAMPLE_RATE_HZ=130`.

## Model Versions

| Artifact | Path | Schema |
|---|---|---|
| Stage 0 v2 | `Models/segment_quality_v2.joblib` | 23 signal-derived features |
| Stage 1 v6 | `Models/beat_tabular_v6.joblib` | 37 signal-derived features |
| Predictions | `Data/Processed/beat_tabular_preds.parquet` | `p_artifact` + `p_artifact_tabular` + `predicted_artifact` + `uncertainty_tabular` |

## Validation Numbers

| Model | PR-AUC | ROC-AUC | Notes |
|---|---|---|---|
| v1 (old, 40 features, leaky) | 0.7885 | 0.9875 | Trained on leaky schema |
| v6 (new, 37 features, clean) | 0.6686 | 0.9924 | Trained on signal-derived only; early stopping at iter 5 |

## Deployment Artifact
The HRV downstream analysis should consume:
- **`Data/Processed/beat_tabular_preds.parquet`** — one row per beat (58,569,724 rows)
- Key columns: `peak_id`, `p_artifact` (float32, [0,1]), `predicted_artifact` (bool), `uncertainty_tabular` (float32)
- Predicted artifact prevalence: 348,981 / 58,569,724 = **0.60%** at threshold=0.3472

## Reproducibility
Reproduce the full pipeline from any future state with:
```bash
source /Users/tannereddy/.envs/hrv/bin/activate
bash Scripts/utils/run_pipeline_logged.sh \
  --processed-dir Data/Processed \
  --segment-quality-model Models/segment_quality_v2.joblib \
  --beat-model Models/beat_tabular_v6.joblib
```
(Steps 1–3 can be skipped with fresh data by passing pre-existing peaks.parquet/labels.parquet.)
