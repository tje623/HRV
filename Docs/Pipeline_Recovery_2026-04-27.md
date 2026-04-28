# Pipeline Recovery — 2026-04-27 / Prompt E — 2026-04-28

## What Changed

### Prompt D (2026-04-27): Initial clean schema

#### Features Dropped (leakage / data quality)
| Feature | Reason |
|---|---|
| `artifact_fraction` | Counted `label=="artifact"` beats — unannotated beats default to "clean", so this was label-leaky |
| `f_imf_entropy`, `f_imf_mean`, `f_imf_variance` | EMD-derived features had numerical instability; dropped permanently |
| `segment_artifact_fraction`, `segment_clean_beat_count` | Same leakage as `artifact_fraction` |
| `segment_quality_pred` | Stage 0 output used as Stage 1 input — circular dependency |
| `sqi_qrs` filter | Previously filtered on `label=="clean"` — changed to `~hard_filtered` only |

#### Features Added (Prompt D)
| Feature | Description |
|---|---|
| `segment_zcr` | Zero-crossing rate of bandpass-filtered ECG — measures noise/irregularity |
| `segment_spectral_entropy` | Shannon entropy of Welch PSD — high = broadband noise |
| `segment_qrs_density` | Detected peaks vs expected at `HR_MODAL_LOW_BPM` — low = missed beats |
| `segment_flatline_fraction` | Fraction of 1-second windows with std < threshold — detects disconnection |
| `segment_amplitude_range` | p99 − p1 ADU — robust amplitude measure |

### Prompt E (2026-04-28): Fix residual leakage + global template

#### Root cause of v6 degradation
v6 trained with `qrs_corr_to_template` — a per-segment QRS template built via `_build_segment_templates`, which filtered on `label=="clean"`. Unannotated beats default to "clean", making the template contaminated. `qrs_corr_to_template` reached gain 3,005,120 vs #2 at 35,931 (56× gap), causing iteration-5 early stopping and 0.60% predicted prevalence.

#### Changes (Prompt E)
| Change | Detail |
|---|---|
| Removed `qrs_corr_to_template` | Deleted `_build_segment_templates` from `beat_features.py`; feature was leaking labels |
| Added `global_corr_clean` | Built by `global_templates.py` from `reviewed=True AND label=="clean" AND ~hard_filtered` — genuinely label-free |
| Fixed `segment_amplitude_range` | p99−p1 → p95−p5 to resist ADC saturation spikes (p99−p1 max was 20× median) |
| Added pipeline steps 6b/6c | `global_templates build` + `correlate` inserted before Stage 1 train/predict |
| LightGBM hyperparameters | lr 0.05→0.01 (beat-specific), metric order `[auc, binary_logloss]` (early-stop on logloss), feature_fraction 0.8→0.4 (prevents global_corr_clean saturation) |

### Sample Rate Lock
All comments, docstrings, and help strings updated from 125 Hz → 130 Hz. `WINDOW_SIZE_SAMPLES=130`, `QRS_WINDOW_SAMPLES=65` derived from `SAMPLE_RATE_HZ=130`.

---

## Model Versions

| Artifact | Path | Schema | Notes |
|---|---|---|---|
| Stage 0 v2 | `Models/segment_quality_v2.joblib` | 23 signal-derived features | Unchanged from Prompt D |
| Stage 1 v6 | `Models/beat_tabular_v6.joblib` | 37 features incl. leaky `qrs_corr_to_template` | Superseded by v8 |
| Stage 1 v8 | `Models/beat_tabular_v8.joblib` | 37 features: 36 beat + `global_corr_clean` | **Current production model** |
| Predictions | `Data/Processed/beat_tabular_preds.parquet` | `p_artifact` + `p_artifact_tabular` + `predicted_artifact` + `uncertainty_tabular` | From v8 |

---

## Validation Numbers

| Model | PR-AUC | ROC-AUC | Best iter | #1/#2 ratio | Notes |
|---|---|---|---|---|---|
| v1 (40 features, leaky) | 0.7885 | 0.9875 | — | — | Trained on leaky schema |
| v6 (37 features, per-seg template) | 0.6686 | 0.9924 | 5 | 56× | `qrs_corr_to_template` leaked labels; iter-5 stop |
| v8 (37 features, global template) | 0.4901 | 0.9808 | 48 | 1.56× | Clean; `window_energy_ratio` #1, `global_corr_clean` #3 |

**Global template separation** (verified clean, 180,296 reviewed-clean beats):
- Reviewed-clean median `global_corr_clean`: +0.93
- Reviewed-artifact median `global_corr_clean`: −0.18

---

## Deployment Artifact
The HRV downstream analysis should consume:
- **`Data/Processed/beat_tabular_preds.parquet`** — one row per beat (58,569,724 rows)
- Key columns: `peak_id`, `p_artifact` (float32, [0,1]), `predicted_artifact` (bool), `uncertainty_tabular` (float32)
- Predicted artifact prevalence: 365,114 / 58,569,724 = **0.62%** at threshold=0.3690
- Max predicted probability: 0.4444 (model is conservative; probabilities compressed near zero for clean beats)

---

## Reproducibility
Reproduce the full pipeline from any future state with:
```bash
source /Users/tannereddy/.envs/hrv/bin/activate
bash Scripts/utils/run_pipeline_logged.sh \
  --processed-dir Data/Processed \
  --segment-quality-model Models/segment_quality_v2.joblib \
  --beat-model Models/beat_tabular_v8.joblib
```
(Steps 1–3 can be skipped with fresh data by passing pre-existing peaks.parquet/labels.parquet.)

Note: steps 6b (`global_templates build`) and 6c (`global_templates correlate`) use `--chunk-segments 1` and `--chunk-segments 200` respectively to avoid loading the full `ecg_samples.parquet` into RAM. Default chunk size spans the full segment index range and causes ~80 GB memory usage.
