# Prompt E Summary — 2026-04-28

1. **Root cause confirmed**: `qrs_corr_to_template` (per-segment template via `label=="clean"`) leaked labels; gain 3,005,120 vs #2 at 35,931 (56×); caused iter-5 early stopping in v6.
2. **beat_features.py**: deleted `_build_segment_templates` and `qrs_corr_to_template`; `_compute_qrs_similarity` now produces only `qrs_corr_prev` / `qrs_corr_next` (no label dependency).
3. **segment_features.py**: `segment_amplitude_range` switched from p99−p1 → p95−p5 (p99−p1 max was 20× median due to ADC saturation spikes).
4. **global_templates.py build**: 180,296 reviewed-clean beats (`reviewed=True AND label=="clean" AND ~hard_filtered`) used to build `global_template.joblib`; run with `--chunk-segments 1` to avoid 80 GB RAM spike.
5. **global_templates.py correlate**: `global_corr_clean` computed for all 58,569,724 beats in 2,924 chunks; saved to `global_template_features.parquet`.
6. **Separation verified**: reviewed-clean median `global_corr_clean` = +0.93; reviewed-artifact median = −0.18; no label contamination.
7. **beat_artifact_tabular.py**: `--global-template-features` added to both train and predict paths; predict loads GTF once and joins each streaming batch; raises if model expects `global_corr_clean` but flag is missing.
8. **run_pipeline_logged.sh**: steps 6b (global_templates build) and 6c (correlate) inserted; steps 7/8 pass `--global-template-features`.
9. **LightGBM fix**: lr 0.05→0.01 (beat-specific), metric order `[auc, binary_logloss]` (early stop on logloss not AUC), feature_fraction 0.8→0.4 (prevents global_corr_clean saturation).
10. **beat_tabular_v8 training** (full dataset, 37 features): best iteration 48 ✓ (target >30); #1/#2 ratio 1.56× ✓ (target <5×); `global_corr_clean` ranked #3 behind `window_energy_ratio` (#1) and `rr_local_sd_5` (#2); PR-AUC=0.4901, ROC-AUC=0.9808.
11. **Full-dataset predictions**: 365,114 / 58,569,724 flagged = **0.62%** at threshold=0.3690; max p_artifact=0.4444 (model conservative; probabilities compressed by low training artifact rate 1.83%).
12. **Deployment**: `Data/Processed/beat_tabular_preds.parquet` (58,569,724 rows); production model `Models/beat_tabular_v8.joblib`; Stage 0 unchanged at `Models/segment_quality_v2.joblib`.
