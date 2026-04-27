# Prompt C — Retrain Stage 0 + Stage 1 on Subset, Validate, Decide Forward

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Stage 0 (segment quality) and Stage 1 (beat tabular) on the smoke_test subset under the cleaned feature contract, produce an honest PR-AUC comparison vs baselines, and recommend whether the model meaningfully beats the rule-based baseline.

**Architecture:**
- Python venv: `/Users/tannereddy/.envs/hrv/bin/activate` — the only env with scipy + lightgbm + joblib.
- All scripts invoked as `python Scripts/models/...` with venv active.
- Stage 0 uses `segment_features.parquet` (23 columns, 35808 rows); labeled subset is sparse (261 non-unknown segments: 148 clean, 112 noisy_ok, 1 bad) — results on Stage 0 will be limited by this.
- Stage 1 uses `beat_features.parquet` (37 columns, ~3.77M beats); reviewed subset is 22,486 beats (857 artifact, 20,936 clean/other reviewed).
- Both scripts already implement: reviewed filter, temporal split, in_bad_region exclusion, interpolated exclusion, hard_filtered exclusion from training.
- Baselines computed from a short Python analysis script (no new model files).

**Tech Stack:** LightGBM 4.6, Python 3.10, pandas, numpy, scipy, scikit-learn, pyarrow

**Key data facts:**
- `Data/Subsets/smoke_test/Processed/` — working directory for all I/O
- `beat_features.parquet` — MISSING, must be produced first
- `segment_features.parquet` — present, 35808×23
- `segments.parquet` — quality_label: 148 clean, 112 noisy_ok, 1 bad, 35176 unknown
- `labels.parquet` — reviewed=True: 857 artifact, 20936 clean, 555 missed_original, 138 interpolated
- `global_corr_clean` does NOT exist in beat_features — Baseline 2 is skipped
- `Models/` lives at `/Volumes/xHRV/Models/`

---

### Task 1: Produce beat_features.parquet

**Files:**
- Run: `Scripts/features/beat_features.py`
- Produces: `Data/Subsets/smoke_test/Processed/beat_features.parquet`
- Audit log: `Docs/run_logs/<timestamp>/`

- [ ] **Step 1: Run beat_features via the pipeline logger (which activates the venv)**

```bash
bash Scripts/utils/run_pipeline_logged.sh --subset smoke_test
```

But this will fail at segment_quality_predict (expected). Instead, run beat_features directly with the venv:

```bash
source /Users/tannereddy/.envs/hrv/bin/activate && \
python Scripts/features/beat_features.py \
  --processed-dir Data/Subsets/smoke_test/Processed \
  --segment-quality-preds /dev/null \
  --output Data/Subsets/smoke_test/Processed/beat_features.parquet \
  2>&1 | tee Docs/run_logs/beat_features_smoke.log
```

NOTE: `--segment-quality-preds /dev/null` is needed only if it's required; check `--help` first. If the arg is optional, omit it.

- [ ] **Step 2: Verify beat_features.parquet was written**

```bash
source /Users/tannereddy/.envs/hrv/bin/activate && python - <<'EOF'
import pandas as pd
df = pd.read_parquet("Data/Subsets/smoke_test/Processed/beat_features.parquet")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
assert df.shape[1] == 37, f"Expected 37 columns, got {df.shape[1]}"
print("✓ beat_features OK")
EOF
```

Expected: 3773379 rows × 37 columns.

- [ ] **Step 3: Commit**

```bash
git add Data/Subsets/  # only if not gitignored
git status  # beat_features.parquet is in Data/Subsets/ which is .gitignored — nothing to add
```

---

### Task 2: Train Stage 0 (segment_quality) on smoke_test

**Files:**
- Modify: `Scripts/models/segment_quality.py` — stale docstring says "22 feature columns"; update to "23 feature columns" only if it causes confusion. The logic is dynamic — no code change needed.
- Produces: `Models/smoke_test_segment_quality.joblib`

The `train()` function already drops unknown labels (they don't appear in `LABEL_ENCODER`, so they map to NaN and are dropped at line ~237). This satisfies the "filter to reviewed only" requirement.

- [ ] **Step 1: Confirm the reviewed-only filter works as expected**

```bash
source /Users/tannereddy/.envs/hrv/bin/activate && python - <<'EOF'
import pandas as pd
from Scripts.models.segment_quality import LABEL_ENCODER  # just check
segs = pd.read_parquet("Data/Subsets/smoke_test/Processed/segments.parquet")
known = segs[segs["quality_label"].isin(LABEL_ENCODER.keys())]
print(f"Labeled (non-unknown): {len(known)}")
print(known["quality_label"].value_counts())
EOF
```

Expected: 261 non-unknown (148 clean, 112 noisy_ok, 1 bad).

- [ ] **Step 2: Run Stage 0 training**

```bash
source /Users/tannereddy/.envs/hrv/bin/activate && \
python Scripts/utils/run_audit.py \
  --name stage0_train \
  --log-dir Docs/run_logs/prompt_c \
  -- \
  python Scripts/models/segment_quality.py train \
    --segment-features Data/Subsets/smoke_test/Processed/segment_features.parquet \
    --segments Data/Subsets/smoke_test/Processed/segments.parquet \
    --output Models/smoke_test_segment_quality.joblib \
    --val-fraction 0.2
```

- [ ] **Step 3: Check feature importances output**

Inspect the printed "Top 15 Features" block from stdout. Confirm:
- No `artifact_fraction`, `f_imf_entropy`, `f_imf_mean`, `f_imf_variance` appear
- Top features include signal-derived ones: `sqi_qrs`, `segment_zcr`, `segment_spectral_entropy`, `segment_qrs_density`, `beat_count`, `rr_roughness_*`, etc.

If ANY label-derived feature appears → STOP and surface it.

- [ ] **Step 4: Verify model file**

```bash
source /Users/tannereddy/.envs/hrv/bin/activate && python - <<'EOF'
import joblib, os
path = "Models/smoke_test_segment_quality.joblib"
assert os.path.exists(path), "Model not found!"
artifact = joblib.load(path)
print("feature_columns:", artifact["feature_columns"])
print("trained_at:", artifact["trained_at"])
print("macro_f1:", artifact["val_metrics"].get("macro_f1"))
print(f"Model size: {os.path.getsize(path)/1024:.1f} KB")
print("✓ Model artifact OK")
EOF
```

---

### Task 3: Predict Stage 0 over entire subset

**Files:**
- Produces: `Data/Subsets/smoke_test/Processed/segment_quality_preds.parquet`

- [ ] **Step 1: Run Stage 0 predict**

```bash
source /Users/tannereddy/.envs/hrv/bin/activate && \
python Scripts/utils/run_audit.py \
  --name stage0_predict \
  --log-dir Docs/run_logs/prompt_c \
  -- \
  python Scripts/models/segment_quality.py predict \
    --segment-features Data/Subsets/smoke_test/Processed/segment_features.parquet \
    --model Models/smoke_test_segment_quality.joblib \
    --output Data/Subsets/smoke_test/Processed/segment_quality_preds.parquet
```

- [ ] **Step 2: Report predicted class distribution and confusion matrix**

```bash
source /Users/tannereddy/.envs/hrv/bin/activate && python - <<'EOF'
import pandas as pd

preds = pd.read_parquet("Data/Subsets/smoke_test/Processed/segment_quality_preds.parquet")
segs  = pd.read_parquet("Data/Subsets/smoke_test/Processed/segments.parquet")

print("=== Predicted distribution (all 35,808 segments) ===")
print(preds["quality_label"].value_counts())
pct = preds["quality_label"].value_counts(normalize=True).mul(100).round(1)
print(pct)

# Confusion matrix on labeled subset only
labeled = segs[segs["quality_label"].isin(["clean", "noisy_ok", "bad"])]
merged = labeled.merge(preds[["segment_idx", "quality_label"]], on="segment_idx", suffixes=("_true", "_pred"))
print("\n=== Confusion matrix (labeled subset, n={}) ===".format(len(merged)))
cm = pd.crosstab(merged["quality_label_true"], merged["quality_label_pred"], margins=True)
print(cm)
EOF
```

---

### Task 4: Train Stage 1 (beat_artifact_tabular) on smoke_test

**Files:**
- Produces: `Models/smoke_test_beat_tabular.joblib`
- Verify: existing exclusion logic (reviewed filter, in_bad_region, interpolated, temporal split)

- [ ] **Step 1: Verify existing exclusion logic is present**

Read lines 382-445 of `Scripts/models/beat_artifact_tabular.py`. Confirm:
- Bad segment exclusion: line ~387 (`quality_label == "bad"`)
- in_bad_region exclusion: line ~404
- Interpolated exclusion: line ~412
- Reviewed filter: line ~423
- Temporal split: line ~447
All 5 should already be present from the April 20 overhaul (confirmed in Prompt B review).

- [ ] **Step 2: Run Stage 1 training**

```bash
source /Users/tannereddy/.envs/hrv/bin/activate && \
python Scripts/utils/run_audit.py \
  --name stage1_train \
  --log-dir Docs/run_logs/prompt_c \
  -- \
  python Scripts/models/beat_artifact_tabular.py train \
    --beat-features Data/Subsets/smoke_test/Processed/beat_features.parquet \
    --labels Data/Subsets/smoke_test/Processed/labels.parquet \
    --segment-quality-preds Data/Subsets/smoke_test/Processed/segment_quality_preds.parquet \
    --output Models/smoke_test_beat_tabular.joblib \
    --val-fraction 0.2
```

- [ ] **Step 3: Capture training output**

From stdout, note:
- PR-AUC (primary metric)
- ROC-AUC
- n_train / n_val reviewed beats
- Artifact prevalence in val set (should be 1-3%)
- Top feature importances (top 5 must be signal-derived)

- [ ] **Step 4: Verify model file**

```bash
source /Users/tannereddy/.envs/hrv/bin/activate && python - <<'EOF'
import joblib, os
path = "Models/smoke_test_beat_tabular.joblib"
assert os.path.exists(path), "Model not found!"
artifact = joblib.load(path)
print("PR-AUC:", artifact["val_metrics"].get("pr_auc"))
print("ROC-AUC:", artifact["val_metrics"].get("roc_auc"))
print("Top 5 features:")
for e in artifact["val_metrics"]["top_features"][:5]:
    print(f"  {e['feature']}: {e['gain']:.1f}")
print(f"Size: {os.path.getsize(path)/1024:.0f} KB")
print("✓ Model artifact OK")
EOF
```

---

### Task 5: Compute baselines + comparison table

**Files:**
- Create: `Scripts/utils/eval_baselines.py` — standalone script, no new classes

The script computes three baselines on the validation reviewed subset:
1. Random — PR-AUC = prevalence
2. global_corr_clean threshold — SKIPPED (feature does not exist)
3. Rule-based: `segment_zcr > 0.5 OR segment_qrs_density < 0.3 OR segment_spectral_entropy > T` — threshold T tuned on train half

All evaluation is beat-level (each beat inherits its segment's signal-quality features via a join).

- [ ] **Step 1: Write eval_baselines.py**

```python
#!/usr/bin/env python3
"""
Scripts/utils/eval_baselines.py — Baseline PR-AUC comparisons for Prompt C.

Computes baselines on the validation reviewed subset and prints comparison table.
Run from /Volumes/xHRV with the hrv venv active.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib

PROCESSED = Path("Data/Subsets/smoke_test/Processed")
MODELS    = Path("Models")

def load_val_reviewed(val_fraction: float = 0.2) -> pd.DataFrame:
    """Return the reviewed beat rows from the temporal validation segment split."""
    labels  = pd.read_parquet(PROCESSED / "labels.parquet")
    peaks   = pd.read_parquet(PROCESSED / "peaks.parquet", columns=["peak_id", "segment_idx"])
    seg_feat = pd.read_parquet(PROCESSED / "segment_features.parquet")
    if seg_feat.index.name == "segment_idx":
        seg_feat = seg_feat.reset_index()

    # Restrict to reviewed beats
    reviewed = labels[labels["reviewed"] | (labels["label"] == "artifact")].copy()
    reviewed["target"] = (reviewed["label"] == "artifact").astype(int)
    # Drop interpolated
    reviewed = reviewed[reviewed["label"] != "interpolated"]

    # Join segment_idx
    merged = reviewed.merge(peaks, on="peak_id", how="inner")

    # Temporal split — same logic as beat_artifact_tabular.py
    unique_segs = sorted(merged["segment_idx"].unique())
    n_train = max(1, int(len(unique_segs) * (1 - val_fraction)))
    val_segs = set(unique_segs[n_train:])
    val_df = merged[merged["segment_idx"].isin(val_segs)].copy()

    # Join segment features for rule-based baseline
    val_df = val_df.merge(
        seg_feat[["segment_idx", "segment_zcr", "segment_qrs_density", "segment_spectral_entropy"]],
        on="segment_idx",
        how="left",
    )
    return val_df


def tune_spectral_entropy_threshold(train_val_fraction: float = 0.2) -> float:
    """Tune spectral entropy threshold on training half only."""
    labels  = pd.read_parquet(PROCESSED / "labels.parquet")
    peaks   = pd.read_parquet(PROCESSED / "peaks.parquet", columns=["peak_id", "segment_idx"])
    seg_feat = pd.read_parquet(PROCESSED / "segment_features.parquet")
    if seg_feat.index.name == "segment_idx":
        seg_feat = seg_feat.reset_index()

    reviewed = labels[labels["reviewed"] | (labels["label"] == "artifact")].copy()
    reviewed["target"] = (reviewed["label"] == "artifact").astype(int)
    reviewed = reviewed[reviewed["label"] != "interpolated"]
    merged = reviewed.merge(peaks, on="peak_id", how="inner")

    unique_segs = sorted(merged["segment_idx"].unique())
    n_train = max(1, int(len(unique_segs) * (1 - train_val_fraction)))
    train_segs = set(unique_segs[:n_train])
    train_df = merged[merged["segment_idx"].isin(train_segs)].copy()
    train_df = train_df.merge(
        seg_feat[["segment_idx", "segment_spectral_entropy"]],
        on="segment_idx", how="left"
    )

    # Grid search: try thresholds for segment_spectral_entropy upper bound
    # Higher entropy = more noisy = more likely artifact
    best_t, best_auc = 6.0, 0.0
    for t in np.linspace(3.0, 6.5, 35):
        scores = (train_df["segment_spectral_entropy"] > t).astype(float).values
        if scores.sum() == 0:
            continue
        auc = average_precision_score(train_df["target"].values, scores)
        if auc > best_auc:
            best_auc, best_t = auc, t
    return float(best_t)


def main() -> None:
    sys.path.insert(0, "Scripts")

    print("Loading validation reviewed subset...")
    val_df = load_val_reviewed(val_fraction=0.2)

    y_val = val_df["target"].values
    prevalence = float(y_val.mean())
    n_total = len(y_val)
    n_artifact = int(y_val.sum())

    print(f"\nValidation reviewed subset: {n_total:,} beats, {n_artifact} artifact ({prevalence*100:.2f}%)")

    # ── Baseline 1: Random (PR-AUC = prevalence) ─────────────────────────
    random_pr_auc = prevalence
    random_roc_auc = 0.5

    # ── Baseline 2: global_corr_clean — SKIPPED (feature absent) ─────────
    b2_pr_auc = float("nan")
    b2_roc_auc = float("nan")

    # ── Baseline 3: Rule-based ─────────────────────────────────────────────
    print("Tuning spectral entropy threshold on training split...")
    se_thresh = tune_spectral_entropy_threshold(0.2)
    print(f"  Best spectral_entropy threshold: {se_thresh:.3f}")

    rule_scores = (
        (val_df["segment_zcr"] > 0.5) |
        (val_df["segment_qrs_density"] < 0.3) |
        (val_df["segment_spectral_entropy"] > se_thresh)
    ).astype(float).values

    if rule_scores.sum() > 0 and (y_val.sum() > 0):
        rule_pr_auc  = float(average_precision_score(y_val, rule_scores))
        rule_roc_auc = float(roc_auc_score(y_val, rule_scores))
    else:
        rule_pr_auc  = float("nan")
        rule_roc_auc = float("nan")

    # ── LGBM Stage 1 from artifact: load model predictions ───────────────
    model_path = MODELS / "smoke_test_beat_tabular.joblib"
    beat_feat  = pd.read_parquet(PROCESSED / "beat_features.parquet")
    if beat_feat.index.name == "peak_id":
        beat_feat = beat_feat.reset_index()

    artifact = joblib.load(model_path)
    clf           = artifact["model"]
    feature_cols  = artifact["feature_columns"]
    train_medians = artifact["train_medians"]

    # Restrict to val_df peak_ids
    val_peak_ids = set(val_df["peak_id"].values)
    val_feat = beat_feat[beat_feat["peak_id"].isin(val_peak_ids)].copy()

    # Impute
    for col in feature_cols:
        if val_feat[col].isna().any():
            val_feat[col] = val_feat[col].fillna(train_medians.get(col, 0.0))

    # Predict probabilities
    proba = clf.predict_proba(val_feat[feature_cols])[:, 1]

    # Align to val_df order
    prob_series = pd.Series(proba, index=val_feat["peak_id"].values)
    y_lgbm = prob_series.reindex(val_df["peak_id"].values).values

    valid_mask = ~np.isnan(y_lgbm)
    if valid_mask.sum() > 0 and y_val[valid_mask].sum() > 0:
        lgbm_pr_auc  = float(average_precision_score(y_val[valid_mask], y_lgbm[valid_mask]))
        lgbm_roc_auc = float(roc_auc_score(y_val[valid_mask], y_lgbm[valid_mask]))
    else:
        lgbm_pr_auc  = float("nan")
        lgbm_roc_auc = float("nan")

    # ── Print comparison table ─────────────────────────────────────────────
    rows = [
        ("random baseline",              random_pr_auc,  random_roc_auc),
        ("rule-based on new features",   rule_pr_auc,    rule_roc_auc),
        ("global_corr_clean threshold",  b2_pr_auc,      b2_roc_auc),
        ("smoke_test beat tabular LGBM", lgbm_pr_auc,    lgbm_roc_auc),
    ]

    print(f"\n{'Model':<32} {'Val PR-AUC':>12} {'Val ROC-AUC':>12}")
    print("-" * 58)
    for name, pr, roc in rows:
        pr_str  = f"{pr:.4f}"  if not np.isnan(pr)  else "n/a"
        roc_str = f"{roc:.4f}" if not np.isnan(roc) else "n/a"
        print(f"{name:<32} {pr_str:>12} {roc_str:>12}")
    print("-" * 58)

    # ── Recommendation ─────────────────────────────────────────────────────
    print("\n=== RECOMMENDATION ===")
    if not np.isnan(lgbm_pr_auc) and not np.isnan(rule_pr_auc):
        delta = lgbm_pr_auc - rule_pr_auc
        if delta > 0.05:
            print(f"LGBM beats rule-based by {delta:.4f} PR-AUC. "
                  "Stage 1 is healthy → proceed to full dataset (Prompt D).")
        elif delta > 0.0:
            print(f"LGBM marginally beats rule-based (+{delta:.4f}). "
                  "Consider whether complexity is worth it before scaling to full dataset.")
        else:
            print(f"Rule-based matches or outperforms LGBM (delta={delta:.4f}). "
                  "Consider replacing Stage 1 with the rule-based detector — "
                  "simpler, faster, no retraining required.")
    else:
        print("Insufficient data to compare (NaN metrics). Check val set size and artifact count.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run eval_baselines.py**

```bash
source /Users/tannereddy/.envs/hrv/bin/activate && \
python Scripts/utils/eval_baselines.py 2>&1 | tee Docs/run_logs/prompt_c/eval_baselines.log
```

- [ ] **Step 3: Capture and report the comparison table**

Copy the printed table and recommendation into the session summary.

---

### Task 6: Commit all new artifacts and scripts

- [ ] **Step 1: Stage new files**

```bash
git add Scripts/utils/eval_baselines.py \
        Docs/superpowers/plans/2026-04-27-prompt-c-retrain-stage0-stage1.md \
        Docs/run_logs/prompt_c/
# Do NOT add Models/ binary files unless tracking them is intentional
```

- [ ] **Step 2: Check what's staged**

```bash
git diff --cached --stat
```

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
Train Stage 0 + Stage 1 on smoke_test subset, compute baseline comparisons

- Run beat_features to produce beat_features.parquet (37 cols, ~3.77M beats)
- Stage 0 smoke_test_segment_quality.joblib: trained on 261 labeled segments
  (148 clean, 112 noisy_ok, 1 bad); unknown segments filtered automatically
- Stage 0 predict: distribution over 35,808 segments
- Stage 1 smoke_test_beat_tabular.joblib: reviewed=True filter, temporal split,
  in_bad_region + interpolated + hard_filtered exclusions all confirmed
- eval_baselines.py: random / rule-based / LGBM comparison table
- Recommendation: see run log

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Push**

```bash
git push origin main
```

---

## Self-Review

**Spec coverage check:**
1. ✅ Train Stage 0, reviewed-only filter, save to smoke_test_segment_quality.joblib — Task 2
2. ✅ Predict Stage 0, report distribution + confusion matrix — Task 3
3. ✅ Train Stage 1, verified exclusion logic, temporal split, save to smoke_test_beat_tabular.joblib — Task 4
4. ✅ Baseline 1 (random) — Task 5
5. ✅ Baseline 2 (global_corr_clean) — SKIPPED with note (feature absent) — Task 5
6. ✅ Baseline 3 (rule-based, threshold tuned on train half) — Task 5
7. ✅ Comparison table — Task 5
8. ✅ Top 5 importances all signal-derived (verification in Task 2 step 3 and Task 4 step 3)
9. ✅ Artifact prevalence check (1-3%) — Task 4 step 3
10. ✅ Model files non-empty — Task 2 step 4, Task 4 step 4
11. ✅ Recommendation — Task 5 step 3

**Placeholder scan:** No TBDs, no "add validation" phrases, all code shown.

**Type consistency:** `val_df["peak_id"]` consistent with `beat_feat["peak_id"]` reset_index pattern.
