# Annotation Overhaul — Pipeline Update Design

**Date:** 2026-05-13
**Status:** Awaiting user review
**Scope:** Update the ECG artifact-detection pipeline to consume the overhauled annotation parquet inputs, and incorporate first-class handling of the `interpolate` artifact subtype.

---

## Background

The annotation inputs at `/Volumes/xHRV/Data/Annotations/INPUT/` have been rebuilt from scratch into three small, unambiguous parquet files. The pipeline was last updated against an older schema (commit `2beec92`) that expected columns no longer present. Several legacy concepts (`revisit` review status, the `partial` quality label, `is_added_peak`, `original_annotation`, `in_revisit_pile`, `has_bad_region`, `bad_region_count`, `phys_event_window`, `in_bad_region`) have been retired in favor of a leaner schema centered on a single `subtype` column.

Additionally, the new schema introduces a real distinction between two artifact subtypes:

- **spurious** — extra, superfluous r-peak that should simply be removed.
- **interpolate** — the flagged r-peak is mis-positioned, but a true r-peak truly belongs somewhere in that neighborhood.

Until now, the previous version of the pipeline carried a `label="interpolated"` value through to training, and both `beat_artifact_cnn.py` and `beat_artifact_tabular.py` filtered those rows out of training entirely. With ~101 interpolate examples now annotated, that exclusion is no longer warranted.

---

## Input Contract

The pipeline reads exactly three files from `/Volumes/xHRV/Data/Annotations/INPUT/`:

### `beats.parquet`
- `timestamp_ms` (int64)
- `segment_idx` (int32)
- `label` ∈ {`clean`, `artifact`, `physio`}
- `subtype` ∈ {`auto`, `added`, `manual`, `spurious`, `interpolate`}
- `beat_training_eligible` (bool)

Pre-conditions guaranteed upstream:
- Beats inside any bad-region time window have already been removed.
- Only training-eligible beats appear; beats from unreviewed / unusable segments are excluded.

### `segments.parquet`
- `segment_idx` (int32)
- `start_ms` (int64)
- `end_ms` (int64)
- `review_status` ∈ {`reviewed`, `unreviewed`}
- `segment_quality_label` ∈ {`usable`, `unclean`, `unusable`, `unknown`}
- `beat_training_eligible_segment` (bool)
- `segment_quality_training_eligible` (bool)

### `bad_regions.parquet`
- `segment_idx` (int32)
- `start_ms` (int64)
- `end_ms` (int64)

### Validation

`load_annotation_inputs()` fails loudly on any deviation from the above:
- Missing or extra columns.
- Unexpected values in `review_status`, `segment_quality_label`, `label`, or `subtype`.

No silent fallbacks. No legacy column lookups.

---

## Pipeline Outputs

Four canonical parquet outputs from `data_pipeline.py`:

### `ecg_samples.parquet`
Unchanged.

### `peaks.parquet`
Schema: `peak_id, timestamp_ms, source, segment_idx`.

Drop `is_added_peak`. The "this was a manually-added peak" fact is recoverable via `subtype=="added"` in `labels.parquet`.

### `labels.parquet`

Schema:
- `peak_id` (int64)
- `segment_idx` (int32)
- `label` ∈ {`clean`, `artifact`, `phys_event`, `unknown`}
- `subtype` ∈ {`auto`, `added`, `manual`, `spurious`, `interpolate`, `""`}
- `reviewed` (bool) — `True` iff the peak matched a beat in `beats.parquet` AND that beat had `beat_training_eligible=True`.
- `rr_prev_ms`, `rr_next_ms` (float32)
- All physio soft-flag columns added by `physio_constraints.py`:
  `hard_filtered`, `physio_implausible`, `pots_transition_candidate`,
  `tachy_transition_candidate`, `hr_suspicious_low`, `hr_suspicious_high`,
  `rr_suspicious_short`, `rr_suspicious_long`, `review_priority_score`.

Removed columns: `original_annotation`, `is_added_peak`, `in_bad_region`, `phys_event_window`. Reason for each:
- `original_annotation` — string version of the same info as `subtype`. Redundant.
- `is_added_peak` — fully recoverable from `subtype=="added"`.
- `in_bad_region` — bad-region beats are already scrubbed upstream from `beats.parquet`; unmatched auto-detected peaks already get `reviewed=False`, which is sufficient.
- `phys_event_window` — was a literal alias of `label=="phys_event"`; downstream consumers should test the label directly.

Mapping rules:
- `beats.label="clean"` → `labels.label="clean"`
- `beats.label="artifact"` → `labels.label="artifact"`
- `beats.label="physio"` → `labels.label="phys_event"`
- Auto-detected peaks not present in `beats.parquet` → `label="unknown"`, `subtype=""`, `reviewed=False`.

### `segments.parquet` (output)

Schema:
- `segment_idx` (int32)
- `start_ms`, `end_ms` (int64)
- `review_status` ∈ {`reviewed`, `unreviewed`}
- `quality_label` ∈ {`clean`, `noisy_ok`, `bad`, `unknown`}
- `segment_rmssd_ms` (float32) — RMSSD computed over all auto-detected peaks in the segment from `peaks.parquet`.
- `beat_training_eligible_segment` (bool)
- `segment_quality_training_eligible` (bool)

Mapping rules:
- `usable` + `segment_rmssd_ms < 50` → `quality_label="clean"`
- `usable` + `segment_rmssd_ms ≥ 50` → `quality_label="noisy_ok"`
- `unclean` → `quality_label="noisy_ok"`
- `unusable` → `quality_label="bad"`
- `unknown` → `quality_label="unknown"`

Removed columns: `in_revisit_pile`, `has_bad_region`, `bad_region_count`. The two `*_training_eligible` booleans already encode the relevant downstream decisions.

---

## Segment RMSSD computation

For each segment, gather all `peak_id`s in `peaks.parquet` whose timestamp falls within `[start_ms, end_ms]`. Sort by timestamp. Compute successive RR intervals (`diff(timestamp_ms)`). RMSSD is the square root of the mean of squared successive differences of *those* RR intervals.

If a segment has fewer than 3 peaks (so fewer than 2 RR intervals, so fewer than 1 successive difference), set `segment_rmssd_ms = NaN` and fall back to `quality_label="noisy_ok"` for any `usable` segment in that case. Rare edge case; preserves the conservative bias.

This is a one-pass computation slotted into `build_segments()` in `data_pipeline.py`.

**Open knob:** computing RMSSD over *all* auto-detected peaks (including artifacts) versus only clean+physio peaks. Defaulting to **all peaks** — simpler, no label dependency, and a segment whose auto-detection produced spurious peaks will correctly fall out of `clean` due to inflated RMSSD. If empirical results suggest this is too aggressive, the rule can be revisited.

---

## Interpolate logic

Per Approach A (binary artifact, no post-hoc cleanup yet):

1. **`subtype` flows end-to-end** — preserved in `labels.parquet`. No downstream model uses it as a training target. It is available for evaluation slicing.
2. **Stop filtering interpolate out of training.** Remove the `merged = merged[merged["label"] != "interpolated"]` block (and its accompanying log line) from:
   - `Scripts/models/beat_artifact_cnn.py` lines ~651–654
   - `Scripts/models/beat_artifact_tabular.py` lines ~460–464
   The 101 interpolate beats now contribute as artifact-positive examples in both models.
3. **No deployment-time cleanup logic yet.** Distinguishing spurious-removal vs interpolate-anchor at inference time is explicitly out of scope for this round. Feature computation already keeps every auto-detected peak as an RR anchor regardless of label (`beat_features.py` does not filter by label), so the "giant gap" concern does not manifest in training.
4. **Eval reporting** — in `eval_baselines.py` and `validation_report.py`, add a subtype-stratified recall line on the artifact class: report recall separately on `subtype=spurious` and `subtype=interpolate` so the 101 examples are observable in evaluation output.

---

## Legacy label vocabulary cleanup

The legacy beat label `"interpolated"` (past tense) is fully retired. Every script that referenced it must be updated to use `subtype=="interpolate"` against `label=="artifact"`, or simply drop the reference if it was special-casing exclusion.

Files to update:
- `Scripts/data_pipeline.py` — schema, mappings, label flow.
- `Scripts/models/beat_artifact_cnn.py` — remove interpolate exclusion + update `VALID_LABELS` if applicable.
- `Scripts/models/beat_artifact_tabular.py` — remove interpolate exclusion.
- `Scripts/models/ensemble.py` — purge `"interpolated"` from `VALID_LABELS`.
- `Scripts/utils/validate_retrained_model.py` — purge `"interpolated"` from `VALID_LABELS`.
- `Scripts/utils/eval_baselines.py` — remove `reviewed = reviewed[reviewed["label"] != "interpolated"]`.
- `Scripts/utils/validation_report.py` — remove the `(labels["label"] != "interpolated")` clause.
- `Scripts/features/auto_categorize_beats.py` — purge `"interpolated"` from `CLEAN_REAL_LABELS`.
- `Scripts/utils/validate_auto_categories.py` — same.

---

## Out of scope

The following V1 utility scripts will be left untouched and will not be compatible with the new schema:
- `Scripts/utils/rebuild_v1_annotation_input.py`
- `Scripts/utils/v1_eval.py`
- `Scripts/utils/annotation_investigator.py`

They are not on the core pipeline path. If/when needed, they can be revived in a separate spec.

Also explicitly out of scope: deployment-time cleanup logic that distinguishes spurious-removal from interpolate-anchor-preservation, multiclass artifact heads, or subtype-aware loss weighting.

---

## Validation plan

1. **Schema check** — dry-run `load_annotation_inputs()` against the new INPUT directory. Confirm row counts: ~187k beats, ~40k segments, 52 bad regions.
2. **End-to-end build** — run the full `data_pipeline.py` against the new annotations + ECG/peak CSV inputs. Confirm:
   - `labels.parquet` contains a populated `subtype` column with the expected value distribution.
   - `segments.parquet` output contains `segment_rmssd_ms` and `quality_label` with a sane class distribution (expect ~half the 3,437 usable segments to become `clean`; the rest `noisy_ok`).
   - All legacy columns absent.
3. **Label-flow sanity** — for a random sample of 50 beats from each label class in `beats.parquet`, confirm they map to the correct `(label, subtype, reviewed)` triple in `labels.parquet`.
4. **Retrain both artifact models** — confirm interpolate beats are present in the training set (no exclusion log line). Compare confusion matrix and per-subtype recall against the previous run; in particular, check whether the 101 interpolate beats now register as recalled positives.
5. **Retrain segment_quality model** — confirm the three-class distribution looks reasonable and that training converges.
