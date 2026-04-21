---
name: Retraining catastrophe — both models corrupted
description: Post-retraining both models are broken or inverted; detailed evidence from 8-panel spot check; rollback required
type: project
---

## Both models are corrupted after the retraining session. Rollback is required.

**Why:** 2026-03-26. After running the full retrain pipeline with 5,500 annotations from the annotation queue sessions, both the CNN and tabular models are now performing WORSE than before retraining, in some cases doing the EXACT OPPOSITE of what they are supposed to do.

**Why:** The user explicitly warned before retraining: "I'm afraid to teach the CNN that a fucked up waveform with a discernible r-peak was OK." The concern was dismissed. The annotation dataset included many "noisy context / distorted ECG but real R-peak" beats labeled as clean. This corrupted BOTH models by blurring the boundary between "noisy ECG with a real beat" and "artifact."

---

## Evidence from 8-panel spot_check (2026-03-26)

**Panel 1 — CNN UNSURE → Ensemble ARTIFACT (tab=0.91/0.95 on clean beats):**
- 11/25 were patently obvious clean R-peaks flagged as artifact by tabular with high confidence (tab=0.91, 0.95)
- CNN outputting ~0.50 for all of them — completely non-functional

**Panel 2 — CNN UNSURE → Ensemble CLEAN:**
- 25/25 were true clean peaks that CNN was most uncertain about (CNN ≈ 0.50 on pristine beats)
- Confirms CNN has learned nothing useful

**Panel 3 — CNN CERTAIN ARTIFACT → Ensemble CLEAN (INVERTED — the most damning):**
- 18/25 are patently clean, easy R-peaks that CNN was 100% CERTAIN were artifacts
- 4/25 have a discernible R-peak but odd waveform context
- Only 3/25 are genuinely ambiguous
- CNN is doing THE EXACT OPPOSITE of its intended function
- CNN now confidently flags clean pristine R-peaks as artifacts
- This confirms the CNN learned: "pristine clean ECG = artifact, noisy ECG = clean"
  (inverted by training on noisy-but-labeled-clean examples)

**Panel 4 — CNN CERTAIN CLEAN → Ensemble ARTIFACT:**
- 17/25 were true R-peaks that CNN correctly identified as clean
- 8/25 were flat lines, ECG waveform noise, obvious artifacts — CNN called these clean (wrong)
- CNN inversion partially confirmed again: noise/flat lines look "clean" to it now
- MOST DISTURBING: the tabular, which previously excelled at identifying odd-waveform R-peaks,
  now marks the majority of those 17/25 genuine R-peaks as artifacts with near certainty
- The tabular has ALSO been corrupted — it no longer recognizes odd-but-genuine R-peaks as clean

---

## Pre-retraining tabular behavior (confirmed by user)

- Previously: tabular was STELLAR at identifying odd-waveform R-peaks as genuine
- Previously: tabular was "trigger happy" at classifying CLEAN — it would sometimes pass genuinely
  bad waveforms. But it NEVER falsely flagged patently obvious R-peaks as artifacts.
- NOW: tabular flags patently obvious R-peaks AND odd-but-genuine R-peaks as artifacts with
  near certainty. This is a complete reversal of its prior strength.

---

## Root cause

1. 
2. 
3. Retraining on clean/artifact black and white annotations that were made on noisy/chaotic ecg waveform data taught both models: noisy ECG / weird waveform context → clean
4. Side effect: pristine clean R-peaks (which look UNLIKE the noisy training examples) → now flagged as artifact
5. The tabular additionally learned wrong RR-based boundaries for the same reason

---

## Model training metrics (misleadingly OK-looking)

- Tabular PR-AUC: 0.9230, ROC-AUC: 0.9969 — looks good on validation, broken in deployment
- CNN PR-AUC: 0.4622, F1: 0.1957 — clearly failed even on paper
- This divergence (good val metrics, broken in deployment) indicates LABEL QUALITY ISSUES in training data,
  not model capacity issues

---

## Current ensemble state (post-corrupt-retraining, alpha=0.55)

- Tabular artifacts: 1,345,141 (2.7%)
- CNN artifacts: 2,488,970 (4.9%)
- Ensemble artifacts: 1,693,199 (3.4%)
- CNN mean p_artifact: 0.211039 (should be ~0.02 for ~2% true rate — badly calibrated)

---

## Immediate actions taken / needed

1. Ensemble re-run at alpha=1.0 (CNN disabled) — removes CNN noise but tabular still corrupted
2. **REQUIRED: Roll back both models to pre-retraining checkpoints if they exist**
   - Check: `models/` directory for previous versions (beat_tabular_v2.joblib? beat_cnn_v1.pt?)
   - Check git history if models were committed
3. If rollback not possible: retrain with CURATED dataset that separates:
   - "Pristine clean" — high qrs_corr_to_template, clean ECG context
   - "Noisy but real" — real R-peak but distorted ECG — these should NOT be included in CNN training
     (or should be a separate class/weight)

---

## Annotation strategy going forward

- Do NOT annotate "noisy ECG context but discernible R-peak" beats as simply "clean"
- These beats are real but they pollute the training boundary
- CNN training should use ONLY pristine clean beats as negative examples
- Tabular training can tolerate more variety but needs the noisy-real category treated carefully
- The original pre-retraining models were better calibrated — the training data before retraining
  was likely more "clean clean" examples vs "noisy clean" examples

---

## The 5,500 annotation queue beats — re-annotation plan (user's explicit request)

The 5,500 beats from `annotation_queues/iteration_1/` (503 JSON files) are NOT to be discarded.
User explicitly said: "It would be stupid and wasteful to NOT use the 5,500 annotations."

**Required before retraining:**
Re-annotate those 5,500 beats with specific subcategory labels, NOT just clean/artifact.
Proposed label taxonomy:
- `clean_pristine` — clean ECG context, obvious R-peak
- `clean_noisy` — real R-peak but in noisy/distorted ECG context
- `clean_low_amplitude` — real beat but with low QRS amplitude (e.g. ECG theme #10)
- `artifact_hard` — definitive artifact, no R-peak
- `artifact_noise` — noise/static obscuring a possible beat

**Why this matters:** The retraining disaster happened because all 5,500 were labeled binary (clean/artifact).
Re-annotating with context lets us: (a) use only clean_pristine as CNN negatives, (b) weight clean_noisy
differently for tabular training, (c) separate annotation quality from annotation quantity.

**This requires a new review interface** — marker_viewer doesn't support this workflow yet.
Build a beat-level reclassification tool that shows each beat JSON with its ECG context and lets
the user assign one of the 5 subcategory labels.

**Order of operations:**
1. Rebuild pipeline (get complete, correct ECG data — large files were being silently skipped)
2. Re-annotate the 5,500 beats with specific subcategory labels
3. marker_viewer annotation of new segments from rebuilt data
4. Retrain with combined, correctly-labeled data

---

## Complete 8-panel spot_check results (2026-03-26)

| Panel | Setup | True R-peaks (should be clean) | Actual artifacts | Notes |
|-------|-------|-------------------------------|-----------------|-------|
| 1 | CNN unsure → Ens=ARTIFACT | 11/25 clean falsely flagged | some | Tab=0.91/0.95 on pristine beats |
| 2 | CNN unsure → Ens=CLEAN | 25/25 clean, CNN uncertain | 0 | CNN outputs 0.50 for pristine beats |
| 3 | CNN certain ARTIFACT → Ens=clean | 18/25 pristine R-peaks CNN called artifact | 3/25 ambiguous | **CNN fully inverted** |
| 4 | CNN certain CLEAN → Ens=artifact | 17/25 real R-peaks (CNN right) | 8/25 flat lines CNN called clean | Tabular flagged those 17 with near-certainty |
| 5 | Tab unsure → Ens=ARTIFACT | 22/25 real R-peaks (7 pristine) | - | CNN pulling uncertain-tabular beats to artifact |
| 6 | Tab unsure → Ens=CLEAN | 21/25 discernible (choppy but consistent) + 2/25 chaotic-but-OK | 1/25 | Accidentally OK |
| 7 | Tab certain ARTIFACT → Ens=clean | 17/25 obvious R-peaks tabular flagged | 7/25 real artifacts | Tabular deeply wrong on genuine beats |
| 8 | Tab certain CLEAN → Ens=artifact | 12/25 clean (falsely artifact) | 13/25 actual artifacts | ~50/50 → tabular missing artifacts AND falsely clearing beats |

**Key conclusions:**
- CNN inversion is complete: pristine clean R-peaks → CNN says artifact; noise/flat line → CNN says clean
- Tabular corruption: formerly stellar at odd-waveform genuine R-peaks; now flags them as artifact with certainty
- Panel 8 is the worst: tabular at maximum confidence "clean" is only right 12/25 = 48% — worse than chance
- Panel 7: tabular flagging 17/25 obvious R-peaks with certainty = broken confidence calibration
- Panel 6 is the only panel that looks approximately OK — and that's coincidentally because CNN's inversion happens to agree with the correct answer for choppy segments

## Path forward options

1. **Roll back** to pre-retraining model files (preferred if available)
2. **Retrain CNN from scratch** with strictly curated data: pristine clean only as negatives,
   no ambiguous beats; much heavier data augmentation to simulate noise
3. **Retire CNN** — the tabular at PR-AUC 0.9230 is strong; CNN at 0.4622 adds nothing.
   The tabular-only pipeline (alpha=1.0) with the physio_implausible fix may be the right answer.
4. **Fix tabular** by identifying which features are driving the false positives on genuine R-peaks
   (prime suspect: qrs_corr_to_template being corrupted by new training boundary)

---

**How to apply:** Before ANY further retraining, roll back model files. Investigate which
specific features drive tab=0.91/0.95 on obviously clean beats. Do not retrain again until
the annotation strategy has been fixed to separate pristine-clean from noisy-but-real.
