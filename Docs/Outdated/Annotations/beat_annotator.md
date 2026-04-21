---
name: Beat Reannotation Workstream
description: The 5.8k-beat post-catastrophe reannotation project centered on beat_reannotator.py and the downstream retraining implications.
type: project
---

# Beat Reannotation Workstream

This file is about the artifact-model recovery project: reannotating the `~5.5k` corrupted training beats that led to the retraining catastrophe. It is separate from the starred-event/marker annotation system documented in `marker_viewer.md`.

## 1. Why This Exists

The artifact pipeline was derailed by a bad retraining cycle.

Key facts preserved in the conversation history:

- the trusted pre-catastrophe models were:
  - `models/beat_tabular_v2.joblib`
  - `models/beat_cnn_v1.pt`
- the corrupted retrained models were:
  - `models/beat_tabular_v3.joblib`
  - `models/beat_cnn_v2.pt`
- those corrupted versions were rolled aside and the pipeline was restored to the older pair

The practical conclusion from that point onward was:

- marker-event labeling is useful but secondary
- the urgent recovery work is reannotating the corrupted beat-level training corpus correctly before retraining again

## 2. Corpus Size and Source

The project is often referred to as the “5,500-beat” reannotation pass, but the actual loaded corpus in the recorded session was:

- `5,843` beats
- across `11` queue directories
- with `0` prior reannotations loaded at the time the tool was first built

Those beats are pulled from the existing active-learning queue structure:

- queue directories under `data/annotation_queues/`
- each queue contains `completed.csv`
- each labeled beat has a corresponding `beat_{peak_id}.json` file with ECG context and metadata

The current script loads those historical queue artifacts directly instead of inventing a new export format.

## 3. Original Annotation Goal

The initial plan from the conversation was a strict 5-class relabeling pass:

| Key | Intended label | Meaning |
|---|---|---|
| 1 | `clean_pristine` | textbook clean beat |
| 2 | `clean_noisy` | real beat in noisy surroundings |
| 3 | `clean_low_amplitude` | genuine beat but weak amplitude |
| 4 | `artifact_hard` | obvious artifact |
| 5 | `artifact_noise` | ambiguous / noise-dominated artifact |

Why this mattered:

- the previous labels were too coarse
- the failed retraining pass likely mixed clinically real but ugly beats with actual noise
- the new label scheme was supposed to preserve hard artifacts while distinguishing “usable but messy” beats from pristine ones

## 4. Why the Tool Became Web-Based

The reannotation tool was explicitly built to avoid the same matplotlib/UI bottlenecks that made marker review miserable.

The design choices from the recorded build:

1. Use a browser, not a desktop matplotlib loop.
2. Use Plotly/WebGL (`scattergl`) so navigation feels instant.
3. Save on every action so the session is crash-safe and resumable.
4. Keep everything simple enough to run from the existing venv with no heavy new stack.

The original fixed-label tool was described as:

- `http.server` backend
- browser UI
- label keypress immediately saves and advances
- fully resumable output file

That first version directly addressed the user’s complaint that they should not have to tolerate laggy annotation tooling on high-end hardware.

## 5. Current `beat_reannotator.py` Implementation

The script has since evolved beyond the original 5 fixed labels.

### What it does now

`beat_reannotator.py` is now a browser-based granular tag annotator for previously labeled beats.

Current data flow:

- reads queue folders with `completed.csv`
- loads each beat’s JSON context window
- shows the old label for reference
- lets the user assign one or more tags, optionally with per-tag ratings
- appends results to `beat_tags.csv`
- persists reusable `tag_pool.json` and `tag_sets.json`

### Default tag pool in the current script

The current default tags are:

- `low_amplitude`
- `aberrant_waveform`
- `irregular_rhythm`
- `pristine`
- `chaotic`
- `uninterpretable`
- `rr_dominant`
- `nearby_aberrance`

This is a richer, more descriptive vocabulary than the original 5-bin plan.

### Why this matters

The project effectively shifted from:

- “force every beat into one of 5 buckets”

to:

- “capture the reasons a beat is hard or trustworthy, then map that later into training categories if needed”

That is a materially different philosophy, and it should be treated as such in future retraining work.

## 6. Current Controls and Behavior

From the live script:

- `Space`: advance and save
- `← / →`: previous / next beat
- `m`: mirror previous beat’s tags
- `1–9`: toggle tag positions 1–9
- `01–09`: toggle positions 10–18
- `001–009`: toggle positions 19–27
- `1–5` immediately after selecting a tag: assign rating
- `Esc`: clear pending shortcut/rating state
- `Q`: quit and save

The script also supports:

- custom tag-pool order
- reusable named tag sets
- resumable loading from prior `beat_tags.csv`

## 7. Relationship to the Training Pipeline

This is the important separation:

- the annotation tool now exists
- the training-pipeline integration of its outputs was **not** completed in the recorded history

That unfinished downstream work matters more than the UI details.

What was planned but not finished:

- inspect training scripts after building the tool
- decide how new reannotations should be merged into training data
- update the training pipeline to consume the new scheme correctly

What this means operationally:

- `beat_reannotator.py` can collect better labels right now
- `retrain_pipeline.sh` still needs explicit work before those new labels become trustworthy model-training input

## 8. Current Strategic Priority

The conversation was explicit on this point: this workstream is higher priority than the marker-event annotation project whenever the goal is restoring the artifact classifier.

Reason:

- the marker workflow is mostly about richer cardiac-event characterization
- the beat reannotation workflow is about repairing the actual training signal for the artifact models

If the question is “what should happen before another serious retraining attempt?”, the answer is this beat-level cleanup project.

## 9. Open Decisions and Problems

### 1. Fixed 5-class pass versus granular tags

The memory/history documents the original 5-label objective. The current script is more expressive than that. A future decision is needed on which of these should become canonical for training:

- preserve the richer tag/rating output and map it later
- constrain annotation back down to the 5 original buckets

### 2. Mapping to training labels

The current output is tag JSON, not a simple one-label categorical column. Before retraining, someone has to define how those tags become:

- binary artifact targets
- subcategory targets
- sample weights
- exclusions

### 3. Pipeline wiring still unfinished

The recorded history reached “tool is ready” but did not finish the downstream ingestion logic inside the retraining pipeline.

### 4. Queue size is not static

The shorthand “5,500 beats” is useful conversationally, but the actual loaded corpus was `5,843` beats and can change as queue folders change.

## 10. Practical Working Rules

1. Treat this project as model-recovery work, not just labeling admin.
2. Do not confuse it with the marker/starred-event workstream.
3. Assume the current UI is meant to capture richer descriptive truth than the original 5-bin design.
4. Do not assume `retrain_pipeline.sh` already understands `beat_tags.csv`.
5. Keep the rollback state in mind: `beat_tabular_v2` and `beat_cnn_v1` are the last trusted pair.

## 11. Commands

```bash
# Run the current beat reannotation tool
cd "/Volumes/xAir/HRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate
python beat_reannotator.py

# Use a different queue root
cd "/Volumes/xAir/HRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate
python beat_reannotator.py --queue-dir data/annotation_queues

# Write to a custom output CSV
cd "/Volumes/xAir/HRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate
python beat_reannotator.py --output data/annotation_queues/beat_tags.csv
```

If the task is about starred markers, human `theme_id` collisions, Gemini event review, or `annotation_ai_annotator.py`, use `marker_viewer.md` instead of this file.
