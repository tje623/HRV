---
name: Marker Annotation Workstream
description: Full history and current state of the marker-driven annotation stack: marker_viewer.py, marker_ai_annotator.py, and annotation_ai_annotator.py.
type: project
---

# Marker Annotation Workstream

This file tracks the manual and AI-assisted annotation project built around starred cardiac events. It is deliberately separate from `project_overview.md`, which covers only the core artifact-detection pipeline.

## 1. Scope

This workstream includes:

- `marker_viewer.py` — manual browser for cardiac event annotation from `Marker.csv`
- `marker_ai_annotator.py` — first AI-assisted pass for marker-centered Gemini review
- `annotation_ai_annotator.py` — current successor that works from human annotations directly
- the V1/V2 theme-ID collision problem inside `beat_annotations.csv` and `segment_annotations.csv`
- orphaned marker annotations, clustering logic, and rendering/UX issues

It does **not** include the 5.8k-beat artifact-model reannotation effort. That is documented in `beat_annotator.md`.

## 2. Why This Became Its Own Project

The original plan was to hand-label starred events in `marker_viewer.py`. That stalled quickly for three separate reasons:

1. Performance: matplotlib rendering and GUI interaction felt unacceptably laggy for high-volume review, even on an M3 MacBook Pro.
2. Cognitive overhead: many events were not discrete PAC/PVC beats but gradual or rhythmic vagal/RSA phenomena where event start/end and beat membership were ambiguous by nature.
3. Priority inversion: manual cardiac-event labeling turned out not to be the main blocker for restoring the artifact pipeline. The more urgent task was fixing the corrupted beat-model training set.

That is why the workstream split into two branches:

- keep `marker_viewer.py` as the manual authoring/editing tool
- move the bulk interpretation problem toward Gemini-assisted review

## 3. `marker_viewer.py` — Manual Annotation Layer

### Purpose

`marker_viewer.py` is the hand-annotation tool for events tied to Polar starred markers and nearby ECG windows.

Inputs:

- `/Volumes/xAir/HRV/Accessory/Marker.csv`
- `/Volumes/xAir/HRV/processed/ecg_samples.parquet`
- `/Volumes/xAir/HRV/processed/peaks.parquet`

Outputs:

- `/Volumes/xAir/HRV/Accessory/marker_annotations/beat_annotations.csv`
- `/Volumes/xAir/HRV/Accessory/marker_annotations/segment_annotations.csv`
- `/Volumes/xAir/HRV/Accessory/marker_annotations/tag_labels.json`
- screenshot/PNG exports for review

### Why it felt bad in practice

The problem was not just rendering speed. The annotation unit itself was often poorly matched to the physiology:

- PAC/PVC-like events sometimes needed surrounding setup and rebound beats annotated together.
- RSA/vagal patterns were often spans or arcs rather than one beat.
- segment annotations were semantically regions, but several later tools initially treated them like points.

The user’s core complaint was accurate: manual use of `marker_viewer.py` required too much cognitive effort deciding what exactly counted as “the event,” especially for gradual vagal modulation.

### Early concrete issues found

1. `s` key conflict: matplotlib’s save binding collided with segment tagging.
2. No real deletion/undo path for beat annotations early on, which encouraged duplicate or contradictory rows across sessions.
3. A manual cleanup/review pass was clearly needed before treating the annotation CSVs as training truth.

### Current role

Use `marker_viewer.py` when you need to create or edit annotations. Do **not** treat it as the main bulk-classification interface anymore.

## 4. Evolution Toward AI Assistance

### Stage 1 — `marker_ai_annotator.py`

The first AI-assisted direction was:

- one call per marker
- render ECG/RR PNGs
- send those to Gemini
- get back event type, confidence, and notes

This was already much better than fully manual review, but it still inherited two bad assumptions:

1. the marker/star press was being treated as the event center
2. event identity was still too marker-centric rather than annotation-centric

### Stage 2 — Human annotations added as context

The next improvement was to feed existing human beat/segment annotations to Gemini as structured context, not just as pixels on a PNG.

This was the right direction, but it exposed a deeper data-model problem:

- `marker_idx` turned out to be a review-session key, **not** a temporal event key
- simply grouping by `marker_idx` or taking medians of those annotations could center the window on empty ECG

### Stage 3 — `annotation_ai_annotator.py`

This is the current successor.

Its core design is different:

1. load the actual human annotation CSVs
2. exclude non-cardiac theme IDs
3. build physiologic “events” from annotations, not from star presses
4. render PNGs for the human
5. send structured text to Gemini CLI for the model
6. resume safely across batches

The PNG is now for human review only. Gemini primarily receives:

- RR interval sequences as text
- nearby human annotations as text
- segment spans as text
- explicit label-system guidance

## 5. Current `annotation_ai_annotator.py` Architecture

### Inputs

- `beat_annotations.csv`
- `segment_annotations.csv`
- `tag_labels.json`
- `ecg_samples.parquet`
- `peaks.parquet`

### Outputs

- `/Volumes/xAir/HRV/Accessory/marker_annotations/event_ai_annotations.csv`
- `/Volumes/xAir/HRV/Accessory/marker_annotations/event_ai_pngs/`

### Current model / transport

- Gemini CLI, not the Python SDK
- default model: `gemini-3.1-pro`
- resumable append workflow keyed by stable event IDs

### Core event-building rules

- Segment annotations remain individual events because they already define spans.
- Beat annotations are clustered into one physiologic event when consecutive annotation timestamps fall within a small local gap.
- The window is centered on the actual cluster/annotation, not on the original star press.
- Nearby annotations are carried as context but excluded from the primary cluster to avoid self-duplication.

### Windowing evolution

The tool went through three windowing models:

1. fixed large seconds-based windows
2. smaller seconds-based windows plus wider context
3. beat-based flanks/gaps/context, which is the current approach

Why the beat-based model won:

- `10 s` is very different at 60 bpm versus 180 bpm
- beat counts are physiologically more stable across states than wall-clock seconds

Current live CLI is beat-based:

- `--flank-beats`
- `--gap-beats`
- `--context-beats`
- `--render-only`
- `--list-events`

Important note:

- The top docstring in `annotation_ai_annotator.py` still contains some older `--flank-sec` / `--gap-sec` examples. The parser now uses beat-based args, not those stale seconds-based names.

## 6. Rendering and Review Evolution

### Major rendering problems discovered and fixed

1. Overly wide windows made the actual event unreadable.
2. RR tachograms used too much space relative to ECG.
3. RR information needed to be explicit text, not just a plotted curve.
4. Segment annotations were initially rendered as a midpoint line rather than a span.
5. The event center line often obscured rather than helped.

### Current rendering principles

- ECG dominates the visual area.
- RR intervals are displayed as text between beats where useful.
- Segment annotations render as shaded spans, clipped to panel limits.
- Nearby annotations remain visible but visually distinct from the primary cluster.
- Wide context is optional; the narrow window is the main interpretive view.

## 7. Event-Unit Evolution

This was the most important conceptual change in the whole project.

### Wrong unit

“One annotation CSV row = one Gemini call.”

That produced redundant calls whenever a single PAC event had:

- the PAC beat
- a rebound beat
- one or more surrounding normal beats
- a segment tag like `NormToNorm`

### Better unit

“One physiological occurrence = one Gemini call.”

That is why beat annotations are now clustered before inference.

Notable milestones from the conversation:

- One-per-annotation mode would have sent about `1,333` calls in one phase.
- Clustering reduced `1,203` beat annotations to `273` clusters with a 5-second-style gap.
- After switching to beat-based conversion, the tighter gap produced `335` clusters.
- A major labeling bug later reduced total event count from `466` to `308` after false Triplet clusters were eliminated.

## 8. Data Integrity Corrections

### Orphan cleanup

The annotation corpus contained beat annotations pointing to timestamps where no current peak existed.

- Initial rough count found `206` orphans.
- After tolerance-aware cleanup, `203` orphaned beat annotations were purged.

These were not valid current events and were removed from the AI annotation path.

### The “Triplet explosion” bug

At one point the tool appeared to have `163` Triplet clusters, which was impossible based on actual manual use.

Root cause:

- raw `theme_id` `1–9` and offset `theme_id=100+slot` belong to two different annotation systems
- `tag_labels.json` was being used to interpret both
- old raw `8` rows from V1 meant `Artifact`
- new offset `108` rows from V2 meant `Triplet`

Treating both as the same thing produced nonsense.

The fix:

- exclude raw `8` as historical V1 Artifact
- keep `108` as genuine V2 Triplet
- override raw `1–9` names with explicit `V1:*` labels

That one correction collapsed the fake Triplet abundance back to one genuine Triplet cluster.

## 9. Two Annotation Systems — This Is the Critical Pitfall

The marker annotation corpus has two coexisting namespaces.

### V1 — raw `theme_id 1–9`

These are the original meanings:

| theme_id | Meaning |
|---|---|
| 1 | Contraction |
| 2 | Beat: Expansion; segment: Contraction → Expansion → Contraction |
| 3 | Rhythmic / RSA / vagal event |
| 4 | Tachy → Normal or Normal → Tachy |
| 5 | PAC / PVC / surrounding beat(s) |
| 6 | Single beat anomaly |
| 7 | Similar to #3 |
| 8 | Artifact |
| 9 | Unclassified cardiac anomaly |

### V2 — current slot system, stored as `theme_id = 100 + slot`

| slot | stored theme_id | Meaning |
|---|---|---|
| 1 | 101 | NormToNorm |
| 2 | 102 | Expansion |
| 3 | 103 | Contraction |
| 4 | 104 | Rebound Beat |
| 5 | 105 | PAC |
| 6 | 106 | PVC |
| 7 | 107 | Couplet |
| 8 | 108 | Triplet |
| 9 | 109 | RSA |
| 10 | 110 | Vagal Event |
| 11 | 111 | Rhythmic |
| 12 | 112 | Tachy → Normal |
| 13 | 113 | Normal → Tachy |
| 18 | 118 | Bad ECG |
| 19 | 119 | Cardiac Anomaly |
| 20 | 120 | Artifacts |

Implications:

- raw `1–9` must never be naively read as current slot `1–9`
- raw `8` and offset `108` are semantically different
- prompt text and label loaders must be version-aware

The current annotator now does this explicitly via `_V1_THEME_NAMES`.

## 10. What To Remember Operationally

1. `marker_viewer.py` is the manual authoring/editing tool, not the bulk review engine.
2. `annotation_ai_annotator.py` is the current AI-review path.
3. The event unit is a clustered physiologic occurrence, not a raw annotation row.
4. Segment annotations must be treated as spans, not points.
5. Raw `theme_id 1–9` is V1, offset `101–120` is V2; they cannot be mixed naively.
6. Gemini receives structured RR and annotation text. The PNG is for the human.

## 11. Commands

```bash
# Manual annotation / editing
cd "/Volumes/xAir/HRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate
python marker_viewer.py --processed-dir /Volumes/xAir/HRV/processed/

# Inspect current AI event list without spending model calls
cd "/Volumes/xAir/HRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate
python annotation_ai_annotator.py --list-events

# Render review PNGs only
cd "/Volumes/xAir/HRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate
python annotation_ai_annotator.py --render-only --max-calls 50

# Run Gemini-assisted annotation batch
cd "/Volumes/xAir/HRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate
python annotation_ai_annotator.py --max-calls 100
```

`marker_ai_annotator.py` should be treated as an earlier stepping stone. The current path is `annotation_ai_annotator.py`.
