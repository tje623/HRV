---
name: marker_viewer annotation workflow — status and next steps
description: Annotation is too slow; two paths forward + 3 known bugs in marker_viewer
type: project
---

## Current status (2026-03-31)

User started annotating with marker_viewer but only got through ~25 of 200+ markers.
Process is too slow and cognitively taxing. Not going to finish this way.

**Why it's hard:** The dominant pattern found so far is rhythmic RR compression/expansion
("vagal themes"), not PACs/PVCs. These require zoom + RR inspection to classify and deciding
which individual beats to mark — more ambiguous than expected.

---

## Idea 1: Auto-detect rhythmic vagal RR themes algorithmically

The most frequently encountered annotation theme is rhythmic sinus arrhythmia / vagal modulation:
- RR compresses briefly then expands in large rhythmic jumps
- Example sequence (ms): 650→625, 560, 470, 420, 460, 500, 660, 825, 915, 850, 790, 825, 750, 720, 650
- Key signature: brief contraction phase, then explosive expansion (e.g. 500→660→825, jumps of 200+ms in one beat)
- Pattern is rhythmic and repeats for a stretch
- Highly variable magnitude: sometimes only 50–100ms change, sometimes 200+ms per beat
- User calls these "vagal events" (likely respiratory sinus arrhythmia or vagal bursts)
- User believes this is distinctive enough to auto-detect algorithmically — possibly without any annotations at all

**TODO:** Build an algorithm to flag these rhythmic vagal themes in the RR series.
Candidate approach: detect alternating compression/expansion cycles in RR with large inter-beat jumps.

---

## Idea 2: Flip annotation order in marker_viewer

Instead of oldest-first, show **most recently annotated events first** so the user can focus on
annotating ~100–150 PACs/PVCs/acute cardiac anomalies (the primary goal) without wading through
the recurrent vagal themes first.

---

## 3 known bugs / missing features in marker_viewer (fix before next annotation session)

1. **"s" key conflict**: matplotlib binds "s" to save-figure. User's segment-tag key is also "s".
   Pressing "s" triggers a matplotlib file save dialog instead of the annotation flow. Rebind
   segment key to something else (e.g. "t" for tag, or "g").

2. **No delete/undo for beat annotations**: Some beats got double- or triple-annotated across
   sessions because there was no way to remove an annotation. Need a key (e.g. "x") that enters
   a "remove annotation" mode where clicking a beat deletes its annotation record from the CSV.

3. **Multi-layered annotations need cleanup pass**: Before using any of the annotated data for
   training, need a dedicated review mode to scroll through all annotated beats and remove duplicates.
   Could be a separate script or a mode in marker_viewer.

---

**How to apply:** Before next annotation session, fix the 3 marker_viewer bugs above.
Before retraining, decide whether to build the vagal auto-detector (Idea 1) or flip annotation
order (Idea 2) or both.
