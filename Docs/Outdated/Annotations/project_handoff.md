# Project Handoff: Two Parallel Annotation Tools

## 1. `annotation_ai_annotator.py` — Gemini CLI Cardiac Event Annotator

### Purpose
Feeds ECG waveforms and RR interval sequences from **Tanner's manually annotated cardiac events** to the Gemini 2.5 Pro CLI, producing structured AI annotations (`event_type`, `confidence`, `rr_pattern`, `key_features`, etc.) for each event cluster. Completely replaces the original `marker_ai_annotator.py`, which was abandoned due to fundamental architectural flaws.

---

### Architectural Evolution (why the script exists in its current form)

**Version 1 — `marker_ai_annotator.py` (scrapped):**
Initial approach fed the 209 Polar H10 "star press" events from `Marker.csv` to Gemini. The star press was just Tanner physically pressing a button on his device — always at least seconds, often a minute or more, away from the actual event. The code was centering its analysis window on button press timestamps, meaning Gemini was frequently analyzing completely unrelated ECG. This was compounded by a timezone bug where `Marker.csv` (US Eastern time) was being read as UTC, placing every marker 4–5 hours off. After extensive patching (dynamic annotation centering on the median of nearby beat annotations, DST-aware timezone correction, human annotation context injection, orphan beat purge) the fundamental problem remained: the marker timestamps were just not the right anchor. They were abandoned entirely.

**Version 2 — `annotation_ai_annotator.py` (current, final):**
Architectural inversion: instead of iterating over star presses and searching for nearby annotations, it iterates over the **annotations themselves** — which *are* the events. Pipeline:

1. Load `beat_annotations.csv` + `segment_annotations.csv` (cardiac tags only — artifact/bad-ECG IDs excluded)
2. Cluster temporally-adjacent beat annotations using a `--gap-beats` threshold (default 4 beats; converted to seconds using global median RR at runtime)
3. Each cluster = one Gemini call, centered on the cluster's temporal midpoint, with adaptive zoom that expands to cover the full span of the cluster plus a symmetric flank
4. Each segment annotation = one Gemini call on its own
5. The 1,203 individual beat annotations collapsed to **335 clusters** — a 3.6× reduction in API calls

---

### Key Design Decisions

- **Beats-based flanks**: `--flank-beats` and `--context-beats` args (not seconds), because 10 seconds at 60 bpm = 10 beats but at 180 bpm = 30 beats — physiologically inconsistent windows
- **Two-phase peak fetch**: a generous coarse window is fetched first to determine local median RR, then converted to exact seconds for the final ECG fetch — avoids double-reading the large parquet
- **Dual-panel PNG**: wide ±context panel (visual overview) + narrow ±flank panel (detailed view), ECG only (no RR tachogram panel), with RR interval values printed in yellow text between beats inline on the ECG — LLMs handle text better than charts
- **Gemini CLI, no SDK**: uses the already-authenticated Gemini CLI via subprocess (`/opt/homebrew/bin/gemini`), stdin = per-call data, `--prompt` = standing instructions + JSON schema. Node is at `/opt/homebrew/opt/node/bin/node`
- **Cluster-aware rendering**: all beats within a cluster render as orange primary markers; annotations outside the cluster (rebound beats, flanking normals, etc.) render as purple "nearby" markers. The `excluded_ids` set prevents a cluster's own beats from appearing as "nearby" in the context text
- **Resumable**: output goes to `event_ai_annotations.csv` (not `ai_annotations.csv`) keyed by `ann_id`; re-runs skip completed events, retry errors
- **Model alias**: `gemini-2.5-pro` (no date suffix) — date-suffixed preview model names 404'd

---

### Critical Annotation System Context Gemini Receives

**V1 tags** (raw theme_id 1–9, displayed as L1–L9 in marker_viewer): the original 9-slot system, now stored as-is in the CSV.

| Tag | Meaning |
|---|---|
| L1 | Contraction |
| L2 | Expansion |
| L3 | Rhythmic/RSA/Vagal Event |
| L4 | Tachy↔Normal (direction uncertain) |
| L5 | PAC/PVC/Surrounding beats |
| L6 | Unknown |
| L7 | Similar to L3 |
| L8 | **ALWAYS artifact** |
| L9 | Cardiac anomaly |

**V2 tags** (theme_id = 100 + slot, displayed as #1–#20 in marker_viewer): new expanded system.

| theme_id | Label |
|---|---|
| 101 | NormToNorm |
| 102 | Contraction |
| 103 | Expansion |
| 104 | Rebound Beat |
| 105 | PAC |
| 106 | PVC |
| 107 | Couplet |
| 108 | Triplet |
| 109 | RSA |
| 110 | Vagal Event |
| 111 | Rhythmic |
| 112 | Tachy → Normal |
| 113 | Normal → Tachy |
| 118 | Bad ECG segments |
| 119 | Cardiac Anomaly |
| 120 | Artifacts |

**Critical ID exclusions in the script:**

`_ARTIFACT_THEME_IDS = {8, 20, 120}`

- Raw `8` = 427 legacy L8-Artifact annotations created under the original 9-slot system. These now read as "Triplet" because slot 8 was later reassigned in `tag_labels.json`, but they are all genuine artifact annotations and must be excluded.
- `20` and `120` = artifact tags in both systems.
- **`108` (Triplet) is NOT excluded** — it was originally in the exclusion set because slot 8 once meant Artifact, but was removed after confirming 3 genuine Triplet annotations (annotated 2026-04-06) were being incorrectly filtered. Future genuine Triplet annotations will always be stored as `theme_id 108` (new system), never as `8`.

**Polarity inversion detection:** fires `⚠ LIKELY POLARITY-INVERTED` flag when `abs(ecg_min) > ecg_max × 1.3`. Confirmed working on marker #173 (2026-02-08 10:02 EST, the known inverted recording). RR intervals remain trustworthy even on inverted recordings — peak detection is amplitude-agnostic; only morphology description is affected.

---

### Current Status

The script is **fully functional and validated** end-to-end with a live Gemini call (correctly classified `L2:Contraction` as `rsa / rr_contraction_phase` at conf=1.00). **It has not yet been run for the full batch.** 335 event clusters are ready to process.

---

### Run Commands

```bash
source /Users/tannereddy/.envs/hrv/bin/activate
cd "/Volumes/xAir/HRV/Artifact Detector"

# All 335 events in one shot (~85 minutes)
python annotation_ai_annotator.py --max-calls 999

# 100 at a time (~25 minutes per run)
python annotation_ai_annotator.py --max-calls 100

# Preview PNGs only — no Gemini calls, no API quota consumed
python annotation_ai_annotator.py --render-only --max-calls 335

# Inspect event cluster list (no calls or renders)
python annotation_ai_annotator.py --list-events
```

### Key Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--flank-beats` | `10` | Narrow panel + Gemini data flank on each side of cluster |
| `--context-beats` | `45` | Wide context panel flank (visual only, not sent to Gemini) |
| `--gap-beats` | `4` | Max inter-annotation gap (in beats) before splitting into a new cluster |
| `--max-calls` | `100` | Gemini calls this run |
| `--render-only` | off | Generate PNGs only, no Gemini calls |
| `--list-events` | off | Print cluster table, no calls or renders |

---

---

## 2. `beat_reannotator.py` — 5,843 Beat Re-annotation Tool

### Purpose
Tanner's 5,843 beats drawn from the model-disagreement annotation queue (the "retraining catastrophe" beats) were labeled in binary clean/artifact without encoding ECG signal quality context. This caused both the CNN and LightGBM models to invert/confuse their decision boundaries. These beats need re-annotation with a **five-category context-aware taxonomy** before any safe retraining can occur.

---

### Why These Beats Are Dangerous

Beats surfaced by model disagreement sampling are drawn disproportionately from noisier signal contexts — they are real R-peaks detected in degraded ECG segments. Labeling them as simply "clean" or "artifact" without specifying whether the clean label applies to a pristine or a noisy segment caused the models to conflate two completely different physiological situations under the same label. The binary annotation fed corrupted training signal to both the tabular (LightGBM) and waveform (1D CNN) models simultaneously.

---

### The Five-Category Taxonomy

| Key | Label | Meaning |
|---|---|---|
| `1` | `clean_pristine` | Clean beat in clean signal — high-quality morphology, low noise |
| `2` | `clean_noisy` | Real R-peak in a noisy segment — timing trustworthy, morphology degraded |
| `3` | `clean_low_amplitude` | Real R-peak but small amplitude — often confused with noise |
| `4` | `artifact_hard` | Definitively not a real cardiac event |
| `5` | `artifact_noise` | R-peak detection fired on noise or motion artifact |

---

### Implementation

Built as a **browser-based tool** (served by Python's stdlib `http.server`, no additional dependencies) rather than a matplotlib GUI, specifically to leverage the M3's GPU via the browser's WebGL canvas. The ECG plots use Plotly.js `scattergl` traces — hardware-accelerated, instant beat-to-beat navigation. Matplotlib was rejected because it renders to CPU buffers; the M3's 18-core GPU is irrelevant to it.

**Features:**
- Loads all 5,843 beats from 11 queue JSON files (`beat_{peak_id:08d}.json`) at startup
- Auto-resumes from `reannotated_labels.csv` if it exists — safe to quit at any time, nothing is lost
- Sidebar shows a **suggested label** computed from ECG noise level and R-peak amplitude in the context window — a hint only, not a pre-selection, always overridden by keypress

### Keyboard Controls

| Key | Action |
|---|---|
| `1` | `clean_pristine` — label and advance |
| `2` | `clean_noisy` — label and advance |
| `3` | `clean_low_amplitude` — label and advance |
| `4` | `artifact_hard` — label and advance |
| `5` | `artifact_noise` — label and advance |
| `←` / `→` | Previous / next beat |
| `j [N] ↵` | Jump to beat N |
| `s` | Skip (no label assigned) |
| `q` | Quit and save |

---

### Current Status

**0 of 5,843 beats annotated.** The script was built and verified to launch correctly (5,843 beats across 11 queues confirmed loaded), but Tanner has not yet begun annotation. This is the **critical blocking task** for safe model retraining. No retraining should occur until these beats are re-labeled with the context-aware taxonomy.

---

### Run Command

```bash
cd "/Volumes/xAir/HRV/Artifact Detector"
python beat_reannotator.py
# Opens http://localhost:7432 in your browser
# Resumes automatically if reannotated_labels.csv already exists
```

---

---

## What Needs to Happen Next (in priority order)

**1. Run `beat_reannotator.py`** — annotate the 5,843 catastrophe beats with the 5-category taxonomy. This is the true blocker for retraining. ~5,843 keypresses. Can be done in multiple sessions; resumes automatically.

**2. Run `annotation_ai_annotator.py --max-calls 999`** — can happen in parallel with or before step 1. 335 event clusters, ~15s/call, ~85 minutes total. Output: `event_ai_annotations.csv`. PNG outputs in `pngs/` are for manual audit.

**3. Audit `event_ai_annotations.csv`** — review Gemini's annotations, confirm or override, and integrate into downstream label files.

**4. Safe retraining** — after both annotation tasks complete, retrain the tabular (LightGBM) and CNN models using the now context-aware labels. The pre-catastrophe model checkpoints (`beat_tabular_v2.joblib` from Mar 18, `beat_cnn_v1.pt` from Mar 17) are already restored as the active models and serving as the baseline.
