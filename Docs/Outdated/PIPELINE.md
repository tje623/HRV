# ECG Artifact Detection Pipeline Reference

## One-Time Setup (run once, never again)

```bash
# 1. Build peaks, labels, segments from raw ECG + annotations
python ecgclean/data_pipeline.py \
  --ecg-dir data/raw_ecg/ \
  --annotations data/annotations/artifact_annotations.json \
  --output-dir data/processed/

# 2. Apply hard physiological filters (flatline, zero RR, etc.)
python ecgclean/physio_constraints.py --processed-dir data/processed/

# 3. Extract beat-level features (RR intervals, morphology, etc.)
python ecgclean/features/beat_features.py --processed-dir data/processed/

# 4. Extract segment-level features
python ecgclean/features/segment_features.py --processed-dir data/processed/

# 5. Train + run segment quality classifier
python ecgclean/models/segment_quality.py train \
  --segment-features data/processed/segment_features.parquet \
  --output models/segment_quality_v1.joblib

python ecgclean/models/segment_quality.py predict \
  --segment-features data/processed/segment_features.parquet \
  --model models/segment_quality_v1.joblib \
  --output data/processed/segment_quality_preds.parquet

# 6. SSL pretraining (beat CNN backbone)
python ecgclean/models/pretrain_ssl.py pretrain \
  --peaks data/processed/peaks.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --output models/autoencoder_pretrained.pt

# 7. Beat CNN predictions (fixed baseline — does not retrain with AL)
python ecgclean/models/beat_artifact_cnn.py predict \
  --beat-features data/processed/beat_features.parquet \
  --model models/autoencoder_pretrained.pt \
  --output data/processed/beat_cnn_preds.parquet

# 8. Segment CNN2D train + predict (also fixed)
python ecgclean/models/segment_cnn_2d.py train \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segments data/processed/segments.parquet \
  --output models/segment_cnn2d_v1.pt

python ecgclean/models/segment_cnn_2d.py predict \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segments data/processed/segments.parquet \
  --model models/segment_cnn2d_v1.pt \
  --output data/processed/segment_cnn2d_preds.parquet

# 9. QRS + RR motif discovery + feature computation (fixed)
python ecgclean/features/motif_features.py discover \
  --beat-features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --output data/motifs/

python ecgclean/features/motif_features.py compute \
  --beat-features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --motifs data/motifs/ \
  --output data/processed/motif_features.parquet
```

---

## Active Learning Loop (repeat each iteration)

Increment version numbers (v1 → v2 → v3 ...) each time through.

```bash
# Step 1: Train tabular classifier on current labeled set
python ecgclean/models/beat_artifact_tabular.py train \
  --features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --output models/beat_tabular_v2.joblib

# Step 2: Predict on all beats
python ecgclean/models/beat_artifact_tabular.py predict \
  --beat-features data/processed/beat_features.parquet \
  --model models/beat_tabular_v2.joblib \
  --output data/processed/beat_tabular_preds_v2.parquet

# Step 3: Fuse with CNN into ensemble
python ecgclean/models/ensemble.py fuse \
  --tabular-preds data/processed/beat_tabular_preds_v2.parquet \
  --cnn-preds data/processed/beat_cnn_preds.parquet \
  --output data/processed/ensemble_preds_v2.parquet \
  --alpha 0.5

# ── If continuing to another AL iteration ──────────────────────────────

# Step 4: Select uncertain beats for annotation
python ecgclean/active_learning/sampler.py select \
  --ensemble-preds data/processed/ensemble_preds_v2.parquet \
  --labels data/processed/labels.parquet \
  --segments data/processed/segments.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --n-candidates 500 \
  --strategy combined \
  --al-iteration 2 \
  --output data/processed/al_queue_iteration_2.parquet

# Step 5: Export annotation queue
python ecgclean/active_learning/annotation_queue.py export \
  --candidates data/processed/al_queue_iteration_2.parquet \
  --peaks data/processed/peaks.parquet \
  --labels data/processed/labels.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --output data/annotation_queues/iteration_2/

# Step 6: Review beats interactively
python ecgclean/active_learning/review_queue.py \
  --queue-dir data/annotation_queues/iteration_2/ \
  --output data/annotation_queues/iteration_2/completed.csv

# Step 7: Import your labels back into labels.parquet
python ecgclean/active_learning/annotation_queue.py import \
  --completed data/annotation_queues/iteration_2/completed.csv \
  --expected-ids data/processed/al_queue_iteration_2.parquet \
  --labels data/processed/labels.parquet \
  --al-iteration 2

# → Go back to Step 1 with v3, v3, v3 ...
```

**When done with AL**, stop after Step 3. Your final `ensemble_preds_vN.parquet` is ready for HRV analysis.

---

## Comparing Model Performance Across Iterations

Metrics are stored inside each `.joblib` artifact. To compare iterations:

```python
import joblib

for version, path in [
    ("v1", "models/beat_tabular_v1.joblib"),
    ("v2", "models/beat_tabular_v2.joblib"),
]:
    art = joblib.load(path)
    m = art["val_metrics"]
    opt = m.get("at_optimal", {})
    print(f"\n── {version} ({art['trained_at'][:10]}) ──")
    print(f"  PR-AUC:   {m.get('pr_auc', float('nan')):.4f}")
    print(f"  ROC-AUC:  {m.get('roc_auc', float('nan')):.4f}")
    print(f"  F1:       {opt.get('f1', 0):.4f}  "
          f"P={opt.get('precision', 0):.3f}  R={opt.get('recall', 0):.3f}")
    print(f"  Threshold: {art['optimal_threshold']:.3f}")
    cm = m.get("confusion_matrix", [])
    if cm:
        print(f"  Confusion: TN={cm[0][0]}  FP={cm[0][1]}  "
              f"FN={cm[1][0]}  TP={cm[1][1]}")
```

---

## When to Stop

- **PR-AUC** is the main signal (it's robust to class imbalance — you have ~1.4% artifacts).
  Diminishing returns typically look like < 0.01–0.02 improvement iteration over iteration.
- **Recall** matters more than precision for this use case: you'd rather flag a clean beat
  as suspicious than miss a real artifact.
- **Gray-area beats** (ones you couldn't confidently call either way) contribute noise to
  training. If you're consistently unsure about the candidates the sampler is surfacing,
  that's a sign the remaining uncertainty is irreducible — a good stopping signal.
- Practically: 3–5 iterations is usually sufficient for a well-behaved AL loop.
