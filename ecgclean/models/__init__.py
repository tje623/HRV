"""
ecgclean.models — Model training and inference for ECG artifact detection.

Submodules:
    segment_quality        — Stage 0: segment-level noise gate (LightGBM multiclass)
    beat_artifact_tabular  — Stage 1: beat-level artifact classifier (LightGBM binary)
    beat_artifact_cnn      — Stage 2a: hybrid CNN + tabular beat-level classifier (PyTorch Lightning)
    segment_cnn_2d         — Stage 0b: 2D CNN segment quality from CWT scalograms (PyTorch Lightning)
    ensemble               — Stage 3: ensemble fusion (linear blend + logistic meta-classifier)
    pretrain_ssl           — Stage 3b: self-supervised denoising autoencoder pretraining
"""
