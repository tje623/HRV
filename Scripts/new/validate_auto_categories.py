#!/usr/bin/env python3
"""
scripts/validate_auto_categories.py — Sanity-check auto-categories against labels.

Answers: "Do the auto-categories make sense? Do they agree with existing labels?
Where do they disagree, and why?"

Output goes to both stdout and --output (text report file).

Usage
-----
    python scripts/validate_auto_categories.py \\
        --auto-categories data/processed/auto_categories.parquet \\
        --labels          data/processed/labels.parquet \\
        --beat-features   data/processed/beat_features_merged.parquet \\
        --output          data/auto_categorization/validation_report.txt
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.validate_auto_categories")

# ── Label vocabulary ──────────────────────────────────────────────────────────
CLEAN_REAL_LABELS   = frozenset({"clean", "missed_original", "interpolated", "phys_event"})
ARTIFACT_LABEL      = "artifact"
CLEAN_CAT_NAMES     = frozenset({"pristine", "clean_normal", "clean_low_amplitude", "clean_noisy"})
ARTIFACT_CAT_NAMES  = frozenset({"artifact_morphology", "artifact_noise",
                                  "artifact_general", "baseline_wander"})

KEY_FEATURES = [
    "global_corr_clean",
    "r_peak_snr",
    "window_hf_noise_rms",
    "window_iqr",
    "window_wander_slope",
    "window_zcr",
]
N_SURPRISING = 100


# ── Tee: write to stdout AND file simultaneously ──────────────────────────────

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text: str) -> None:
        for s in self.streams:
            s.write(text)
    def flush(self) -> None:
        for s in self.streams:
            s.flush()


# ── ASCII histogram ───────────────────────────────────────────────────────────

def _ascii_histogram(
    series_dict: dict[str, np.ndarray],
    title: str,
    n_bins: int = 25,
    bar_width: int = 40,
) -> str:
    """Return a multi-group ASCII histogram string."""
    all_vals = np.concatenate([v for v in series_dict.values() if len(v)])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return f"  {title}: (no data)\n"

    lo = float(np.percentile(all_vals, 1))
    hi = float(np.percentile(all_vals, 99))
    if lo == hi:
        return f"  {title}: (constant value {lo:.4g})\n"

    bins = np.linspace(lo, hi, n_bins + 1)
    symbols = ["█", "▓", "░", "·"]
    names   = list(series_dict.keys())

    hists: list[np.ndarray] = []
    for name in names:
        v = series_dict[name]
        v = v[np.isfinite(v)]
        h, _ = np.histogram(v, bins=bins)
        hists.append(h / max(h.max(), 1))

    lines = [f"\n  {title}"]
    # Legend
    legend_parts = [f"{symbols[i % len(symbols)]}={names[i]}" for i in range(len(names))]
    lines.append("  " + "  ".join(legend_parts[:4]))

    for i in range(n_bins):
        label = f"{bins[i]:7.3f}"
        bars  = "  ".join(
            symbols[j % len(symbols)] * int(hists[j][i] * bar_width)
            for j in range(len(names))
        )
        lines.append(f"  {label} |{bars}")
    return "\n".join(lines) + "\n"


# ── Main validation ───────────────────────────────────────────────────────────

def validate(
    auto_cats_path: Path,
    labels_path:    Path,
    features_path:  Path,
    output_path:    Path,
) -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading auto-categories from %s", auto_cats_path)
    auto_cats = pd.read_parquet(auto_cats_path)

    logger.info("Loading labels from %s", labels_path)
    labels = pd.read_parquet(labels_path)
    if labels.index.name == "peak_id" and "peak_id" not in labels.columns:
        labels = labels.reset_index()

    logger.info("Loading features from %s", features_path)
    features = pd.read_parquet(features_path)
    if features.index.name == "peak_id":
        features = features.reset_index()

    # ── Join ───────────────────────────────────────────────────────────────────
    # Inner join: keep only beats present in all three sources
    df = auto_cats.merge(labels, on="peak_id", how="inner")
    df = df.merge(features[["peak_id"] + [c for c in KEY_FEATURES if c in features.columns]],
                  on="peak_id", how="left")
    n_total = len(df)
    logger.info("Joined dataset: %d beats", n_total)

    has_reviewed = "reviewed" in df.columns
    reviewed_mask = df["reviewed"].astype(bool) if has_reviewed else pd.Series([False] * n_total)

    # ── Binary prediction from auto-category ─────────────────────────────────
    df["auto_binary"] = df["category"].apply(
        lambda c: 0 if c in CLEAN_CAT_NAMES else (1 if c in ARTIFACT_CAT_NAMES else -1)
    )
    df["label_binary"] = df["label"].apply(
        lambda l: 0 if l in CLEAN_REAL_LABELS else (1 if l == ARTIFACT_LABEL else -1)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as report_file:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, report_file)  # type: ignore[assignment]
        try:
            _run_report(df, reviewed_mask, features, output_path, n_total)
        finally:
            sys.stdout = old_stdout

    logger.info("Report written → %s", output_path)


def _run_report(
    df: pd.DataFrame,
    reviewed_mask: pd.Series,
    features: pd.DataFrame,
    output_path: Path,
    n_total: int,
) -> None:
    W = 72
    print()
    print("═" * W)
    print("  AUTO-CATEGORY VALIDATION REPORT")
    print("═" * W)
    print(f"  Total beats in joined dataset : {n_total:,}")
    print(f"  Reviewed beats                : {int(reviewed_mask.sum()):,}")
    print()

    # ── Per-category stats ────────────────────────────────────────────────────
    print("─" * W)
    print("  PER-CATEGORY STATISTICS")
    print("─" * W)

    all_cats = sorted(df["category"].unique())
    feat_avail = [f for f in KEY_FEATURES if f in df.columns]

    # Header
    hdr_parts = [f"{'Category':<28}", f"{'N':>8}", f"{'%':>6}"]
    for f in feat_avail:
        hdr_parts.append(f"{f[:10]:>11}")
    print("  " + "  ".join(hdr_parts))
    print("  " + "─" * (W - 2))

    for cat in all_cats:
        mask = df["category"] == cat
        n    = int(mask.sum())
        pct  = 100.0 * n / max(n_total, 1)
        row  = [f"  {cat:<28}", f"{n:>8,}", f"{pct:>5.1f}%"]
        for f in feat_avail:
            if f in df.columns:
                med = float(df.loc[mask, f].median())
                row.append(f"{med:>11.3f}")
        print("".join(row))

    # Label distribution per category
    print()
    print("─" * W)
    print("  LABEL DISTRIBUTION WITHIN EACH CATEGORY (reviewed beats)")
    print("─" * W)
    rev_df = df[reviewed_mask]
    all_labels = sorted(rev_df["label"].unique())
    print(f"  {'Category':<28}  " + "  ".join(f"{l[:14]:>14}" for l in all_labels))
    print("  " + "─" * (W - 2))
    for cat in all_cats:
        mask = (rev_df["category"] == cat)
        n    = int(mask.sum())
        if n == 0:
            continue
        counts_str = "  ".join(
            f"{int((rev_df.loc[mask, 'label'] == l).sum()):>14,}"
            for l in all_labels
        )
        print(f"  {cat:<28}  {counts_str}   (n={n:,})")

    # ── Agreement computation ─────────────────────────────────────────────────
    print()
    print("─" * W)
    print("  AGREEMENT: auto-category vs labels (reviewed beats only)")
    print("─" * W)

    # Only use beats where both have a clear binary mapping
    cmp = rev_df[(rev_df["auto_binary"] != -1) & (rev_df["label_binary"] != -1)].copy()
    n_cmp = len(cmp)
    if n_cmp > 0:
        n_agree    = int((cmp["auto_binary"] == cmp["label_binary"]).sum())
        n_c2a      = int(((cmp["auto_binary"] == 1) & (cmp["label_binary"] == 0)).sum())
        n_a2c      = int(((cmp["auto_binary"] == 0) & (cmp["label_binary"] == 1)).sum())

        try:
            from sklearn.metrics import confusion_matrix as _cm
            cm = _cm(cmp["label_binary"], cmp["auto_binary"], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            print(f"  Beats with unambiguous binary mapping : {n_cmp:,}")
            print(f"  Agreement (both same)                 : {n_agree:,}  ({100.0*n_agree/n_cmp:.2f}%)")
            print(f"  auto=clean  but label=artifact (FN)  : {fn:,}")
            print(f"  auto=artifact but label=clean  (FP)  : {fp:,}")
            print()
            print(f"  Confusion matrix  (rows=auto, cols=label)")
            print(f"    auto clean   | TN={tn:>7,}  FN={fn:>7,}")
            print(f"    auto artifact| FP={fp:>7,}  TP={tp:>7,}")

            if n_cmp > 0 and (int(cmp['label_binary'].sum()) > 0):
                from sklearn.metrics import average_precision_score, roc_auc_score
                y_true = cmp["label_binary"].values
                y_score = cmp["auto_binary"].values.astype(float)
                n_pos = int(y_true.sum())
                n_neg = n_cmp - n_pos
                if n_pos > 0 and n_neg > 0:
                    pr  = float(average_precision_score(y_true, y_score))
                    roc = float(roc_auc_score(y_true, y_score))
                    print(f"\n  PR-AUC  (reviewed, binary) : {pr:.4f}")
                    print(f"  ROC-AUC (reviewed, binary) : {roc:.4f}")
        except ImportError:
            print(f"  Agreement: {n_agree}/{n_cmp} ({100.0*n_agree/n_cmp:.2f}%)")
            print(f"  clean→artifact : {n_c2a:,}")
            print(f"  artifact→clean : {n_a2c:,}")
    else:
        print("  (no reviewed beats with unambiguous binary mapping)")

    # ── Surprising disagreements ───────────────────────────────────────────────
    print()
    print("─" * W)
    print(f"  TOP {N_SURPRISING} MOST SURPRISING DISAGREEMENTS")
    print("─" * W)

    # Disagreement = auto_binary != label_binary, among reviewed beats
    # Surprise score = category_confidence × is_disagreeing
    disagree_mask = (
        reviewed_mask
        & (df["auto_binary"] != -1)
        & (df["label_binary"] != -1)
        & (df["auto_binary"] != df["label_binary"])
    )
    disagree_df = df[disagree_mask].copy()
    disagree_df["surprise_score"] = disagree_df["category_confidence"].astype(float)
    top_surprises = disagree_df.nlargest(N_SURPRISING, "surprise_score")

    print(f"  Total disagreeing reviewed beats : {len(disagree_df):,}")
    print(f"  Writing top {N_SURPRISING} to surprising_disagreements.csv …")

    surprises_csv = output_path.parent / "surprising_disagreements.csv"
    feature_cols_for_csv = [c for c in features.columns
                             if c != "peak_id" and c in df.columns]
    surprise_out_cols = (
        ["peak_id", "label", "auto_category", "category_confidence", "surprise_score"]
        + [f for f in feature_cols_for_csv if f in top_surprises.columns]
    )
    # Rename for clarity
    surprise_write = top_surprises[
        [c for c in surprise_out_cols if c in top_surprises.columns]
    ].rename(columns={"category": "auto_category"})
    # category column might still be named "category" if not renamed
    if "category" in top_surprises.columns and "auto_category" not in top_surprises.columns:
        top_surprises = top_surprises.rename(columns={"category": "auto_category"})
        surprise_write = top_surprises[
            [c for c in surprise_out_cols if c in top_surprises.columns]
        ]

    surprise_write.to_csv(surprises_csv, index=False)
    print(f"  Saved → {surprises_csv}")

    # Print a short preview (use cat_col = whichever name exists)
    print()
    if len(top_surprises) > 0:
        cat_col = "auto_category" if "auto_category" in top_surprises.columns else "category"
        preview_cols = ["peak_id", "label", cat_col, "category_confidence"]
        for f in ["global_corr_clean", "r_peak_snr"]:
            if f in top_surprises.columns:
                preview_cols.append(f)
        hdr = "  " + "  ".join(f"{c[:18]:<18}" for c in preview_cols)
        print(hdr)
        for _, row in top_surprises.head(10).iterrows():
            vals = "  ".join(
                f"{str(row[c])[:18]:<18}" if c in row.index else f"{'N/A':<18}"
                for c in preview_cols
            )
            print(f"  {vals}")
        if len(top_surprises) > 10:
            print(f"  … ({len(top_surprises)} total — see {surprises_csv.name})")

    # ── ASCII histograms ───────────────────────────────────────────────────────
    print()
    print("─" * W)
    print("  FEATURE DISTRIBUTIONS BY AUTO-CATEGORY")
    print("─" * W)
    print("  (showing clean vs artifact groupings + up to 4 individual categories)")

    for feat in feat_avail:
        if feat not in df.columns:
            continue
        # Group 1: all clean categories vs all artifact categories
        clean_vals = df.loc[df["category"].isin(CLEAN_CAT_NAMES), feat].dropna().values
        art_vals   = df.loc[df["category"].isin(ARTIFACT_CAT_NAMES), feat].dropna().values
        unc_vals   = df.loc[df["category"] == "uncertain", feat].dropna().values

        series: dict[str, np.ndarray] = {}
        if len(clean_vals): series["clean"]    = clean_vals
        if len(art_vals):   series["artifact"] = art_vals
        if len(unc_vals):   series["uncertain"] = unc_vals

        print(_ascii_histogram(series, title=feat, n_bins=20, bar_width=30))

    print()
    print("═" * W)
    print("  VALIDATION COMPLETE")
    print("═" * W)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check auto-categories against existing labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--auto-categories", type=Path, required=True,
                        help="auto_categories.parquet from auto_categorize_beats.py apply")
    parser.add_argument("--labels",          type=Path, required=True,
                        help="labels.parquet")
    parser.add_argument("--beat-features",   type=Path, required=True,
                        help="beat_features_merged.parquet (for feature values)")
    parser.add_argument("--output",          type=Path, required=True,
                        help="Output text report path")
    args = parser.parse_args()

    for flag, path in [
        ("--auto-categories", args.auto_categories),
        ("--labels",          args.labels),
        ("--beat-features",   args.beat_features),
    ]:
        if not path.exists():
            logger.error("File not found for %s: %s", flag, path)
            sys.exit(1)

    validate(
        auto_cats_path = args.auto_categories,
        labels_path    = args.labels,
        features_path  = args.beat_features,
        output_path    = args.output,
    )


if __name__ == "__main__":
    main()
