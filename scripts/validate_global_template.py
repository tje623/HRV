#!/usr/bin/env python3
"""
scripts/validate_global_template.py — Global Template Discriminability Diagnostic

Answers: "Does the global clean QRS template actually separate clean beats
from artifact beats?"

Loads global_template_features.parquet (per-beat Pearson correlation to the
global template), merges with labels, and produces a self-contained text
report covering:

  • Per-group summary statistics (clean / artifact / hard_filtered)
  • ASCII density histograms showing distribution overlap
  • ROC-AUC and PR-AUC treating global_corr_clean as a standalone detector
  • Suggested operating thresholds (high-confidence clean, high-confidence
    artifact, and the ambiguous zone between them)

Usage
-----
    cd "/Volumes/xHRV/Artifact Detector"
    source /Users/tannereddy/.envs/hrv/bin/activate

    python scripts/validate_global_template.py \\
        --templates               data/templates/global_templates.joblib \\
        --beat-features           data/processed/beat_features.parquet \\
        --labels                  data/processed/labels.parquet \\
        --global-template-features data/processed/global_template_features.parquet

Notes
-----
  • --beat-features is optional — only used to report feature-count context.
  • --templates    is optional — only used to show template build metadata.
  • If either optional file is absent the script continues without them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Optional imports with graceful fallbacks ──────────────────────────────────
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    import joblib as _joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_W = 78  # total report width

def _rule(char: str = "═") -> str:
    return char * _W

def _header(title: str) -> str:
    pad   = _W - 2 - len(title)
    left  = pad // 2
    right = pad - left
    return f"{'═' * left} {title} {'═' * right}"

def _section(title: str) -> str:
    return f"\n{_header(title)}\n"


def _fmt_pct(n: int, total: int) -> str:
    return f"{n:>8,}  ({100 * n / total:5.1f}%)" if total > 0 else f"{n:>8,}"


def _summary_stats(values: np.ndarray, name: str) -> str:
    """Return a formatted multi-line summary of a 1-D float array."""
    if len(values) == 0:
        return f"  {name}: <no data>\n"
    pcts = np.percentile(values, [5, 25, 50, 75, 95])
    lines = [
        f"  {name}  (n={len(values):,})",
        f"    mean ± std : {values.mean():.4f} ± {values.std():.4f}",
        f"    median     : {pcts[2]:.4f}",
        f"    p5 / p25   : {pcts[0]:.4f} / {pcts[1]:.4f}",
        f"    p75 / p95  : {pcts[3]:.4f} / {pcts[4]:.4f}",
        f"    min / max  : {values.min():.4f} / {values.max():.4f}",
    ]
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# ASCII HISTOGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def _ascii_histogram(
    groups: dict[str, np.ndarray],
    n_bins: int = 40,
    bar_width: int = 50,
) -> str:
    """Overlay ASCII histogram for multiple groups on a shared bin axis.

    Each group is normalised to a density (area ≈ 1) so groups with very
    different counts are still visually comparable.  Each bin row shows one
    bar per group, scaled to ``bar_width`` characters.

    Args:
        groups: Mapping of label → 1-D float array of correlation values.
        n_bins: Number of equal-width bins spanning the joint data range.
        bar_width: Maximum bar length in characters.

    Returns:
        Multi-line string ready to print.
    """
    all_vals = np.concatenate(list(groups.values()))
    if len(all_vals) == 0:
        return "  <no data>\n"

    lo = float(np.percentile(all_vals, 0.5))   # clip extreme outliers for display
    hi = float(np.percentile(all_vals, 99.5))
    if lo >= hi:
        lo, hi = float(all_vals.min()), float(all_vals.max())
    if lo >= hi:
        return "  <all values identical>\n"

    edges = np.linspace(lo, hi, n_bins + 1)
    bin_width = edges[1] - edges[0]

    # Compute density histograms
    densities: dict[str, np.ndarray] = {}
    for name, vals in groups.items():
        if len(vals) == 0:
            densities[name] = np.zeros(n_bins)
            continue
        clipped = np.clip(vals, lo, hi)
        counts, _ = np.histogram(clipped, bins=edges)
        total = counts.sum()
        densities[name] = counts / (total * bin_width) if total > 0 else counts.astype(float)

    global_max = max(d.max() for d in densities.values() if len(d) > 0)
    if global_max == 0:
        return "  <zero density everywhere>\n"

    # Symbol and colour map
    symbols = {"clean": "█", "artifact": "▓", "hard_filtered": "░"}
    default_syms = ["▪", "▫", "◆", "◇"]
    sym_list = list(symbols.values()) + default_syms
    name_sym: dict[str, str] = {}
    for i, name in enumerate(groups):
        name_sym[name] = symbols.get(name, sym_list[i % len(sym_list)])

    lines: list[str] = []

    # Legend
    legend_parts = [f"  {sym}={name} (n={len(vals):,})"
                    for (name, vals), sym in zip(groups.items(), name_sym.values())]
    lines.append("  " + "   ".join(legend_parts))
    lines.append("")

    # Axis label
    lines.append(f"  {'corr':>6}  {'density (normalised)':^{bar_width}}")
    lines.append(f"  {'':>6}  {lo:.3f}{' ' * (bar_width - 11)}{hi:.3f}")
    lines.append(f"  {'':>6}  {'─' * bar_width}")

    for b in range(n_bins):
        mid   = (edges[b] + edges[b + 1]) / 2
        row   = f"  {mid:>6.3f}  "
        stacked = ""
        for name in groups:
            d   = densities[name][b]
            bar = int(round(bar_width * d / global_max))
            sym = name_sym[name]
            # Overlay: replace existing spaces with this group's symbol
            for j in range(bar):
                if j < len(stacked):
                    if stacked[j] == " ":
                        stacked = stacked[:j] + sym + stacked[j + 1:]
                else:
                    stacked += sym
            # Pad to full bar_width with spaces for next group overlay
            stacked = stacked.ljust(bar_width)
        lines.append(row + stacked.rstrip())

    lines.append(f"  {'':>6}  {'─' * bar_width}")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _threshold_analysis(
    clean_corr: np.ndarray,
    artifact_corr: np.ndarray,
) -> str:
    """Suggest operating thresholds based on percentile criteria.

    High-confidence clean  : corr ≥ p5 of known-clean    → captures 95%+ of clean
    High-confidence artifact: corr ≤ p90 of known-artifact → captures 90%+ of artifacts

    Args:
        clean_corr: Correlation values for confirmed-clean beats.
        artifact_corr: Correlation values for confirmed-artifact beats.

    Returns:
        Formatted string describing the thresholds and ambiguous zone.
    """
    lines: list[str] = []

    if len(clean_corr) == 0 or len(artifact_corr) == 0:
        return "  Insufficient data for threshold analysis.\n"

    # "High confidence clean" — threshold above which 95% of clean beats fall
    thresh_clean = float(np.percentile(clean_corr, 5))
    n_clean_above = int((clean_corr >= thresh_clean).sum())
    n_art_above   = int((artifact_corr >= thresh_clean).sum())
    pct_clean_above = 100 * n_clean_above / len(clean_corr)
    pct_art_above   = 100 * n_art_above   / len(artifact_corr)

    # "High confidence artifact" — threshold below which 90% of artifact beats fall
    thresh_artifact = float(np.percentile(artifact_corr, 90))
    n_clean_below = int((clean_corr <= thresh_artifact).sum())
    n_art_below   = int((artifact_corr <= thresh_artifact).sum())
    pct_clean_below = 100 * n_clean_below / len(clean_corr)
    pct_art_below   = 100 * n_art_below   / len(artifact_corr)

    lines += [
        f"  High-confidence CLEAN threshold  (p5  of clean)    : corr ≥ {thresh_clean:+.4f}",
        f"    → retains  {pct_clean_above:5.1f}% of clean beats  "
        f"({n_clean_above:,} / {len(clean_corr):,})",
        f"    → passes   {pct_art_above:5.1f}% of artifact beats "
        f"({n_art_above:,} / {len(artifact_corr):,})",
        "",
        f"  High-confidence ARTIFACT threshold (p90 of artifact): corr ≤ {thresh_artifact:+.4f}",
        f"    → captures {pct_art_below:5.1f}% of artifact beats "
        f"({n_art_below:,} / {len(artifact_corr):,})",
        f"    → flags     {pct_clean_below:5.1f}% of clean beats  "
        f"({n_clean_below:,} / {len(clean_corr):,})",
        "",
    ]

    # Ambiguous zone
    lo_zone = min(thresh_artifact, thresh_clean)
    hi_zone = max(thresh_artifact, thresh_clean)

    if thresh_artifact >= thresh_clean:
        lines.append(
            "  ⚠  Thresholds OVERLAP (artifact p90 ≥ clean p5) — the template has\n"
            "     limited discriminability.  The 'ambiguous zone' spans the entire\n"
            f"    range [{lo_zone:+.4f}, {hi_zone:+.4f}]."
        )
    else:
        n_clean_ambig = int(((clean_corr > thresh_artifact) & (clean_corr < thresh_clean)).sum())
        n_art_ambig   = int(((artifact_corr > thresh_artifact) & (artifact_corr < thresh_clean)).sum())
        pct_c_ambig   = 100 * n_clean_ambig / len(clean_corr)
        pct_a_ambig   = 100 * n_art_ambig   / len(artifact_corr)
        lines += [
            f"  Ambiguous zone  [{thresh_artifact:+.4f}, {thresh_clean:+.4f}]:",
            f"    clean beats in zone   : {n_clean_ambig:,}  ({pct_c_ambig:.1f}% of clean)",
            f"    artifact beats in zone: {n_art_ambig:,}  ({pct_a_ambig:.1f}% of artifact)",
        ]

    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate discriminability of the global QRS template correlation score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--global-template-features",
        type=str,
        required=True,
        help="Path to global_template_features.parquet (peak_id, global_corr_clean)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels.parquet",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default=None,
        help="(Optional) Path to global_templates.joblib — used for metadata display only",
    )
    parser.add_argument(
        "--beat-features",
        type=str,
        default=None,
        help="(Optional) Path to beat_features.parquet — used for feature-count context only",
    )
    args = parser.parse_args()

    # ── Input validation ──────────────────────────────────────────────────────
    missing = [p for p in [args.global_template_features, args.labels]
               if not Path(p).exists()]
    if missing:
        for p in missing:
            print(f"ERROR: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    # ── Report header ─────────────────────────────────────────────────────────
    print()
    print(_rule())
    print("  GLOBAL TEMPLATE VALIDATION REPORT")
    print(_rule())
    print()

    # ── Optional: template metadata ───────────────────────────────────────────
    if args.templates and Path(args.templates).exists() and _JOBLIB_OK:
        tmpl = _joblib.load(args.templates)
        print("Template metadata:")
        print(f"  built_at      : {tmpl.get('built_at', 'unknown')}")
        print(f"  n_clean_beats : {tmpl.get('n_clean_beats', 'unknown'):,}" if isinstance(tmpl.get('n_clean_beats'), int) else f"  n_clean_beats : {tmpl.get('n_clean_beats', 'unknown')}")
        print(f"  sample_rate   : {tmpl.get('sample_rate_hz', 'unknown')} Hz")
        print(f"  window_size   : {tmpl.get('window_size', 'unknown')} samples")
        print(f"  version       : {tmpl.get('version', 'unknown')}")
        tc = tmpl.get("template_clean")
        if tc is not None:
            print(f"  template stats: min={tc.min():.4f}  max={tc.max():.4f}  std={tc.std():.4f}")
        print()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading global_template_features …", end=" ", flush=True)
    gtf = pd.read_parquet(args.global_template_features, columns=["peak_id", "global_corr_clean"])
    print(f"{len(gtf):,} rows")

    print("Loading labels …", end=" ", flush=True)
    labels = pd.read_parquet(
        args.labels,
        columns=["peak_id", "label", "hard_filtered", "reviewed"]
        if "reviewed" in pd.read_parquet(args.labels, columns=["peak_id"]).columns
        else ["peak_id", "label", "hard_filtered"],
    )
    print(f"{len(labels):,} rows")

    # Re-read with correct columns (the above was a cheap probe)
    try:
        labels = pd.read_parquet(
            args.labels,
            columns=["peak_id", "label", "hard_filtered", "reviewed"],
        )
        has_reviewed = True
    except Exception:
        labels = pd.read_parquet(
            args.labels,
            columns=["peak_id", "label", "hard_filtered"],
        )
        has_reviewed = False

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = gtf.merge(labels, on="peak_id", how="inner")
    n_total = len(df)
    print(f"Merged: {n_total:,} beats with both template correlation and label")

    if n_total == 0:
        print("ERROR: No rows after merge — check peak_id alignment.", file=sys.stderr)
        sys.exit(1)

    # ── Optional: beat-features context ───────────────────────────────────────
    if args.beat_features and Path(args.beat_features).exists():
        import pyarrow.parquet as pq
        bf_schema = pq.read_schema(args.beat_features)
        print(f"beat_features.parquet: {len(bf_schema.names)} columns "
              f"(global template features are separate — join on peak_id to combine)")
    print()

    # ── Label distribution ────────────────────────────────────────────────────
    print(_section("DATASET COMPOSITION"))
    print(f"  Total beats in merged dataset: {n_total:,}")
    print()
    for lbl, cnt in df["label"].value_counts().items():
        print(f"  label={lbl!r:<22} {_fmt_pct(cnt, n_total)}")
    print()
    hf_n = int(df["hard_filtered"].sum())
    print(f"  hard_filtered=True             {_fmt_pct(hf_n, n_total)}")
    if has_reviewed:
        rev_n = int(df["reviewed"].sum())
        print(f"  reviewed=True                  {_fmt_pct(rev_n, n_total)}")
    print()

    # ── Define evaluation groups ──────────────────────────────────────────────
    # Use reviewed-only clean/artifact when available for cleaner separation.
    if has_reviewed:
        clean_mask    = (df["label"] == "clean")    & (~df["hard_filtered"]) & df["reviewed"]
        artifact_mask = (df["label"] == "artifact") & (~df["hard_filtered"]) & df["reviewed"]
        clean_label   = "clean (reviewed, ~hard_filtered)"
        artifact_label = "artifact (reviewed, ~hard_filtered)"
    else:
        clean_mask    = (df["label"] == "clean")    & (~df["hard_filtered"])
        artifact_mask = (df["label"] == "artifact") & (~df["hard_filtered"])
        clean_label   = "clean (~hard_filtered)"
        artifact_label = "artifact (~hard_filtered)"

    hf_mask = df["hard_filtered"].astype(bool)

    clean_corr    = df.loc[clean_mask,    "global_corr_clean"].values.astype(np.float64)
    artifact_corr = df.loc[artifact_mask, "global_corr_clean"].values.astype(np.float64)
    hf_corr       = df.loc[hf_mask,       "global_corr_clean"].values.astype(np.float64)

    # ── Per-group statistics ──────────────────────────────────────────────────
    print(_section("SUMMARY STATISTICS — global_corr_clean"))
    print(_summary_stats(clean_corr,    clean_label))
    print(_summary_stats(artifact_corr, artifact_label))
    print(_summary_stats(hf_corr,       "hard_filtered=True"))

    # All unreviewed clean (to show contamination risk)
    if has_reviewed:
        unrev_clean_mask = (df["label"] == "clean") & (~df["hard_filtered"]) & (~df["reviewed"])
        unrev_corr = df.loc[unrev_clean_mask, "global_corr_clean"].values.astype(np.float64)
        print(_summary_stats(unrev_corr, "clean (UN-reviewed, ~hard_filtered)"))

    # ── ASCII histogram ───────────────────────────────────────────────────────
    print(_section("DISTRIBUTION OVERLAP — ASCII HISTOGRAM"))

    hist_groups: dict[str, np.ndarray] = {}
    if len(clean_corr) > 0:
        hist_groups["clean"] = clean_corr
    if len(artifact_corr) > 0:
        hist_groups["artifact"] = artifact_corr
    if len(hf_corr) > 0:
        hist_groups["hard_filtered"] = hf_corr

    if hist_groups:
        print(_ascii_histogram(hist_groups, n_bins=40, bar_width=52))
    else:
        print("  No data for histogram.\n")

    # ── ROC-AUC / PR-AUC ─────────────────────────────────────────────────────
    print(_section("DISCRIMINABILITY — ROC-AUC & PR-AUC"))

    if not _SKLEARN_OK:
        print("  sklearn not available — skipping AUC metrics.\n")
    elif len(clean_corr) == 0 or len(artifact_corr) == 0:
        print("  Need both clean and artifact beats for AUC — skipping.\n")
    else:
        # Binary labels: 1 = artifact, 0 = clean
        # Score = NEGATIVE correlation (lower corr → more likely artifact)
        y_true  = np.concatenate([np.zeros(len(clean_corr)),    np.ones(len(artifact_corr))])
        y_score = np.concatenate([-clean_corr,                  -artifact_corr])

        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc  = average_precision_score(y_true, y_score)
        prevalence = len(artifact_corr) / (len(clean_corr) + len(artifact_corr))

        print(f"  Evaluation set: {len(clean_corr):,} clean + {len(artifact_corr):,} artifact beats")
        print(f"  Artifact prevalence in eval set: {100 * prevalence:.2f}%")
        print()
        print(f"  ROC-AUC  : {roc_auc:.4f}   (random = 0.50)")
        print(f"  PR-AUC   : {pr_auc:.4f}   (random = {prevalence:.4f}  [= prevalence])")
        print()

        # Interpretation guidance
        if roc_auc >= 0.90:
            interp = "Excellent — global template is a strong standalone detector."
        elif roc_auc >= 0.80:
            interp = "Good — useful as an additional feature; not sufficient alone."
        elif roc_auc >= 0.65:
            interp = "Moderate — adds signal but limited standalone power."
        else:
            interp = "Poor — template may not be representative; check build quality."
        print(f"  Interpretation: {interp}")
        print()

        # Lift over random baseline
        pr_lift = pr_auc / prevalence if prevalence > 0 else float("nan")
        print(f"  PR-AUC lift over random baseline: {pr_lift:.2f}×")
        print()

    # ── Threshold analysis ────────────────────────────────────────────────────
    print(_section("SUGGESTED OPERATING THRESHOLDS"))
    print(_threshold_analysis(clean_corr, artifact_corr))

    # ── Hard-filtered vs artifact overlap ─────────────────────────────────────
    if len(hf_corr) > 0 and len(artifact_corr) > 0:
        print(_section("HARD-FILTERED VS LABELED ARTIFACT"))
        print("  Hard-filtered beats are excluded from training but are worth checking:")
        print(f"  Mean corr — hard_filtered : {hf_corr.mean():.4f}")
        print(f"  Mean corr — artifact      : {artifact_corr.mean():.4f}")
        overlap = float(np.mean(hf_corr < np.percentile(artifact_corr, 75)))
        print(f"  Fraction of hard_filtered below artifact p75: {100*overlap:.1f}%")
        print()

    # ── Quick verdict ─────────────────────────────────────────────────────────
    print(_rule("─"))
    print("  VERDICT")
    print(_rule("─"))

    if not _SKLEARN_OK or len(clean_corr) == 0 or len(artifact_corr) == 0:
        print("  Cannot compute verdict — missing clean or artifact data.")
    else:
        sep_gap = clean_corr.mean() - artifact_corr.mean()
        print(f"  Mean separation (clean − artifact): {sep_gap:+.4f}")

        if sep_gap > 0.10 and roc_auc >= 0.80:
            verdict = (
                "INTEGRATE — template correlation is discriminative enough to\n"
                "  add as a feature in the next model retraining pass."
            )
        elif sep_gap > 0.05 or roc_auc >= 0.65:
            verdict = (
                "MARGINAL — consider adding as a weak feature, but investigate\n"
                "  template quality (n_clean_beats, build date, reviewed fraction)."
            )
        else:
            verdict = (
                "DO NOT INTEGRATE YET — template shows insufficient separation.\n"
                "  Rebuild with more verified-clean beats or check data alignment."
            )
        print(f"  {verdict}")
    print(_rule("─"))
    print()


if __name__ == "__main__":
    main()
