#!/usr/bin/env python3
"""
Build a named subset by symlinking ECG (and matching Peaks) CSVs whose
timestamp range overlaps [start, end).

Usage:
    python Scripts/build_test_subset.py \
        --name smoke_test --start 2025-06-01 --end 2025-07-01
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ECG_DIR = ROOT / "Data" / "ECG"
PEAKS_DIR = ROOT / "Data" / "Peaks"
SUBSETS_DIR = ROOT / "Data" / "Subsets"


def _read_first_last_ms(csv_path: Path) -> tuple[int, int] | None:
    """Read first and last DateTime (epoch ms) from a CSV file.

    Reads only the header + first data row forward, then seeks to the end of
    the file and scans backward for the last non-empty line — O(1) I/O.
    """
    try:
        with csv_path.open("rb") as fb:
            # ── First row ────────────────────────────────────────────────
            raw_header = fb.readline().decode("utf-8", errors="replace").strip().replace("\x00", "")
            if not raw_header:
                return None
            cols = [c.strip() for c in raw_header.split(",")]
            try:
                dt_col = cols.index("DateTime")
            except ValueError:
                return None

            raw_first = fb.readline().decode("utf-8", errors="replace").strip().replace("\x00", "")
            if not raw_first:
                return None
            first_fields = raw_first.split(",")
            if len(first_fields) <= dt_col:
                return None
            try:
                first_val = int(first_fields[dt_col])
            except ValueError:
                return None

            # ── Last row — binary seek backward ──────────────────────────
            fb.seek(0, 2)
            file_size = fb.tell()
            if file_size == 0:
                return None

            chunk = 4096
            pos = file_size
            last_line = b""
            while pos > 0:
                read_size = min(chunk, pos)
                pos -= read_size
                fb.seek(pos)
                data = fb.read(read_size)
                # Strip trailing newline on the very last read
                combined = data + last_line
                lines = combined.split(b"\n")
                # Walk lines from the end looking for a non-empty one
                for line in reversed(lines[1:]):  # skip potential partial leading chunk
                    stripped = line.strip().replace(b"\x00", b"")
                    if stripped:
                        last_line = stripped
                        break
                else:
                    last_line = lines[0] + last_line
                    continue
                break
            else:
                # Reached start of file — use what we have
                last_line = last_line or b""

            raw_last = last_line.decode("utf-8", errors="replace")
            last_fields = raw_last.split(",")
            if len(last_fields) <= dt_col:
                return None
            try:
                last_val = int(last_fields[dt_col])
            except ValueError:
                return None

    except OSError as exc:
        print(f"  WARNING: could not read {csv_path.name}: {exc}", file=sys.stderr)
        return None

    return first_val, last_val


def _ecg_stem_to_peaks_stem(ecg_stem: str) -> str:
    """ECG files use '_' as the date-range separator; Peaks use ' - '."""
    return ecg_stem.replace("_", " - ", 1)


def _ensure_gitignore(root: Path) -> None:
    marker = "Data/Subsets/"
    gitignore = root / ".gitignore"
    if gitignore.exists():
        if marker in gitignore.read_text():
            return
    with gitignore.open("a") as fh:
        fh.write(
            "\n# ── Test subsets (symlinks, never commit) ─────────────────────────────────\n"
            f"{marker}\n"
        )
    print(f"Added '{marker}' to .gitignore")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a named subset by symlinking ECG/Peaks files in a date window."
    )
    parser.add_argument("--name", required=True, help="Subset name")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive, UTC)")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive, UTC)")
    args = parser.parse_args()

    start_dt = datetime.datetime.fromisoformat(args.start).replace(tzinfo=datetime.timezone.utc)
    end_dt = datetime.datetime.fromisoformat(args.end).replace(tzinfo=datetime.timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    subset_dir = SUBSETS_DIR / args.name
    ecg_out = subset_dir / "ECG"
    peaks_out = subset_dir / "Peaks"
    proc_out = subset_dir / "Processed"

    for d in (ecg_out, peaks_out, proc_out):
        d.mkdir(parents=True, exist_ok=True)

    ecg_files = sorted(ECG_DIR.glob("*.csv"))
    if not ecg_files:
        print(f"ERROR: No CSVs found in {ECG_DIR}", file=sys.stderr)
        return 1

    print(f"Scanning {len(ecg_files)} ECG files for [{args.start}, {args.end})...")

    n_ecg = 0
    n_peaks = 0
    total_duration_ms = 0

    for ecg_path in ecg_files:
        result = _read_first_last_ms(ecg_path)
        if result is None:
            continue
        first_ms, last_ms = result

        # Overlap check: file range must intersect [start_ms, end_ms)
        if last_ms < start_ms or first_ms >= end_ms:
            continue

        link = ecg_out / ecg_path.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(ecg_path)
        n_ecg += 1

        clamped_first = max(first_ms, start_ms)
        clamped_last = min(last_ms, end_ms)
        total_duration_ms += max(0, clamped_last - clamped_first)

        # Match peaks file (separator transform: _ → " - ")
        peaks_stem = _ecg_stem_to_peaks_stem(ecg_path.stem)
        peaks_path = PEAKS_DIR / (peaks_stem + ".csv")
        if peaks_path.exists():
            p_link = peaks_out / peaks_path.name
            if p_link.is_symlink() or p_link.exists():
                p_link.unlink()
            p_link.symlink_to(peaks_path)
            n_peaks += 1

    _ensure_gitignore(ROOT)

    total_days = total_duration_ms / (1_000 * 86_400)

    print(f"\nSubset '{args.name}' → {subset_dir}")
    print(f"  ECG files linked  : {n_ecg}")
    print(f"  Duration (days)   : {total_days:.2f}")
    print(f"  Peaks files linked: {n_peaks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
