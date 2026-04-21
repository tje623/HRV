import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    HR_CSV, IHR_INTERVALS_JSON, HR_DEC, DT_HR_FORMAT, RR_CSV,
)

import os
import json
import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
''' Interval Ledger System (Copied from hrv.py for consistency)                   '''
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IntervalLedger:
    """Stores processed coverage as merged, non-overlapping [beg_ms, end_ms] intervals."""
    def __init__(self, path: str):
        self.path = path
        self.schema = 1
        self.intervals = []
        self._loaded = False

    @staticmethod
    def _merge(intervals):
        if not intervals: return []
        intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged = [intervals[0][:]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    @staticmethod
    def _subtract(base_interval, covered):
        S, E = base_interval
        gaps = []
        cur = S
        for s, e in covered:
            if e < S or s > E: continue
            if s > cur:
                gaps.append([cur, s - 1])
            cur = max(cur, e + 1)
            if cur > E: break
        if cur <= E:
            gaps.append([cur, E])
        return gaps

    def load(self):
        if self._loaded: return
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f: data = json.load(f)
                if isinstance(data, dict) and "intervals" in data:
                    self.intervals = self._merge([[int(s), int(e)] for s, e in data["intervals"]])
            else: self.intervals = []
        except Exception as e:
            print(f"[WARNING] ❗ Could not load interval ledger at '{self.path}': {e}. Starting empty.")
            self.intervals = []
        self._loaded = True

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {"schema": self.schema, "intervals": self._merge(self.intervals)}
        with open(self.path, "w") as f: json.dump(payload, f, indent=2)

    def add_intervals(self, new_intervals_ms):
        self.load()
        cleaned = [[int(s), int(e)] for s, e in new_intervals_ms if s is not None and e is not None]
        self.intervals = self._merge(self.intervals + cleaned)
        self.save()

    def unprocessed_subintervals(self, beg_ms: int, end_ms: int):
        self.load()
        if beg_ms is None or end_ms is None or end_ms < beg_ms: return []
        covered = [[max(s, beg_ms), min(e, end_ms)] for s, e in self.intervals if e >= beg_ms and s <= end_ms]
        return self._subtract([beg_ms, end_ms], self._merge(covered))

def filter_new_rows_by_intervals(full_df: pd.DataFrame, intervals_path: str):
    """Filters a DataFrame to include only rows not covered by the IntervalLedger."""
    if 'DateTime' not in full_df.columns:
        raise ValueError("❗ Expected 'DateTime' column in RR dataframe.")

    # Ensure DateTime is a numeric millisecond timestamp for filtering
    ms_series = pd.to_numeric(full_df['DateTime'], errors='coerce')
    if ms_series.isna().all():
        print("[ERROR] ⛔ Could not parse any timestamps from 'DateTime' column.")
        return pd.DataFrame(), []

    # Get the time range of the data
    min_ms_all, max_ms_all = int(ms_series.min()), int(ms_series.max())

    ledger = IntervalLedger(intervals_path)
    gaps = ledger.unprocessed_subintervals(min_ms_all, max_ms_all)

    if not gaps:
        print("[INFO] ✅ No unprocessed intervals found. Data is up-to-date.")
        return pd.DataFrame(), []

    print(f"[INFO] 📌 Found {len(gaps)} unprocessed time interval(s). Filtering new data...")

    parts = []
    for start_gap, end_gap in gaps:
        mask = (ms_series >= start_gap) & (ms_series <= end_gap)
        chunk = full_df.loc[mask]
        if not chunk.empty:
            parts.append(chunk)

    if not parts:
        print("[INFO] 📌 Gaps present in ledger but no corresponding data rows found.")
        return pd.DataFrame(), []

    new_df = pd.concat(parts, axis=0).sort_values('DateTime').reset_index(drop=True)
    return new_df, gaps


def calculate_and_append_ihr(new_rr_df, output_hr_csv_path, hr_decimal_places=1):
    """
    Calculates IHR for new RR data and appends it to a master CSV file.
    """
    if new_rr_df.empty:
        print("[INFO] No new data to process.")
        return

    print(f"[INFO] 🔄 Processing {len(new_rr_df)} new RR intervals for IHR calculation...")

    # 1. Parse DateTime and sort chronologically
    dt_parsed = pd.to_datetime(new_rr_df['DateTime'], unit='ms', errors='coerce')
    initial_count = len(new_rr_df)

    # Drop rows where timestamp could not be parsed and report it
    valid_dt_mask = dt_parsed.notna()
    new_rr_df = new_rr_df[valid_dt_mask].copy()
    dt_parsed = dt_parsed[valid_dt_mask]
    dropped_count = initial_count - len(new_rr_df)
    if dropped_count > 0:
        print(f"[WARNING] ⚠️ Dropped {dropped_count} row(s) due to unparseable timestamps.")

    new_rr_df.sort_values('DateTime', inplace=True)

    # 2. Calculate IHR
    rr_ms = pd.to_numeric(new_rr_df['RR'], errors='coerce')
    initial_count = len(new_rr_df)

    # Calculate HR only for valid RR intervals
    valid_rr_mask = rr_ms.gt(0) & rr_ms.notna()
    hr = pd.Series(np.nan, index=new_rr_df.index, dtype='float')
    hr.loc[valid_rr_mask] = (60_000.0 / rr_ms[valid_rr_mask]).round(hr_decimal_places)

    # Prepare the new data for output
    new_hr_data = pd.DataFrame({'DateTime': dt_parsed, 'HR': hr})

    # Drop rows where HR calculation failed (bad RR value) and report it
    new_hr_data.dropna(subset=['HR'], inplace=True)
    dropped_count = initial_count - len(new_hr_data)
    if dropped_count > 0:
        print(f"[WARNING] ⚠️ Dropped {dropped_count} row(s) due to invalid RR values (e.g., zero, negative, or non-numeric).")

    if new_hr_data.empty:
        print("[WARNING] No valid IHR data was generated from the new rows.")
        return

    # 3. Load existing data if it exists
    if os.path.exists(output_hr_csv_path):
        print(f"[INFO] 🔄 Loading existing data from {os.path.basename(output_hr_csv_path)}...")
        existing_df = pd.read_csv(output_hr_csv_path)
        existing_df['DateTime'] = pd.to_datetime(existing_df['DateTime'], errors='coerce')
        combined_df = pd.concat([existing_df, new_hr_data], ignore_index=True)
    else:
        print("[INFO] 📌 No existing IHR file found. Creating a new one.")
        combined_df = new_hr_data

    # 4. De-duplicate and sort the combined data
    combined_df.drop_duplicates(subset=['DateTime'], keep='last', inplace=True)
    combined_df.sort_values(by='DateTime', inplace=True)

    # 5. Format columns for saving
    # Tidy HR serialization: integers without .0, keep non-zero decimals
    def _tidy_hr(v):
        if pd.isna(v): return v
        f = float(v)
        return int(f) if f.is_integer() else f

    combined_df['HR'] = combined_df['HR'].apply(_tidy_hr)
    combined_df['DateTime'] = combined_df['DateTime'].dt.strftime(DT_HR_FORMAT)

    # 6. Save to master CSV
    try:
        combined_df.to_csv(output_hr_csv_path, index=False, na_rep='NaN')
        print(f"[SUCCESS] ✅ Instantaneous heart rate data saved to: '{os.path.basename(output_hr_csv_path)}'")
        print(f"[INFO] Output file now contains {len(combined_df)} total data points.")
    except Exception as e:
        print(f"[ERROR] ⛔ Failed to save output CSV '{output_hr_csv_path}': {e}")
        raise

def main():
    # Resolve paths and settings from config
    rr_csv_path = RR_CSV
    hr_csv_path = HR_CSV
    intervals_path = str(IHR_INTERVALS_JSON)
    hr_decimals = HR_DEC

    # 1. Load the full source RR data
    print(f"[INFO] Loading full RR dataset from {os.path.basename(rr_csv_path)}...")
    try:
        full_rr_df = pd.read_csv(rr_csv_path)
    except FileNotFoundError:
        print(f"[ERROR] ⛔ Input RR data file not found at: {rr_csv_path}")
        return
    except Exception as e:
        print(f"[ERROR] ❌ Failed to load or parse RR data from {rr_csv_path}: {e}")
        return

    # 2. Filter for only new, unprocessed rows using the ledger
    new_rr_df, processed_gaps = filter_new_rows_by_intervals(full_rr_df, intervals_path)

    if new_rr_df.empty:
        return # filter_new_rows will have already printed a status message

    # 3. Process new data and append to the master IHR file
    try:
        calculate_and_append_ihr(new_rr_df, hr_csv_path, hr_decimals)
    except Exception as e:
        print(f"[ERROR] ⛔ A critical error occurred during IHR calculation and saving: {e}")
        print("[INFO] 💾 Ledger will not be updated due to the error.")
        return

    # 4. Update the ledger with the newly processed time intervals
    ledger = IntervalLedger(intervals_path)
    ledger.add_intervals(processed_gaps)
    print(f"[INFO] ✅ Interval ledger has been updated.")

if __name__ == "__main__":
    main()