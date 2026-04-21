import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_ECG_DIR, MARKER_CSV, DEVICE_HR_CSV, DEVICE_RR_CSV, ECG_DIR,
    COL_MAPPING, DT_CSVS_FORMAT, DT_CSVS_FILENAME,
)

import os
import pandas as pd
import glob
import datetime as dt
import send2trash
import warnings
import subprocess

# --- Define Paths from Config ---
input_dir = RAW_ECG_DIR                  # Directory containing raw CSVs
marker_output_path = MARKER_CSV          # Path for the consolidated Marker CSV
hr_output_path = DEVICE_HR_CSV           # Path for the consolidated HR CSV
rr_output_path = DEVICE_RR_CSV           # Path for the consolidated Device RR CSV
ecg_output_dir = ECG_DIR                 # Output directory for cleaned ECG files

# --- Create Directories ---
# Ensure the directory for preprocessed files exists
os.makedirs(os.path.dirname(marker_output_path), exist_ok=True)
# Ensure the output directory for cleaned ECG files exists
os.makedirs(ecg_output_dir, exist_ok=True)

# --- Initialize DataFrames ---
# Load existing data or create new DataFrames. Use column names from config mapping.
marker_col = COL_MAPPING.get('marker', 'Marker') # Get marker column name, default to 'Marker'
hr_col = COL_MAPPING.get('hr', 'HR')             # Get HR column name, default to 'HR'
rr_col = COL_MAPPING.get('rr', 'InputRR')        # Get RR column name, default to 'InputRR'
dt_col = COL_MAPPING.get('time', 'DateTime')     # Get DateTime column name, default to 'DateTime'
ecg_col = COL_MAPPING.get('ecg', 'ECG')          # Get ECG column name, default to 'ECG'


if os.path.exists(marker_output_path):
    try:
        marker_df = pd.read_csv(marker_output_path)
        # Ensure columns match expected names
        if 'DateTime' not in marker_df.columns and dt_col in marker_df.columns:
             marker_df = marker_df.rename(columns={dt_col: 'DateTime'})

        # Marker output is now a single DateTime column stored as datetimes.
        if 'DateTime' in marker_df.columns:
            marker_df = marker_df[['DateTime']].copy()
            marker_df['DateTime'] = pd.to_datetime(marker_df['DateTime'], errors='coerce')
            marker_df = marker_df.dropna(subset=['DateTime'])
        else:
            marker_df = pd.DataFrame(columns=['DateTime'])

        marker_existing_count = len(marker_df)
        print(f"Loaded existing marker data from: {marker_output_path} ({marker_existing_count} rows)")
    except Exception as e:
        print(f"Error loading {marker_output_path}: {e}. Creating new DataFrame.")
        marker_df = pd.DataFrame(columns=['DateTime'])
        marker_existing_count = 0
else:
    marker_df = pd.DataFrame(columns=['DateTime'])
    marker_existing_count = 0
    print(f"Creating new marker data file: {marker_output_path}")

if os.path.exists(hr_output_path):
    try:
        hr_df = pd.read_csv(hr_output_path)
        hr_existing_count = len(hr_df)
        print(f"Loaded existing HR data from: {hr_output_path} ({hr_existing_count} rows)")
        if 'HR' not in hr_df.columns and hr_col in hr_df.columns:
             hr_df = hr_df.rename(columns={hr_col: 'HR'})
        if 'DateTime' not in hr_df.columns and dt_col in hr_df.columns:
             hr_df = hr_df.rename(columns={dt_col: 'DateTime'})
    except Exception as e:
        print(f"Error loading {hr_output_path}: {e}. Creating new DataFrame.")
        hr_df = pd.DataFrame(columns=['DateTime', 'HR'])
        hr_existing_count = 0
else:
    hr_df = pd.DataFrame(columns=['DateTime', 'HR'])
    hr_existing_count = 0
    print(f"Creating new HR data file: {hr_output_path}")

if os.path.exists(rr_output_path):
    try:
        # Explicitly specify dtype for DateTime column to ensure int64
        dtype_dict = {'DateTime': 'int64'}
        rr_df = pd.read_csv(rr_output_path, dtype=dtype_dict)
        rr_existing_count = len(rr_df)
        print(f"Loaded existing DeviceRR data from: {rr_output_path} ({rr_existing_count} rows)")
        # Rename columns for internal consistency if they match the config mapping but not the internal names
        if 'InputRR' not in rr_df.columns and rr_col in rr_df.columns:
             rr_df = rr_df.rename(columns={rr_col: 'InputRR'})
        if 'DateTime' not in rr_df.columns and dt_col in rr_df.columns:
             rr_df = rr_df.rename(columns={dt_col: 'DateTime'})

        # Convert formatted DateTime strings back to millisecond timestamps for consistent processing
        if not rr_df.empty and 'DateTime' in rr_df.columns:
            print(f"InputRR DataFrame DateTime types before conversion: {rr_df['DateTime'].apply(type).unique()}")

            def convert_to_ms_timestamp(x):
                if isinstance(x, str):
                    try:
                        # Use format from config for parsing
                        dt_obj = dt.datetime.strptime(x, DT_CSVS_FORMAT)
                        return int(dt_obj.timestamp() * 1000)
                    except ValueError: # Handle potential format mismatches or non-date strings
                        try: # Try a more generic approach if specific format fails
                            from dateutil import parser
                            dt_obj = parser.parse(x)
                            return int(dt_obj.timestamp() * 1000)
                        except Exception as e_parse:
                            print(f"Could not parse date string '{x}' to timestamp: {e_parse}. Returning None.")
                            return None # Return None if conversion fails
                    except Exception as e:
                        print(f"Error converting '{x}' to timestamp: {e}. Returning None.")
                        return None
                elif isinstance(x, (int, float)) and pd.notna(x):
                    # Assume it might already be a timestamp (e.g., ms or ns)
                    if x > 1e15: # Likely nanoseconds
                        return int(x // 1_000_000)
                    elif x < 1e12: # Likely seconds
                        return int(x * 1000)
                    else: # Assume milliseconds
                        return int(x)
                return None # Return None for other types or NaNs

            print("Converting DateTime in loaded DeviceRR data to milliseconds timestamp...")
            rr_df['DateTime'] = rr_df['DateTime'].apply(convert_to_ms_timestamp)
            rr_df = rr_df.dropna(subset=['DateTime']) # Remove rows where conversion failed
            if not rr_df.empty:
                 rr_df['DateTime'] = rr_df['DateTime'].astype('int64') # Ensure int64 type
                 print(f"InputRR DataFrame DateTime types after conversion: {rr_df['DateTime'].apply(type).unique()}")

    except Exception as e:
        print(f"Error loading {rr_output_path}: {e}. Creating new DataFrame.")
        rr_df = pd.DataFrame(columns=['DateTime', 'InputRR']) # Use internal standard names
        rr_df = rr_df.astype({'DateTime': 'int64'}) # Ensure proper dtype from start
        rr_existing_count = 0
else:
    rr_df = pd.DataFrame(columns=['DateTime', 'InputRR']) # Use internal standard names
    rr_df = rr_df.astype({'DateTime': 'int64'}) # Ensure proper dtype from start
    rr_existing_count = 0
    print(f"Creating new DeviceRR data file: {rr_output_path}")


# --- Process Raw CSV Files ---
try:
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {input_dir}")
except Exception as e:
    print(f"Error accessing input directory {input_dir}: {e}")
    csv_files = []

# Track stats across all files
total_hr_processed = 0
total_rr_processed = 0
total_marker_processed = 0
processed_files_count = 0

# Initialize lists to hold chunks for concatenation
hr_appends = [hr_df]
rr_appends = [rr_df]
marker_appends = [marker_df]

def read_raw_csv_with_trailing_field_repair(file_path):
    """Repair rows where an unquoted comma split the final CSV field in two."""
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as raw_file:
        header_line = raw_file.readline().rstrip('\r\n')

    if not header_line:
        raise ValueError("CSV file is empty")

    raw_columns = header_line.split(',')
    extra_col = '__extra_csv_field__'
    while extra_col in raw_columns:
        extra_col = f'_{extra_col}'

    # Allow one overflow field, then merge it back into the last declared column.
    df = pd.read_csv(
        file_path,
        header=None,
        skiprows=1,
        names=raw_columns + [extra_col],
        low_memory=False,
    )

    extra_rows = df[extra_col].notna()
    repaired_rows = int(extra_rows.sum())
    if repaired_rows:
        target_col = raw_columns[-1]
        if df[target_col].dtype != 'object':
            df[target_col] = df[target_col].astype('object')
        overlap_rows = extra_rows & df[target_col].notna() & (df[target_col] != '')
        fill_rows = extra_rows & ~overlap_rows

        if overlap_rows.any():
            df.loc[overlap_rows, target_col] = df.loc[overlap_rows, target_col].astype(str).str.cat(
                df.loc[overlap_rows, extra_col].astype(str),
                sep=','
            )

        if fill_rows.any():
            df.loc[fill_rows, target_col] = df.loc[fill_rows, extra_col]

        print(
            f"  Repaired {repaired_rows} malformed row(s) by merging an extra trailing field into "
            f"'{target_col}'."
        )

    df.drop(columns=[extra_col], inplace=True)
    return df

for file_path in csv_files:
    print(f"Processing: {os.path.basename(file_path)}")

    # Read the CSV file
    try:
        # Address DtypeWarning by setting low_memory=False or specifying dtypes if known
        df = pd.read_csv(file_path, low_memory=False)
        processed_files_count += 1
    except pd.errors.ParserError as e:
        try:
            print(f"  Parser error reading {os.path.basename(file_path)}: {e}. Retrying with repair.")
            df = read_raw_csv_with_trailing_field_repair(file_path)
            processed_files_count += 1
        except Exception as repair_e:
            print(f"  Error reading {os.path.basename(file_path)} after repair attempt: {repair_e}. Skipping.")
            continue
    except Exception as e:
        print(f"  Error reading {os.path.basename(file_path)}: {e}. Skipping.")
        continue

    # --- Data Cleaning Steps ---
    # Step 1: Rename columns using mapping from config
    # Create a reverse mapping for convenience if needed, but direct mapping is fine
    # Use the mapping defined in COL_MAPPING
    columns_to_rename = {cfg_key: cfg_val for cfg_key, cfg_val in COL_MAPPING.items() if cfg_key in df.columns}
    df.rename(columns=columns_to_rename, inplace=True)
    # Now use the *target* names (values from COL_MAPPING) for subsequent operations
    # Get the potentially renamed column names:
    dt_col_renamed = COL_MAPPING.get('time', 'DateTime')
    ecg_col_renamed = COL_MAPPING.get('ecg', 'ECG')
    hr_col_renamed = COL_MAPPING.get('hr', 'HR')
    rr_col_renamed = COL_MAPPING.get('rr', 'InputRR')
    marker_col_renamed = COL_MAPPING.get('marker', 'Marker')

    # Step 2: Convert DateTime from nanoseconds to milliseconds with validation
    if dt_col_renamed in df.columns:
        # Ensure the column is numeric first
        df[dt_col_renamed] = pd.to_numeric(df[dt_col_renamed], errors='coerce')
        df = df.dropna(subset=[dt_col_renamed]) # Drop rows where conversion failed

        # Additional validation: remove timestamps that are clearly not valid
        def is_reasonable_timestamp(timestamp):
            """Check if timestamp could be nanoseconds, milliseconds, or seconds"""
            if pd.isna(timestamp):
                return False
            # Convert to absolute value for comparison
            abs_ts = abs(timestamp)
            # Valid ranges:
            # Nanoseconds: > 1e15 (roughly after 2001)
            # Milliseconds: 1e10 to 1e15 (1970-2286 roughly)
            # Seconds: 1e8 to 1e12 (1973-2286 roughly)
            # Anything smaller is likely invalid
            return abs_ts >= 1e8

        # Filter out clearly invalid timestamps
        before_filter = len(df)
        df = df[df[dt_col_renamed].apply(is_reasonable_timestamp)]
        filtered_out = before_filter - len(df)
        if filtered_out > 0:
            print(f"  Filtered out {filtered_out} rows with invalid timestamps")

        # Convert ns to ms and ensure int64
        df[dt_col_renamed] = (df[dt_col_renamed] // 1_000_000).astype('int64')

        # Sort DataFrame by DateTime (oldest to newest)
        df.sort_values(dt_col_renamed, inplace=True)
    else:
        print(f"  Warning: DateTime column '{dt_col_renamed}' not found in {os.path.basename(file_path)}. Skipping time-based operations.")
        continue # Skip file if no DateTime column

    # Drop duplicate rows based on DateTime and ECG columns
    if ecg_col_renamed in df.columns:
        subset_cols = [dt_col_renamed, ecg_col_renamed]
        if not all(col in df.columns for col in subset_cols):
            print(f"  Warning: Cannot drop duplicates based on DateTime and ECG as one column is missing.")
            subset_cols = [dt_col_renamed] # Drop based on time only if ECG is missing

        before_drop = len(df)
        df.drop_duplicates(subset=subset_cols, keep='first', inplace=True)
        dropped = before_drop - len(df)
        if dropped > 0:
            print(f"  Removed {dropped} duplicate rows.")
    else:
         # If no ECG column, drop duplicates based on DateTime only
         before_drop = len(df)
         df.drop_duplicates(subset=[dt_col_renamed], keep='first', inplace=True)
         dropped = before_drop - len(df)
         if dropped > 0:
             print(f"  Removed {dropped} duplicate rows (based on DateTime only).")


    # Step 3: Convert ECG values to integers (multiply by 1000 to save space)
    if ecg_col_renamed in df.columns:
        df[ecg_col_renamed] = (pd.to_numeric(df[ecg_col_renamed], errors='coerce') * 1000).round().astype('Int64')

    # Step 4: Convert HR and InputRR columns to integers (if they exist and are not NaN)
    if hr_col_renamed in df.columns:
        df[hr_col_renamed] = pd.to_numeric(df[hr_col_renamed], errors='coerce').astype('Int64')

    if rr_col_renamed in df.columns:
        df[rr_col_renamed] = pd.to_numeric(df[rr_col_renamed], errors='coerce').astype('Int64')

    # Step 5 & 6: Extract and save Marker column, then drop it
    marker_count = 0
    if marker_col_renamed in df.columns:
        # Filter rows where Marker is not blank/null
        marker_rows = df[df[marker_col_renamed].notna() & (df[marker_col_renamed] != '')].copy()
        marker_count = len(marker_rows)

        if not marker_rows.empty:
            # Keep Marker output as a single DateTime column using datetime values.
            marker_rows_to_append = pd.DataFrame({
                'DateTime': pd.to_datetime(
                    marker_rows[dt_col_renamed].apply(
                        lambda x: dt.datetime.fromtimestamp(x / 1000) if pd.notna(x) else None
                    ),
                    errors='coerce'
                )
            }).dropna(subset=['DateTime'])

            # Append to marker DataFrame
            if not marker_rows_to_append.empty:
                marker_df = pd.concat([marker_df, marker_rows_to_append], ignore_index=True)
                total_marker_processed += len(marker_rows_to_append)

        # Drop Marker column from original DataFrame
        df.drop(columns=[marker_col_renamed], inplace=True, errors='ignore')

    # Step 7: Extract and save HR column
    hr_count = 0
    if hr_col_renamed in df.columns:
        # Filter rows where HR is not blank/null
        hr_rows = df[df[hr_col_renamed].notna()].copy()
        hr_count = len(hr_rows)
        total_hr_processed += hr_count

        if not hr_rows.empty:
            # Convert DateTime to required string format using config format
            hr_rows[dt_col_renamed] = hr_rows[dt_col_renamed].apply(
                lambda x: dt.datetime.fromtimestamp(x/1000).strftime(DT_CSVS_FORMAT) if pd.notna(x) else None
            )
            # Select and rename columns for appending ('DateTime', 'HR')
            hr_rows_to_append = hr_rows[[dt_col_renamed, hr_col_renamed]].rename(
                columns={dt_col_renamed: 'DateTime', hr_col_renamed: 'HR'}
            )
            # Append to HR DataFrame
            if not hr_rows_to_append.empty:
                hr_df = pd.concat([hr_df, hr_rows_to_append], ignore_index=True)

    # Extract and save InputRR (DeviceRR) column
    rr_count = 0
    if rr_col_renamed in df.columns:
        # Filter rows where InputRR is not blank/null
        rr_rows = df[df[rr_col_renamed].notna()].copy()
        rr_count = len(rr_rows)
        total_rr_processed += rr_count

        if not rr_rows.empty:
            # Keep DateTime as milliseconds timestamp for RR data and ensure int64
            rr_rows[dt_col_renamed] = rr_rows[dt_col_renamed].astype('int64')
            # Select and rename columns for appending ('DateTime', 'InputRR')
            rr_rows_to_append = rr_rows[[dt_col_renamed, rr_col_renamed]].rename(
                columns={dt_col_renamed: 'DateTime', rr_col_renamed: 'InputRR'}
            )
            # Ensure DateTime column is int64 in the rows to append
            rr_rows_to_append['DateTime'] = rr_rows_to_append['DateTime'].astype('int64')
            # Append to RR appends list
            if not rr_rows_to_append.empty:
                rr_appends.append(rr_rows_to_append)

    print(f"  Extracted: HR={hr_count}, RR={rr_count}, Marker={marker_count}")

    # Step 8: Save cleaned ECG data (DateTime, ECG) to a new file
    if not df.empty and ecg_col_renamed in df.columns and dt_col_renamed in df.columns:
        # Validate timestamps before using them for filename generation
        def is_valid_ms_timestamp(timestamp):
            """Check if timestamp is a valid millisecond timestamp (reasonable range)"""
            if not isinstance(timestamp, (int, float)) or pd.isna(timestamp):
                return False
            # Valid range: roughly 1970 to 2050 (in milliseconds)
            # 1970: 0, 2050: ~2,524,608,000,000
            return 0 <= timestamp <= 2_600_000_000_000

        # Filter for valid timestamps and get first/last
        valid_timestamps = df[df[dt_col_renamed].apply(is_valid_ms_timestamp)][dt_col_renamed]

        if len(valid_timestamps) >= 2:
            first_timestamp = valid_timestamps.iloc[0]
            last_timestamp = valid_timestamps.iloc[-1]

            try:
                first_formatted = dt.datetime.fromtimestamp(first_timestamp/1000).strftime(DT_CSVS_FILENAME)
                last_formatted = dt.datetime.fromtimestamp(last_timestamp/1000).strftime(DT_CSVS_FILENAME)
                # Create new filename format: first_timestamp_last_timestamp
                output_filename = f"{first_formatted}_{last_formatted}.csv"

            except Exception as fmt_e:
                print(f"  Error formatting filename timestamps: {fmt_e}. Using default name.")
                output_filename = f"cleaned_{os.path.basename(file_path)}"
        else:
            print(f"  Warning: Not enough valid timestamps found in {os.path.basename(file_path)}. Using default name.")
            output_filename = f"cleaned_{os.path.basename(file_path)}"

        # Create a new DataFrame with just the required columns (using renamed cols)
        output_df = df[[dt_col_renamed, ecg_col_renamed]].copy()
        # Rename columns back to standard 'DateTime', 'ECG' for the output file if desired, or keep as is
        output_df = output_df.rename(columns={dt_col_renamed: 'DateTime', ecg_col_renamed: 'ECG'})

        # Save to output directory (cfg.PRECLEANED_DIR)
        output_path = os.path.join(ecg_output_dir, output_filename)
        try:
            output_df.to_csv(output_path, index=False)
            print(f"  Saved cleaned ECG: {output_filename}")

            # Set file creation and modification dates based on the recording's first and last rows
            if len(valid_timestamps) >= 2:
                try:
                    # Get datetime objects for first and last timestamps
                    first_dt = dt.datetime.fromtimestamp(first_timestamp / 1000)
                    last_dt = dt.datetime.fromtimestamp(last_timestamp / 1000)
                    
                    # Set access + modification time to the last row's timestamp
                    os.utime(output_path, (last_dt.timestamp(), last_dt.timestamp()))
                    
                    # Set creation date on macOS to the first row's timestamp
                    subprocess.run(
                        ["SetFile", "-d", first_dt.strftime("%m/%d/%Y %H:%M:%S"), str(output_path)],
                        check=True,
                        capture_output=True
                    )
                    print("  Set file OS creation and modification timestamps successfully.")
                except Exception as time_e:
                    print(f"  Warning: Could not set OS file timestamps for {output_filename}: {time_e}")

        except Exception as save_e:
            print(f"  Error saving cleaned ECG file {output_filename}: {save_e}")
    elif ecg_col_renamed not in df.columns:
         print(f"  Skipping ECG file save for {os.path.basename(file_path)} as ECG column ('{ecg_col_renamed}') is missing.")


# --- Final Deduplication and Saving ---
print("\n--- Finalizing Consolidated Files ---")

# Concat lists back to main DataFrames
if len(rr_appends) > 1 or (len(rr_appends) == 1 and not rr_appends[0].empty):
    rr_df = pd.concat(rr_appends, ignore_index=True)
if len(hr_appends) > 1 or (len(hr_appends) == 1 and not hr_appends[0].empty):
    hr_df = pd.concat(hr_appends, ignore_index=True)
if len(marker_appends) > 1 or (len(marker_appends) == 1 and not marker_appends[0].empty):
    marker_df = pd.concat(marker_appends, ignore_index=True)

# Process DeviceRR (InputRR) data: Sort, Deduplicate, Format DateTime, Save
if not rr_df.empty:
    print(f"DeviceRR: Pre-existing rows: {rr_existing_count}, New rows added: {total_rr_processed}, Total before deduplication: {len(rr_df)}")
    # Ensure DateTime is int64 before sorting
    rr_df['DateTime'] = pd.to_numeric(rr_df['DateTime'], errors='coerce')
    rr_df = rr_df.dropna(subset=['DateTime'])
    rr_df['DateTime'] = rr_df['DateTime'].astype('int64')

    # Ensure InputRR values are numeric and use Nullable Integer type to preserve NaNs
    rr_df['InputRR'] = pd.to_numeric(rr_df['InputRR'], errors='coerce').astype('Int64')

    # Sort by DateTime (milliseconds) first
    rr_df.sort_values('DateTime', inplace=True)
    before_drop = len(rr_df)
    # Drop duplicates based on DateTime AND InputRR value, keeping the last occurrence
    rr_df.drop_duplicates(subset=['DateTime', 'InputRR'], keep='last', inplace=True)
    dropped = before_drop - len(rr_df)
    print(f"DeviceRR: Duplicates removed: {dropped}, Final deduplicated rows: {len(rr_df)}")

    # Ensure DateTime column is int64 before saving
    rr_df['DateTime'] = rr_df['DateTime'].astype('int64')
    print(f"DeviceRR: Keeping DateTime as milliseconds timestamp with dtype: {rr_df['DateTime'].dtype}")

    # Move original file to trash before saving
    if os.path.exists(rr_output_path):
        try:
            send2trash.send2trash(rr_output_path)
            print(f"DeviceRR: Moved existing file to trash: {rr_output_path}")
        except Exception as e:
            print(f"DeviceRR: Error moving existing file to trash: {e}")

    # Save the final deduplicated data
    try:
        rr_df.to_csv(rr_output_path, index=False)
        print(f"DeviceRR: Saved to: {rr_output_path}")
    except Exception as e:
        print(f"DeviceRR: Error saving final file: {e}")
else:
    print("DeviceRR: No data to save.")


# Process Marker data: Sort, Deduplicate, Save
if not marker_df.empty:
    print(f"Marker: Pre-existing rows: {marker_existing_count}, New rows added: {total_marker_processed}, Total before deduplication: {len(marker_df)}")
    # Ensure DateTime column is stored as datetimes before sorting and saving.
    marker_df['DateTime'] = pd.to_datetime(marker_df['DateTime'], errors='coerce')
    marker_df = marker_df.dropna(subset=['DateTime'])

    # Sort by DateTime
    marker_df.sort_values('DateTime', inplace=True, na_position='first') # Sort before drop
    before_drop = len(marker_df)
    # Remove duplicates based on DateTime, keeping the last entry
    marker_df.drop_duplicates(subset=['DateTime'], keep='last', inplace=True)
    dropped = before_drop - len(marker_df)
    print(f"Marker: Duplicates removed: {dropped}, Final deduplicated rows: {len(marker_df)}")

    # Move original file to trash
    if os.path.exists(marker_output_path):
        try:
            send2trash.send2trash(marker_output_path)
            print(f"Marker: Moved existing file to trash: {marker_output_path}")
        except Exception as e:
            print(f"Marker: Error moving existing file to trash: {e}")

    # Save final data
    try:
        # Write marker timestamps in ISO order so spreadsheet apps sort them chronologically.
        marker_df.to_csv(marker_output_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Marker: Saved to: {marker_output_path}")
    except Exception as e:
        print(f"Marker: Error saving final file: {e}")
else:
    print("Marker: No data to save.")


# Process HR data: Sort, Deduplicate, Save
if not hr_df.empty:
    print(f"HR: Pre-existing rows: {hr_existing_count}, New rows added: {total_hr_processed}, Total before deduplication: {len(hr_df)}")
    # Ensure HR values are numeric (float first for NaNs, then int)
    hr_df['HR'] = pd.to_numeric(hr_df['HR'], errors='coerce')
    hr_df['HR'] = hr_df['HR'].apply(lambda x: int(x) if pd.notna(x) else x)

    # Sort by DateTime string
    hr_df.sort_values('DateTime', inplace=True, na_position='first') # Sort before drop
    before_drop = len(hr_df)
    # Drop duplicates based on DateTime only, keeping the last entry for each timestamp
    hr_df.drop_duplicates(subset=['DateTime'], keep='last', inplace=True)
    dropped = before_drop - len(hr_df)
    print(f"HR: Duplicates removed: {dropped}, Final deduplicated rows: {len(hr_df)}")

    # Move original file to trash
    if os.path.exists(hr_output_path):
        try:
            send2trash.send2trash(hr_output_path)
            print(f"HR: Moved existing file to trash: {hr_output_path}")
        except Exception as e:
            print(f"HR: Error moving existing file to trash: {e}")

    # Save final data
    try:
        hr_df.to_csv(hr_output_path, index=False)
        print(f"HR: Saved to: {hr_output_path}")
    except Exception as e:
        print(f"HR: Error saving final file: {e}")
else:
    print("HR: No data to save.")


# --- Final Summary and Cleanup ---
print("\n--- SUMMARY ---")
print(f"Processed {processed_files_count} CSV files from: {input_dir}")
print(f"Total unique HR values saved: {len(hr_df) if not hr_df.empty else 0}")
print(f"Total unique RR values saved: {len(rr_df) if not rr_df.empty else 0}")
print(f"Total unique marker values saved: {len(marker_df) if not marker_df.empty else 0}")
print(f"Cleaned ECG files saved to: {ecg_output_dir}")

# Move processed raw CSV files to trash
print("\n--- Moving Processed Raw Files to Trash ---")
moved_count = 0
error_count = 0
for file_path in csv_files:
    try:
        send2trash.send2trash(file_path)
        moved_count += 1
    except Exception as e:
        print(f"Error moving {os.path.basename(file_path)} to trash: {e}")
        error_count += 1

print(f"Moved {moved_count} raw files to trash.")
if error_count > 0:
    print(f"Failed to move {error_count} files.")

print("\nProcessing complete!")
