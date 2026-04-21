import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MPL_RC_PARAMS, NORM, MAX_INTERPOLATION, ANALYSIS_TF,
    MIN_WEEKLY_HRS, MIN_DAILY_HRS, MIN_OVERALL_HRS,
    VIZ_DIR, XL_MET, CIR_METRICS, OUTPUTS_DIR,
)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import traceback
import matplotlib.pyplot as plt
import matplotlib as mpl
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from itertools import chain

warnings.filterwarnings('ignore')

# --- Configuration & Global Variables ---

mpl.rcParams.update(MPL_RC_PARAMS)

# --- Core Functions ---

def find_excel_file(directory):
    """Return the Metrics Excel workbook in `directory` (matching XL_MET*.xlsx)."""
    print(f"[INFO] 🔍 Searching for Metrics workbook in {directory}…")
    # Prefer only the Metrics workbook(s)
    metrics_files = sorted(
        [f for f in Path(directory).glob(f"{XL_MET}*.xlsx") if f.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if metrics_files:
        print(f"[INFO] ✅ Found Metrics workbook: {metrics_files[0].name}")
        return metrics_files[0]

    print(f"[ERROR] ❌ No Excel files found in {directory}")
    return None

def setup_output_directories(timeframe):
    """Create a structured set of output directories for the given timeframe."""
    print(f"[INFO] ⏳ Setting up output directories for timeframe: {timeframe}...")
    base_viz_dir = Path(VIZ_DIR) / timeframe
    output_dirs = {
        'base': base_viz_dir,
        'weekly': base_viz_dir / 'Weekly',
        'daily': base_viz_dir / 'Daily'
    }
    try:
        for path in output_dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] ✅ Created/verified output directory structure under: {base_viz_dir}")
        return {key: str(value) for key, value in output_dirs.items()}
    except Exception as e:
        print(f"[ERROR] ❗ Failed to create output directories: {e}")
        return None

def analyze_circadian_rhythm(df, metric_cols, max_interpolation_pct=MAX_INTERPOLATION):
    """Analyze circadian rhythm with interpolation limit."""
    df_processed = df.copy()
    available_metrics = [metric for metric in metric_cols if metric in df_processed.columns]
    if not available_metrics:
        print("[ERROR] ❗ No valid metrics found for circadian analysis")
        return {}

    if not isinstance(df_processed.index, pd.DatetimeIndex):
        print("[ERROR] ❗ DataFrame must have a DatetimeIndex for this function.")
        return {}

    df_hourly = df_processed.resample('H').mean(numeric_only=True)
    max_interp_limit = int(24 * max_interpolation_pct / 100)

    for col in available_metrics:
        if col in df_hourly.columns and df_hourly[col].isna().any():
            original_na = df_hourly[col].isna().sum()
            df_hourly[col] = df_hourly[col].interpolate(method='time', limit=max_interp_limit, limit_direction='both')
            filled_na = original_na - df_hourly[col].isna().sum()
            if filled_na > 0:
                print(f"[INFO] 💧 Interpolated {filled_na} missing values for {col}.")

    df_hourly['hour'] = df_hourly.index.hour
    results = {}

    for metric in available_metrics:
        subdf = df_hourly[[metric, 'hour']].dropna()
        if subdf.empty or subdf['hour'].nunique() < 3:
            results[metric] = {'amplitude': np.nan, 'acrophase': np.nan, 'r_squared': np.nan}
            continue

        hourly_stats = subdf.groupby('hour')[metric].agg(['mean', 'std', 'count'])
        time_radians = subdf['hour'] * 2 * np.pi / 24
        X = np.column_stack([np.ones(len(time_radians)), np.cos(time_radians), np.sin(time_radians)])
        y = subdf[metric].values

        try:
            model, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            mesor, amplitude = model[0], np.sqrt(model[1]**2 + model[2]**2)
            acrophase_hours = (np.arctan2(-model[2], model[1]) * 24 / (2 * np.pi)) % 24
            y_pred = X @ model
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        except (np.linalg.LinAlgError, ValueError):
            mesor, amplitude, acrophase_hours, r_squared = [np.nan] * 4

        results[metric] = {
            'hourly_stats': hourly_stats, 'mesor': mesor, 'amplitude': amplitude,
            'acrophase': acrophase_hours, 'r_squared': r_squared
        }
    return results

def plot_circadian_patterns(analysis_results, output_dir, analysis_label=""):
    """Plot circadian patterns and save to a structured directory."""
    if not analysis_results:
        print(f"[WARNING] ⚠️ No results to plot for {analysis_label}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figsize = MPL_RC_PARAMS.get('figure.figsize', (10, 6))
    dpi = MPL_RC_PARAMS.get('savefig.dpi', 150)

    for metric, results in analysis_results.items():
        if 'hourly_stats' not in results or results['hourly_stats'].empty:
            continue

        hourly_stats = results['hourly_stats']
        plt.figure(figsize=figsize)
        plt.plot(hourly_stats.index, hourly_stats['mean'], 'o-', label='Hourly Mean')
        if 'std' in hourly_stats and not hourly_stats['std'].isna().all():
            plt.fill_between(hourly_stats.index, hourly_stats['mean'] - hourly_stats['std'], hourly_stats['mean'] + hourly_stats['std'], alpha=0.2, label='Mean ± SD')

        if all(pd.notna(results.get(k)) for k in ['mesor', 'amplitude', 'acrophase']):
            mesor, amp, acr = results['mesor'], results['amplitude'], results['acrophase']
            x_fit = np.linspace(0, 23.99, 200)
            y_fit = mesor + amp * np.cos((x_fit - acr) * (2 * np.pi / 24))
            plt.plot(x_fit, y_fit, 'r--', label='Cosinor Fit')
            stats_text = (f'MESOR: {mesor:.2f}\nAmplitude: {amp:.2f}\n'
                          f'Peak Time: {acr:.1f} h\n'
                          f'R²: {results.get("r_squared", np.nan):.3f}')
            plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

        title = f"{analysis_label} Pattern: {metric}"
        plt.title(title, fontsize=14, weight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel(metric)
        plt.xticks(range(0, 25, 2))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(-0.5, 23.5)
        plt.legend()

        safe_metric = metric.replace("/", "_").replace(" ", "_")

        if analysis_label.startswith('Weekly'):
             filename = f'Weekly_{safe_metric}.png'
        else: # Daily
             filename = f'{safe_metric}.png'

        filepath = output_path / filename
        try:
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"    ✅ Saved plot: {filepath.name}")
        except Exception as e:
            print(f"[ERROR] ❗ Failed to save plot {filepath}: {e}")
        plt.close()

def generate_segment_pdf(segment_results, output_dir, title_label, segment_type):
    """Generates a single, multi-page PDF for one analysis segment with a 2-column layout."""
    pdf_path = Path(output_dir) / f"{title_label.split('(')[0].strip()}_Report.pdf"
    print(f"    ⏳ Generating PDF report: {pdf_path.name}")

    doc = SimpleDocTemplate(str(pdf_path), pagesize=landscape(letter), topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.5*inch, rightMargin=0.5*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['h1'], fontSize=16, alignment=1, spaceAfter=12)

    table_header_style = ParagraphStyle('TableHeader', parent=styles['Normal'], fontName='Helvetica-Bold', alignment=1)
    table_body_style = ParagraphStyle('TableBody', parent=styles['Normal'], alignment=1)

    story = [Paragraph(f"Circadian Analysis: {title_label}", title_style)]

    # Define the 3-page layout
    metric_pages = CIR_METRICS

    for page_num, metrics_on_page in enumerate(metric_pages):
        left_col_flowables = []
        right_col_flowables = []

        for metric in metrics_on_page:
            safe_metric = metric.replace("/", "_").replace(" ", "_")
            results = segment_results.get(metric, {})

            # --- Build Left Column (Data Tables) ---
            if all(k in results and pd.notna(results[k]) for k in ['mesor', 'amplitude', 'r_squared']):
                circ_data = [
                    [Paragraph(p, table_header_style) for p in [metric, 'Value']],
                    [Paragraph('MESOR', table_body_style), Paragraph(f"{results['mesor']:.2f}", table_body_style)],
                    [Paragraph('Amplitude', table_body_style), Paragraph(f"{results['amplitude']:.2f}", table_body_style)],
                    [Paragraph('Peak Time', table_body_style), Paragraph(f"{results['acrophase']:.1f} h", table_body_style)],
                    [Paragraph('R² Strength', table_body_style), Paragraph(f"{results['r_squared']:.3f}", table_body_style)],
                ]
                circ_table = Table(circ_data, colWidths=[1.5*inch, 1.0*inch])
                circ_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
                left_col_flowables.append(circ_table)
            else:
                left_col_flowables.append(Paragraph(f"{metric}: Cosinor fit could not be determined.", styles['Normal']))
            left_col_flowables.append(Spacer(1, 0.25*inch))

            # --- Build Right Column (Graphs) ---
            img_filename = f'Weekly_{safe_metric}.png' if segment_type == 'Weekly' else f'{safe_metric}.png'
            img_path = Path(output_dir) / img_filename
            if img_path.exists():
                try:
                    img = Image(str(img_path), width=5.5*inch, height=2.75*inch)
                    right_col_flowables.append(img)
                except Exception:
                    right_col_flowables.append(Paragraph(f"Could not load image for {metric}.", styles['Normal']))
            else:
                right_col_flowables.append(Paragraph(f"Image not found for {metric}.", styles['Normal']))
            right_col_flowables.append(Spacer(1, 0.25*inch))

        # Create the 2-column layout table for the page
        page_layout_table = Table([[left_col_flowables, right_col_flowables]], colWidths=[2.7*inch, 5.8*inch])
        page_layout_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
        story.append(page_layout_table)

        if page_num < len(metric_pages) - 1:
            story.append(PageBreak())

    try:
        doc.build(story)
    except Exception as e:
        print(f"    [ERROR] ❗ Failed to build PDF for {title_label}: {e}")

def _analyze_time_segments(df, metric_cols, output_dirs, segment_type, min_hours):
    """Refactored function to analyze segments and now also returns a count of skipped segments."""
    print(f"\n[INFO] --- Analyzing {segment_type} Patterns ---")
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"[ERROR] ❗ DataFrame must have a DatetimeIndex for {segment_type.lower()} analysis.")
        return {}, 0

    segment_results = {}
    skipped_count = 0

    if segment_type == 'Weekly':
        dates = pd.date_range(start=df.index.min().normalize(), end=df.index.max().normalize(), freq='W-MON')
        output_base_dir = output_dirs['weekly']
    else: # Daily
        dates = df.index.normalize().unique()
        output_base_dir = output_dirs['daily']

    print(f"[INFO] ℹ️ Found {len(dates)} potential {segment_type.lower()} segments for analysis.")

    for i, date in enumerate(dates):
        if segment_type == 'Weekly':
            segment_start = date
            segment_end = date + timedelta(days=7) - timedelta(seconds=1)
            segment_label = f"Week {i+1} ({segment_start.strftime('%b %d')} - {segment_end.strftime('%b %d')})"
            plot_label = 'Weekly'
        else: # Daily
            segment_start = date
            segment_end = date + timedelta(days=1) - timedelta(seconds=1)
            segment_label = f"{date.strftime('%Y-%m-%d (%a)')}"
            plot_label = 'Daily'

        segment_df = df.loc[segment_start:segment_end]

        if segment_df.empty or len(segment_df.select_dtypes(include=[np.number]).resample('H').mean()) < min_hours:
            skipped_count += 1
            continue

        print(f"\n[INFO] 🔍 Processing {segment_type} segment: {segment_label}")
        try:
            results = analyze_circadian_rhythm(segment_df, metric_cols)
            if results:
                segment_results[segment_label] = results
                output_dir = os.path.join(output_base_dir, segment_label)
                plot_circadian_patterns(results, output_dir, analysis_label=plot_label)

                # NEW: Generate a PDF for this specific segment
                generate_segment_pdf(results, output_dir, segment_label, segment_type)

        except Exception as e:
            print(f"[ERROR] ❗ Error analyzing segment {segment_label}: {e}")
            traceback.print_exc()

    return segment_results, skipped_count

def main():
    print("[INFO] ▶️ Initializing Script #2: Circadian Analysis...")
    input_directory = OUTPUTS_DIR
    latest_excel = find_excel_file(input_directory)
    if not latest_excel:
        print("[ERROR] ❗ No Excel file found. Cannot proceed.")
        return

    print(f"[INFO] 📄 Using Excel file: {latest_excel.name}")
    output_dirs = setup_output_directories(ANALYSIS_TF)
    if not output_dirs:
        return

    try:
        df_analysis = pd.read_excel(latest_excel, sheet_name=f"{ANALYSIS_TF}", parse_dates=['StartTime'])
        df_analysis = df_analysis.dropna(subset=['StartTime']).set_index('StartTime').sort_index()
    except Exception as e:
        print(f"[ERROR] ❗ Failed to read or process sheet for timeframe '{ANALYSIS_TF}': {e}")
        return

    if len(df_analysis.select_dtypes(include=[np.number]).resample('H').mean()) < MIN_OVERALL_HRS:
        print(f"[WARNING] ⚠️ Insufficient data for robust analysis (< {MIN_OVERALL_HRS} hourly points).")

    circadian_metrics = list(chain.from_iterable(CIR_METRICS))
    if not circadian_metrics:
        print("[ERROR] ❗ CIR_METRICS not defined in cfg.py. Cannot proceed.")
        return

    print(f"[INFO] ℹ️ Using metrics from cfg.py: {circadian_metrics}")

    try:
        _, weekly_skipped = _analyze_time_segments(df_analysis, circadian_metrics, output_dirs, 'Weekly', MIN_WEEKLY_HRS)
        _, daily_skipped = _analyze_time_segments(df_analysis, circadian_metrics, output_dirs, 'Daily', MIN_DAILY_HRS)

        print("\n-------------------- ANALYSIS SUMMARY --------------------")
        print(f"[INFO] 📊 Weekly Segments Skipped: {weekly_skipped} (due to < {MIN_WEEKLY_HRS} hrs of data)")
        print(f"[INFO] 📊 Daily Segments Skipped: {daily_skipped} (due to < {MIN_DAILY_HRS} hrs of data)")
        print("--------------------------------------------------------")

    except Exception as e:
        print(f"[ERROR] ❗ Critical error during analysis execution: {e}")
        traceback.print_exc()
        print("\n[INFO] ❌ Script #2 finished with errors.")
        return

    print("\n[INFO] ✅ Script #2 finished successfully.")

if __name__ == "__main__":
    main()