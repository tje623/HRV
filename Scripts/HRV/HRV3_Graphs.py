import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    VIZ_DIR, VIZ_SUBDIRS, XL_MET, HRV3_SHEET_NAMES, HRV3_PLOT_TYPES,
    CHNG_PT_PENALTY, CHNG_PT_COLORS, NONLINEAR_METRICS, FREQ_METRICS,
    KEY_METRICS, MPL_RC_PARAMS, OUTPUTS_DIR,
)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Prevent figures from displaying on screen
plt.ioff()

# Function to find the most recent Excel file recursively
def find_most_recent_excel_file(directory):
    """
    Return the most recent HRV Metrics workbook in `directory` (and subdirs),
    matching files that start with XL_MET and end with .xlsx.

    If no matching file is found, return None and print a helpful error.
    """
    metrics_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx') and file.startswith(XL_MET):
                metrics_files.append(os.path.join(root, file))

    if not metrics_files:
        print(f"[ERROR] ❌ No Metrics workbooks matching '{XL_MET}*.xlsx' in {directory} (or subdirs).")
        return None

    metrics_files.sort(key=os.path.getmtime, reverse=True)
    return metrics_files[0]

def detect_change_points(df, metric='RMSSD', output_dir=None, timeframe='1H'):
    """Detect significant change points and plot them against a proper time axis."""
    try:
        import ruptures as rpt
    except ImportError:
        print("ruptures package not installed. Skipping change point detection.")
        return None, None

    # Prepare data
    df = df.sort_values('StartTime').reset_index(drop=True)
    signal = df[metric].interpolate().values

    # Detect change points
    model = "rbf"  # Radial Basis Function kernel
    algo = rpt.Pelt(model=model).fit(signal)
    result = algo.predict(pen=CHNG_PT_PENALTY)

    # --- Manual Plotting for Date Axis ---
    fig, ax = plt.subplots(figsize=(18, 7))
    dates = df['StartTime']

    # Plot the signal against time
    ax.plot(dates, signal, label=metric, color='steelblue', linewidth=1.5)

    # Shade the background to show the segments
    change_points_indices = [0] + result
    colors = CHNG_PT_COLORS
    for i in range(len(change_points_indices) - 1):
        start_idx = change_points_indices[i]
        end_idx = change_points_indices[i+1] - 1
        if end_idx >= len(dates): end_idx = len(dates) - 1
        ax.axvspan(dates.iloc[start_idx], dates.iloc[end_idx], facecolor=colors[i % 2], alpha=0.5)

    # Format the plot
    ax.set_title(f'Change Point Detection for {metric} ({timeframe})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.margins(x=0.01) # Add a small margin to the x-axis

    # Format x-axis dates for clarity
    fig.autofmt_xdate(rotation=30)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{timeframe}_{metric}_change_points.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path, result

def create_correlation_heatmap(df, valid_cols, output_dir=None, timeframe='1H'):
    """Create a standalone, portrait-orientation correlation heatmap."""
    num_metrics = len(valid_cols)
    annot_size = 4 if num_metrics > 40 else 5 if num_metrics > 30 else 6 if num_metrics > 20 else 8

    fig, ax = plt.subplots(figsize=(12, 16))  # Portrait orientation
    corr_matrix = df[valid_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
        annot=True, fmt='.2f', square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": annot_size}
    )
    ax.set_title(f'Correlation Matrix of HRV Metrics ({timeframe})', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"Correlation_Heatmap.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def create_pca_biplot(df, valid_cols, output_dir=None, timeframe='1H'):
    """Create a standalone, landscape-orientation PCA biplot with scaled arrows."""
    X = df[valid_cols].copy()
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Hour'] = df['StartTime'].dt.hour

    fig, ax = plt.subplots(figsize=(16, 10))  # Landscape orientation

    scatter = ax.scatter(
        pca_df['PC1'], pca_df['PC2'], c=pca_df['Hour'],
        cmap='viridis', alpha=0.6, s=40
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hour of Day', fontsize=12)

    # Dynamically scale arrows to fill the plot without overlapping
    arrow_scale = np.max(np.abs(X_pca)) * 0.9

    for i, feature in enumerate(valid_cols):
        ax.arrow(
            0, 0, pca.components_[0, i] * arrow_scale, pca.components_[1, i] * arrow_scale,
            head_width=0.1, head_length=0.1, fc='red', ec='red'
        )
        ax.text(
            pca.components_[0, i] * arrow_scale * 1.15, pca.components_[1, i] * arrow_scale * 1.15,
            feature, color='red', fontsize=9, ha='center', va='center'
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title(f'PCA of HRV Metrics ({timeframe})', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"PCA_Biplot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_nonlinear_metrics(df, output_dir=None, timeframe='1H'):
    """Plot nonlinear HRV metrics over time."""
    df = df.copy()
    df.set_index('StartTime', inplace=True)

    nonlinear_metrics = NONLINEAR_METRICS
    available_metrics = [m for m in nonlinear_metrics if m in df.columns]

    if not available_metrics:
        print(f"No nonlinear metrics found in the {timeframe} data")
        return None

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 3 * n_metrics), sharex=True)
    if n_metrics == 1: axes = [axes]

    for i, metric in enumerate(available_metrics):
        hours = df.index.hour
        axes[i].scatter(df.index, df[metric], c=hours, cmap='plasma', alpha=0.8, s=20, label=f'{metric} (raw)')
        daily_avg = df[metric].resample('D').mean()
        axes[i].plot(daily_avg.index, daily_avg, 'r-', linewidth=2, label='Daily Avg')
        axes[i].set_ylabel(metric, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Format the shared x-axis
    last_ax = axes[-1]
    last_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    last_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    last_ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=15))
    plt.setp(last_ax.get_xticklabels(), rotation=30, ha='right')

    plt.suptitle(f'Nonlinear HRV Metrics Time Series ({timeframe})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_path = os.path.join(output_dir, f"Nonlinear_Metrics.png")
    plt.savefig(output_path)
    plt.close()
    return output_path

def plot_frequency_domain(df, output_dir=None, timeframe='5M'):
    """Plot frequency domain components over time and their relative proportions."""
    df = df.copy()
    df.set_index('StartTime', inplace=True)

    freq_components = FREQ_METRICS
    available_comps = [comp for comp in freq_components if comp in df.columns]
    if not available_comps:
        print(f"No frequency components found in the {timeframe} data")
        return None

    num_plots = 2 + ('LFHF' in df.columns)
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots), sharex=True)
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if num_plots == 3 else None

    for comp in available_comps:
        ax1.plot(df.index, df[comp], linewidth=2, label=comp)
    ax1.set_ylabel('Absolute Power (ms²)', fontweight='bold')
    ax1.set_title('HRV Frequency Components Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    df_rel = df[available_comps].copy()
    total = df_rel.sum(axis=1)
    for comp in available_comps:
        df_rel[f'{comp}%'] = df_rel[comp] / total * 100

    rel_cols = [f'{comp}%' for comp in available_comps]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(available_comps)))
    df_rel[rel_cols].plot.area(ax=ax2, stacked=True, alpha=0.7, color=colors)
    ax2.set_ylabel('Relative Power (%)', fontweight='bold')
    ax2.set_title('Relative Distribution of Frequency Components', fontsize=14, fontweight='bold')
    ax2.legend(available_comps)
    ax2.grid(True, alpha=0.3)

    if ax3:
        ax3.plot(df.index, df['LFHF'], 'r-', linewidth=2)
        ax3.set_ylabel('LF/HF Ratio', fontweight='bold')
        ax3.set_title('Sympathovagal Balance (LF/HF Ratio)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='Balance Point (1.0)')
        ax3.legend()

    # Format the shared x-axis
    last_ax = axes[-1]
    last_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    last_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    last_ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=15))
    plt.setp(last_ax.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"Frequency_Domain.png")
    plt.savefig(output_path)
    plt.close()
    return output_path

def plot_hrv_heatmap(df, metric_name, output_dir=None, timeframe='1H'):
    """Create a heatmap showing HRV metrics by hour of day and day of week."""
    df = df.copy()
    df['Hour'] = df['StartTime'].dt.hour
    df['DayOfWeek'] = df['StartTime'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    pivot_df = df.pivot_table(values=metric_name, index='Hour', columns='DayOfWeek', aggfunc='mean')
    available_days = [day for day in day_order if day in pivot_df.columns]
    pivot_df = pivot_df.reindex(columns=available_days)

    plt.figure(figsize=(12, 10))
    cmap = 'magma' if metric_name == 'RMSSD' else 'viridis'
    sns.heatmap(
        pivot_df, cmap=cmap, annot=True, fmt='.1f', linewidths=0.5,
        cbar_kws={'label': f'Mean {metric_name}'}
    )
    plt.title(f'{metric_name} by Hour and Day of Week ({timeframe})', fontsize=16, fontweight='bold')
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Hour of the Day', fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"Heatmap_{metric_name}.png")
    plt.savefig(output_path)
    plt.close()
    return output_path

def plot_daily_weekly_rhythms(df, metric_names, output_dir=None, timeframe='1H'):
    """Plot key HRV metrics to visualize daily and weekly patterns with a clear time axis."""
    df = df.copy()
    df.set_index('StartTime', inplace=True)
    available_metrics = [m for m in metric_names if m in df.columns]

    if not available_metrics:
        print(f"None of the specified rhythm metrics found in {timeframe} data")
        return None

    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(15, 4 * len(available_metrics)), sharex=True)
    if len(available_metrics) == 1: axes = [axes]

    color_cycle = plt.cm.tab10.colors
    for i, metric in enumerate(available_metrics):
        hours = df.index.hour
        scatter = axes[i].scatter(df.index, df[metric], c=hours, cmap='twilight_shifted', alpha=0.7, s=20)
        cbar = plt.colorbar(scatter, ax=axes[i])
        cbar.set_label('Hour of Day')

        if df[metric].max() > 1000:
            axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        rolling_avg = df[metric].rolling(window=f'7D').mean()
        axes[i].plot(
            rolling_avg.index, rolling_avg, '-', color=color_cycle[i % len(color_cycle)],
            linewidth=2.5, label='7-day Rolling Avg'
        )
        axes[i].set_ylabel(metric, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].margins(x=0.01)

    # Format the shared x-axis for better readability
    last_ax = axes[-1]
    last_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    last_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    last_ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=15))
    plt.setp(last_ax.get_xticklabels(), rotation=30, ha='right')

    plt.suptitle(f'HRV Daily and Weekly Patterns ({timeframe})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_path = os.path.join(output_dir, "Daily_Weekly_Patterns.png")
    plt.savefig(output_path)
    plt.close()
    return output_path


def main():
    plt.rcParams.update(MPL_RC_PARAMS)
    file_path = find_most_recent_excel_file(OUTPUTS_DIR)
    if not file_path:
        # Nothing to visualize; exit gracefully
        return
    print(f"Using most recent Excel file: {os.path.basename(file_path)}")

    # Load all Excel sheets
    try:
        dfs = pd.read_excel(file_path, sheet_name=HRV3_SHEET_NAMES, engine="openpyxl")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading Excel file or sheets: {e}")
        return

    # A set to track created directories to avoid redundant checks
    created_dirs = set()

    for sheet_name, df in dfs.items():
        print(f"\n--- Processing sheet: {sheet_name} ---")
        timeframe = sheet_name.split()[0]

        if "StartTime" not in df.columns:
            print(f"Warning: 'StartTime' column not found in sheet {sheet_name}. Skipping.")
            continue

        try:
            df["StartTime"] = pd.to_datetime(df["StartTime"])
        except Exception as e:
            print(f"Warning: Could not convert StartTime to datetime in {sheet_name}: {e}. Skipping.")
            continue

        # Rename column with special characters for better font compatibility
        if 'DFA 𝛼1' in df.columns:
            df.rename(columns={'DFA 𝛼1': 'DFA_alpha1'}, inplace=True)

        # Create timeframe-specific subdirectories for this sheet's plots
        plot_folders_to_create = [*HRV3_PLOT_TYPES, *VIZ_SUBDIRS.values()]
        for plot_type in plot_folders_to_create:
            timeframe_specific_dir = os.path.join(VIZ_DIR, plot_type, timeframe)
            if timeframe_specific_dir not in created_dirs:
                os.makedirs(timeframe_specific_dir, exist_ok=True)
                created_dirs.add(timeframe_specific_dir)

        # --- Basic Per-Metric Visualizations ---
        numeric_cols = df.select_dtypes(include=["number"]).columns
        color_palette = sns.color_palette("husl", len(numeric_cols))

        print(f"Generating basic plots for {len(numeric_cols)} metrics...")
        for i, col in enumerate(numeric_cols):
            safe_col_name = col.replace("/", "_").replace("\\", "_").replace(":", "_").replace("%", "pct")
            color = color_palette[i]

            try:
                # 1. Line Plot (Time Series)
                plt.figure(figsize=(12, 6))
                sns.lineplot(x=df["StartTime"], y=df[col], color=color)
                plt.title(f"{col} Over Time ({timeframe})")
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                save_path = os.path.join(VIZ_DIR, "Line Plots", timeframe, f"{safe_col_name}.png")
                plt.savefig(save_path)
                plt.close()

                # 2. Histogram
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col], bins=50, kde=True, color=color)
                plt.title(f"Distribution of {col} ({timeframe})")
                plt.tight_layout()
                save_path = os.path.join(VIZ_DIR, "Histograms", timeframe, f"{safe_col_name}.png")
                plt.savefig(save_path)
                plt.close()

            except Exception as e:
                print(f"  - Error creating basic plot for column {col}: {e}")
                continue
        print("Done with basic plots.")

        # --- Advanced Visualizations ---
        print("Generating advanced visualizations...")
        try:
            # Get valid numeric columns for multivariate analysis
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            valid_cols = [col for col in numeric_cols if df[col].isna().mean() < 0.3 and df[col].nunique() > 1]

            # Time series charts
            time_series_dir = os.path.join(VIZ_DIR, VIZ_SUBDIRS["time_series"], timeframe)
            key_metrics = KEY_METRICS
            plot_daily_weekly_rhythms(df, key_metrics, time_series_dir, timeframe)
            plot_nonlinear_metrics(df, time_series_dir, timeframe)
            if timeframe in ['5M', '15M', '30M', '1H', '3H']:
                plot_frequency_domain(df, time_series_dir, timeframe)
            print("  - Time series charts created.")

            # Heatmaps
            heatmaps_dir = os.path.join(VIZ_DIR, VIZ_SUBDIRS["heatmaps"], timeframe)
            for metric in ['RMSSD', 'SDNN']:
                if metric in df.columns:
                    plot_hrv_heatmap(df, metric, heatmaps_dir, timeframe)
            print("  - Heatmap charts created.")

            # Correlation, PCA, and Change Points
            circadian_dir = os.path.join(VIZ_DIR, VIZ_SUBDIRS["circadian"], timeframe)
            if len(valid_cols) > 2:
                create_correlation_heatmap(df, valid_cols, circadian_dir, timeframe)
                create_pca_biplot(df, valid_cols, circadian_dir, timeframe)
                print("  - Correlation matrix and PCA created.")

            for metric in ['RMSSD', 'SDNN']:
                if metric in df.columns:
                    detect_change_points(df, metric, circadian_dir, timeframe)
            print("  - Change point detection completed.")

        except Exception as e:
            print(f"Error creating advanced visualizations for {sheet_name}: {e}")

    print(f"\nAll visualizations saved in: {VIZ_DIR}")
    print(f"Used data file: {file_path}")

if __name__ == "__main__":
    main()