"""
config.py — Centralized configuration for the xHRV artifact detection &
HRV analysis pipeline.

All scripts import from this module rather than hardcoding paths or constants.

Import boilerplate
------------------
Scripts in /Volumes/xHRV/Scripts/ (root level):
    from config import <names>

Scripts in subdirectories (models/, features/, learning/, etc.):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import <names>
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VOLUME ROOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT = Path("/Volumes/xHRV")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DIRECTORIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPTS_DIR      = ROOT / "Scripts"
DATA_DIR         = ROOT / "Data"
OUTPUTS_DIR      = ROOT / "Outputs"
PROCESSED_DIR    = ROOT / "Processed"       # canonical Parquet tables
MODELS_DIR       = ROOT / "Artifact Detector" / "Models"

# input data
ECG_DIR          = DATA_DIR / "ECG"
PEAKS_DIR        = DATA_DIR / "Peaks"
ANNOTATIONS_DIR  = DATA_DIR / "Annotations"
ACCESSORY_DIR    = DATA_DIR / "Accessory"

# annotation sub-dirs
ANNOTATIONS_V3_DIR      = ANNOTATIONS_DIR / "V3"
ANNOTATIONS_MARKERS_DIR = ANNOTATIONS_DIR / "Markers"
STACKS_DIR              = ANNOTATIONS_DIR / "Stacks"

# accessory sub-dirs
AUTODETECTED_DIR = ACCESSORY_DIR / "Autodetected"

# output sub-dirs
VIZ_DIR          = OUTPUTS_DIR / "Visualizations"
QUEUE_DIR        = DATA_DIR / "Queue"
CHECKPOINT_DIR   = ACCESSORY_DIR / "Checkpoints"
ENTROPIC_DIR     = ACCESSORY_DIR / "Entropic"
HRV_DIR_ACC      = ACCESSORY_DIR / "HRV"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CANONICAL PARQUET FILES  (written by data_pipeline.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ECG_SAMPLES_PARQUET      = PROCESSED_DIR / "ecg_samples.parquet"
PEAKS_PARQUET            = PROCESSED_DIR / "peaks.parquet"
LABELS_PARQUET           = PROCESSED_DIR / "labels.parquet"
SEGMENTS_PARQUET         = PROCESSED_DIR / "segments.parquet"

# derived feature tables
BEAT_FEATURES_PARQUET    = PROCESSED_DIR / "beat_features.parquet"
SEGMENT_FEATURES_PARQUET = PROCESSED_DIR / "segment_features.parquet"
GLOBAL_TEMPLATES_PARQUET = PROCESSED_DIR / "global_templates.parquet"
MOTIF_FEATURES_PARQUET   = PROCESSED_DIR / "motif_features.parquet"

# model prediction outputs
SEGMENT_QUALITY_PARQUET  = PROCESSED_DIR / "segment_quality_preds.parquet"
BEAT_TABULAR_PREDS       = PROCESSED_DIR / "beat_tabular_preds.parquet"
BEAT_CNN_PREDS           = PROCESSED_DIR / "beat_cnn_preds.parquet"
ENSEMBLE_PREDS           = PROCESSED_DIR / "ensemble_preds.parquet"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODEL FILES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TABULAR_MODEL          = MODELS_DIR / "beat_artifact_tabular.joblib"
CNN_MODEL              = MODELS_DIR / "beat_artifact_cnn.pt"
SEGMENT_QUALITY_MODEL  = MODELS_DIR / "segment_quality.joblib"
PRETRAINED_ENCODER     = MODELS_DIR / "encoder_pretrained_weights.pt"
PRETRAINED_AUTOENCODER = MODELS_DIR / "autoencoder_pretrained.pt"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ANNOTATION FILES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEAT_TAGS_CSV           = ANNOTATIONS_V3_DIR / "beat_tags.csv"
CUSTOM_TAGS_JSON        = ANNOTATIONS_V3_DIR / "custom_tags.json"
TAG_POOL_JSON           = ANNOTATIONS_V3_DIR / "tag_pool.json"
TAG_SETS_JSON           = ANNOTATIONS_V3_DIR / "tag_sets.json"
REANNOTATED_LABELS_CSV  = ANNOTATIONS_V3_DIR / "reannotated_labels.csv"

MARKER_BEATS_CSV        = ANNOTATIONS_MARKERS_DIR / "beats.csv"
MARKER_SEGMENTS_CSV     = ANNOTATIONS_MARKERS_DIR / "segments.csv"
MARKER_AI_CSV           = ANNOTATIONS_MARKERS_DIR / "ai.csv"
MARKER_TAGS_JSON        = ANNOTATIONS_MARKERS_DIR / "tags.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ACCESSORY / DEVICE FILES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MARKER_CSV           = ACCESSORY_DIR / "Marker.csv"
DEVICE_HR_CSV        = ACCESSORY_DIR / "DeviceHR.csv"
DEVICE_RR_CSV        = ACCESSORY_DIR / "DeviceRR.csv"
RSA_EVENTS_CSV       = AUTODETECTED_DIR / "rsa_events.csv"
VAGAL_ARRESTS_CSV    = AUTODETECTED_DIR / "vagal_arrests.csv"

INTERVALS_JSON       = ACCESSORY_DIR / "HRV Intervals.json"
IHR_INTERVALS_JSON   = ACCESSORY_DIR / "IHR_Intervals.json"
HRV_LEDGER_JSON      = ACCESSORY_DIR / "HRV Processed.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HRV OUTPUT FILES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HR_CSV         = OUTPUTS_DIR / "HRV4_Instantaneous.csv"
XL_MET         = OUTPUTS_DIR / "HRV Metrics.xlsx"
XL_STAT        = OUTPUTS_DIR / "HRV Statistics.xlsx"
RAW_ECG_DIR    = Path("/Users/tannereddy/Downloads/Raw ECG")

# Artifact-filtered RR intervals CSV — primary input to the HRV analysis pipeline.
# Generated downstream from ensemble_preds.parquet.
RR_CSV         = ACCESSORY_DIR / "RR_Intervals.csv"

# Base filename used when auto-renaming date-stamped Excel outputs.
XL_BASE: str   = "HRV Metrics"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIGNAL PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_RATE_HZ: int       = 130    # Polar H10 effective ECG sample rate

BEAT_WINDOW_MS: int       = 1000   # 1-second CNN context window
QRS_WINDOW_MS: int        = 500    # 0.5-second QRS morphology window

# Derived sample counts — always consistent with SAMPLE_RATE_HZ
WINDOW_SIZE_SAMPLES: int  = int(BEAT_WINDOW_MS / 1000 * SAMPLE_RATE_HZ)   # 130 @ 130 Hz
QRS_WINDOW_SAMPLES: int   = int(QRS_WINDOW_MS  / 1000 * SAMPLE_RATE_HZ)   #  65 @ 130 Hz

PEAK_SNAP_SAMPLES: int    = 8      # +-8 samples = ~+-62 ms snap tolerance

ECG_BANDPASS_LOW_HZ: float  = 3.0
ECG_BANDPASS_HIGH_HZ: float = 40.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TIMESTAMP CONSTANTS  (all in MILLISECONDS — no nanoseconds in this codebase)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEGMENT_DURATION_MS: int           = 60_000       # 60 seconds per segment
DEDUP_TOLERANCE_MS: int            = 10           # 10 ms dedup window
ANNOTATION_MATCH_TOLERANCE_MS: int = 80           # 80 ms peak-to-annotation snap
MIN_VALID_TIMESTAMP_MS: int        = 1_577_836_800_000   # 2020-01-01 00:00:00 UTC


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PATIENT PHYSIOLOGY  (POTS — patient-specific)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HR_BRADY_BPM: float           = 65.0    # bradycardic threshold
HR_TACHY_BPM: float           = 135.0   # tachycardic threshold (190s are genuine physiology)
HR_SUSPICIOUS_LOW_BPM: float  = 45.0    # hard flag
HR_SUSPICIOUS_HIGH_BPM: float = 205.0   # POTS patient regularly hits 186-190+ BPM
HR_MODAL_LOW_BPM: float       = 85.0    # typical floor
HR_MODAL_HIGH_BPM: float      = 115.0   # typical ceiling

RR_FALLBACK_MS: float          = 600.0   # median HR ~100 BPM
RR_SUSPICIOUS_SHORT_MS: float  = 275.0   # ~218 BPM, soft flag
RR_SUSPICIOUS_LONG_MS: float   = 2250.0  # ~27 BPM, soft flag
RR_SANDWICH_SHORT_MS: float    = 175.0   # spurious-peak sandwich threshold
RR_NORMAL_LOW_MS: float        = 300.0   # ~200 BPM upper limit
RR_NORMAL_HIGH_MS: float       = 1800.0  # ~33.3 BPM lower limit

POTS_TRANSITION_WINDOW_SEC: float      = 20.0
POTS_MAX_DELTA_HR_PER_SEC: float       = 25.0
TACHY_TRANSITION_MIN_RISE_BPM: float   = 40.0
TACHY_TRANSITION_LOOK_AHEAD_SEC: float = 60.0
TACHY_TRANSITION_SMOOTH_BEATS: int     = 10

POTS_MIN_FRAC_DECREASING: float  = 0.65
POTS_MIN_RR_DECLINE_MS: float    = 200.0
POTS_MIN_FRAC_INCREASING: float  = 0.65
POTS_MIN_RR_RECOVERY_MS: float   = 200.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PIPELINE THRESHOLDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARTIFACT_FRACTION_BAD: float   = 0.50   # >50% artifact -> bad segment
MIN_VALIDATED_BEATS_CLEAN: int = 10     # min clean beats for "clean" segment

# ECG polarity detection
ECG_INVERSION_THRESHOLD: float  = -100.0  # mV; negative dominant amplitude -> invert
ECG_INVERSION_WINDOW_SEC: float = 1.0
ECG_INVERSION_MIN_WINDOWS: int  = 5.0
ECG_POLARITY_SAMPLE_STEP: int   = 10    # read 1-in-N rows for file-wide scan

# detect_peaks
REFRACTORY_MS: int           = 225     # minimum ms between R-peaks
MERGE_TOLERANCE_MS: int      = 50      # ensemble detector merge window
PEAK_CHUNK_MIN: int          = 5       # chunk duration in minutes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODEL TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LGBM_N_ESTIMATORS_BEAT: int  = 1000
LGBM_N_ESTIMATORS_SEG: int   = 500
LGBM_LEARNING_RATE: float    = 0.05
LGBM_NUM_LEAVES_BEAT: int    = 63
LGBM_NUM_LEAVES_SEG: int     = 31
LGBM_RANDOM_STATE: int       = 42
VAL_FRACTION: float          = 0.2     # 80/20 temporal train/val split

CNN_BATCH_SIZE: int          = 256
CNN_MAX_EPOCHS: int          = 50
CNN_LEARNING_RATE: float     = 1e-3
CNN_DROPOUT: float           = 0.3

ENSEMBLE_ALPHA: float        = 0.5     # tabular weight (1-alpha = CNN weight)
ENSEMBLE_THRESHOLD: float    = 0.5

ECG_CHUNK_SIZE: int          = 10_000_000
BEAT_BATCH_SIZE: int         = 10_000_000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HRV PIPELINE  (unified from HRV/config.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COL_MAPPING = {
    'time':   'DateTime',
    'ecg':    'ECG',
    'hr':     'HR',
    'rr':     'InputRR',
    'marker': 'Marker',
}

MAX_GAP: int             = 10
NEIGHBORING_BEATS: int   = 20
MAX_CONDITIONAL_GAP: int = 15
LOW_VAR_THRESHOLD: float = 300.0
LOW_VAR_BEATS: int       = 30
MS_IN_SECOND: int        = 1000

TIMEFRAMES = ['5M', '15M', '30M', '1H', '2H', '3H', '4H', '6H', '12H', '24H']

MIN_INT = {
    '5M': 250, '15M': 750, '30M': 1500, '1H': 3000, '2H': 6500,
    '3H': 9750, '4H': 15000, '6H': 24000, '12H': 48000, '24H': 96000,
}

STATS = {
    'Count', 'MedRR', 'MinRR', 'MaxRR', 'MxDMn', 'MAD', 'MCV',
    'IQR', 'Q20', 'Q80', 'CV%',
}

BM = {
    'td': ['Count', 'MedRR', 'MinRR', 'MaxRR', 'MxDMn', 'IQR', 'Q20', 'Q80',
           'MAD', 'MCV', 'SDSD', 'SDNN', 'RMSSD', 'pNN20', 'pNN50', 'CV%',
           'CVSD', 'DC', 'AC', 'SDANN5', 'SDNNI5'],
    'fd': ['Total Power', 'HF', 'HF%', 'HFp', 'LFHF', 'LF', 'LF%', 'LFp',
           'VLF', 'VLF%', 'VLFp', 'ULF'],
    'nl': ['PermEn', 'SampEn', 'MFE', 'CorrDim', 'DF𝛼1', 'DF𝛼2'],
    'gm': ['SD1', 'SD2', 'SD1SD2', 'CSI', 'CVI', 'AMo50%', 'HTI', 'TINN'],
}

BMS = {
    'base': {'Count', 'MedRR', 'MinRR', 'MaxRR', 'MxDMn', 'IQR', 'Q20', 'Q80',
             'MAD', 'MCV', 'SDSD', 'SDNN', 'RMSSD', 'pNN20', 'pNN50', 'CV%',
             'CVSD', 'DC', 'AC', 'HF', 'HF%', 'HFp', 'LF', 'LF%', 'LFp',
             'LFHF', 'PermEn', 'SampEn', 'DF𝛼1', 'SD1', 'SD2', 'SD1SD2',
             'CSI', 'CVI'},
    '≤1h': {'MFE', 'CorrDim'},
    '≥3h': {'VLF', 'VLF%', 'VLFp', 'Total Power', 'DF𝛼2'},
    '24h': {'ULF', 'SDANN5', 'SDNNI5', 'AMo50%', 'HTI', 'TINN'},
}

TF = {
    '5M':  BMS['base'] | BMS['≤1h'],
    '15M': BMS['base'] | BMS['≤1h'],
    '30M': BMS['base'] | BMS['≤1h'],
    '1H':  BMS['base'] | BMS['≤1h'],
    '2H':  BMS['base'] | BMS['≤1h'],
    '3H':  BMS['base'] | BMS['≥3h'],
    '4H':  BMS['base'] | BMS['≥3h'],
    '6H':  BMS['base'] | BMS['≥3h'],
    '12H': BMS['base'] | BMS['≥3h'],
    '24H': BMS['base'] | BMS['≥3h'] | BMS['24h'],
}

ENTROPIC_INTERVALS = {
    '5M':  {'delta': timedelta(minutes=5),  'dimension': 3, 'min_intervals': 250,
            'cp_fractions': [1.0]},
    '15M': {'delta': timedelta(minutes=15), 'dimension': 3, 'min_intervals': 750,
            'cp_fractions': [0.5, 1.0]},
    '30M': {'delta': timedelta(minutes=30), 'dimension': 3, 'min_intervals': 1500,
            'cp_fractions': [0.33, 0.66, 1.0]},
    '1H':  {'delta': timedelta(hours=1),    'dimension': 4, 'min_intervals': 3400,
            'cp_fractions': [0.25, 0.5, 0.75, 1.0]},
}

DEC = 1
DECX = {
    'Count': 0, 'MAD': 2, 'MCV': 2, 'RMSSD': 2, 'DC': 2, 'AC': 2,
    'HTI': 0, 'TINN': 0, 'Total Power': 0, 'ULF': 0, 'VLF': 0, 'VLFp': 3,
    'LF': 0, 'LF%': 2, 'LFp': 3, 'HF': 0, 'HF%': 2, 'HFp': 3, 'LFHF': 2,
    'PermEn': 3, 'SampEn': 3, 'MFE': 3, 'CorrDim': 3,
    'DF𝛼1': 3, 'DF𝛼2': 3,
    'SDANN5': 0, 'SDNNI5': 0, 'SD1': 2, 'SD2': 2, 'SD1SD2': 2,
    'CSI': 2, 'CVI': 2,
}

NORM = {
    'SDNN':  {'low': 30,  'normal': (30, 150),  'high': 150,  'unit': 'ms'},
    'RMSSD': {'low': 27,  'normal': (27, 72),   'high': 72,   'unit': 'ms'},
    'pNN50': {'low': 3,   'normal': (3, 30),    'high': 30,   'unit': '%'},
    'pNN20': {'low': 1,   'normal': (1, 20),    'high': 20,   'unit': '%'},
    'LFHF':  {'low': 0.5, 'normal': (0.5, 2.0), 'high': 2.0, 'unit': 'ratio'},
}

ANALYSIS_TF       = '2H'
MAX_INTERPOLATION = 10
MIN_WEEKLY_HRS    = 120
MIN_DAILY_HRS     = 17
MIN_OVERALL_HRS   = 24

CIR_METRICS       = [['SDNN', 'RMSSD'], ['pNN20', 'LFHF'], ['LF', 'HF']]
KEY_METRICS       = ['SDNN', 'RMSSD', 'LFHF']
NONLINEAR_METRICS = ['SD1', 'SD2', 'SD1SD2', 'SampEn', 'DF𝛼1', 'MFE', 'CorrDim']
FREQ_METRICS      = ['ULF', 'VLF', 'LF', 'HF']

CHNG_PT_PENALTY  = 10
CHNG_PT_COLORS   = ['#ADD8E6', '#FFC0CB']
HRV3_PLOT_TYPES  = ["Line Plots", "Scatter Plots", "Histograms"]
HRV3_SHEET_NAMES = ["5M", "15M", "30M", "1H", "2H", "3H", "4H", "6H", "12H", "24H"]

MPL_RC_PARAMS = {
    'figure.figsize':   (12, 6),
    'figure.dpi':       100,
    'savefig.dpi':      300,
    'font.family':      'STIXGeneral',
    'mathtext.fontset': 'stix',
    'axes.grid':        True,
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
}

DT_CSVS_FORMAT   = '%-m-%-d-%y %-I:%M:%S %p'
DT_CSVS_FILENAME = '%-m-%-d, %-I.%M%p'
DT_XL_FORMAT     = '%-m/%-d/%y %-I:%M %p'
DT_HR_FORMAT     = '%-m/%-d/%y %-I:%M:%S %p'
DT_XL_FILENAME   = '%b %-d %-I.%M%p'

VIZ_SUBDIRS = {
    "time_series": "Time Series Charts",
    "heatmaps":    "Heatmaps",
    "circadian":   "Circadian Analysis",
    "line":        "Line Plots",
    "scatter":     "Scatter Plots",
    "histogram":   "Histograms",
}

HR_DEC = 1
SEQUENT_SCRIPTS: set = set()
