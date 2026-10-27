"""Shared paths, feature list and settings for every script in this project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FONTS = ROOT / "assets" / "fonts"

SEASON_ROWS = RAW / "bbref_advanced_season_rows.csv"
RAPM_FILE = RAW / "nbarapm_5y_ending_2026.csv"
MODEL_DATA = PROCESSED / "model_data.csv"

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
MIN_MINUTES = 2500        # total minutes over the 5 seasons
MIN_POSSESSIONS = 5000    # RAPM possessions over the 5 seasons

TARGET = "rapm_5y"

# column in the season file -> column in the model file -> label used in the README
FEATURES = {
    "ts_pct": ("ts_pct", "TS%"),
    "fg3a_per_fga_pct": ("three_par", "3PAr"),
    "fta_per_fga_pct": ("ftr", "FTr"),
    "trb_pct": ("trb_pct", "TRB%"),
    "ast_pct": ("ast_pct", "AST%"),
    "stl_pct": ("stl_pct", "STL%"),
    "blk_pct": ("blk_pct", "BLK%"),
    "tov_pct": ("tov_pct", "TOV%"),
    "usg_pct": ("usg_pct", "USG%"),
}
X_COLS = [v[0] for v in FEATURES.values()]
LABELS = {v[0]: v[1] for v in FEATURES.values()}

# order used for the correlation matrix in the README
CORR_ORDER = [TARGET, "ast_pct", "usg_pct", "tov_pct", "ts_pct", "ftr",
              "three_par", "stl_pct", "blk_pct", "trb_pct"]

SPLIT_SEED = 42           # 80-20 train-test split
TEST_SIZE = 0.20
KFOLD_REPEATS = 200       # repeated 10-fold cross-validation (numpy default_rng(0))


def label(col):
    return "RAPM" if col == TARGET else LABELS.get(col, col)
