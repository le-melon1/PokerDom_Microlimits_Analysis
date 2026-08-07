"""Project-wide constants for the microlimit analysis pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

# Target limit: NL2 equivalent, 1/2 RUB blinds.
SMALL_BLIND = 1.0
BIG_BLIND = 2.0
CURRENCY = "RUB"
TABLE_SIZES = (6, 7, 8)  # short-handed tables kept in sample intentionally

# Rake modeling (heavily impacts microlimit winrate, per project brief: up to 10-15 bb/100).
RAKE_CAP_BB = 5.0
RAKE_PERCENT = 0.05

# Minimum sample size before trusting a dynamic villain label (e.g. "Maniac", "Station").
MIN_HANDS_FOR_LABEL = 100

# WTSD range typically inflated at microlimits per project brief.
FIELD_WTSD_RANGE = (0.35, 0.45)

POSITIONS_6MAX = ("BTN", "SB", "BB", "UTG", "MP", "CO")
POSITIONS_8MAX = ("BTN", "SB", "BB", "UTG", "UTG+1", "MP", "MP+1", "CO")

STREETS = ("preflop", "flop", "turn", "river")
