"""Single source of truth for project paths + small shared helpers.

Import from here rather than re-deriving paths in each script. teams.csv moved
once already and its path was duplicated across two files; centralising it means
the next move is a one-line change instead of a grep.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = RAW_DIR / "results"
REFERENCE_DIR = ROOT / "data" / "reference"
PROCESSED_DIR = ROOT / "data" / "processed"

TEAMS_CSV = REFERENCE_DIR / "teams.csv"
MATCHES_CSV = PROCESSED_DIR / "matches_combined.csv"
ELO_RATINGS_CSV = PROCESSED_DIR / "elo_ratings.csv"
ELO_HISTORY_CSV = PROCESSED_DIR / "elo_history.csv"


def load_matches(path=MATCHES_CSV, **kwargs):
    """Read matches_combined.csv with the dtypes it MUST have.

    `season` is read as str: otherwise season code "0001" silently becomes int 1.
    Use this instead of a bare pd.read_csv so that gotcha can't be forgotten.
    """
    return pd.read_csv(path, dtype={"season": str}, parse_dates=["date"], **kwargs)


class LookaheadError(AssertionError):
    """Raised when training data contains rows at or after the prediction cutoff."""


def assert_no_lookahead(train, cutoff, label=""):
    """Hard guarantee that a fit cannot see the future.

    The project's nested-tuning discipline was previously maintained by
    convention alone. Convention is not executable, and the one thing that would
    make every backtest number in this repository meaningless is a training set
    that quietly includes the season it is predicting. This makes the guarantee
    a runtime property instead.

    Note the boundary is INCLUSIVE of the cutoff: a match kicking off at the
    cutoff instant has not been played when the prediction is made, and
    time_weights clips negative ages to zero, so such a row would silently be
    given full weight.
    """
    if cutoff is None or train is None or len(train) == 0:
        return
    if "date" not in train.columns:
        return
    late = train["date"] >= pd.Timestamp(cutoff)
    n = int(late.sum())
    if n:
        first = train.loc[late, "date"].min()
        raise LookaheadError(
            f"{label or 'fit'}: {n} training row(s) at or after the cutoff "
            f"{pd.Timestamp(cutoff).date()} (earliest {pd.Timestamp(first).date()}). "
            "Training data must be strictly earlier than the prediction date.")


def active_teams(matches, season=None, league=None):
    """Set of teams that actually played, optionally within one season/league.

    Elo ratings for long-dormant clubs (Tranmere, Wimbledon...) drift to the pool
    mean and are meaningless. Always filter through this before ranking, rather
    than ranking against every team that has ever appeared.
    """
    m = matches
    if season is not None:
        m = m[m["season"] == season]
    if league is not None:
        m = m[m["league"] == league]
    return set(m["home_team"]) | set(m["away_team"])
