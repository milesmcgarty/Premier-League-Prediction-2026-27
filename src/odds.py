"""Bookmaker odds -> de-vigged probabilities, and the comparison-set guard.

OPENING vs CLOSING matters more than anything else here. football-data.co.uk ships
both: the plain columns (B365H) are the OPENING price, the *C* columns (B365CH) are
the CLOSING price. The closing line is the recognised benchmark in forecasting
literature, because it has absorbed team news, weather and the market's own money
right up to kick-off, and is measurably sharper than the open. A model compared only
against opening odds is being graded against a softer benchmark than it should be.

Coverage is NOT uniform across seasons, so the baseline column matters:
  B365   opening, from season 0203 (2002-03) at ~100%   <- longest history
  Avg    opening consensus, from 1920 (2019-20)
  B365C  CLOSING, from 1920 (2019-20)                   <- the real benchmark
  AvgC   CLOSING consensus, from 1920 (2019-20)         <- the sharpest of all
  WH     from 0001 but 0% in 2526 and 80% in 2425       <- unusable as a spine
"""
import numpy as np
import pandas as pd

ODDS_GROUPS = {
    "B365": ["B365H", "B365D", "B365A"],
    "WH": ["WHH", "WHD", "WHA"],
    "Avg": ["AvgH", "AvgD", "AvgA"],
    "B365C": ["B365CH", "B365CD", "B365CA"],
    "AvgC": ["AvgCH", "AvgCD", "AvgCA"],
}

# Longest-history benchmark, used where all 9 report seasons are needed.
PRIMARY_BOOK = "B365"
# The benchmark that actually counts, where available (1920 onward).
CLOSING_BOOK = "B365C"
CLOSING_CONSENSUS = "AvgC"


def clean_odds(df, verbose=True):
    """Null unusable odds triplets. Keeps the MATCH -- only the price is bad.

    Two things are nulled, per bookmaker, as a whole triplet:

    1. Impossible prices. A decimal odd of 1.0 implies certainty and anything
       below it implies probability > 1. Two B365H values are a literal 0.0
       (Blackpool-Derby 2013-04-27, Brentford-Blackburn 2019-02-02), and 1/0 is
       an infinite implied probability, which turns any log loss into inf/NaN.
    2. Partial triplets. Implied probabilities are normalised across H/D/A, so
       a triplet missing one leg cannot be de-vigged and is useless anyway.

    Dropping the rows instead would be wrong: the results are perfectly good and
    the model should still be scored on them.
    """
    for book, cols in ODDS_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if len(present) != 3:
            continue
        vals = df[present]
        impossible = (vals <= 1.0).any(axis=1)
        partial = vals.isna().any(axis=1) & vals.notna().any(axis=1)
        if verbose and impossible.any():
            print(f"  [{book}] nulled {int(impossible.sum())} impossible "
                  f"triplet(s) (an odd <= 1.0)")
        if verbose and partial.any():
            print(f"  [{book}] nulled {int(partial.sum())} partial triplet(s)")
        df.loc[impossible | partial, present] = np.nan
    return df


def has_odds(df, book):
    """Boolean mask: this match has a complete, usable triplet for `book`."""
    cols = ODDS_GROUPS[book]
    if not all(c in df.columns for c in cols):
        return pd.Series(False, index=df.index)
    return df[cols].notna().all(axis=1) & (df[cols] > 1.0).all(axis=1)


def market_probs(df, book):
    """De-vigged (H, D, A) probabilities as an (n, 3) array; NaN where absent.

    Overround removal is multiplicative: take 1/odds and normalise so the three
    sum to 1. Bookmakers price in a margin, so raw 1/odds sums to ~1.05 and
    would score as an invalid distribution.

    This is the standard approach and is what we compare against. It does assume
    the margin is spread proportionally across the three outcomes; Shin and
    power methods relax that (they load more margin onto longshots). Worth
    revisiting in Phase 5 if the market baseline is being flattered on heavy
    favourites, but proportional is the honest default.
    """
    cols = ODDS_GROUPS[book]
    raw = 1.0 / df[cols].to_numpy(dtype=float)
    return raw / raw.sum(axis=1, keepdims=True)


def overround(df, book):
    """Raw bookmaker margin: sum of 1/odds. ~1.05 means a 5% book."""
    cols = ODDS_GROUPS[book]
    return (1.0 / df[cols]).sum(axis=1)


def comparison_set(df, books):
    """Mask of matches usable by EVERY listed book, so model and market are
    scored on identical fixtures.

    This is not hygiene, it is a correctness requirement. Missing odds are not
    randomly distributed -- they cluster on obscure fixtures, which are exactly
    the hard-to-predict ones. Scoring the market on a shrunken set while the
    model is scored on everything would hand the market an easier exam and make
    the comparison meaningless.
    """
    mask = pd.Series(True, index=df.index)
    for b in books:
        mask &= has_odds(df, b)
    return mask


def report_coverage(df, books, label=""):
    """Print how many matches survive, and what was lost. Always print this."""
    n = len(df)
    print(f"\nOdds coverage{' - ' + label if label else ''} (n = {n} matches)")
    for b in books:
        k = int(has_odds(df, b).sum())
        print(f"  {b:>5}: {k:>6} / {n}  ({k / n:6.1%})")
    both = int(comparison_set(df, books).sum())
    print(f"  {'ALL':>5}: {both:>6} / {n}  ({both / n:6.1%})  <- comparison set")
    if both < n:
        print(f"  NOTE: {n - both} match(es) excluded from the market comparison.")
    return both
