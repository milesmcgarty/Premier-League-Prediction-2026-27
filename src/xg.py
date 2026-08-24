"""Phase 9: match-level expected goals from Understat.

WHY xG. Goals are a noisy measure of how well a team played -- a 1-0 win off one
shot and a 1-0 win off twenty are identical to a goals-based model. Expected
goals measures chance quality rather than conversion, so it is a less noisy
estimate of the same underlying team strength.

COVERAGE, verified rather than assumed (2026-08-24):
  - Understat read_schedule() returns 380 matches per season with home_xg and
    away_xg, for every season from 2014-15 to 2025-26. Twelve seasons.
  - Understat is PREMIER LEAGUE ONLY. There is no Championship xG, confirmed via
    available_leagues(). That matters: the joint two-division fit is what
    identifies the division gap and rates promoted teams, so xG cannot simply
    replace goals throughout without measuring the two divisions with different
    instruments.
  - FBref genuinely does not expose xG in soccerdata 1.9.0 -- its four stat types
    (standard, keeper, shooting, playing_time) contain none. The Phase 1 note was
    correct on this, unlike its Transfermarkt claim.

Scraping is browser-automated and slow, so results are cached to
data/processed/xg_matches.csv and only re-fetched on demand.
"""
import sys

import pandas as pd

from paths import PROCESSED_DIR, TEAMS_CSV, load_matches

XG_CSV = PROCESSED_DIR / "xg_matches.csv"
FIRST_SEASON = "1415"
LEAGUE = "ENG-Premier League"


def _season_codes(first=FIRST_SEASON):
    """Every season code from `first` to the latest present in our match data."""
    order = sorted(load_matches()["season"].unique(), key=lambda s: int(s[:2]))
    return [s for s in order if int(s[:2]) >= int(first[:2])]


def _name_map():
    """Understat name -> canonical. Falls back to an exact canonical match.

    teams.csv's `understat` column was only ever filled for one season's squad
    list, but Understat spells most clubs exactly as our canonical names do, so
    an exact match is a safe second pass. Anything still unmatched is reported
    rather than silently dropped.
    """
    t = pd.read_csv(TEAMS_CSV)
    m = dict(zip(t["understat"].dropna(), t.loc[t["understat"].notna(), "canonical_name"]))
    for c in t["canonical_name"]:
        m.setdefault(c, c)
    return m


def fetch_xg(seasons=None, verbose=True):
    """Match-level xG for `seasons`. Requires network + browser automation."""
    import soccerdata as sd

    seasons = seasons or _season_codes()
    frames = []
    for s in seasons:
        if verbose:
            print(f"  fetching {s} ...", flush=True)
        try:
            sch = sd.Understat(leagues=LEAGUE, seasons=s).read_schedule()
        except Exception as e:
            print(f"    FAILED {s}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        d = sch.reset_index()
        d = d[["date", "home_team", "away_team", "home_goals", "away_goals",
               "home_xg", "away_xg"]].copy()
        d["season"] = s
        frames.append(d)
    if not frames:
        raise RuntimeError("No Understat data fetched.")
    return pd.concat(frames, ignore_index=True)


def canonicalise(df, strict=True):
    """Map Understat team names to canonical. Raises on anything unmapped."""
    m = _name_map()
    names = sorted(set(df["home_team"]) | set(df["away_team"]))
    unmapped = [n for n in names if n not in m]
    if unmapped:
        msg = (f"{len(unmapped)} Understat team name(s) missing from "
               f"{TEAMS_CSV.name}: {unmapped}. Add them to the `understat` "
               "column before trusting the output -- unmapped names become NaN "
               "and the club silently disappears.")
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)
    df = df.copy()
    df["home_team"] = df["home_team"].map(m)
    df["away_team"] = df["away_team"].map(m)
    return df, unmapped


def save_xg(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(XG_CSV, index=False)
    return XG_CSV


def load_xg():
    if not XG_CSV.exists():
        raise FileNotFoundError(
            f"No xG table at {XG_CSV}. Run:  py src/xg.py  to fetch it.")
    return pd.read_csv(XG_CSV, dtype={"season": str}, parse_dates=["date"])


def attach_xg(matches, xg=None):
    """Join xG onto a match table by season, teams. Unmatched rows keep NaN."""
    if xg is None:
        xg = load_xg()
    key = ["season", "home_team", "away_team"]
    return matches.merge(xg[key + ["home_xg", "away_xg"]], on=key, how="left")


if __name__ == "__main__":
    seasons = sys.argv[1:] or _season_codes()
    print(f"Fetching Understat xG for {len(seasons)} seasons "
          f"({seasons[0]} -> {seasons[-1]})")
    raw = fetch_xg(seasons)
    print(f"\n  {len(raw)} matches fetched")

    df, unmapped = canonicalise(raw, strict=False)
    if unmapped:
        print(f"\n  {len(unmapped)} UNMAPPED name(s): {unmapped}")
        print("  Add them to the `understat` column of teams.csv, then re-run.")
        print("  (soccerdata caches, so the re-run will be fast.)")
        sys.exit(1)

    path = save_xg(df)
    print(f"  saved to {path.relative_to(PROCESSED_DIR.parent.parent)}")
    print(f"\n  seasons: {df['season'].nunique()}, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  mean goals {df[['home_goals', 'away_goals']].mean().round(3).tolist()}"
          f"   mean xG {df[['home_xg', 'away_xg']].mean().round(3).tolist()}")
