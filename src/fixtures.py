"""Phase 7: the live fixture + results table for the current season.

Source is the FPL API, which publishes ALL 380 Premier League fixtures for the
season up front and fills in scores as they are played. football-data.co.uk's
fixtures.csv only carries the next few days, so it cannot support a full-season
simulation in August -- but it is the source for Championship fixtures, which
FPL does not cover.

The output is one table per season with a row per fixture, home_goals/away_goals
null until played. That single table is what the weekly harness re-reads: fill
in results, re-run, snapshot.
"""
import json
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from paths import ROOT, TEAMS_CSV

FIXTURES_DIR = ROOT / "data" / "fixtures"
FPL_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
CURRENT_SEASON = "2627"
UA = {"User-Agent": "Mozilla/5.0 (PL-prediction-project; personal analysis)"}


def _get_json(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _fpl_name_map():
    t = pd.read_csv(TEAMS_CSV)
    m = dict(zip(t["fpl"].dropna(), t.loc[t["fpl"].notna(), "canonical_name"]))
    return m


def fetch_fpl_fixtures(season=CURRENT_SEASON):
    """All Premier League fixtures for the season, with results where played.

    Raises if any FPL team name is missing from teams.csv -- the same discipline
    as load_results. An unmapped name would otherwise become NaN and silently
    delete a club, and promoted sides are exactly where new spellings appear.
    """
    boot = _get_json(FPL_BOOTSTRAP)
    fx = _get_json(FPL_FIXTURES)
    tid = {t["id"]: t["name"] for t in boot["teams"]}
    name_map = _fpl_name_map()

    unmapped = sorted({n for n in tid.values() if n not in name_map})
    if unmapped:
        raise ValueError(
            f"FPL team name(s) missing from {TEAMS_CSV.name}: {unmapped}. "
            "Add them (the fpl column is season-specific and shifts every year "
            "as clubs are promoted and relegated) before trusting the output."
        )

    rows = []
    for f in fx:
        h, a = tid.get(f["team_h"]), tid.get(f["team_a"])
        ko = f.get("kickoff_time")
        rows.append({
            "date": pd.to_datetime(ko).tz_localize(None) if ko else pd.NaT,
            "gameweek": f.get("event"),
            "home_team": name_map[h],
            "away_team": name_map[a],
            "league": "Prem",
            "season": season,
            "home_goals": f["team_h_score"] if f["finished"] else None,
            "away_goals": f["team_a_score"] if f["finished"] else None,
            "finished": bool(f["finished"]),
        })
    df = pd.DataFrame(rows).sort_values(["date", "home_team"]).reset_index(drop=True)
    return df


def save_fixtures(df, season=CURRENT_SEASON):
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    p = FIXTURES_DIR / f"{season}.csv"
    df.to_csv(p, index=False)
    return p


def load_fixtures(season=CURRENT_SEASON):
    p = FIXTURES_DIR / f"{season}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"No fixture table at {p}. Run:  py src\fixtures.py  to fetch it.")
    return pd.read_csv(p, dtype={"season": str}, parse_dates=["date"])


def played_matches(fixtures):
    """Finished fixtures, in the same shape as matches_combined rows."""
    p = fixtures[fixtures["home_goals"].notna() & fixtures["away_goals"].notna()]
    return p[["date", "home_team", "away_team", "home_goals", "away_goals",
              "league", "season"]].copy()


if __name__ == "__main__":
    print(f"Fetching {CURRENT_SEASON} fixtures from the FPL API...")
    df = fetch_fpl_fixtures()
    path = save_fixtures(df)
    n_done = int(df["finished"].sum())
    print(f"  {len(df)} fixtures, {n_done} played, {len(df)-n_done} remaining")
    print(f"  {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  {df['home_team'].nunique()} teams")
    print(f"  saved to {path.relative_to(ROOT)}")
    print(f"\nfetched at {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print("\nFirst gameweek:")
    print(df.head(10)[["date", "gameweek", "home_team", "away_team"]].to_string(index=False))
