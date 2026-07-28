import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
MATCHES = ROOT / "data" / "processed" / "matches_combined.csv"


def season_ppg(df):
    """Points per game for every team in every season+league."""
    rows = []
    for (season, league), m in df.groupby(["season", "league"]):
        teams = pd.unique(m[["home_team", "away_team"]].values.ravel())
        for t in teams:
            home = m[m["home_team"] == t]
            away = m[m["away_team"] == t]
            pts = 0
            games = len(home) + len(away)
            # home points
            pts += (home["home_goals"] > home["away_goals"]).sum() * 3
            pts += (home["home_goals"] == home["away_goals"]).sum() * 1
            # away points
            pts += (away["away_goals"] > away["home_goals"]).sum() * 3
            pts += (away["away_goals"] == away["home_goals"]).sum() * 1
            rows.append({"season": season, "league": league, "team": t,
                         "ppg": pts / games if games else 0})
    return pd.DataFrame(rows)


def season_order(s):
    """Sort key: '0001' -> 2000, '9900' -> 1999 handling not needed (data starts 2000)."""
    return int(s[:2])


if __name__ == "__main__":
    df = pd.read_csv(MATCHES, dtype={"season": str}, parse_dates=["date"])
    df = df.dropna(subset=["home_goals", "away_goals"])

    ppg = season_ppg(df)
    seasons = sorted(ppg["season"].unique(), key=season_order)

    promoted_gaps = []   # Champ PPG -> PL PPG (team went UP)
    relegated_gaps = []  # PL PPG -> Champ PPG (team came DOWN)

    for i in range(len(seasons) - 1):
        s0, s1 = seasons[i], seasons[i + 1]
        this_yr = ppg[ppg["season"] == s0].set_index("team")
        next_yr = ppg[ppg["season"] == s1].set_index("team")

        for team in this_yr.index:
            if team not in next_yr.index:
                continue
            l0 = this_yr.loc[team, "league"]
            l1 = next_yr.loc[team, "league"]
            p0 = this_yr.loc[team, "ppg"]
            p1 = next_yr.loc[team, "ppg"]

            if l0 == "Champ" and l1 == "Prem":       # promoted
                promoted_gaps.append(p0 - p1)
            elif l0 == "Prem" and l1 == "Champ":     # relegated
                relegated_gaps.append(p1 - p0)       # PPG they GAINED dropping down

    print("="*55)
    print("CROSS-DIVISION QUALITY GAP (measured from your data)")
    print("="*55)

    import statistics as st
    print(f"\nPromoted teams (n={len(promoted_gaps)}):")
    print(f"  Avg PPG DROP going Champ -> PL: {st.mean(promoted_gaps):.3f}")
    print(f"  (they earned {st.mean(promoted_gaps):.2f} fewer points/game in the PL)")

    print(f"\nRelegated teams (n={len(relegated_gaps)}):")
    print(f"  Avg PPG GAIN going PL -> Champ: {st.mean(relegated_gaps):.3f}")

    combined = (st.mean(promoted_gaps) + st.mean(relegated_gaps)) / 2
    print(f"\nCombined average PPG gap between divisions: {combined:.3f}")

    # Rough PPG -> Elo conversion. In Elo, a 1.0 PPG difference over a season
    # corresponds to a substantial rating gap. We estimate: an average PL team
    # (~1.36 PPG) vs the gap gives the implied Elo offset. A common empirical
    # mapping puts the full PL-Champ gap around 150-250 Elo points.
    print(f"\n(We'll convert this PPG gap to an Elo adjustment in the next step.)")