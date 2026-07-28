import pandas as pd
from elo import EloEngine
from paths import MATCHES_CSV as MATCHES, PROCESSED_DIR

DIVISION_GAP = 232   # measured empirically from 25 years of promotions/relegations
TARGET_MEAN = 1500   # recenter the whole pool here at the end (cosmetic)


def reanchor_leagues(elo, season_teams, division_gap=DIVISION_GAP):
    """Slide the Championship pool so its mean sits `division_gap` below the PL mean.
    Preserves each team's position within its league; only the offset is corrected."""
    prem = [elo.get(t) for t, lg in season_teams.items() if lg == "Prem"]
    champ = [elo.get(t) for t, lg in season_teams.items() if lg == "Champ"]
    if not prem or not champ:
        return
    m_prem = sum(prem) / len(prem)
    m_champ = sum(champ) / len(champ)
    error = division_gap - (m_prem - m_champ)
    for t, lg in season_teams.items():
        if lg == "Champ":
            elo.ratings[t] = elo.get(t) - error


def build_ratings(k_factor=20, home_adv=70, regression=0.25, division_gap=DIVISION_GAP):
    df = pd.read_csv(MATCHES, dtype={"season": str}, parse_dates=["date"])
    df = df.dropna(subset=["home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)

    elo = EloEngine(k_factor=k_factor, home_advantage=home_adv)

    season_league = {}
    for (season, league), grp in df.groupby(["season", "league"]):
        teams = pd.unique(grp[["home_team", "away_team"]].values.ravel())
        for t in teams:
            season_league.setdefault(season, {})[t] = league

    seen_teams = set()
    history = []
    current_season = None

    for _, r in df.iterrows():
        if r["season"] != current_season:
            if current_season is not None and regression > 0:
                mean = sum(elo.ratings.values()) / len(elo.ratings)
                for team in elo.ratings:
                    elo.ratings[team] += (mean - elo.ratings[team]) * regression

            # NEW-ENTRANT FIX: first-ever appearance starts at the FLOOR of its
            # league's current pool, not the 1500 default.
            this_season = season_league[r["season"]]
            for team, lg in this_season.items():
                if team not in seen_teams:
                    pool = [elo.get(t) for t, l in this_season.items()
                            if l == lg and t in seen_teams]
                    if pool:
                        elo.ratings[team] = min(pool)
                    seen_teams.add(team)

            reanchor_leagues(elo, this_season, division_gap)
            current_season = r["season"]
        else:
            seen_teams.add(r["home_team"])
            seen_teams.add(r["away_team"])

        elo.update_one_match(r["home_team"], r["away_team"],
                             int(r["home_goals"]), int(r["away_goals"]))

        history.append({"date": r["date"], "season": r["season"],
                        "team": r["home_team"], "rating": elo.get(r["home_team"])})
        history.append({"date": r["date"], "season": r["season"],
                        "team": r["away_team"], "rating": elo.get(r["away_team"])})

    # FINAL RECENTER: uniform shift so mean = TARGET_MEAN. Rankings unchanged.
    mean = sum(elo.ratings.values()) / len(elo.ratings)
    shift = TARGET_MEAN - mean
    for team in elo.ratings:
        elo.ratings[team] += shift

    return elo, pd.DataFrame(history)


if __name__ == "__main__":
    elo, history = build_ratings()

    final_season = history["season"].iloc[-1]
    df = pd.read_csv(MATCHES, dtype={"season": str})
    final = df[(df["season"] == final_season) & (df["league"] == "Prem")]
    pl_teams = set(final["home_team"]) | set(final["away_team"])

    ranked = sorted(((t, r) for t, r in elo.ratings.items() if t in pl_teams),
                    key=lambda x: -x[1])

    print("="*50)
    print(f"PREMIER LEAGUE POWER RANKING (end of {final_season}):")
    print("="*50)
    for i, (team, rating) in enumerate(ranked, 1):
        print(f"  {i:2d}. {team:28s} {rating:.0f}")

    champ_final = df[(df["season"] == final_season) & (df["league"] == "Champ")]
    champ_teams = set(champ_final["home_team"]) | set(champ_final["away_team"])
    champ_ranked = sorted(((t, r) for t, r in elo.ratings.items() if t in champ_teams),
                          key=lambda x: -x[1])
    print("\nTop 5 Championship sides (should sit below the PL pack):")
    for i, (team, rating) in enumerate(champ_ranked[:5], 1):
        print(f"  {i:2d}. {team:28s} {rating:.0f}")

# --- Save ratings + history so we don't rebuild every time ---
    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Final ratings: one row per team, sorted strongest first
    ratings_df = (pd.DataFrame(
        [{"team": t, "elo": round(r, 1)} for t, r in elo.ratings.items()]
    ).sort_values("elo", ascending=False).reset_index(drop=True))
    ratings_df.to_csv(out_dir / "elo_ratings.csv", index=False)

    # 2. Full history: every team's rating after every match (for trajectory plots)
    history.to_csv(out_dir / "elo_history.csv", index=False)

    print(f"\nSaved {len(ratings_df)} final ratings to elo_ratings.csv")
    print(f"Saved {len(history)} history rows to elo_history.csv")

    #Elo ratings for teams absent from recent seasons are stale/meaningless — filter to active teams when using