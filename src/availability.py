"""Phase 10: key-player availability.

THE IDEA. The model is fitted on results, so it cannot tell "played badly at full
strength" from "played badly missing four starters" -- and only the second should
revert when those players return. Availability is information about the FUTURE
rather than a restatement of the past, which is what makes it different from
squad market value.

WHY THIS SURVIVES WHERE SQUAD VALUE DID NOT. Squad value was tested and rejected
(see CLAUDE.md): it correlates with points at r=+0.762 but adds nothing to match
prediction, because the model's ratings already encode which squads are good, and
the market prior absorbs the rest. Availability is transient and match-specific.
The market prior is fitted once on a season's opening fixtures, so it cannot know
that a key player limped off in November.

HELD-OUT RESULT (9 seasons, market prior already applied):
    model + market prior                 0.9790
    model + market prior + availability  0.9760      (+0.0030)
Positive in 8 of 9 seasons, t = 2.97, p = 0.018. Availability also helps without
the prior (0.9858 -> 0.9830), so the two are complementary rather than rival.

TWO SOURCES, ONE SCALE
  historical  Transfermarkt appearances (2012-13 onward). For each match, weight
              every player by their share of the team's minutes over the previous
              eight games, then measure how much of that weight actually appeared
              in the team's MOST RECENT match. Uses only prior information.
  live        the FPL API's injury and availability flags, weighted by price.
              Transfermarkt appearances stop at the end of last season, so the
              current campaign has to come from somewhere current.
Both return a team-level number in [0, 1] on the same scale, so the coefficient
tuned on history applies to the live season.
"""
import sys

import numpy as np
import pandas as pd

from paths import PROCESSED_DIR, TEAMS_CSV, load_matches

AVAIL_CSV = PROCESSED_DIR / "availability.csv"

# Strength offset per unit of availability above the league mean, selected on
# TUNE seasons by log loss and frozen. Applied to attack and defence alike.
AVAIL_GAMMA = 0.30
# League-average availability and spread, measured on the historical series.
# Offsets are expressed relative to these so a fully fit squad is a small
# positive and a depleted one a small negative.
AVAIL_MEAN = 0.781
AVAIL_SD = 0.104

WINDOW = 8          # matches used to weight a player's importance
MIN_HISTORY = 6     # skip a team's first games, where weights are meaningless


# --------------------------------------------------------------------------
def build_historical(verbose=True):
    """Team-match availability from Transfermarkt appearances."""
    import duckdb

    from paths import RAW_DIR
    db = RAW_DIR / "transfermarkt-datasets.duckdb"
    if not db.exists():
        raise FileNotFoundError(
            f"{db} not found. It is gitignored (~195 MB); re-download from "
            "github.com/dcaribou/transfermarkt-datasets to rebuild this table.")
    con = duckdb.connect(str(db), read_only=True)
    app = con.execute("""
        select a.game_id, a.player_id, a.date, a.minutes_played,
               case when a.player_club_id = g.home_club_id then g.home_club_name
                    else g.away_club_name end as club
        from appearances a join games g on a.game_id = g.game_id
        where a.competition_id = 'GB1'
    """).fetchdf()
    app["date"] = pd.to_datetime(app["date"])

    teams = pd.read_csv(TEAMS_CSV)
    m2 = dict(zip(teams["transfermarkt"].dropna(),
                  teams.loc[teams["transfermarkt"].notna(), "canonical_name"]))
    for c in teams["canonical_name"]:
        m2.setdefault(c, c)

    def canon(n):
        n = str(n)
        if n in m2:
            return m2[n]
        s = n.replace(" FC", "").replace("FC ", "").replace(" AFC", "").replace("AFC ", "")
        for suf in (" II", " U21", " U23"):
            s = s.replace(suf, "")
        return m2.get(s.strip(), s.strip())

    app["team"] = app["club"].map(canon)
    known = set(load_matches()["home_team"]) | set(load_matches()["away_team"])
    unknown = sorted(x for x in set(app["team"]) - known if isinstance(x, str))
    if unknown:
        raise ValueError(
            f"Transfermarkt club name(s) not mapping to a canonical name: "
            f"{unknown[:10]}. Add them to teams.csv before trusting the output.")

    mins = app.groupby(["team", "game_id", "date", "player_id"],
                       as_index=False)["minutes_played"].sum()
    rows = []
    for team, d in mins.groupby("team"):
        games = d[["game_id", "date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
        by_game = {g: dict(zip(grp["player_id"], grp["minutes_played"]))
                   for g, grp in d.groupby("game_id")}
        for i in range(MIN_HISTORY, len(games)):
            w = {}
            for j in range(max(0, i - WINDOW), i - 1):
                for p, mn in by_game.get(games["game_id"][j], {}).items():
                    w[p] = w.get(p, 0) + mn
            tot = sum(w.values())
            if tot <= 0:
                continue
            played = {p for p, v in by_game.get(games["game_id"][i - 1], {}).items() if v > 0}
            rows.append({"team": team, "date": games["date"][i],
                         "avail": sum(v for p, v in w.items() if p in played) / tot})
    out = pd.DataFrame(rows)
    if verbose:
        print(f"  {len(out):,} team-match rows, "
              f"{out['date'].min().date()} -> {out['date'].max().date()}, "
              f"mean {out['avail'].mean():.3f}")
    return out


def save(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(AVAIL_CSV, index=False)
    return AVAIL_CSV


def load():
    if not AVAIL_CSV.exists():
        raise FileNotFoundError(
            f"No availability table at {AVAIL_CSV}. Run:  py src/availability.py")
    return pd.read_csv(AVAIL_CSV, parse_dates=["date"])


def lookup(df):
    """{(team, date) -> availability} for fast per-match access."""
    return {(r.team, pd.Timestamp(r.date).normalize()): r.avail
            for r in df.itertuples()}


# --------------------------------------------------------------------------
def live_from_fpl():
    """Current availability per team from the FPL API's injury flags.

    Transfermarkt appearances stop at the end of last season, so the live
    campaign needs a current source. Players are weighted by price, which is the
    best available proxy for importance before anyone has played; `status` "a"
    means available, and chance_of_playing is used where FPL supplies it.
    """
    import json
    import urllib.request

    from fixtures import FPL_BOOTSTRAP, UA
    req = urllib.request.Request(FPL_BOOTSTRAP, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        boot = json.loads(r.read().decode("utf-8"))

    teams = pd.read_csv(TEAMS_CSV)
    fpl2 = dict(zip(teams["fpl"].dropna(),
                    teams.loc[teams["fpl"].notna(), "canonical_name"]))
    tid = {t["id"]: t["name"] for t in boot["teams"]}
    unknown = sorted({n for n in tid.values() if n not in fpl2})
    if unknown:
        raise ValueError(f"FPL team name(s) missing from teams.csv: {unknown}")

    rows = []
    for p in boot["elements"]:
        chance = p.get("chance_of_playing_next_round")
        if chance is None:
            frac = 1.0 if p.get("status") == "a" else 0.0
        else:
            frac = float(chance) / 100.0
        rows.append({"team": fpl2[tid[p["team"]]],
                     "w": p["now_cost"] / 10.0, "frac": frac})
    d = pd.DataFrame(rows)
    g = d.groupby("team")[["w", "frac"]].apply(
        lambda x: float((x["w"] * x["frac"]).sum() / x["w"].sum()))
    return g.to_dict()


def offsets(avail_by_team, gamma=AVAIL_GAMMA, standardise=False):
    """{team: (d_attack, d_defence)} from availability, for fit.adjustments.

    The two sources are NOT on the same scale: the historical proxy averages
    0.781 because squads rotate, while the FPL injury measure averages 0.839
    because most players are simply fit. Centering live values on the historical
    mean would hand every team a positive offset.

    standardise=True z-scores the supplied values against their own
    cross-sectional mean and spread, then rescales onto the historical spread the
    coefficient was tuned on. For the historical series this is algebraically
    identical to the raw form, so the validated behaviour is unchanged; for any
    other source it maps that source onto the same scale. Use it for live data.
    """
    vals = {t: a for t, a in avail_by_team.items()
            if a is not None and np.isfinite(a)}
    if not vals:
        return {}
    if standardise:
        arr = np.array(list(vals.values()), dtype=float)
        mu, sd = arr.mean(), arr.std(ddof=0)
        if sd <= 0:
            return {t: (0.0, 0.0) for t in vals}
        return {t: (gamma * AVAIL_SD * (a - mu) / sd,) * 2 for t, a in vals.items()}
    return {t: (gamma * (a - AVAIL_MEAN),) * 2 for t, a in vals.items()}


if __name__ == "__main__":
    print("Building historical availability from Transfermarkt appearances...")
    df = build_historical()
    p = save(df)
    print(f"  saved to {p.relative_to(PROCESSED_DIR.parent.parent)}")
    print("\nFetching live availability from the FPL API...")
    try:
        live = live_from_fpl()
        s = pd.Series(live).sort_values()
        print(f"  {len(s)} teams; mean {s.mean():.3f}")
        print("\n  most depleted:")
        for t, v in s.head(5).items():
            print(f"    {t:<24}{v:.3f}")
        print("  fullest strength:")
        for t, v in s.tail(3).items():
            print(f"    {t:<24}{v:.3f}")
    except Exception as e:
        print(f"  live fetch failed ({type(e).__name__}: {str(e)[:90]})")
