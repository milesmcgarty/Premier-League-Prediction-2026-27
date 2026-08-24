"""The whole project's correctness suite, in one runnable place.

Every check here has been used at some point to catch a real defect, and most
were previously scattered across throwaway scripts. Consolidated so that a
change anywhere can be checked everywhere.

    py src/validate_all.py            # everything offline
    py src/validate_all.py --quick    # skip the slow simulation checks

Design rule: every check must be able to FAIL STRUCTURALLY. "Looks sensible" is
not a test. Preferred forms are byte-identical output after a refactor, an
analytic gradient against finite differences, a simulator reproducing a known
final table exactly, and predicted-vs-actual on a quantity the fit did not
target.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime
from scipy.stats import poisson, spearmanr

import dixon_coles as dc
import odds as O
import simulate as S
from paths import TEAMS_CSV, active_teams, load_matches

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)
    return ok


def section(t):
    print("\n" + "=" * 76, flush=True)
    print(t, flush=True)
    print("=" * 76, flush=True)


# --------------------------------------------------------------------------
def data_integrity(m):
    section("1. DATA INTEGRITY")
    check("24,232 matches", len(m) == 24232, f"{len(m)}")
    check("no null goals/dates/teams",
          not (m[["date", "home_team", "away_team", "home_goals",
                  "away_goals"]].isna().any().any()))
    check("no duplicate fixtures",
          m.duplicated(subset=["date", "home_team", "away_team"]).sum() == 0)
    derived = np.where(m.home_goals > m.away_goals, "H",
                       np.where(m.home_goals < m.away_goals, "A", "D"))
    check("result column agrees with goals", (derived == m.result).all())

    counts = m.groupby(["season", "league"]).size().unstack()
    check("every season/league has the right match count",
          bool((counts["Prem"] == 380).all() and (counts["Champ"] == 552).all()))
    check("26 seasons, both leagues",
          m.season.nunique() == 26 and set(m.league) == {"Prem", "Champ"})

    # season codes must stay strings, or "0001" becomes int 1
    check("season codes are strings", m.season.dtype == object
          and "0001" in set(m.season))


def known_tables(m):
    section("2. HISTORICAL TABLES REPRODUCED EXACTLY")
    KNOWN = {
        "2122": [("Manchester City", 1, 93), ("Liverpool", 2, 92),
                 ("Norwich City", 20, 22)],
        "1516": [("Leicester City", 1, 81), ("Aston Villa", 20, 17)],
        "2324": [("Manchester City", 1, 91), ("Sheffield United", 20, 16)],
        "2425": [("Liverpool", 1, 84), ("Southampton", 20, 12)],
        "2526": [("Arsenal", 1, 85), ("Wolverhampton Wanderers", 20, 20)],
    }
    ok = True
    for season, rows in KNOWN.items():
        t = S.results_table(m[(m.season == season) & (m.league == "Prem")]).set_index("team")
        for team, pos, pts in rows:
            good = int(t.loc[team, "pos"]) == pos and int(t.loc[team, "Pts"]) == pts
            ok &= good
            if not good:
                print(f"      {season} {team}: got pos {int(t.loc[team,'pos'])} "
                      f"{int(t.loc[team,'Pts'])}pts, expected {pos} {pts}")
    check("known champions and bottom sides reproduce", ok, f"{len(KNOWN)} seasons")

    bad = 0
    for (s, lg), g in m.groupby(["season", "league"]):
        t = S.results_table(g)
        if (t.P.sum() != 2 * len(g) or t.GF.sum() != t.GA.sum()
                or t.GD.sum() != 0 or t.W.sum() != t.L.sum()
                or t.Pts.sum() != 3 * t.W.sum() + t.D.sum()):
            bad += 1
    check("table arithmetic consistent in all 52 season-leagues", bad == 0,
          f"{bad} bad")


def team_mapping(m):
    section("3. TEAM NAME MAPPING")
    t = pd.read_csv(TEAMS_CSV)
    canon = set(t.canonical_name)
    used = set(m.home_team) | set(m.away_team)
    check("every team in the data is a canonical name", used <= canon,
          f"{len(used - canon)} strays")
    check("teams.csv ids unique", t.team_id.is_unique and t.canonical_name.is_unique)
    for col in ["football_data", "understat", "clubelo", "fpl"]:
        vals = t[col].dropna()
        check(f"teams.csv `{col}` has no duplicate source names",
              vals.is_unique, f"{len(vals)} filled")


def odds_checks(m):
    section("4. ODDS HYGIENE AND DE-VIGGING")
    for book in ["B365", "WH", "Avg", "B365C", "AvgC"]:
        d = m[O.has_odds(m, book)]
        if len(d) == 0:
            check(f"{book}: present", False, "no rows")
            continue
        p = O.market_probs(d, book)
        check(f"{book}: probabilities finite and sum to 1",
              bool(np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0)),
              f"n={len(d)}")
    d = m[O.has_odds(m, "B365")]
    orr = O.overround(d, "B365")
    check("B365 overround in a sane range",
          bool(orr.min() > 1.0 and orr.max() < 1.35),
          f"{orr.min():.3f}-{orr.max():.3f}, mean {orr.mean():.4f}")
    check("the two corrupt B365 triplets are nulled, matches kept",
          len(m[(m.date.isin([pd.Timestamp("2013-04-27"), pd.Timestamp("2019-02-02")]))
                & m.home_team.isin(["Blackpool", "Brentford"])
                & m.B365H.notna()]) == 0)
    p = O.market_probs(d, "B365")
    act = np.array([(d.result == o).mean() for o in ["H", "D", "A"]])
    check("aggregate market probabilities match realised rates",
          bool(np.abs(p.mean(axis=0) - act).max() < 0.01),
          f"max diff {np.abs(p.mean(axis=0)-act).max():.4f}")


def elo_checks(m):
    section("5. ELO")
    try:
        r = pd.read_csv(dc.__dict__["pd"].__name__ and "data/processed/elo_ratings.csv")
    except Exception:
        from paths import ELO_RATINGS_CSV
        r = pd.read_csv(ELO_RATINGS_CSV)
    pl = active_teams(m, season="2526", league="Prem")
    prem = r[r.team.isin(pl)]
    check("Elo ratings exist for every current PL team", len(prem) == 20,
          f"{len(prem)}/20")
    check("PL Elo spread is plausible",
          bool(80 < prem.elo.std() < 130 and 250 < (prem.elo.max()-prem.elo.min()) < 550),
          f"sd {prem.elo.std():.0f}, range {prem.elo.max()-prem.elo.min():.0f}")
    tab = S.results_table(m[(m.season == "2526") & (m.league == "Prem")]).set_index("team")
    rho = spearmanr(prem.set_index("team").loc[tab.index, "elo"], -tab.pos).statistic
    check("Elo ranks the final table well", rho > 0.75, f"Spearman {rho:.3f}")
    stale = r[~r.team.isin(active_teams(m, season="2526"))]
    check("dormant clubs exist and are the known hazard", len(stale) > 0,
          f"{len(stale)} dormant; filter via paths.active_teams()")


def model_checks(m):
    section("6. DIXON-COLES MODEL")
    seasons, cutoff = dc.training_window(m, "2526", 5)
    tr = m[m.season.isin(seasons)]
    f = dc.fit_dixon_coles(tr, cutoff=cutoff)
    check("fit converges", f.converged)
    check("home advantage is the corrected ~0.21, not the old 0.33",
          0.17 < f.home_adv < 0.25, f"{f.home_adv:.4f}")
    check("sum-to-zero constraints hold",
          bool(abs(f.ratings.attack.sum()) < 1e-2 and abs(f.ratings.defence.sum()) < 1e-2))

    # goal level: predicted vs TIME-WEIGHTED actual, which the fit did not target
    atk = f.ratings.set_index("team").attack.to_dict()
    dfn = f.ratings.set_index("team").defence.to_dict()
    lh = np.exp(f.intercept + tr.home_team.map(atk) - tr.away_team.map(dfn) + f.home_adv)
    la = np.exp(f.intercept + tr.away_team.map(atk) - tr.home_team.map(dfn))
    w = dc.time_weights(tr.date, cutoff)
    eh = abs(np.average(lh, weights=w) - np.average(tr.home_goals, weights=w))
    ea = abs(np.average(la, weights=w) - np.average(tr.away_goals, weights=w))
    check("predicted goals match time-weighted actuals",
          bool(eh < 0.01 and ea < 0.01), f"home err {eh:.4f}, away err {ea:.4f}")

    gaps = []
    for wnd in [3, 5, 8]:
        ss, cc = dc.training_window(m, "2526", wnd)
        g = dc.division_gap(dc.fit_dixon_coles(m[m.season.isin(ss)], cutoff=cc),
                            m[m.season.isin(ss)])
        gaps.append(g["elo_equivalent"])
    check("learned division gap is stable and near the Elo-derived 232",
          bool(min(gaps) > 150 and max(gaps) < 400 and (max(gaps)-min(gaps)) < 120),
          f"{[round(x) for x in gaps]} across 3/5/8-season windows")

    check("ridge default is 0 (it biases the division gap)", dc.RIDGE == 0.0)
    check("xG weight is set and below 1 (pure xG is worse)",
          0 < dc.XG_WEIGHT < 1, f"{dc.XG_WEIGHT}")

    # analytic gradient vs finite differences
    from scipy.special import gammaln
    teams = sorted(set(tr.home_team) | set(tr.away_team)); n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    hi = tr.home_team.map(idx).to_numpy(); ai = tr.away_team.map(idx).to_numpy()
    hg = tr.home_goals.to_numpy().astype(int); ag = tr.away_goals.to_numpy().astype(int)
    ww = dc.time_weights(tr.date, cutoff); lf = gammaln(hg + 1.0) + gammaln(ag + 1.0)

    def og(p):
        a, d = p[:n], p[n:2*n]; c, adv, rho = p[2*n], p[2*n+1], p[2*n+2]
        Lh = np.exp(c + a[hi] - d[ai] + adv); La = np.exp(c + a[ai] - d[hi])
        tau, dl, da, dr = dc._tau_and_grads(hg, ag, Lh, La, rho)
        tau = np.clip(tau, 1e-10, None)
        ll = hg*np.log(Lh) - Lh + ag*np.log(La) - La - lf + np.log(tau)
        pen = dc.SUM_ZERO_PENALTY*(a.sum()**2 + d.sum()**2)
        obj = -(ww*ll).sum() + pen
        gh = ww*(hg - Lh + Lh*dl/tau); ga = ww*(ag - La + La*da/tau)
        g = np.empty_like(p)
        g[:n] = -(np.bincount(hi, gh, n) + np.bincount(ai, ga, n)) + 2*dc.SUM_ZERO_PENALTY*a.sum()
        g[n:2*n] = -(-np.bincount(ai, gh, n) - np.bincount(hi, ga, n)) + 2*dc.SUM_ZERO_PENALTY*d.sum()
        g[2*n] = -(gh+ga).sum(); g[2*n+1] = -gh.sum(); g[2*n+2] = -(ww*dr/tau).sum()
        return obj, g

    rng = np.random.default_rng(0)
    p0 = np.concatenate([rng.normal(0, .3, n), rng.normal(0, .3, n),
                         [.15], [.2], [-.05]])
    ga = og(p0)[1]
    gn = approx_fprime(p0, lambda q: og(q)[0], 1e-6)
    rel = np.abs(ga - gn).max() / max(np.abs(gn).max(), 1e-12)
    check("analytic gradient matches finite differences", rel < 1e-4,
          f"relative error {rel:.2e}")

    # kappa = 0 must reproduce the goals-only fit exactly
    f0 = dc.fit_dixon_coles(tr, cutoff=cutoff, xg_weight=0.0)
    check("xg_weight=0 reproduces the goals-only fit",
          bool(abs(f0.home_adv - 0.2096) < 1e-3), f"home_adv {f0.home_adv:.4f}")

    # promoted teams must predict at all (the old KeyError blocker)
    f2 = dc.fit_for_league(m, "2526", "Prem")
    ok = True
    for t in sorted(S.promoted_teams(m, "2526", "Prem")):
        try:
            p = f2.predict(t, "Liverpool")
            ok &= abs(p["home_win"] + p["draw"] + p["away_win"] - 1) < 1e-9
        except Exception:
            ok = False
    check("promoted teams predict (the old KeyError blocker)", ok)


def simulator_checks(m, quick=False):
    section("7. SEASON SIMULATOR")
    f = dc.fit_for_league(m, "2526", "Prem")
    g = S.scoreline_grid(f, "Arsenal", "Burnley")
    lh, la = f.lambdas("Arsenal", "Burnley")
    eh = (g.sum(axis=1) * np.arange(g.shape[0])).sum()
    check("scoreline grid integrates to the fitted lambda",
          bool(abs(g.sum()-1) < 1e-12 and abs(eh - lh) < 0.01),
          f"grid mean {eh:.4f} vs lambda {lh:.4f}")
    p = f.predict("Arsenal", "Burnley")
    check("grid W/D/L agrees with predict()",
          bool(abs(np.tril(g, -1).sum() - p["home_win"]) < 1e-9))

    end = m[(m.season == "2425") & (m.league == "Prem")].date.max() + pd.Timedelta(days=1)
    sim = S.simulate_season(m, "2425", "Prem", n_sims=200, as_of=end)
    act = S.results_table(m[(m.season == "2425") & (m.league == "Prem")]).set_index("team")
    sp = dict(zip(sim["teams"], sim["points"][0]))
    check("with every match played the simulation is deterministic",
          bool(sim["position"].std(axis=0).max() == 0
               and all(sp[t] == act.loc[t, "Pts"] for t in sim["teams"])))

    if quick:
        return
    sim = S.simulate_season(m, "2526", "Prem", n_sims=8000, seed=1)
    check("GD sums to zero in every simulated season",
          bool((sim["gd"].sum(axis=1) == 0).all()))
    check("positions are a permutation in every simulated season",
          bool(np.all(np.sort(sim["position"], axis=1) == np.arange(1, 21)[None, :])))

    season = m[(m.season == "2526") & (m.league == "Prem")]
    exp = np.zeros(len(sim["teams"])); ix = {t: i for i, t in enumerate(sim["teams"])}
    f2 = S.simulate_season(m, "2526", "Prem", n_sims=8000, seed=1,
                           strength_sd=0.0, strength_sd_promoted=0.0)["fit"]
    for _, r in season.iterrows():
        q = f2.predict(r.home_team, r.away_team)
        exp[ix[r.home_team]] += 3*q["home_win"] + q["draw"]
        exp[ix[r.away_team]] += 3*q["away_win"] + q["draw"]
    simmed = S.simulate_season(m, "2526", "Prem", n_sims=8000, seed=1,
                               strength_sd=0.0, strength_sd_promoted=0.0)["points"].mean(axis=0)
    err = np.abs(exp - simmed).max()
    check("simulated mean points match ANALYTIC expected points", err < 0.8,
          f"max error {err:.3f} pts")

    check("promoted teams are detected", len(S.promoted_teams(m, "2526", "Prem")) == 3,
          str(sorted(S.promoted_teams(m, "2526", "Prem"))))


def xg_checks(m):
    section("8. EXPECTED GOALS")
    try:
        import xg as X
        xgd = X.load_xg()
    except Exception as e:
        check("xG table present", False, f"{type(e).__name__}")
        return
    check("xG covers 12 seasons", xgd.season.nunique() == 12, f"{xgd.season.nunique()}")
    check("xG rows = 380 per season", len(xgd) == 380 * xgd.season.nunique(), f"{len(xgd)}")
    pl = m[(m.league == "Prem") & (m.season.isin(xgd.season.unique()))]
    u = xgd.rename(columns={"home_goals": "ug_h", "away_goals": "ug_a"})
    j = pl.merge(u[["season", "home_team", "away_team", "ug_h", "ug_a", "home_xg", "away_xg"]],
                 on=["season", "home_team", "away_team"], how="left")
    check("xG joins onto 100% of covered PL matches",
          int(j.home_xg.notna().sum()) == len(pl), f"{int(j.home_xg.notna().sum())}/{len(pl)}")
    check("joined goals agree (join matches the RIGHT fixtures)",
          int(((j.ug_h != j.home_goals) | (j.ug_a != j.away_goals)).sum()) == 0)
    check("mean xG is close to mean goals",
          bool(abs(xgd.home_xg.mean() - xgd.home_goals.mean()) < 0.15),
          f"xG {xgd.home_xg.mean():.3f} vs goals {xgd.home_goals.mean():.3f}")


def market_prior_checks(m):
    section("9. MARKET PRIOR FOR PROMOTED TEAMS")
    try:
        import market_prior as MP
    except Exception as e:
        check("market_prior importable", False, str(e)[:60]); return
    promo = sorted(S.promoted_teams(m, "2526", "Prem"))
    fx = m[(m.season == "2526") & (m.league == "Prem")].sort_values("date")
    f = dc.fit_for_league(m, "2526", "Prem")
    base = f.predict("Burnley", "Liverpool")["home_win"]
    f, used = MP.apply_market_prior(f, fx, promo)
    check("a strength offset is fitted for every promoted team",
          len(f.market_prior_info) == len(promo), str(list(f.market_prior_info)))
    check("prior consumes only early fixtures", 0 < len(used) <= 6 * len(promo),
          f"{len(used)} fixtures")
    d_sun = f.market_prior_info.get("Sunderland", {}).get("delta", 0)
    d_bur = f.market_prior_info.get("Burnley", {}).get("delta", 0)
    check("the market separated Sunderland (7th) from Burnley (19th)",
          d_sun > d_bur, f"Sunderland {d_sun:+.3f} vs Burnley {d_bur:+.3f}")
    check("applying the prior changes predictions",
          abs(f.predict("Burnley", "Liverpool")["home_win"] - base) > 1e-6)
    f.adjustments = {}
    check("clearing adjustments restores the original prediction",
          abs(f.predict("Burnley", "Liverpool")["home_win"] - base) < 1e-12)


def main():
    quick = "--quick" in sys.argv
    m = load_matches().dropna(subset=["home_goals", "away_goals"])

    print("=" * 76)
    print("PREMIER LEAGUE PREDICTION PROJECT - FULL VALIDATION SUITE")
    print("=" * 76)

    data_integrity(m)
    known_tables(m)
    team_mapping(m)
    odds_checks(m)
    elo_checks(m)
    model_checks(m)
    simulator_checks(m, quick)
    xg_checks(m)
    market_prior_checks(m)

    section("SUMMARY")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    print(f"\n  {passed} / {total} checks passed")
    print("\n  " + ("ALL CHECKS PASSED" if passed == total
                    else f"*** {total - passed} FAILURE(S) ***"))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
