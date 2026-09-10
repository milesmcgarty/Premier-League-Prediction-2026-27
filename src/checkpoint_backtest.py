"""Does re-running the forecast mid-season actually make it better?

Every validation figure this project quotes comes from a forecast made on
1 August. But the live product is a WEEKLY re-run, and that has never been
graded. A model can be well calibrated pre-season and add nothing at all from
gameweek 12, because at that point this season is ~10% of the fitting weight
and the update is dominated by points already banked.

So separate the two things a mid-season re-run does, with three arms scored at
the same checkpoints against the same final table:

    frozen   the 1 August forecast, never touched again.
             The "why bother re-running" baseline.

    banked   August's STRENGTH estimates, but the simulation starts from the
             real current table. Points to date are counted; nothing has been
             re-learned about how good anyone is.

    refit    the full re-fit including this season's results so far, which is
             what src/harness.py does every week.

frozen -> banked is the value of simply counting the league table, which needs
no model at all. banked -> refit is the value of RE-ESTIMATING TEAM STRENGTH
from this season, and it is the only one of the two that is a claim about the
model. If that second gap is ~0, the weekly re-fit is theatre and the fix is
the update rule, not more layers.

The market prior and the availability layer are deliberately OFF here. Both are
real and both help live, but the market prior is fitted on a season's OPENING
fixtures, so it would sit identically in all three arms and dilute exactly the
contrast this is built to measure.
"""
import sys
import warnings

import numpy as np
import pandas as pd

import dixon_coles as dc
import simulate as S
import xg as X
from backtest import MIN_HISTORY
from paths import PROCESSED_DIR, load_matches

warnings.filterwarnings("ignore")

N_SIMS = 10000
# Checkpoints as MATCHES PLAYED, not gameweeks: rounds are ragged (midweek
# fixtures, postponements) and matches played is what the model actually sees.
CHECKPOINTS = [50, 100, 150, 200, 250, 300]
RESULTS = PROCESSED_DIR / "checkpoint_backtest.csv"


def checkpoint_cutoffs(season_m, thresholds=CHECKPOINTS):
    """(target, as_of, actual_played) per checkpoint.

    as_of is the day AFTER the threshold match, because simulate_season treats
    as_of as exclusive. Actual played usually overshoots the target slightly,
    since a matchday is indivisible -- report it rather than pretend otherwise.
    """
    d = season_m.sort_values("date").reset_index(drop=True)
    out = []
    for k in thresholds:
        if k >= len(d):
            continue
        as_of = pd.Timestamp(d.loc[k - 1, "date"]) + pd.Timedelta(days=1)
        out.append((k, as_of, int((d["date"] < as_of).sum())))
    return out


def _outcomes(final_tab, teams):
    """Actual title / top4 / relegation indicators, in `teams` order."""
    n = len(teams)
    pos = np.array([int(final_tab.loc[t, "pos"]) for t in teams])
    return {"title": (pos == 1).astype(int),
            "top4": (pos <= 4).astype(int),
            "releg": (pos >= n - 2).astype(int)}, pos


def _probs(sim, n_sims):
    pos = sim["position"]
    n = pos.shape[1]
    return {"title": (pos == 1).mean(axis=0),
            "top4": (pos <= 4).mean(axis=0),
            "releg": (pos >= n - 2).mean(axis=0)}, pos


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _rps(pos_draws, actual_pos, n_sims):
    """Ranked probability score over the full ordered position distribution."""
    n = pos_draws.shape[1]
    out = []
    for i in range(n):
        dist = np.bincount(pos_draws[:, i], minlength=n + 1)[1:] / n_sims
        e = np.zeros(n)
        e[actual_pos[i] - 1] = 1
        out.append(((np.cumsum(dist) - np.cumsum(e)) ** 2).sum() / (n - 1))
    return float(np.mean(out))


def run_season(m, season, league="Prem", n_sims=N_SIMS, seed=7, verbose=True):
    """Score all three arms at every checkpoint of one completed season."""
    season_m = m[(m["season"] == season) & (m["league"] == league)]
    final_tab = S.results_table(season_m).set_index("team")

    # Promoted-team dispersion is a property of the season, not the checkpoint,
    # so tune it once. Identical across arms; it cannot flatter any of them.
    hist = m[m["season"] != season]
    sd_p, up_p = S.tune_promoted_sd(hist, season, league, apply_prior=False)
    kw = dict(n_sims=n_sims, seed=seed, strength_sd_promoted=sd_p,
              promoted_up_ratio=up_p)

    aug_fit = dc.fit_for_league(m, season, league)
    aug_sim = S.simulate_season(m, season, league, as_of=None, fit=aug_fit, **kw)
    teams = aug_sim["teams"]
    y, actual_pos = _outcomes(final_tab, teams)
    champ_i = int(np.argmax(y["title"]))
    rel_i = np.flatnonzero(y["releg"])
    frozen_p, frozen_pos = _probs(aug_sim, n_sims)

    rows = []
    for target, as_of, played in checkpoint_cutoffs(season_m):
        done = season_m[season_m["date"] < as_of]

        banked = S.simulate_season(m, season, league, as_of=as_of,
                                   fit=aug_fit, **kw)
        re_fit = dc.fit_for_league(m, season, league, extra=done, cutoff=as_of)
        refit = S.simulate_season(m, season, league, as_of=as_of,
                                  fit=re_fit, **kw)

        for arm, (p, pd_) in (("banked", _probs(banked, n_sims)),
                              ("refit", _probs(refit, n_sims)),
                              ("frozen", (frozen_p, frozen_pos))):
            rows.append({"season": season, "target": target, "played": played,
                         "arm": arm,
                         **{f"ll_{k}": _ll(p[k], y[k]) for k in p},
                         "rps": _rps(pd_, actual_pos, n_sims),
                         "p_champ": float(p["title"][champ_i]),
                         "p_rel_true": float(p["releg"][rel_i].mean())})

    R = pd.DataFrame(rows)
    if verbose:
        champ = final_tab.index[final_tab["pos"] == 1][0]
        print(f"\n{season}  (champions: {champ};  promoted sd {sd_p})")
        piv = R.pivot_table(index="target", columns="arm", values="rps")
        piv = piv[["frozen", "banked", "refit"]]
        print("   position RPS by checkpoint")
        print(piv.round(4).to_string())
    return R


def _summarise(R, verbose=True):
    metrics = ["ll_title", "ll_top4", "ll_releg", "rps"]
    piv = R.pivot_table(index="target", columns="arm", values=metrics)
    if verbose:
        print("\n" + "=" * 78)
        print("CHECKPOINT BACKTEST  --  mean over "
              f"{R['season'].nunique()} held-out seasons")
        print("=" * 78)
        for mt, label in [("rps", "position RPS"), ("ll_title", "title log loss"),
                          ("ll_top4", "top-four log loss"),
                          ("ll_releg", "relegation log loss")]:
            sub = piv[mt][["frozen", "banked", "refit"]]
            sub = sub.assign(**{"bank_gain": sub["frozen"] - sub["banked"],
                                "refit_gain": sub["banked"] - sub["refit"]})
            print(f"\n{label}  (lower is better; gains are positive = better)")
            print(sub.round(4).to_string())

        print("\n" + "-" * 78)
        print("THE QUESTION: does re-estimating strength beat simply counting")
        print("the table? That is the 'refit_gain' column above.")
        w = R.pivot_table(index=["season", "target"], columns="arm", values="rps")
        d = (w["banked"] - w["refit"]).dropna()
        from scipy import stats
        t = stats.ttest_1samp(d, 0.0)
        print(f"\n  refit vs banked, position RPS, n={len(d)} season-checkpoints")
        print(f"    mean gain {d.mean():+.5f}   better in "
              f"{(d > 0).sum()}/{len(d)}   t={t.statistic:+.2f}  p={t.pvalue:.4f}")

        # Where in the season does re-fitting earn its keep? A single
        # average can hide a sign flip, and one season suggested exactly
        # that: refit ahead early, behind after ~250 matches.
        print("\n  by checkpoint:")
        print(f"    {'target':>8}{'gain':>10}{'better':>9}{'t':>8}{'p':>8}")
        for tgt, g in d.groupby(level="target"):
            tt = stats.ttest_1samp(g, 0.0)
            print(f"    {tgt:>8}{g.mean():>+10.5f}"
                  f"{f'{(g > 0).sum()}/{len(g)}':>9}"
                  f"{tt.statistic:>+8.2f}{tt.pvalue:>8.3f}")
    return piv


def boost_season(m, season, boosts, league="Prem", n_sims=N_SIMS, seed=7):
    """Score a range of current-season weight boosts at every checkpoint.

    boost=1.0 is the shipped weekly re-fit, and reproduces it exactly, so the
    A/B is built into the parameterisation rather than bolted alongside it.
    """
    season_m = m[(m["season"] == season) & (m["league"] == league)]
    final_tab = S.results_table(season_m).set_index("team")
    hist = m[m["season"] != season]
    sd_p, up_p = S.tune_promoted_sd(hist, season, league, apply_prior=False)
    kw = dict(n_sims=n_sims, seed=seed, strength_sd_promoted=sd_p,
              promoted_up_ratio=up_p)

    aug_fit = dc.fit_for_league(m, season, league)
    teams = sorted(set(season_m["home_team"]) | set(season_m["away_team"]))
    y, actual_pos = _outcomes(final_tab, teams)

    rows = []
    for target, as_of, played in checkpoint_cutoffs(season_m):
        done = season_m[season_m["date"] < as_of]
        for b in boosts:
            fit = dc.fit_for_league(m, season, league, extra=done, cutoff=as_of,
                                    extra_boost=b)
            sim = S.simulate_season(m, season, league, as_of=as_of, fit=fit, **kw)
            assert sim["teams"] == teams, "team order drifted between arms"
            p, pd_ = _probs(sim, n_sims)
            rows.append({"season": season, "target": target, "played": played,
                         "boost": b,
                         **{f"ll_{k}": _ll(p[k], y[k]) for k in p},
                         "rps": _rps(pd_, actual_pos, n_sims)})
    return pd.DataFrame(rows)


def tune_boost(boosts=(1.0, 1.5, 2.0, 3.0, 5.0), n_sims=8000, verbose=True):
    """Select the boost on TUNE seasons, then report it on REPORT seasons.

    Choosing it on the seasons it is then quoted against is the same leakage as
    tuning a half-life on the test set, and is easy to do by accident here
    because both halves come out of one loop.
    """
    m = X.attach_xg(load_matches().dropna(subset=["home_goals", "away_goals"]),
                    X.load_xg())
    order = sorted(m["season"].unique(), key=lambda s: int(s[:2]))
    testable = order[MIN_HISTORY:]
    sp = len(testable) // 2
    tune, report = testable[:sp], testable[sp:]

    if verbose:
        print(f"TUNE   {tune[0]}-{tune[-1]}  ({len(tune)} seasons)")
        print(f"REPORT {report[0]}-{report[-1]}  ({len(report)} seasons)")

    T = pd.concat([boost_season(m, s, boosts, n_sims=n_sims) for s in tune],
                  ignore_index=True)
    T.to_csv(PROCESSED_DIR / "boost_tune.csv", index=False)
    curve = T.groupby("boost")["rps"].mean()
    best = float(curve.idxmin())
    if verbose:
        print("\nTUNE: mean position RPS by boost")
        for b, v in curve.items():
            print(f"  boost {b:>5.2f}   {v:.5f}"
                  f"{'   <- selected' if b == best else ''}")
        print(f"\n>>> SELECTED extra_boost = {best}  (frozen)\n")

    R = pd.concat([boost_season(m, s, sorted({1.0, best}), n_sims=n_sims)
                   for s in report], ignore_index=True)
    R.to_csv(PROCESSED_DIR / "boost_report.csv", index=False)
    if verbose and best != 1.0:
        print("REPORT (held out): boost", best, "against the shipped boost 1.0")
        piv = R.pivot_table(index="target", columns="boost",
                            values=["rps", "ll_title", "ll_releg"])
        for mt in ("rps", "ll_releg", "ll_title"):
            sub = piv[mt].copy()
            sub["gain"] = sub[1.0] - sub[best]
            print(f"\n  {mt}")
            print(sub.round(5).to_string())
        w = R.pivot_table(index=["season", "target"], columns="boost", values="rps")
        d = (w[1.0] - w[best]).dropna()
        from scipy import stats
        t = stats.ttest_1samp(d, 0.0)
        print(f"\n  overall RPS gain {d.mean():+.5f}  better in "
              f"{(d > 0).sum()}/{len(d)}  t={t.statistic:+.2f}  p={t.pvalue:.4f}")
    elif verbose:
        print("REPORT: TUNE selected boost 1.0 -- the shipped update rule is "
              "already the best of those tested. Nothing to change.")
    return T, R


def run(seasons=None, n_sims=N_SIMS, verbose=True):
    m = X.attach_xg(load_matches().dropna(subset=["home_goals", "away_goals"]),
                    X.load_xg())
    order = sorted(m["season"].unique(), key=lambda s: int(s[:2]))
    testable = order[MIN_HISTORY:]
    seasons = seasons or testable[len(testable) // 2:]

    out = [run_season(m, s, n_sims=n_sims, verbose=verbose) for s in seasons]
    R = pd.concat(out, ignore_index=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS, index=False)
    if verbose:
        print(f"\nraw results -> {RESULTS}")
    _summarise(R, verbose=verbose)
    return R


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "boost":
        tune_boost()
    else:
        run(seasons=args or None)
