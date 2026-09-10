"""Validation for the SEASON-LEVEL product, which is a different thing entirely
from the match model.

Every headline number this project quoted for a long time was match log loss.
That says almost nothing about whether title, top-four and relegation
probabilities are any good: they are a different product, aggregated over 380
correlated fixtures, and they fail in different ways. A model can post a
respectable per-match log loss while producing season probabilities that are
badly overconfident, because the errors compound in the same direction all year.

For each held-out season this runs the pipeline as of 1 August using only prior
data, produces the three published markets plus the full position distribution,
and scores them against what actually happened.

BENCHMARKS, in ascending difficulty:
    uniform             every club equally likely
    base rate           historical frequency (1/20, 4/20, 3/20)
    last season's table the previous finishing order, mapped to probabilities
    bookmaker           runs automatically for any COMPLETED season that has an
                        archived market view under data/outrights/<season>/.
                        None exist yet for a completed season, because archiving
                        only began in September 2026. Every weekly harness run
                        now freezes that week's prices, so this arm switches on
                        by itself once 2026-27 finishes. Until then the season
                        product has never been measured against the only
                        benchmark that really counts, and market_status() says
                        so explicitly rather than letting it be forgotten.
"""
import sys
import warnings

import numpy as np
import pandas as pd

import dixon_coles as dc
import simulate as S
import xg as X
from backtest import MIN_HISTORY
from paths import load_matches

warnings.filterwarnings("ignore")

N_SIMS = 20000
BASE_RATE = {"title": 1 / 20, "top4": 4 / 20, "releg": 3 / 20}


def market_status(verbose=True):
    """How close are we to being able to score against a bookmaker?"""
    import outrights as OR
    m = load_matches().dropna(subset=["home_goals", "away_goals"])
    complete = set()
    for s_, g in m[m["league"] == "Prem"].groupby("season"):
        if len(g) >= 380:
            complete.add(s_)
    rows = []
    for d in sorted((OR.ARCHIVE_DIR).glob("*")) if OR.ARCHIVE_DIR.exists() else []:
        if not d.is_dir():
            continue
        caps = OR.list_archive(d.name)
        rows.append({"season": d.name, "captures": len(caps),
                     "earliest": caps[0].stem if caps else None,
                     "season_complete": d.name in complete})
    st = pd.DataFrame(rows)
    usable = st[st["season_complete"]] if len(st) else st
    if verbose:
        print("\nMARKET COMPARISON STATUS")
        if len(st) == 0:
            print("  no archived market views at all")
        else:
            print(st.to_string(index=False))
        print(f"  seasons scoreable against a bookmaker: {len(usable)}")
        if len(usable) == 0:
            print("  -> archiving began Sept 2026; the first scoreable season is")
            print("     2026-27, once it finishes. Nothing to do but keep running")
            print("     the weekly harness, which freezes prices automatically.")
    return st


def score_against_market(season, model_probs, verbose=True):
    """Score model against the archived market view for one completed season."""
    import outrights as OR
    got, when = OR.earliest_capture(season)
    if not got:
        return None
    m = load_matches().dropna(subset=["home_goals", "away_goals"])
    played = m[(m["season"] == season) & (m["league"] == "Prem")]
    if len(played) < 380:
        return None
    tab = S.results_table(played).set_index("team")
    n = len(tab)
    out = {"season": season, "captured": when}
    for mkt, test in [("title", lambda p: p == 1), ("top4", lambda p: p <= 4),
                      ("releg", lambda p: p >= n - 2)]:
        if mkt not in got or mkt not in model_probs:
            continue
        teams = [t for t in got[mkt] if t in tab.index and t in model_probs[mkt]]
        y = np.array([int(test(int(tab.loc[t, "pos"]))) for t in teams])
        pm = np.array([model_probs[mkt][t] for t in teams])
        pk = np.array([got[mkt][t] for t in teams])
        out[mkt] = {"model_ll": _scores(pm, y)[0], "market_ll": _scores(pk, y)[0],
                    "n": len(teams)}
    if verbose:
        print(f"\n{season} (market as of {when})")
        for mkt in ("title", "top4", "releg"):
            if mkt in out:
                d = out[mkt]
                print(f"    {mkt:<8} model {d['model_ll']:.4f}  "
                      f"market {d['market_ll']:.4f}  "
                      f"diff {d['model_ll']-d['market_ll']:+.4f}")
    return out


def _scores(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return ll, float(((p - y) ** 2).mean())


def run(seasons=None, n_sims=N_SIMS, seed=5, verbose=True):
    m = X.attach_xg(load_matches().dropna(subset=["home_goals", "away_goals"]),
                    X.load_xg())
    order = sorted(m["season"].unique(), key=lambda s: int(s[:2]))
    testable = order[MIN_HISTORY:]
    seasons = seasons or testable[len(testable) // 2:]

    rows, rps_model, rps_unif = [], [], []
    for s in seasons:
        prev = order[order.index(s) - 1]
        fit = dc.fit_for_league(m, s, "Prem")
        sim = S.simulate_season(m, s, "Prem", n_sims=n_sims, seed=seed, fit=fit)
        teams, pos, n = sim["teams"], sim["position"], len(sim["teams"])
        tab = S.results_table(
            m[(m["season"] == s) & (m["league"] == "Prem")]).set_index("team")
        prev_tab = S.results_table(m[(m["season"] == prev) & (m["league"] == "Prem")])
        prev_rank = {t: i + 1 for i, t in enumerate(prev_tab["team"])}

        for i, t in enumerate(teams):
            actual = int(tab.loc[t, "pos"])
            rows.append({
                "season": s, "team": t, "actual_pos": actual,
                "p_title": (pos[:, i] == 1).mean(),
                "p_top4": (pos[:, i] <= 4).mean(),
                "p_releg": (pos[:, i] >= n - 2).mean(),
                "y_title": int(actual == 1), "y_top4": int(actual <= 4),
                "y_releg": int(actual >= n - 2),
                "prev_rank": prev_rank.get(t),
            })
            # ordered position RPS, the product actually published
            dist = np.bincount(pos[:, i], minlength=n + 1)[1:] / n_sims
            e = np.zeros(n); e[actual - 1] = 1
            rps_model.append((((np.cumsum(dist) - np.cumsum(e)) ** 2).sum()) / (n - 1))
            u = np.cumsum(np.full(n, 1 / n))
            rps_unif.append((((u - np.cumsum(e)) ** 2).sum()) / (n - 1))

    R = pd.DataFrame(rows)
    out = {"n_rows": len(R), "n_seasons": R["season"].nunique(),
           "rps_model": float(np.mean(rps_model)),
           "rps_uniform": float(np.mean(rps_unif))}
    out["rps_skill"] = 1 - out["rps_model"] / out["rps_uniform"]

    for mkt in ("title", "top4", "releg"):
        p, y = R[f"p_{mkt}"].to_numpy(), R[f"y_{mkt}"].to_numpy()
        ll_m, br_m = _scores(p, y)
        ll_b, br_b = _scores(np.full_like(p, BASE_RATE[mkt]), y)
        if mkt == "releg":
            py = np.where(R["prev_rank"] >= 18, 0.55, 0.08)
        elif mkt == "top4":
            py = np.where(R["prev_rank"] <= 4, 0.65, 0.09)
        else:
            py = np.where(R["prev_rank"] == 1, 0.40, 0.032)
        ll_y, _ = _scores(py, y)
        out[mkt] = {"model_ll": ll_m, "base_ll": ll_b, "lastyear_ll": ll_y,
                    "model_brier": br_m, "base_brier": br_b,
                    "skill_vs_base": 1 - ll_m / ll_b}

    if verbose:
        print("=" * 74)
        print(f"SEASON-LEVEL VALIDATION  ({out['n_seasons']} held-out seasons, "
              f"{out['n_rows']} club-seasons)")
        print("=" * 74)
        print(f"{'market':<10}{'model LL':>10}{'base LL':>10}{'last yr':>10}"
              f"{'skill':>9}{'model Brier':>13}")
        print("-" * 62)
        for mkt in ("title", "top4", "releg"):
            d = out[mkt]
            print(f"{mkt:<10}{d['model_ll']:>10.4f}{d['base_ll']:>10.4f}"
                  f"{d['lastyear_ll']:>10.4f}{d['skill_vs_base']:>8.0%}"
                  f"{d['model_brier']:>13.4f}")
        print(f"\nposition RPS  model {out['rps_model']:.4f}  "
              f"uniform {out['rps_uniform']:.4f}  "
              f"skill {out['rps_skill']:.1%}")
        market_status()
        se = lambda p_: np.sqrt(p_ * (1 - p_) / n_sims)
        print(f"\nMonte Carlo error at n={n_sims:,}: "
              f"p=0.02 +/-{1.96*se(0.02)*100:.2f}pp, "
              f"p=0.50 +/-{1.96*se(0.50)*100:.2f}pp")
    return out, R


if __name__ == "__main__":
    run()
