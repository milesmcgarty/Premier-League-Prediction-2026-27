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
    bookmaker           NOT AVAILABLE -- no historical August outright prices
                        exist in this repository, and that remains the single
                        most important missing comparison. Until it is run, the
                        season product has never been measured against the only
                        benchmark that really counts.
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
        print("\nNO BOOKMAKER COMPARISON: no historical August outright prices are")
        print("held in this repository, so the season product has never been")
        print("measured against a market. That remains the biggest open gap.")
        se = lambda p_: np.sqrt(p_ * (1 - p_) / n_sims)
        print(f"\nMonte Carlo error at n={n_sims:,}: "
              f"p=0.02 +/-{1.96*se(0.02)*100:.2f}pp, "
              f"p=0.50 +/-{1.96*se(0.50)*100:.2f}pp")
    return out, R


if __name__ == "__main__":
    run()
