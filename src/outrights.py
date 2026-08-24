"""Season-long (outright) market odds as a strength prior.

WHY THIS BEATS MATCH ODDS IN AUGUST. The match-odds prior reads a team's strength
from the fixtures a bookmaker has priced. In mid-August that is ONE fixture per
team, which gives 10 constraints for 20 unknowns: the two sides of a match get
exactly mirrored offsets and the disagreement cannot be attributed to either of
them. Outright markets -- title, top four, relegation -- price every team
SEPARATELY, so 20 teams give 20 independent constraints. That is the difference
between "someone in Newcastle vs Liverpool is mispriced" and "Newcastle are
mispriced".

METHOD. Rather than invert the simulator analytically (title probability is a
badly behaved function of strength), fit by iteration: simulate, compare our
probabilities to the market's in logit space, nudge each team's offset, recentre
so the offsets sum to zero, repeat. Converges in a few dozen cheap simulations.

SOURCES. No free feed carries outrights without a key, so there are two paths:
  - the-odds-api.com, if ODDS_API_KEY is set (free tier, 500 requests/month)
  - a hand-filled CSV at data/reference/outrights_<season>.csv, which is 20 rows
    copied off any bookmaker or odds-comparison page
Either way the numbers are de-vigged before use, exactly as match odds are.
"""
import os
import sys

import numpy as np
import pandas as pd

import dixon_coles as dc
import simulate as S
from paths import REFERENCE_DIR, load_matches

MARKETS = ("title", "top4", "releg")
DEFAULT_ITERS = 30
DEFAULT_SIMS = 4000
LR = 0.35          # damped update; the mapping from offset to probability is steep
MAX_DELTA = 0.9    # keep offsets inside the range the fit can express


def template_path(season):
    return REFERENCE_DIR / f"outrights_{season}.csv"


def write_template(teams, season):
    """Create a CSV for someone to paste decimal odds into."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    p = template_path(season)
    pd.DataFrame({"team": sorted(teams), "title_odds": "", "top4_odds": "",
                  "releg_odds": ""}).to_csv(p, index=False)
    return p


def load_outrights(season, teams):
    """Read the hand-filled CSV, de-vig each market, return {market: {team: p}}."""
    p = template_path(season)
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    out = {}
    totals = {"title": 1.0, "top4": 4.0, "releg": 3.0}
    for mkt, col in [("title", "title_odds"), ("top4", "top4_odds"),
                     ("releg", "releg_odds")]:
        if col not in d.columns:
            continue
        sub = d[["team", col]].dropna()
        sub = sub[pd.to_numeric(sub[col], errors="coerce").notna()]
        if len(sub) < 10:
            continue
        odds = pd.to_numeric(sub[col]).to_numpy(dtype=float)
        if (odds <= 1.0).any():
            raise ValueError(f"{col}: decimal odds must exceed 1.0")
        raw = 1.0 / odds
        # scale so the market sums to the number of places it fills
        p_ = raw * (totals[mkt] / raw.sum())
        p_ = np.clip(p_, 1e-4, 1 - 1e-4)
        out[mkt] = dict(zip(sub["team"], p_))
    return out


def from_odds_api(season_teams):
    """Outright title odds from the-odds-api, if a key is available."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return {}
    import json
    import urllib.request
    url = ("https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
           f"?regions=uk&markets=outrights&oddsFormat=decimal&apiKey={key}")
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    prices = {}
    for ev in data:
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                for o in mk.get("outcomes", []):
                    prices.setdefault(o["name"], []).append(o["price"])
    if not prices:
        return {}
    med = {k: float(np.median(v)) for k, v in prices.items()}
    raw = {k: 1.0 / v for k, v in med.items()}
    tot = sum(raw.values())
    return {"title": {k: v / tot for k, v in raw.items()}}


# --------------------------------------------------------------------------
def _sim_probs(matches, season, league, fit, delta, n_sims, seed=7):
    import copy as _c
    f = _c.copy(fit)
    f.adjustments = {t: (d, d) for t, d in delta.items()}
    sim = S.simulate_season(matches, season, league, n_sims=n_sims, fit=f, seed=seed)
    n = len(sim["teams"])
    pos = sim["position"]
    return sim["teams"], {
        "title": (pos == 1).mean(axis=0),
        "top4": (pos <= 4).mean(axis=0),
        "releg": (pos >= n - 2).mean(axis=0),
    }


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_outright_offsets(matches, season, league, targets, fit=None,
                         n_iter=DEFAULT_ITERS, n_sims=DEFAULT_SIMS, verbose=True):
    """Strength offsets that make the simulator reproduce the market's outrights.

    `targets` is {market: {team: probability}}. Any subset of title/top4/releg
    may be supplied; each is matched in logit space and averaged.
    """
    if fit is None:
        fit = dc.fit_for_league(matches, season, league)
    teams, _ = _sim_probs(matches, season, league, fit, {}, 200)
    delta = {t: 0.0 for t in teams}
    idx = {t: i for i, t in enumerate(teams)}
    hist = []

    for it in range(n_iter):
        _, ours = _sim_probs(matches, season, league, fit, delta, n_sims,
                             seed=7 + it)
        err = np.zeros(len(teams))
        cnt = np.zeros(len(teams))
        for mkt, tgt in targets.items():
            if mkt not in ours:
                continue
            for t, p in tgt.items():
                if t not in idx:
                    continue
                i = idx[t]
                # relegation runs the other way: more likely down = weaker
                sign = -1.0 if mkt == "releg" else 1.0
                err[i] += sign * (_logit(p) - _logit(ours[mkt][i]))
                cnt[i] += 1
        step = np.where(cnt > 0, err / np.maximum(cnt, 1), 0.0)
        rmse = float(np.sqrt(np.mean(step[cnt > 0] ** 2))) if (cnt > 0).any() else 0.0
        hist.append(rmse)
        for t in teams:
            delta[t] = float(np.clip(delta[t] + LR * 0.05 * step[idx[t]],
                                     -MAX_DELTA, MAX_DELTA))
        mu = np.mean(list(delta.values()))
        delta = {t: d - mu for t, d in delta.items()}      # keep sum zero
        if verbose and (it % 5 == 0 or it == n_iter - 1):
            print(f"    iter {it:>3}  logit RMSE {rmse:.4f}")
        if rmse < 0.02:
            break
    return delta, hist


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2627"
    from fixtures import load_fixtures
    fx = load_fixtures(season)
    teams = sorted(set(fx["home_team"]) | set(fx["away_team"]))

    got = from_odds_api(teams) or load_outrights(season, teams)
    if not got:
        p = write_template(teams, season)
        print("No outright odds found.\n")
        print("Two ways to supply them:")
        print("  1. set ODDS_API_KEY   (free tier at the-odds-api.com)")
        print(f"  2. fill in {p.relative_to(REFERENCE_DIR.parent.parent)}")
        print("     -- 20 rows, decimal odds, any bookmaker. Partial is fine:")
        print("        relegation odds alone are enough to fix the bottom of")
        print("        the table, which is where the model is least sure.")
        sys.exit(0)

    print("Markets found:", ", ".join(f"{k} ({len(v)} teams)" for k, v in got.items()))
    m = load_matches().dropna(subset=["home_goals", "away_goals"])
    combined = pd.concat([m[m["season"] != season], fx.assign(season=season)],
                         ignore_index=True)
    print("\nFitting strength offsets to the market's season view...")
    delta, hist = fit_outright_offsets(combined, season, "Prem", got)
    s = pd.Series(delta).sort_values()
    print("\noffsets (negative = market rates them below our model):")
    for t, v in s.items():
        print(f"  {t:<24}{v:+.3f}")
