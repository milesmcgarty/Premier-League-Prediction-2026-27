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
import odds as O
import simulate as S
from paths import REFERENCE_DIR, ROOT, load_matches

ARCHIVE_DIR = ROOT / "data" / "outrights"

MARKETS = ("title", "top4", "releg")
DEFAULT_ITERS = 60
DEFAULT_SIMS = 12000
LR = 0.35          # damped update; the mapping from offset to probability is steep
MAX_DELTA = 0.9    # keep offsets inside the range the fit can express

# Only fit a team from a market that genuinely discriminates it. Bookmakers stop
# pricing outrights past roughly 2000/1, so in a TITLE market every side outside
# the top handful sits on that floor: Coventry and Hull were both 2000/1, which
# says nothing about which is better. Fitting to those prices reads the floor as
# information and boosts the weakest teams -- the first run had Hull City at
# +0.382, i.e. the market supposedly rating them far above our model.
# A team is only used where the market's implied probability clears this floor.
MIN_MARKET_P = 0.010


def template_path(season):
    return REFERENCE_DIR / f"outrights_{season}.csv"


def write_template(teams, season):
    """Create a CSV for someone to paste decimal odds into."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    p = template_path(season)
    pd.DataFrame({"team": sorted(teams), "title_odds": "", "top4_odds": "",
                  "releg_odds": ""}).to_csv(p, index=False)
    return p


def archive(season, as_of=None, source="oddschecker"):
    """Freeze today's outright prices, dated and immutable.

    outrights_<season>.csv is a WORKING file: paste new prices into it and the
    old ones are gone. That is fatal for the one comparison this project cannot
    currently make, because scoring the season product against a market requires
    the market's view AS IT WAS, not as it is once the season has resolved.

    Each capture is written to data/outrights/<season>/<date>.csv and never
    modified. Nine Augusts of these files is what turns "we have never compared
    the season product to a bookmaker" into a closed question. It costs one
    paste a week and nothing else.
    """
    src = template_path(season)
    if not src.exists():
        return None
    d = pd.read_csv(src)
    if d[["title_odds", "top4_odds", "releg_odds"]].notna().sum().sum() == 0:
        return None
    as_of = pd.Timestamp(as_of or pd.Timestamp.utcnow().normalize())
    out = ARCHIVE_DIR / season
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{as_of.strftime('%Y-%m-%d')}.csv"
    if path.exists():
        return path                      # never overwrite a capture
    d = d.copy()
    d["captured"] = as_of.strftime("%Y-%m-%d")
    d["source"] = source
    d.to_csv(path, index=False)
    return path


def list_archive(season):
    """Every captured outright file for a season, oldest first."""
    out = ARCHIVE_DIR / season
    if not out.exists():
        return []
    return sorted(out.glob("*.csv"))


def earliest_capture(season, teams=None):
    """The earliest archived market view for a season, de-vigged.

    'Earliest' is what the comparison needs: a forecast made in August must be
    scored against the market's August prices, not against prices that have
    already absorbed half a season of results.
    """
    files = list_archive(season)
    if not files:
        return {}, None
    return _read_capture(files[0])


def latest_capture(season, teams=None):
    """The most recent archived market view, de-vigged.

    The counterpart to earliest_capture. Scoring uses the earliest, because a
    forecast must be graded against prices that knew no more than it did; a
    LIVE side-by-side wants the newest, because the question there is what the
    market thinks now.
    """
    files = list_archive(season)
    if not files:
        return {}, None
    return _read_capture(files[-1])


def _read_capture(f):
    d = pd.read_csv(f)
    totals = {"title": 1.0, "top4": 4.0, "releg": 3.0}
    out = {}
    for mkt, col in [("title", "title_odds"), ("top4", "top4_odds"),
                     ("releg", "releg_odds")]:
        if col not in d.columns:
            continue
        sub = d[["team", col]].dropna()
        sub = sub[pd.to_numeric(sub[col], errors="coerce").notna()]
        if len(sub) < 10:
            continue
        raw = 1.0 / pd.to_numeric(sub[col]).to_numpy(dtype=float)
        p_ = O.devig(raw, method="power", target=totals[mkt])
        out[mkt] = dict(zip(sub["team"], np.clip(p_, 1e-4, 1 - 1e-4)))
    return out, str(f.stem)


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
        # POWER de-vig, scaled to the number of places the market fills.
        # Proportional normalisation loads too little margin onto longshots; on
        # a title market spanning 4/5 to 2500/1 that distortion is far larger
        # than in a match market. Measured on 22,360 matches, proportional
        # leaves +0.056 residual bias in the top probability bin where the power
        # method leaves -0.0015. Shin is not used because its single-winner
        # derivation does not apply to top-four or relegation.
        p_ = O.devig(raw, method="power", target=totals[mkt])
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
def _sim_probs(matches, season, league, fit, delta, n_sims, seed=7, sim_kw=None):
    """Simulate under `delta`, ADDING it to whatever the fit already carries.

    Two things here were wrong for a long time and both mattered:

    1. This used to ASSIGN `f.adjustments = {t: (d, d) ...}`, discarding the
       market prior and the availability offsets the caller had already put on
       the fit. The offsets were therefore fitted in a world with no injuries,
       and then applied by the harness ON TOP of the injury adjustments -- so a
       depleted squad got the market's correction and the availability penalty
       both, and finished far below the price the offset was meant to reproduce.

    2. It called simulate_season with no `as_of`, simulating all 380 fixtures
       from scratch while production simulates from the real current table. Mid
       season those are different distributions, so the fitter was matching the
       market on a season that had not happened.

    The rule this encodes: THE FITTING SIMULATOR MUST BE THE PRODUCTION
    SIMULATOR. `sim_kw` carries as_of, the tuned promoted dispersion and
    anything else the harness passes, so the only differences left are n_sims
    and the seed.
    """
    import copy as _c
    f = _c.copy(fit)
    merged = dict(getattr(fit, "adjustments", None) or {})
    for t, d in delta.items():
        d0 = merged.get(t, (0.0, 0.0))
        merged[t] = (d0[0] + d, d0[1] + d)
    f.adjustments = merged
    sim = S.simulate_season(matches, season, league, n_sims=n_sims, fit=f,
                            seed=seed, **(sim_kw or {}))
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
                         n_iter=DEFAULT_ITERS, n_sims=DEFAULT_SIMS, verbose=True,
                         sim_kw=None, burn_in=0.5):
    """Strength offsets that make the simulator reproduce the market's outrights.

    `targets` is {market: {team: probability}}. Any subset of title/top4/releg
    may be supplied; each is matched in logit space and averaged.

    CONVERGENCE. This is a stochastic fixed-point iteration: the "gradient" is
    a Monte Carlo estimate, so it has a noise floor. At 4,000 sims a probability
    of 0.05 carries a logit standard error of about 0.07, which is the same size
    as the error being chased -- the iteration used to reach RMSE 0.078 by step
    15, bounce back to 0.144 by step 20, and then ship whatever step 29 happened
    to hold. Three changes make it behave:

      * more sims, so the noise floor sits below the tolerance
      * a decaying step, so late iterations stop overshooting
      * POLYAK AVERAGING over the second half of the run, which is what actually
        removes the oscillation. Taking the best-RMSE iterate instead would be
        selecting on noise and would flatter the reported error.
    """
    if fit is None:
        fit = dc.fit_for_league(matches, season, league)
    teams, _ = _sim_probs(matches, season, league, fit, {}, 200, sim_kw=sim_kw)
    delta = {t: 0.0 for t in teams}
    idx = {t: i for i, t in enumerate(teams)}
    hist = []
    trail = []
    start_avg = int(n_iter * burn_in)

    # Which teams any supplied market actually discriminates. Everyone else is
    # left at exactly zero rather than picking up the recentring constant, which
    # would look like information and is not.
    fitted = {t for tgt in targets.values() for t, p in tgt.items()
              if t in idx and MIN_MARKET_P <= p <= 1 - MIN_MARKET_P}

    for it in range(n_iter):
        _, ours = _sim_probs(matches, season, league, fit, delta, n_sims,
                             seed=7 + it, sim_kw=sim_kw)
        err = np.zeros(len(teams))
        cnt = np.zeros(len(teams))
        for mkt, tgt in targets.items():
            if mkt not in ours:
                continue
            for t, p in tgt.items():
                if t not in idx or p < MIN_MARKET_P or p > 1 - MIN_MARKET_P:
                    continue                      # market does not discriminate here
                i = idx[t]
                # relegation runs the other way: more likely down = weaker
                sign = -1.0 if mkt == "releg" else 1.0
                err[i] += sign * (_logit(p) - _logit(ours[mkt][i]))
                cnt[i] += 1
        step = np.where(cnt > 0, err / np.maximum(cnt, 1), 0.0)
        rmse = float(np.sqrt(np.mean(step[cnt > 0] ** 2))) if (cnt > 0).any() else 0.0
        hist.append(rmse)
        # Step decays as 1/(1 + it/10): big early moves, small late ones, so the
        # iteration settles instead of rattling around the Monte Carlo noise.
        lr = LR * 0.05 / (1.0 + it / 10.0)
        for t in fitted:
            delta[t] = float(np.clip(delta[t] + lr * step[idx[t]],
                                     -MAX_DELTA, MAX_DELTA))
        # recentre over the FITTED teams only, so their average is unchanged
        # relative to the untouched rest of the league
        if fitted:
            mu = np.mean([delta[t] for t in fitted])
            for t in fitted:
                delta[t] -= mu
        if it >= start_avg:
            trail.append(dict(delta))
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(f"    iter {it:>3}  logit RMSE {rmse:.4f}")
        if rmse < 0.02:
            break

    # Polyak average over the trailing iterates. Averaging is unbiased; picking
    # the single best-scoring iterate would be choosing on simulation noise.
    if trail:
        delta = {t: float(np.mean([d[t] for d in trail])) for t in delta}
        if fitted:
            mu = np.mean([delta[t] for t in fitted])
            for t in fitted:
                delta[t] -= mu
    if verbose:
        print(f"    fitted {len(fitted)} of {len(teams)} teams "
              f"(the rest sit at the bookmaker's price floor and are left alone)"
              + (f", averaged over the last {len(trail)} iterates" if trail else ""))
    return {t: d for t, d in delta.items() if t in fitted}, hist


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
