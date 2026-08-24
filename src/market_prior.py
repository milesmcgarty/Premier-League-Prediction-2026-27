"""Phase 8: market-implied strength priors for promoted teams.

THE PROBLEM. A promoted team's rating is derived from Championship form, and
Championship form does not predict Premier League performance -- across 75
promotions the correlation between Championship points and subsequent Premier
League points is +0.004. The model therefore extrapolates confidently from a
predictor that does not predict, which is how it arrived at Sunderland 96.3%
relegation (they finished 7th) and Hull City 93.6% (2026-27).

THE SIGNAL WE ALREADY HAD. Bookmakers price a promoted side's opening fixtures
knowing what we cannot see: summer transfers, the manager, the squad overhaul.
Measured across 63 promoted team-seasons, expected points per game implied by
the market's first six fixtures correlates +0.382 (p=0.002) with that team's
final points, against +0.147 (n.s.) for our own model rating. Adding the market
view lifts R-squared from 0.022 to 0.173.

WHY THIS IS NOT THE BLEND THAT FAILED. blend.py combined a match prediction with
that same match's price, and added nothing -- the market already knew everything
we did. This instead reads the market's view from the handful of fixtures it HAS
priced, converts it into a team RATING, and uses that rating to simulate 380
fixtures including the May ones nobody will price for months. Information flows
from the priced set to the unpriced set, which is the one place the market
cannot help us directly. It is also the architecture Opta actually uses: a power
rating, informed by the market, driving a Monte Carlo simulation.

LEAKAGE. The fixtures whose odds set the prior are excluded from evaluation.
Prior from the first N, scored on everything after.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

import odds as O
from dixon_coles import MAX_GOALS, _tau

# How many of a team's earliest priced fixtures to read the market from.
PRIOR_FIXTURES = 6
# Shrinkage toward zero, selected on TUNE seasons by log loss over promoted-team
# fixtures (prior-building fixtures excluded from scoring). Held-out effect:
# promoted-fixture log loss 0.9668 -> 0.9446, halving the gap to the market on
# those fixtures from +0.0462 to +0.0240; all-fixture log loss 0.9884 -> 0.9829.
SHRINK = 0.75
# Bounds on the strength offset, in log-goal units.
DELTA_BOUNDS = (-1.2, 1.2)


def _match_probs(fit, home, away, d_home, d_away, max_goals=MAX_GOALS):
    """W/D/L for one fixture with per-side strength offsets applied."""
    ah, dh = fit._rating(home)
    aa, da = fit._rating(away)
    lh = np.exp(fit.intercept + (ah + d_home) - (da + d_away) + fit.home_adv)
    la = np.exp(fit.intercept + (aa + d_away) - (dh + d_home))
    g = np.outer(poisson.pmf(np.arange(max_goals + 1), lh),
                 poisson.pmf(np.arange(max_goals + 1), la))
    for i in range(2):
        for j in range(2):
            g[i, j] *= _tau(np.array([i]), np.array([j]),
                            np.array([lh]), np.array([la]), fit.rho)[0]
    g = np.clip(g, 1e-15, None)
    g /= g.sum()
    return np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])


def fit_market_offset(fit, fixtures, team, book="B365", bounds=DELTA_BOUNDS):
    """The strength offset that best reconciles our model with the market.

    Minimises KL(market || model) over this team's priced fixtures. A single
    scalar rather than separate attack and defence terms: with only a handful of
    fixtures, two parameters would fit noise.
    """
    tm = fixtures[(fixtures["home_team"] == team) | (fixtures["away_team"] == team)]
    tm = tm[O.has_odds(tm, book)]
    if len(tm) == 0:
        return None, 0
    mkt = O.market_probs(tm, book)
    is_home = (tm["home_team"] == team).to_numpy()
    rows = list(tm.itertuples(index=False))

    def neg_ll(d):
        tot = 0.0
        for k, r in enumerate(rows):
            dh, da = (d, 0.0) if is_home[k] else (0.0, d)
            p = _match_probs(fit, r.home_team, r.away_team, dh, da)
            tot -= float(mkt[k] @ np.log(np.clip(p, 1e-15, None)))
        return tot

    res = minimize_scalar(neg_ll, bounds=bounds, method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x), len(tm)


def market_adjustments(fit, season_fixtures, teams, book="B365",
                       n_fixtures=PRIOR_FIXTURES, shrink=SHRINK):
    """{team: (d_attack, d_defence)} for `teams`, from their earliest priced games.

    Returns the adjustments plus the fixtures consumed, so the caller can
    exclude them from evaluation.
    """
    adj, used, info = {}, [], {}
    fx = season_fixtures.sort_values("date")
    for t in teams:
        tm = fx[(fx["home_team"] == t) | (fx["away_team"] == t)]
        tm = tm[O.has_odds(tm, book)].head(n_fixtures)
        if len(tm) == 0:
            continue
        d, n = fit_market_offset(fit, tm, t, book=book)
        if d is None:
            continue
        # Scale the correction by how much information supports it. A team's
        # offset is identified by the fixtures the market has priced; with only
        # ONE, the two sides of that match receive exactly mirrored offsets and
        # the disagreement cannot be attributed to either of them. The validated
        # shrink of 0.75 assumed the full n_fixtures. Applying it on a single
        # fixture would import a number the data does not support, so trust
        # grows with coverage and reaches the tuned value at full coverage.
        eff = shrink * min(1.0, n / float(n_fixtures))
        d *= eff
        adj[t] = (d, d)
        info[t] = {"delta": d, "n_fixtures": n, "effective_shrink": round(eff, 3)}
        used.extend(tm.index.tolist())
    return adj, sorted(set(used)), info


def apply_market_prior(fit, season_fixtures, teams, book="B365",
                       n_fixtures=PRIOR_FIXTURES, shrink=SHRINK):
    """Attach market-implied strength offsets for `teams` to `fit`, in place.

    APPLIES TO ALL TEAMS, not just promoted ones. An earlier version restricted
    it to promoted sides, on the reasoning that established teams were already
    well calibrated. That reasoning was about calibration ON AVERAGE ACROSS
    SEASONS, and it missed the case that actually matters: a squad reshaped over
    one summer. The model is fitted on results, so it cannot see that a club has
    sold its best players -- it will keep rating them on last season's form for
    months. Bookmakers reprice immediately.

    Held-out A/B, 3,420 fixtures:
        no prior at all                0.9858   (established-only 0.9937)
        prior on promoted teams only   0.9815   (established-only 0.9937)
        prior on ALL teams             0.9790   (established-only 0.9921)
    Extending it is worth a further +0.0025 overall and +0.0016 on fixtures
    between established sides alone.

    CAVEAT on thin coverage: the offset for a team is identified by the fixtures
    the market has priced. With only ONE priced fixture per team -- the usual
    situation in mid-August -- the two sides of that match get exactly mirrored
    offsets, and the disagreement cannot be attributed to either of them. It
    resolves as more fixtures are priced. Treat single-fixture offsets as weak.
    """
    if teams and not O.has_odds(season_fixtures, book).any():
        print(f"  NOTE: market prior inactive -- no {book} odds on any fixture. "
              "Ratings come from results alone, so recent transfers are invisible "
              "to the model.")
    adj, used, info = market_adjustments(fit, season_fixtures, teams,
                                         book=book, n_fixtures=n_fixtures,
                                         shrink=shrink)
    fit.adjustments = adj
    fit.market_prior_info = info
    return fit, used
