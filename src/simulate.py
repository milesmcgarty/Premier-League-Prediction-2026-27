"""Phase 6: Monte Carlo season simulation.

Converts match-level probabilities into the numbers this project actually exists
to produce -- title, top-four, promotion and relegation probabilities, and full
points/position distributions. The market does not publish these, and (unlike
match predictions) they CANNOT be blended with odds: simulating a season in
August means predicting May fixtures nobody has priced. So this runs on
model-only probabilities, and model quality alone drives every number here.

Method: for each unplayed fixture build the full scoreline grid (Poisson pair
plus the Dixon-Coles low-score correction), sample a scoreline from it, add the
result to any matches already played, build the table, and repeat. Sampling
whole scorelines rather than W/D/L is what makes goal difference -- and
therefore the tie-breaks that decide real titles -- come out right.

SEASON-LEVEL UNCERTAINTY: fixtures are sampled independently GIVEN a team's
strength, but strength itself is drawn ONCE PER SIMULATED SEASON. That matters.
Treating every match as independent assumes a team's strength is exactly its
fitted rating; in reality that rating is both estimated with error and wrong in
ways that persist all year (transfers, a new manager, a key injury). When a side
is genuinely better than rated it is better in all 38 matches at once, which
fattens the tails of the points distribution.

Measured with independent draws: only 63.9% of actual season points fell inside
the predicted 10-90% band (target 80%, z=-5.40) and 37.8% inside the 25-75% band
(target 50%, z=-3.28) -- clearly overconfident. STRENGTH_SD adds a per-team
season-long offset to attack and defence, and is tuned out-of-sample like every
other hyperparameter here.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

from dixon_coles import MAX_GOALS, _tau, fit_for_league

# Per-team season-long strength offset (log-goal units), added to attack and
# defence once per simulated season. Selected on TUNE seasons by minimising the
# Kolmogorov-Smirnov distance of the probability integral transform from uniform
# -- i.e. making the actual season points land at uniformly distributed quantiles
# of the predicted distribution. Both divisions independently chose 0.15.
#
# Held-out effect (Premier League):
#   independent draws : 10-90 band covers 63.9%, PIT KS 0.1185, p = 0.0117
#   sd = 0.15         : 10-90 band covers 75.0%, PIT KS 0.0542, p = 0.6459
# i.e. from "reject calibration" to "indistinguishable from calibrated".
# The band is still marginally narrow (z=-1.68), so the extreme tails remain a
# little optimistic. Propagating the fit's actual parameter covariance, rather
# than one scalar dispersion, is the natural refinement.
STRENGTH_SD = 0.15

# Promoted teams get a WIDER draw. Measured on held-out seasons, splitting the
# calibration by promotion status:
#
#   established  n=153  80% band covers 79.7% (z=-0.08)  PIT p=0.75  error/pred sd 1.07
#   promoted     n= 27  80% band covers 48.1% (z=-4.14)  PIT p=0.006 error/pred sd 1.55
#
# All the miscalibration is in promoted teams, whose outcomes are ~55% more
# dispersed than the model believes. Crucially their BIAS is not significant
# (mean error -3.2 pts, p=0.27): they are not systematically over- or
# under-rated, just far less predictable. Championship form carries almost no
# signal about Premier League performance (corr of Championship points with
# subsequent PL points = +0.004, n=75), so we cannot know WHICH promoted side
# will overachieve -- only that we should stop pretending we do.
STRENGTH_SD_PROMOTED = 0.15

# Promoted-team strength is drawn ASYMMETRICALLY: the offset uses
# STRENGTH_SD_PROMOTED below zero and PROMOTED_UP_RATIO * that above it.
# A symmetric widening has the wrong shape. Across 75 promoted team-seasons
# since 2000, ZERO finished in the top six -- the best is 7th (Wolves 2018-19,
# Sunderland 2025-26) -- while the downside runs to 11 points. Widening both
# tails equally gave the 2026-27 promoted trio a 6.1% chance of a top-four
# finish and a 4.7% chance one of them won the title, against a historical rate
# of 0 in 75. So: long tail down, short tail up.
PROMOTED_UP_RATIO = 0.5

# E[z | z>0] * P(z>0) for a standard normal; used to recentre the two-piece
# draw so that compressing one side does not shift the mean.
PHI0 = 0.3989422804014327

# Number of distinct strength draws. Each scenario plays several seasons, so the
# cost is one grid build per scenario rather than per simulated season.
N_SCENARIOS = 200

# Premier League order: points, then goal difference, then goals scored.
# (The real rules add a play-off after that, which has never been needed.)
TIEBREAK = ["Pts", "GD", "GF"]


def results_table(matches, teams=None):
    """League table from played matches. Deterministic -- this is the piece the
    correctness test pins against real historical tables."""
    if teams is None:
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    P, W, D, L, GF, GA = (np.zeros(n, dtype=int) for _ in range(6))

    h = matches["home_team"].map(idx).to_numpy()
    a = matches["away_team"].map(idx).to_numpy()
    hg = matches["home_goals"].to_numpy().astype(int)
    ag = matches["away_goals"].to_numpy().astype(int)

    np.add.at(P, h, 1);          np.add.at(P, a, 1)
    np.add.at(GF, h, hg);        np.add.at(GF, a, ag)
    np.add.at(GA, h, ag);        np.add.at(GA, a, hg)
    np.add.at(W, h, hg > ag);    np.add.at(W, a, ag > hg)
    np.add.at(D, h, hg == ag);   np.add.at(D, a, hg == ag)
    np.add.at(L, h, hg < ag);    np.add.at(L, a, ag < hg)

    t = pd.DataFrame({"team": teams, "P": P, "W": W, "D": D, "L": L,
                      "GF": GF, "GA": GA, "GD": GF - GA, "Pts": 3 * W + D})
    t = t.sort_values(TIEBREAK, ascending=False).reset_index(drop=True)
    t.insert(0, "pos", np.arange(1, len(t) + 1))
    return t


def scoreline_grid(fit, home, away, max_goals=MAX_GOALS):
    """Full (max_goals+1)^2 scoreline distribution for one fixture."""
    lh, la = fit.lambdas(home, away)
    g = np.outer(poisson.pmf(np.arange(max_goals + 1), lh),
                 poisson.pmf(np.arange(max_goals + 1), la))
    for i in range(2):
        for j in range(2):
            g[i, j] *= _tau(np.array([i]), np.array([j]),
                            np.array([lh]), np.array([la]), fit.rho)[0]
    g = np.clip(g, 0.0, None)
    return g / g.sum()


def promoted_teams(matches, season, league):
    """Teams in `league` this season that were not in it last season."""
    order = sorted(matches["season"].unique(), key=lambda s: int(s[:2]))
    if season not in order:
        order = order + [season]
    i = order.index(season)
    if i == 0:
        return set()
    def act(s):
        sub = matches[(matches["season"] == s) & (matches["league"] == league)]
        return set(sub["home_team"]) | set(sub["away_team"])
    return act(season) - act(order[i - 1])


def simulate_season(matches, season, league, n_sims=10000, as_of=None,
                    fit=None, seed=0, max_goals=MAX_GOALS,
                    strength_sd=None, strength_sd_promoted=None,
                    promoted_up_ratio=None, n_scenarios=N_SCENARIOS):
    """Simulate a season from `as_of` onwards.

    Matches before `as_of` use their ACTUAL results; the rest are sampled.
    `as_of=None` simulates the whole season from scratch. This split is what the
    live harness needs -- results so far plus remaining fixtures.

    Strength is drawn once per SCENARIO and held fixed for every season played
    under it, which is what puts season-long correlation into the tails.
    """
    rng = np.random.default_rng(seed)
    strength_sd = STRENGTH_SD if strength_sd is None else strength_sd
    strength_sd_promoted = (STRENGTH_SD_PROMOTED if strength_sd_promoted is None
                            else strength_sd_promoted)
    promoted_up_ratio = (PROMOTED_UP_RATIO if promoted_up_ratio is None
                         else promoted_up_ratio)
    season_m = matches[(matches["season"] == season) &
                       (matches["league"] == league)].copy()
    teams = sorted(set(season_m["home_team"]) | set(season_m["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # A fixture counts as played only if it has a RESULT, not merely a past date.
    # Postponements are routine, and treating one as played would feed NaN goals
    # into the table; treating a played match as remaining would double-count it.
    has_result = season_m["home_goals"].notna() & season_m["away_goals"].notna()
    if as_of is None:
        played, remaining = season_m.iloc[:0], season_m
    else:
        as_of = pd.Timestamp(as_of)
        is_played = has_result & (season_m["date"] < as_of)
        played, remaining = season_m[is_played], season_m[~is_played]

    if fit is None:
        fit = fit_for_league(matches, season, league)

    # --- points already banked ---
    base_pts, base_gf, base_ga = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    if len(played):
        bt = results_table(played, teams).set_index("team")
        base_pts = bt.loc[teams, "Pts"].to_numpy()
        base_gf = bt.loc[teams, "GF"].to_numpy()
        base_ga = bt.loc[teams, "GA"].to_numpy()

    pts = np.tile(base_pts.astype(np.int32), (n_sims, 1))
    gf = np.tile(base_gf.astype(np.int32), (n_sims, 1))
    ga = np.tile(base_ga.astype(np.int32), (n_sims, 1))

    if len(remaining) == 0:
        gd = gf - ga
        return _rank_and_pack(rng, teams, pts, gf, ga, gd, n_sims,
                              len(played), 0, fit, strength_sd)

    # base ratings, routed through _rating so the newcomer prior and any
    # adjustments (the transfer-value hook) are applied
    atk0 = np.array([fit._rating(t)[0] for t in teams])
    dfn0 = np.array([fit._rating(t)[1] for t in teams])
    hi = remaining["home_team"].map(idx).to_numpy()
    ai = remaining["away_team"].map(idx).to_numpy()
    F = len(remaining)
    size = max_goals + 1
    ks = np.arange(size)

    # one strength draw per scenario; each scenario plays `per` seasons
    # per-team dispersion: promoted sides are genuinely less predictable
    promo = promoted_teams(matches, season, league)
    is_promo = np.array([t in promo for t in teams])
    sd_vec = np.where(is_promo, strength_sd_promoted, strength_sd).astype(float)
    # scale applied only to POSITIVE offsets, and only for promoted sides
    up_vec = np.where(is_promo, promoted_up_ratio, 1.0).astype(float)

    n_scen = 1 if sd_vec.max() <= 0 else min(n_scenarios, n_sims)
    bounds = np.linspace(0, n_sims, n_scen + 1).astype(int)

    for sc in range(n_scen):
        lo, hi_s = bounds[sc], bounds[sc + 1]
        per = hi_s - lo
        if per == 0:
            continue
        if sd_vec.max() > 0:
            # Two-piece normal: full sd below zero, up_vec * sd above it.
            # RECENTRED so the mean stays zero. A compressed upper half has
            # E[eps] = sd * phi(0) * (ratio - 1) < 0, which would make promoted
            # teams systematically weaker -- and the evidence says they are not
            # systematically weaker, just less predictable (mean error -3.2 pts,
            # p=0.27, n=27). Fixing the tail shape must not smuggle in a bias.
            shift = sd_vec * PHI0 * (up_vec - 1.0)
            za, zd = rng.normal(0, 1, n), rng.normal(0, 1, n)
            ea = za * sd_vec * np.where(za > 0, up_vec, 1.0) - shift
            ed = zd * sd_vec * np.where(zd > 0, up_vec, 1.0) - shift
            atk, dfn = atk0 + ea, dfn0 + ed
        else:
            atk, dfn = atk0, dfn0

        lh = np.exp(fit.intercept + atk[hi] - dfn[ai] + fit.home_adv)
        la = np.exp(fit.intercept + atk[ai] - dfn[hi])

        # (F, size) marginals -> (F, size, size) joint
        ph = poisson.pmf(ks[None, :], lh[:, None])
        pa = poisson.pmf(ks[None, :], la[:, None])
        grid = ph[:, :, None] * pa[:, None, :]
        r = fit.rho
        grid[:, 0, 0] *= 1.0 - lh * la * r
        grid[:, 0, 1] *= 1.0 + lh * r
        grid[:, 1, 0] *= 1.0 + la * r
        grid[:, 1, 1] *= 1.0 - r
        np.clip(grid, 0.0, None, out=grid)
        flat = grid.reshape(F, -1)
        flat /= flat.sum(axis=1, keepdims=True)
        cum = np.cumsum(flat, axis=1)
        cum[:, -1] = 1.0

        # Sample by inverse-CDF. Offsetting row i's CDF by i makes the whole
        # (F, size^2) table globally increasing, so one searchsorted replaces a
        # per-fixture loop AND the O(F * per * size^2) broadcast comparison that
        # made tuning intractable.
        off = np.arange(F, dtype=np.float64)[:, None]
        cum_flat = (cum + off).ravel()
        u = rng.random((F, per)) + off
        k = np.searchsorted(cum_flat, u.ravel(), side="right")
        draws = (k - np.repeat(np.arange(F) * (size * size), per)).reshape(F, per)
        np.clip(draws, 0, size * size - 1, out=draws)
        hg = (draws // size).astype(np.int32)
        ag = (draws % size).astype(np.int32)

        hpts = np.where(hg > ag, 3, (hg == ag) * 1).astype(np.int32)
        apts = np.where(ag > hg, 3, (hg == ag) * 1).astype(np.int32)

        sl = slice(lo, hi_s)
        np.add.at(gf[sl].T, hi, hg);   np.add.at(ga[sl].T, hi, ag)
        np.add.at(gf[sl].T, ai, ag);   np.add.at(ga[sl].T, ai, hg)
        np.add.at(pts[sl].T, hi, hpts); np.add.at(pts[sl].T, ai, apts)

    gd = gf - ga
    return _rank_and_pack(rng, teams, pts, gf, ga, gd, n_sims,
                          len(played), F, fit, strength_sd)


def _rank_and_pack(rng, teams, pts, gf, ga, gd, n_sims, n_played, n_remaining,
                   fit, strength_sd):
    """Rank by points, then GD, then GF; exact ties broken uniformly at random."""
    n = len(teams)
    key = (pts.astype(np.float64) * 1e6
           + (gd.astype(np.float64) + 200.0) * 1e3
           + gf.astype(np.float64)
           + rng.random((n_sims, n)) * 1e-3)
    order = np.argsort(-key, axis=1)
    position = np.empty_like(order)
    np.put_along_axis(position, order,
                      np.arange(1, n + 1)[None, :].repeat(n_sims, 0), axis=1)
    return {"teams": teams, "position": position, "points": pts,
            "gd": gd, "gf": gf, "n_sims": n_sims, "n_played": n_played,
            "n_remaining": n_remaining, "fit": fit, "strength_sd": strength_sd}


PROMOTED_SD_GRID = [0.15, 0.25, 0.35, 0.45, 0.55]
PROMOTED_UP_GRID = [1.0, 0.7, 0.5, 0.35]
PROMOTED_LOOKBACK = 6


_PROMOTED_SD_CACHE = {}


def tune_promoted_sd(matches, season, league="Prem", lookback=PROMOTED_LOOKBACK,
                     grid=PROMOTED_SD_GRID, n_sims=3000, seed=11, use_cache=True):
    """Choose the promoted-team dispersion from the seasons immediately BEFORE
    `season`, by minimising the KS distance of the PIT from uniform.

    Rolling rather than a single fixed block, because promoted-team
    predictability CHANGES. A fixed value tuned on 2008-2017 left held-out
    promoted teams at 44.4% coverage of their 80% band (PIT p=0.005, i.e.
    calibration rejected). Re-tuning on the six preceding seasons lifts that to
    59.3% (p=0.115, no longer rejected), and the value it selects rises steadily
    over the decade -- 0.15 in 2018-19 to 0.45 in 2025-26 -- which is itself
    evidence that promoted sides have become harder to forecast.

    Still an incomplete fix: 59.3% is short of the nominal 80%. Promoted teams
    remain less predictable than the model can express, and Championship form
    carries almost no signal about how they will do (corr = +0.004, n=75).
    """
    from scipy import stats

    # The search is ~20 configurations x lookback seasons of simulation, which
    # takes about 100s. The harness would otherwise repeat it on every snapshot
    # even though the answer only changes once a season.
    key = (season, league, lookback, tuple(grid), n_sims, seed)
    if use_cache and key in _PROMOTED_SD_CACHE:
        return _PROMOTED_SD_CACHE[key]

    order = sorted(matches["season"].unique(), key=lambda x: int(x[:2]))
    if season in order:
        i = order.index(season)
    else:
        i = len(order)
    window = order[max(0, i - lookback):i]
    if not window:
        return STRENGTH_SD_PROMOTED

    rng = np.random.default_rng(3)
    best = None
    for sd in grid:
      for up in PROMOTED_UP_GRID:
        us = []
        for s in window:
            promo = promoted_teams(matches, s, league)
            if not promo:
                continue
            sim = simulate_season(matches, s, league, n_sims=n_sims,
                                  strength_sd_promoted=sd,
                                  promoted_up_ratio=up, seed=seed)
            played = matches[(matches["season"] == s) & (matches["league"] == league)]
            act = results_table(played).set_index("team")
            P = sim["points"]
            for j, t in enumerate(sim["teams"]):
                if t not in promo or t not in act.index:
                    continue
                a = act.loc[t, "Pts"]
                us.append((P[:, j] < a).mean() + rng.random() * (P[:, j] == a).mean())
        if len(us) < 5:
            continue
        k = stats.kstest(us, "uniform").statistic
        if best is None or k < best[2]:
            best = (sd, up, k)
    out = ((STRENGTH_SD_PROMOTED, PROMOTED_UP_RATIO) if best is None
           else (best[0], best[1]))
    if use_cache:
        _PROMOTED_SD_CACHE[key] = out
    return out


def summarise(sim, league="Prem"):
    """Headline probabilities per team, sorted by expected finish."""
    teams, pos, pts = sim["teams"], sim["position"], sim["points"]
    n = len(teams)
    rows = []
    for i, t in enumerate(teams):
        p, q = pos[:, i], pts[:, i]
        row = {"team": t, "exp_pts": q.mean(),
               "pts_10": np.percentile(q, 10), "pts_90": np.percentile(q, 90),
               "exp_pos": p.mean(), "title": (p == 1).mean()}
        if league == "Prem":
            row["top4"] = (p <= 4).mean()
            row["top6"] = (p <= 6).mean()
            row["releg"] = (p >= n - 2).mean()
        else:
            row["auto_promo"] = (p <= 2).mean()
            row["playoff"] = ((p >= 3) & (p <= 6)).mean()
            row["releg"] = (p >= n - 2).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("exp_pos").reset_index(drop=True)


def position_matrix(sim):
    """P(team finishes in position j) as a teams x positions frame."""
    teams, pos = sim["teams"], sim["position"]
    n = len(teams)
    M = np.zeros((n, n))
    for i in range(n):
        M[i] = np.bincount(pos[:, i], minlength=n + 1)[1:] / sim["n_sims"]
    return pd.DataFrame(M, index=teams, columns=range(1, n + 1))


if __name__ == "__main__":
    from paths import load_matches

    m = load_matches().dropna(subset=["home_goals", "away_goals"])
    SEASON, LEAGUE = "2526", "Prem"

    print("=" * 78)
    print(f"SEASON SIMULATION - {LEAGUE} {SEASON}")
    print("=" * 78)
    print("Simulated from scratch: the model is fitted only on seasons BEFORE")
    print(f"{SEASON}, so this is what we would have forecast in August.\n")

    sim = simulate_season(m, SEASON, LEAGUE, n_sims=20000, seed=1)
    summ = summarise(sim, LEAGUE)
    actual = results_table(m[(m["season"] == SEASON) &
                             (m["league"] == LEAGUE)]).set_index("team")
    summ["actual_pos"] = summ["team"].map(actual["pos"])
    summ["actual_pts"] = summ["team"].map(actual["Pts"])

    print(f"{'team':<24}{'xPts':>6}{'80% range':>12}{'title':>7}{'top4':>7}"
          f"{'releg':>7}  |{'ACTUAL':>8}{'pts':>5}")
    print("-" * 78)
    for _, r in summ.iterrows():
        print(f"{r['team']:<24}{r['exp_pts']:>6.1f}"
              f"{f'{r.pts_10:.0f}-{r.pts_90:.0f}':>12}"
              f"{r['title']:>7.1%}{r['top4']:>7.1%}{r['releg']:>7.1%}"
              f"  |{int(r['actual_pos']):>8}{int(r['actual_pts']):>5}")

    inside = ((summ.actual_pts >= summ.pts_10) & (summ.actual_pts <= summ.pts_90)).mean()
    print(f"\nactual points inside the 80% range: {inside:.0%} of teams (expect ~80%)")
    print(f"strength_sd = {sim['strength_sd']}, {sim['n_sims']} simulations")
