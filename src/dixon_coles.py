"""Dixon-Coles match prediction.

Fits attack + defence per team by weighted maximum likelihood across BOTH
divisions and multiple seasons, with time decay and a low-score correction.

Why the joint two-division fit (see CLAUDE.md Phase 4 for the full reasoning):
league fixtures are never cross-division, and adding a constant c to every
Championship team's attack AND defence leaves every within-Championship lambda
unchanged. So the relative scale of the two divisions is a flat direction in the
likelihood, identified ONLY by teams that played in both divisions inside the
training window. Measured consequence: a 1-season window has ZERO such teams and
is formally unidentified; 5 seasons has 14. Hence WINDOW_SEASONS >= 3, and a
gentle decay -- an aggressive half-life strips out the older-division matches
that carry all the linking information.

An explicit division-offset parameter is deliberately NOT included: it is exactly
collinear with the attack/defence of any team that never crosses, so it adds no
identifying information and only destabilises the optimiser. The gap is instead
read off post-hoc via division_gap(), which doubles as a diagnostic.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from paths import load_matches

# --- defaults, all overridable ---
WINDOW_SEASONS = 5      # >= 3, else the divisions are disconnected
HALF_LIFE_DAYS = 365    # gentle: preserves cross-division linking weight
MAX_GOALS = 12          # grid size; truncation mass at lambda=3 is ~3e-4

# Per-division hyperparameters, each selected on TUNE seasons for that division
# only (see backtest.py). NOTE: these are two JOINT fits, not two separate models
# -- each still spans both divisions, which is what preserves cross-division
# connectivity. One supplies Premier League predictions, the other Championship.
# Tuning on Premier League log loss and applying the winner to both divisions left
# the Championship overconfident (0.5-0.6 bin at z=-4.05); refitting the
# Championship with its own 8-season window drops that to z=-2.54 and removes
# every |z|>3 bin. The accuracy gain is small (0.0018 log loss); the calibration
# fix is the real reason to do it.
LEAGUE_PARAMS = {
    "Prem": {"window": 5, "half_life": 365},
    "Champ": {"window": 8, "half_life": 365},
}

# RIDGE shrinks attack/defence toward ZERO. Championship teams sit below zero and
# Premier League teams above, so it shrinks THE DIVISION GAP ITSELF -- it biases
# the very quantity the joint fit exists to estimate. Measured on a 5-season
# window: ridge 0 -> 266 Elo-equivalent gap, ridge 1 -> 206, ridge 10 -> 73.
# The Elo engine independently derives 232, so the unregularised fit is the one
# that corroborates. Default off. If thin-data teams ever need stabilising, the
# principled fix is shrinking toward each team's OWN DIVISION mean, which leaves
# the between-division gap untouched -- not this.
RIDGE = 0.0
SUM_ZERO_PENALTY = 1000.0

# rho bounds keep the Dixon-Coles tau factors strictly positive for realistic
# lambdas: tau(0,0) = 1 - lh*la*rho needs rho < 1/(lh*la).
RHO_BOUNDS = (-0.20, 0.10)


def time_weights(dates, cutoff, half_life_days=HALF_LIFE_DAYS):
    """exp(-xi * days_before_cutoff), with xi set by a half-life.

    Deliberately NOT normalised. Raw weights mean sum(w) is the effective sample
    size, which saturates as you add older seasons -- which is the truth, and it
    keeps RIDGE on a stable absolute scale across different window lengths.
    """
    days = (cutoff - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    xi = np.log(2) / half_life_days
    return np.exp(-xi * np.clip(days, 0, None))


def _tau(hg, ag, lh, la, rho):
    """Dixon-Coles low-score correction, vectorised.

    For rho < 0: boosts 0-0 and 1-1, suppresses 1-0 and 0-1 -- Dixon & Coles'
    1997 fix for Poisson under-predicting low-scoring draws. Note the correction
    is normalisation-preserving: the four adjustments sum to exactly zero.

    FINDING (2026-07-28): fitted on 2020-2025 data rho comes out at ~+0.005,
    i.e. essentially nil. The classic motivation does not replicate -- Poisson
    now predicts the overall draw rate to within 0.3%. Worse for the correction,
    0-0 is OVER-predicted (wants rho>0) while 1-1 is UNDER-predicted (wants
    rho<0), so the single parameter is pulled both ways and the MLE lands at
    zero. This is why rho is fitted rather than hardcoded at -0.05: that value
    would actively hurt, inflating an already over-predicted 0-0.
    """
    t = np.ones_like(lh)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    t[m00] = 1.0 - lh[m00] * la[m00] * rho
    t[m01] = 1.0 + lh[m01] * rho
    t[m10] = 1.0 + la[m10] * rho
    t[m11] = 1.0 - rho
    return t


def _tau_and_grads(hg, ag, lh, la, rho):
    """tau plus its partials w.r.t. lambda_home, lambda_away and rho.

    Needed for the analytic gradient. Finite-difference gradients cost ~2n+3
    function evaluations per step, which with ~143 parameters makes the
    hyperparameter search take hours instead of minutes.
    """
    t = np.ones_like(lh)
    dt_dlh = np.zeros_like(lh)
    dt_dla = np.zeros_like(lh)
    dt_drho = np.zeros_like(lh)

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    t[m00] = 1.0 - lh[m00] * la[m00] * rho
    dt_dlh[m00] = -la[m00] * rho
    dt_dla[m00] = -lh[m00] * rho
    dt_drho[m00] = -lh[m00] * la[m00]

    t[m01] = 1.0 + lh[m01] * rho
    dt_dlh[m01] = rho
    dt_drho[m01] = lh[m01]

    t[m10] = 1.0 + la[m10] * rho
    dt_dla[m10] = rho
    dt_drho[m10] = la[m10]

    t[m11] = 1.0 - rho
    dt_drho[m11] = -1.0

    return t, dt_dlh, dt_dla, dt_drho


@dataclass
class DixonColesFit:
    ratings: pd.DataFrame     # team, attack, defence
    intercept: float          # global goal level; see fit_dixon_coles docstring
    home_adv: float
    rho: float
    ess: float                # effective sample size = sum of time weights
    n_matches: int
    converged: bool

    # INTEGRATION HOOK -- {team: (d_attack, d_defence)} added to fitted ratings.
    # This is where squad market value / net transfer spend will plug in. Those
    # features are not a new likelihood term; they are an offset to a team's
    # strength (a promoted side that spent heavily should be shifted up). Because
    # every prediction routes through _rating(), adding them later is a drop-in
    # and can be A/B tested on the existing backtest without touching this class.
    # Currently unused: the fitted ratings are returned unmodified.
    adjustments: dict = field(default_factory=dict)

    def __post_init__(self):
        self._atk = self.ratings.set_index("team")["attack"].to_dict()
        self._def = self.ratings.set_index("team")["defence"].to_dict()
        self.prior = None     # set by attach_newcomer_prior()

    def _rating(self, team):
        if team in self._atk:
            a, d = self._atk[team], self._def[team]
        elif self.prior is not None:
            a, d = self.prior
        else:
            raise KeyError(
                f"'{team}' has no training data and no newcomer prior is attached. "
                "Call attach_newcomer_prior() before predicting."
            )
        da, dd = self.adjustments.get(team, (0.0, 0.0))
        return a + da, d + dd

    def lambdas(self, home_team, away_team, neutral=False):
        ah, dh = self._rating(home_team)
        aa, da = self._rating(away_team)
        adv = 0.0 if neutral else self.home_adv
        return (np.exp(self.intercept + ah - da + adv),
                np.exp(self.intercept + aa - dh))

    def predict(self, home_team, away_team, max_goals=MAX_GOALS, neutral=False):
        """W/D/L probabilities, expected goals and likeliest scoreline."""
        lh, la = self.lambdas(home_team, away_team, neutral=neutral)
        hp = poisson.pmf(np.arange(max_goals + 1), lh)
        ap = poisson.pmf(np.arange(max_goals + 1), la)
        grid = np.outer(hp, ap)

        # apply the correction to the four low-score cells
        for hg in range(2):
            for ag in range(2):
                grid[hg, ag] *= _tau(
                    np.array([hg]), np.array([ag]),
                    np.array([lh]), np.array([la]), self.rho
                )[0]
        grid = np.clip(grid, 0.0, None)
        grid /= grid.sum()   # renormalise: absorbs truncation + correction

        hg, ag = np.unravel_index(grid.argmax(), grid.shape)
        return {
            "home_win": float(np.tril(grid, -1).sum()),
            "draw": float(np.trace(grid)),
            "away_win": float(np.triu(grid, 1).sum()),
            "exp_home_goals": float(lh),
            "exp_away_goals": float(la),
            "likely_score": (int(hg), int(ag)),
        }


def fit_dixon_coles(matches, cutoff=None, half_life_days=HALF_LIFE_DAYS,
                    ridge=RIDGE, verbose=False):
    """Weighted Dixon-Coles MLE: attack, defence, intercept, home adv and rho.

        lambda_home = exp(c + atk_home - def_away + gamma)
        lambda_away = exp(c + atk_away - def_home)

    rho is fitted INSIDE the likelihood rather than hardcoded -- that is what
    makes this Dixon-Coles rather than Poisson with an arbitrary tweak.

    The intercept c is NOT cosmetic. Without it the model has exactly one flat
    direction (atk += k, def += k) but two sum-to-zero constraints, so the second
    binds on real structure and gamma is forced to carry both the overall goal
    level and the home effect. Measured symptom: away goals under-predicted by
    5.3% in both divisions and the home/away ratio inflated to 1.303 vs an actual
    1.213. Adding c creates a second genuine flat direction (c += k, atk -= k),
    so the two constraints become exactly identifying rather than one too many.
    """
    matches = matches.dropna(subset=["home_goals", "away_goals"])
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    hi = matches["home_team"].map(idx).to_numpy()
    ai = matches["away_team"].map(idx).to_numpy()
    hg = matches["home_goals"].to_numpy().astype(int)
    ag = matches["away_goals"].to_numpy().astype(int)

    if cutoff is None:
        cutoff = matches["date"].max()
    w = time_weights(matches["date"], cutoff, half_life_days)

    # params: [attack (n), defence (n), intercept, home_adv, rho]
    init = np.concatenate([np.zeros(n), np.zeros(n), [np.log(1.35)], [0.25], [-0.05]])
    bounds = [(-3, 3)] * (2 * n) + [(-2, 2), (-1, 1), RHO_BOUNDS]

    # constant across iterations, so hoist it out of the likelihood
    log_fact = gammaln(hg + 1.0) + gammaln(ag + 1.0)

    def neg_ll(p):
        atk, dfn = p[:n], p[n:2 * n]
        c, adv, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]

        lh = np.exp(c + atk[hi] - dfn[ai] + adv)
        la = np.exp(c + atk[ai] - dfn[hi])

        tau, dt_dlh, dt_dla, dt_drho = _tau_and_grads(hg, ag, lh, la, rho)
        tau = np.clip(tau, 1e-10, None)

        ll = (hg * np.log(lh) - lh + ag * np.log(la) - la - log_fact + np.log(tau))
        pen = SUM_ZERO_PENALTY * (atk.sum() ** 2 + dfn.sum() ** 2)
        pen += ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))
        obj = -(w * ll).sum() + pen

        # d(loglik)/d(log rate), so the chain rule to attack/defence is just +/-1
        gh = w * (hg - lh + lh * dt_dlh / tau)
        ga = w * (ag - la + la * dt_dla / tau)

        g_atk = np.bincount(hi, gh, n) + np.bincount(ai, ga, n)
        g_dfn = -np.bincount(ai, gh, n) - np.bincount(hi, ga, n)

        grad = np.empty_like(p)
        grad[:n] = -g_atk + 2 * SUM_ZERO_PENALTY * atk.sum() + 2 * ridge * atk
        grad[n:2 * n] = -g_dfn + 2 * SUM_ZERO_PENALTY * dfn.sum() + 2 * ridge * dfn
        grad[2 * n] = -(gh + ga).sum()            # intercept
        grad[2 * n + 1] = -gh.sum()               # home advantage
        grad[2 * n + 2] = -(w * dt_drho / tau).sum()
        return obj, grad

    res = minimize(neg_ll, init, method="L-BFGS-B", jac=True, bounds=bounds,
                   options={"maxiter": 5000, "maxfun": 100000})
    if verbose and not res.success:
        print(f"  [optimiser note] {res.message}")

    ratings = pd.DataFrame({
        "team": teams,
        "attack": res.x[:n],
        "defence": res.x[n:2 * n],
    })
    ratings["strength"] = ratings["attack"] + ratings["defence"]

    return DixonColesFit(
        ratings=ratings,
        intercept=float(res.x[2 * n]),
        home_adv=float(res.x[2 * n + 1]),
        rho=float(res.x[2 * n + 2]),
        ess=float(w.sum()),
        n_matches=len(matches),
        converged=bool(res.success),
    )


def league_of(matches):
    """Each team's most recent division within `matches`."""
    out = {}
    for _, r in matches.sort_values("date").iterrows():
        out[r["home_team"]] = r["league"]
        out[r["away_team"]] = r["league"]
    return out


def division_gap(fit, matches):
    """The LEARNED Prem-Champ gap, reported two ways.

    Returned in log-goal units and as an Elo-equivalent, so it can be compared
    directly against the 232 the Elo engine imposes as a fixed prior. Here the
    gap is estimated from data, so agreement is genuine corroboration.
    """
    lg = league_of(matches)
    r = fit.ratings.copy()
    r["league"] = r["team"].map(lg)
    prem = r[r.league == "Prem"]
    champ = r[r.league == "Champ"]
    if prem.empty or champ.empty:
        return None

    gap = prem["strength"].mean() - champ["strength"].mean()

    # average Prem side vs average Champ side on neutral ground
    lh = np.exp(fit.intercept + prem["attack"].mean() - champ["defence"].mean())
    la = np.exp(fit.intercept + champ["attack"].mean() - prem["defence"].mean())
    g = np.outer(poisson.pmf(np.arange(MAX_GOALS + 1), lh),
                 poisson.pmf(np.arange(MAX_GOALS + 1), la))
    g /= g.sum()
    win, draw = np.tril(g, -1).sum(), np.trace(g)
    exp_score = win + 0.5 * draw
    elo_equiv = -400 * np.log10(1 / min(max(exp_score, 1e-6), 1 - 1e-6) - 1)

    return {"gap_log": float(gap), "exp_score": float(exp_score),
            "elo_equivalent": float(elo_equiv),
            "n_prem": len(prem), "n_champ": len(champ)}


def attach_newcomer_prior(fit, matches, train_seasons, all_matches):
    """Prior for teams with NO training data (League One arrivals).

    Reference class: teams whose first tracked appearance inside the training
    window was in the Championship, i.e. genuinely up from League One. Teams
    RELEGATED from the Prem are also 'new to the Championship' but are much
    stronger, so including them would badly overrate a real League One arrival.
    """
    order = sorted(all_matches.season.unique(), key=lambda s: int(s[:2]))
    arrivals = []
    for s in train_seasons:
        i = order.index(s)
        if i == 0:
            continue
        prev = set(all_matches[all_matches.season == order[i - 1]]["home_team"]) | \
               set(all_matches[all_matches.season == order[i - 1]]["away_team"])
        cur = all_matches[(all_matches.season == s) & (all_matches.league == "Champ")]
        cur_teams = set(cur["home_team"]) | set(cur["away_team"])
        arrivals.extend(cur_teams - prev)

    known = fit.ratings.set_index("team")
    arrivals = [t for t in set(arrivals) if t in known.index]

    if len(arrivals) >= 3:
        fit.prior = (float(known.loc[arrivals, "attack"].mean()),
                     float(known.loc[arrivals, "defence"].mean()))
        fit.prior_source = f"mean of {len(arrivals)} League One arrivals"
    else:
        lg = league_of(matches)
        ch = known[known.index.map(lambda t: lg.get(t) == "Champ")]
        q = ch["strength"].quantile(0.25)
        weak = ch[ch["strength"] <= q]
        fit.prior = (float(weak["attack"].mean()), float(weak["defence"].mean()))
        fit.prior_source = f"fallback: bottom-quartile Championship (n={len(weak)})"
    return fit


def fit_for_league(all_matches, test_season, league, params=None,
                   extra=None, cutoff=None):
    """Fit the joint two-division model using `league`'s tuned hyperparameters.

    THE entry point for prediction. The harness and the backtest both go through
    here so that per-division parameters, the newcomer prior and (later) rating
    adjustments are applied identically everywhere -- rather than each caller
    reassembling the pipeline and drifting out of step.
    """
    p = params or LEAGUE_PARAMS[league]
    seasons, auto_cutoff = training_window(all_matches, test_season, p["window"])
    train = all_matches[all_matches["season"].isin(seasons)]

    # Mid-season, results already played THIS season are the most informative
    # data available -- and because they carry the newest dates they get the
    # heaviest time-decay weight. Excluding them (as the backtest deliberately
    # does) would be right for evaluation and wrong for a live forecast.
    if extra is not None and len(extra):
        keep = [c for c in ["date", "home_team", "away_team", "home_goals",
                            "away_goals", "league", "season"] if c in extra.columns]
        train = pd.concat([train, extra[keep]], ignore_index=True)

    cutoff = pd.Timestamp(cutoff) if cutoff is not None else auto_cutoff
    fit = fit_dixon_coles(train, cutoff=cutoff, half_life_days=p["half_life"])
    return attach_newcomer_prior(fit, train, seasons, all_matches)


def predict_fixtures(all_matches, fixtures, test_season):
    """Predict a set of fixtures, routing each to its division's model.

    `fixtures` needs home_team, away_team and league. Returns the same frame with
    prob_H / prob_D / prob_A and expected goals attached.
    """
    out = fixtures.copy()
    for col in ["prob_H", "prob_D", "prob_A", "exp_home_goals", "exp_away_goals"]:
        out[col] = np.nan
    for league, grp in fixtures.groupby("league"):
        fit = fit_for_league(all_matches, test_season, league)
        for i, r in grp.iterrows():
            p = fit.predict(r["home_team"], r["away_team"])
            out.loc[i, ["prob_H", "prob_D", "prob_A"]] = (
                p["home_win"], p["draw"], p["away_win"])
            out.loc[i, ["exp_home_goals", "exp_away_goals"]] = (
                p["exp_home_goals"], p["exp_away_goals"])
    return out


def training_window(all_matches, test_season, window=WINDOW_SEASONS):
    """The `window` seasons immediately before `test_season`, plus the cutoff date.

    `test_season` may be a season not present in `all_matches` -- which is the
    live case, where we are forecasting a season that has barely started. Then
    the window is simply the most recent `window` seasons and the caller supplies
    the cutoff.
    """
    order = sorted(all_matches.season.unique(), key=lambda s: int(s[:2]))
    if test_season in order:
        i = order.index(test_season)
        cutoff = all_matches[all_matches.season == test_season]["date"].min()
    else:
        i = len(order)
        cutoff = all_matches["date"].max()
    return order[max(0, i - window):i], cutoff


if __name__ == "__main__":
    m = load_matches().dropna(subset=["home_goals", "away_goals"])
    TEST = "2526"

    print("=" * 66)
    print(f"DIVISION-GAP STABILITY CHECK (test season {TEST})")
    print("=" * 66)
    print("If the LEARNED gap is stable across windows and lands near the Elo")
    print("engine's imposed 232, two independent methods agree.\n")
    print(f"{'window':>7} {'matches':>8} {'ESS':>7} {'c':>6} {'home_adv':>9} {'rho':>7} "
          f"{'gap(log)':>9} {'Elo-equiv':>10} {'conv':>5}")
    print("-" * 78)

    fits = {}
    for wnd in [3, 5, 8]:
        seasons, cutoff = training_window(m, TEST, wnd)
        tr = m[m.season.isin(seasons)]
        f = fit_dixon_coles(tr, cutoff=cutoff)
        g = division_gap(f, tr)
        fits[wnd] = (f, g, tr)
        print(f"{wnd:>6}y {f.n_matches:>8} {f.ess:>7.0f} {f.intercept:>6.3f} "
              f"{f.home_adv:>9.3f} {f.rho:>7.3f} {g['gap_log']:>9.3f} "
              f"{g['elo_equivalent']:>10.0f} {str(f.converged):>5}")

    print("\n(Elo engine's imposed gap, for comparison: 232)")

    # --- goal-level calibration: catches a missing/mis-specified intercept ---
    print("\n" + "=" * 66)
    print("GOAL-LEVEL CALIBRATION (predicted vs actual mean goals)")
    print("=" * 66)
    print(f"{'window':>7} {'home pred':>10} {'home act':>9} {'away pred':>10} "
          f"{'away act':>9} {'ratio p/a':>10}")
    print("-" * 60)
    for wnd, (f, g, tr) in fits.items():
        atk = f.ratings.set_index("team")["attack"].to_dict()
        dfn = f.ratings.set_index("team")["defence"].to_dict()
        lh = np.exp(f.intercept + tr.home_team.map(atk).to_numpy()
                    - tr.away_team.map(dfn).to_numpy() + f.home_adv)
        la = np.exp(f.intercept + tr.away_team.map(atk).to_numpy()
                    - tr.home_team.map(dfn).to_numpy())
        print(f"{wnd:>6}y {lh.mean():>10.3f} {tr.home_goals.mean():>9.3f} "
              f"{la.mean():>10.3f} {tr.away_goals.mean():>9.3f} "
              f"{(lh.mean()/la.mean()):>5.3f}/{(tr.home_goals.mean()/tr.away_goals.mean()):.3f}")

    # --- sanity check the 5-season fit ---
    f, g, tr = fits[5]
    seasons, cutoff = training_window(m, TEST, 5)
    lg = league_of(tr)
    r = f.ratings.copy()
    r["league"] = r["team"].map(lg)
    prem = r[r.league == "Prem"].copy()

    print("\n" + "=" * 66)
    print("5-SEASON FIT: does it match reality?")
    print("=" * 66)
    print("\nTop 6 Premier League attacks:")
    for _, x in prem.nlargest(6, "attack").iterrows():
        print(f"   {x.team:26s} {x.attack:+.3f}")
    print("\nTop 6 Premier League defences (higher = better):")
    for _, x in prem.nlargest(6, "defence").iterrows():
        print(f"   {x.team:26s} {x.defence:+.3f}")

    # --- the KeyError blocker: promoted teams must now predict ---
    f = attach_newcomer_prior(f, tr, seasons, m)
    print(f"\nNewcomer prior: {f.prior_source} -> attack {f.prior[0]:+.3f}, "
          f"defence {f.prior[1]:+.3f}")

    print("\n" + "=" * 66)
    print("PROMOTED TEAMS (these used to raise KeyError)")
    print("=" * 66)
    for home, away in [("Burnley", "Liverpool"), ("Leeds United", "Sunderland"),
                       ("Arsenal", "Burnley"), ("Sunderland", "Manchester City")]:
        p = f.predict(home, away)
        print(f"\n{home} vs {away}")
        print(f"   {p['home_win']*100:5.1f}% / {p['draw']*100:5.1f}% / "
              f"{p['away_win']*100:5.1f}%   xG {p['exp_home_goals']:.2f}-"
              f"{p['exp_away_goals']:.2f}   likeliest {p['likely_score'][0]}-{p['likely_score'][1]}")
