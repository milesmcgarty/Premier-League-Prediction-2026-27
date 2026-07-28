"""Phase 5: blending model probabilities with the market.

Method: a log-opinion pool, i.e. a weighted geometric mean, renormalised.

    log p_blend  ~  w * log p_model + (1 - w) * log p_market

Log space rather than linear because a component that assigns near-zero
probability to an outcome should be able to VETO it, not merely be averaged
away. w = 1 is model-only, w = 0 is market-only.

SCOPE LIMIT -- read this before relying on the blend:
odds only exist for fixtures a bookmaker has priced, which in practice means the
next round or two. Simulating a full season in August requires predicting May
fixtures that nobody has priced. So:

    match predictions for the coming week  ->  blended
    season simulation                      ->  model-only, necessarily

That is the boundary of the method, not a defect. It also means the blend does
NOT reduce the value of improving the model itself: the season simulator runs on
unblended probabilities all year, so model quality drives every title and
relegation number the project produces.

RESULT (2026-07-28) -- THE BLEND DOES NOT WORK. Tuned on TUNE seasons, evaluated
on held-out REPORT seasons, against the opening line:

    Premier League   model 0.9868   market 0.9578   blend(w=0.10) 0.9583
    Championship     model 1.0640   market 1.0363   blend(w=0.00) 1.0363

The Premier League blend is 0.0005 WORSE than simply using the market, and the
Championship optimiser set w=0, discarding the model outright. The TUNE curve is
flat across w = 0.00-0.20 (0.9653 / 0.9652 / 0.9652 / 0.9652 / 0.9654), so the
selected weight was arbitrary within noise.

Conclusion, stated plainly: our model carries essentially no information the
market does not already have. Not "the weight leans on the market" -- the
marginal contribution is indistinguishable from zero.

The code is kept because the negative result is worth preserving and re-running
once the model improves (the promoted-team / transfer-value work is the obvious
candidate). If the model ever does add something, w will move off zero and this
is how we will find out.

NOTE: the blend could NOT be tuned against the closing line. Closing odds begin
in 2019-20, which sits entirely inside the REPORT window, so there are no
closing-odds TUNE seasons; tuning on report seasons would be leakage. Since the
opening-line result is decisive and closing is sharper still, this does not
change the conclusion.
"""
import numpy as np

EPS = 1e-15


def blend_probs(p_model, p_market, w):
    """Weighted geometric mean of two probability arrays, renormalised.

    Rows where the market is missing (any NaN) fall back to the model, so this
    can be applied to a mixed set of fixtures safely.
    """
    pm = np.clip(np.asarray(p_model, dtype=float), EPS, None)
    pk = np.asarray(p_market, dtype=float)

    missing = ~np.isfinite(pk).all(axis=1)
    pk_safe = np.where(np.isfinite(pk), pk, EPS)
    pk_safe = np.clip(pk_safe, EPS, None)

    log_b = w * np.log(pm) + (1.0 - w) * np.log(pk_safe)
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    b /= b.sum(axis=1, keepdims=True)

    b[missing] = pm[missing] / pm[missing].sum(axis=1, keepdims=True)
    return b


def tune_weight(p_model, p_market, y, grid=None):
    """Pick w by log loss. MUST be called on tuning data only, never on the
    seasons the result is reported from."""
    from backtest import log_loss
    grid = np.arange(0, 1.001, 0.05) if grid is None else np.asarray(grid)
    scores = [(float(w), log_loss(blend_probs(p_model, p_market, w), y)) for w in grid]
    best = min(scores, key=lambda t: t[1])
    return best[0], scores
