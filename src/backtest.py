"""Phase 4 backtest: rolling out-of-sample evaluation against naive and market.

Design commitments, all of which exist to stop the headline number being a lie:

1. ROLLING, not a single split. Fit on everything before season X (within the
   window), predict X, repeat. One season is 380 matches -- far too noisy to
   resolve a 0.01 log-loss difference.

2. NESTED hyperparameter selection. Both the decay half-life AND the window
   length are chosen on early TUNE seasons only, then frozen and reported on
   held-out REPORT seasons. Choosing the window after seeing test results is the
   same leakage as tuning xi on them.

3. IDENTICAL match sets. Model and market are scored on exactly the same
   fixtures, and the count is printed. Missing odds cluster on obscure fixtures,
   which are the hard ones -- scoring the market on a shrunken set would hand it
   an easier exam.

4. CALIBRATION, not just a single number. Log loss can look respectable while
   the model is systematically overconfident.
"""
import numpy as np
import pandas as pd

import odds as O
from dixon_coles import (attach_newcomer_prior, fit_dixon_coles, training_window)
from paths import load_matches

OUTCOMES = ["H", "D", "A"]

# hyperparameter grid, searched on TUNE seasons only
WINDOW_GRID = [3, 5, 8]
HALF_LIFE_GRID = [180, 365, 730, 1460]

# Opening prices span the whole history; CLOSING prices (*C) exist only from
# 2019-20 but are the recognised benchmark -- they have absorbed team news and
# the market's own money right up to kick-off, and are measurably sharper.
MARKET_BOOKS = ["B365", "Avg", "B365C", "AvgC"]

# The first testable season needs max(WINDOW_GRID) seasons of history so every
# candidate window is evaluated on identical test seasons.
MIN_HISTORY = max(WINDOW_GRID)


def outcome_idx(result):
    return pd.Series(result).map({"H": 0, "D": 1, "A": 2}).to_numpy()


def log_loss(p, y):
    """Mean -log(probability assigned to what actually happened). Lower better."""
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, None)).mean())


def rps(p, y):
    """Ranked probability score for ordered H/D/A. Lower better.

    Works on cumulative probabilities, so being wrong by one step (predicting a
    draw when it was an away win) costs less than being wrong by two. Log loss
    treats those identically, which is why both are reported.
    """
    e = np.zeros_like(p)
    e[np.arange(len(y)), y] = 1.0
    cp, ce = np.cumsum(p, axis=1), np.cumsum(e, axis=1)
    return float((((cp[:, :-1] - ce[:, :-1]) ** 2).sum(axis=1) / (p.shape[1] - 1)).mean())


def predict_season(fit, test):
    """(n, 3) array of H/D/A probabilities for every match in `test`."""
    out = np.empty((len(test), 3))
    for i, (_, r) in enumerate(test.iterrows()):
        p = fit.predict(r["home_team"], r["away_team"])
        out[i] = (p["home_win"], p["draw"], p["away_win"])
    return out


def backtest_season(m, test_season, window, half_life):
    """Fit on the window before `test_season`, predict it. Returns test rows + probs."""
    seasons, cutoff = training_window(m, test_season, window)
    train = m[m.season.isin(seasons)]
    fit = fit_dixon_coles(train, cutoff=cutoff, half_life_days=half_life)
    fit = attach_newcomer_prior(fit, train, seasons, m)

    test = m[m.season == test_season].copy()
    probs = predict_season(fit, test)

    # base rates from TRAINING data only -- using test-season rates would leak
    counts = train["result"].value_counts()
    base = np.array([counts.get(o, 0) for o in OUTCOMES], dtype=float)
    base /= base.sum()
    return test, probs, base, fit


def evaluate(m, seasons, window, half_life, league=None, verbose=False):
    """Pooled log loss / RPS over `seasons`, model vs baselines, identical sets."""
    rows = []
    for s in seasons:
        test, probs, base, _ = backtest_season(m, s, window, half_life)
        if league:
            keep = (test["league"] == league).to_numpy()
            test, probs = test[keep], probs[keep]
        rows.append((test, probs, base))

    test = pd.concat([r[0] for r in rows])
    probs = np.vstack([r[1] for r in rows])
    base = np.vstack([np.tile(r[2], (len(r[0]), 1)) for r in rows])
    y = outcome_idx(test["result"])

    ok = test["result"].notna().to_numpy()
    test, probs, base, y = test[ok], probs[ok], base[ok], y[ok]

    res = {"n_model": len(test),
           "model_ll": log_loss(probs, y), "model_rps": rps(probs, y),
           "base_ll": log_loss(base, y), "base_rps": rps(base, y),
           "unif_ll": log_loss(np.full_like(probs, 1 / 3), y),
           "unif_rps": rps(np.full_like(probs, 1 / 3), y)}

    # --- market comparison on an IDENTICAL match set ---
    for book in MARKET_BOOKS:
        mask = O.has_odds(test, book).to_numpy()
        res[f"n_{book}"] = int(mask.sum())
        if mask.sum() == 0:
            continue
        mp = O.market_probs(test[mask], book)
        res[f"{book}_ll"] = log_loss(mp, y[mask])
        res[f"{book}_rps"] = rps(mp, y[mask])
        # model re-scored on exactly the same subset
        res[f"model_ll_vs_{book}"] = log_loss(probs[mask], y[mask])
        res[f"model_rps_vs_{book}"] = rps(probs[mask], y[mask])

    # --- book vs book, on the intersection ---
    # B365 and Avg cover different spans (Avg only from 2019-20), so their
    # headline numbers above sit on different match sets and cannot be compared
    # to each other. This block puts both books AND the model on one common set,
    # which is the only way to tell an edge against one book's pricing apart
    # from an edge against the market consensus.
    common = (O.has_odds(test, "B365") & O.has_odds(test, "Avg")).to_numpy()
    res["n_common"] = int(common.sum())
    if common.sum() > 0:
        res["common_B365_ll"] = log_loss(O.market_probs(test[common], "B365"), y[common])
        res["common_Avg_ll"] = log_loss(O.market_probs(test[common], "Avg"), y[common])
        res["common_B365_rps"] = rps(O.market_probs(test[common], "B365"), y[common])
        res["common_Avg_rps"] = rps(O.market_probs(test[common], "Avg"), y[common])
        res["common_model_ll"] = log_loss(probs[common], y[common])
        res["common_model_rps"] = rps(probs[common], y[common])
    return res, test, probs, y


def calibration_table(probs, y, nbins=10):
    """Bin predicted probability vs realised frequency, all outcomes pooled."""
    p = probs.reshape(-1)
    e = np.zeros_like(probs)
    e[np.arange(len(y)), y] = 1.0
    hit = e.reshape(-1)
    b = pd.cut(p, np.arange(0, 1.01, 1 / nbins))
    d = pd.DataFrame({"p": p, "hit": hit, "bin": b})
    g = d.groupby("bin", observed=True).agg(n=("hit", "size"), pred=("p", "mean"),
                                            actual=("hit", "mean"))
    g["diff"] = g["actual"] - g["pred"]
    g["se"] = np.sqrt(g["pred"] * (1 - g["pred"]) / g["n"])
    g["z"] = g["diff"] / g["se"]
    return g


if __name__ == "__main__":
    m = load_matches().dropna(subset=["home_goals", "away_goals"])
    order = sorted(m["season"].unique(), key=lambda s: int(s[:2]))
    testable = order[MIN_HISTORY:]
    split = len(testable) // 2
    TUNE, REPORT = testable[:split], testable[split:]

    print("=" * 74)
    print("PHASE 4 BACKTEST")
    print("=" * 74)
    print(f"Testable seasons : {len(testable)}  ({testable[0]} -> {testable[-1]})")
    print(f"TUNE   (hyperparameters chosen here) : {len(TUNE)}  {TUNE[0]} -> {TUNE[-1]}")
    print(f"REPORT (held out, never seen)        : {len(REPORT)}  {REPORT[0]} -> {REPORT[-1]}")

    # ---------- 1. nested tuning: window AND half-life, on TUNE only ----------
    print("\n" + "=" * 74)
    print("1. HYPERPARAMETER SEARCH  (Premier League log loss on TUNE seasons)")
    print("=" * 74)
    print("Both window and half-life are selected here. Selecting either after")
    print("seeing REPORT results would be leakage.\n")
    print(f"{'window':>7} {'half-life':>10} {'log loss':>10} {'RPS':>8}")
    print("-" * 38)

    grid = []
    for wnd in WINDOW_GRID:
        for hl in HALF_LIFE_GRID:
            r, *_ = evaluate(m, TUNE, wnd, hl, league="Prem")
            grid.append({"window": wnd, "half_life": hl,
                         "ll": r["model_ll"], "rps": r["model_rps"]})
            print(f"{wnd:>6}y {hl:>9}d {r['model_ll']:>10.4f} {r['model_rps']:>8.4f}")

    G = pd.DataFrame(grid).sort_values("ll").reset_index(drop=True)
    BEST_W = int(G.loc[0, "window"])
    BEST_HL = int(G.loc[0, "half_life"])
    print(f"\n>>> SELECTED: window {BEST_W} seasons, half-life {BEST_HL} days "
          f"(TUNE log loss {G.loc[0, 'll']:.4f})")
    print("    Now frozen. Everything below is out-of-sample.")

    # ---------- 2. held-out results, by division ----------
    for league in ["Prem", "Champ"]:
        print("\n" + "=" * 74)
        print(f"2. HELD-OUT RESULTS - {league}  ({REPORT[0]} -> {REPORT[-1]})")
        print("=" * 74)
        r, test, probs, y = evaluate(m, REPORT, BEST_W, BEST_HL, league=league)

        print(f"\nMatch counts (model and market MUST be scored on the same set):")
        print(f"  model scored on          : {r['n_model']}")
        print(f"  B365 available           : {r.get('n_B365', 0)}")
        print(f"  Avg  available           : {r.get('n_Avg', 0)}")

        print(f"\n{'predictor':<28} {'log loss':>10} {'RPS':>9} {'n':>7}")
        print("-" * 57)
        print(f"{'uniform (1/3 each)':<28} {r['unif_ll']:>10.4f} {r['unif_rps']:>9.4f} "
              f"{r['n_model']:>7}")
        print(f"{'base rates (train only)':<28} {r['base_ll']:>10.4f} {r['base_rps']:>9.4f} "
              f"{r['n_model']:>7}")
        print(f"{'OUR MODEL':<28} {r['model_ll']:>10.4f} {r['model_rps']:>9.4f} "
              f"{r['n_model']:>7}")
        for book in MARKET_BOOKS:
            if f"{book}_ll" not in r:
                continue
            print(f"\n  -- vs {book}, identical {r[f'n_{book}']} matches --")
            print(f"{'  market ' + book:<28} {r[f'{book}_ll']:>10.4f} "
                  f"{r[f'{book}_rps']:>9.4f} {r[f'n_{book}']:>7}")
            print(f"{'  our model (same set)':<28} {r[f'model_ll_vs_{book}']:>10.4f} "
                  f"{r[f'model_rps_vs_{book}']:>9.4f} {r[f'n_{book}']:>7}")
            d = r[f"model_ll_vs_{book}"] - r[f"{book}_ll"]
            print(f"{'  difference':<28} {d:>+10.4f} "
                  f"{r[f'model_rps_vs_{book}'] - r[f'{book}_rps']:>+9.4f}")
            print(f"   ({'model worse' if d > 0 else 'MODEL BETTER - suspect leakage'} "
                  f"by {abs(d):.4f} log loss)")

        # ---------- 2b. B365 vs Avg on a common set ----------
        if r.get("n_common", 0) > 0:
            print(f"\n  -- B365 vs Avg vs model, common {r['n_common']} matches --")
            print(f"  {'  market B365':<26} {r['common_B365_ll']:>10.4f} "
                  f"{r['common_B365_rps']:>9.4f}")
            print(f"  {'  market Avg (consensus)':<26} {r['common_Avg_ll']:>10.4f} "
                  f"{r['common_Avg_rps']:>9.4f}")
            print(f"  {'  our model':<26} {r['common_model_ll']:>10.4f} "
                  f"{r['common_model_rps']:>9.4f}")
            print(f"   model - B365 : {r['common_model_ll'] - r['common_B365_ll']:+.4f} LL")
            print(f"   model - Avg  : {r['common_model_ll'] - r['common_Avg_ll']:+.4f} LL")
            print(f"   Avg  - B365  : {r['common_Avg_ll'] - r['common_B365_ll']:+.4f} LL "
                  f"(which book prices better)")

        # ---------- 3. per-season stability ----------
        print(f"\n  Per-season ({league}) - is any edge stable or a couple of good years?")
        print(f"  {'season':>7} {'n':>5} {'model LL':>9} {'B365 LL':>9} {'diff':>8} "
              f"{'model RPS':>10} {'B365 RPS':>9}")
        print("  " + "-" * 62)
        for s in REPORT:
            rs, *_ = evaluate(m, [s], BEST_W, BEST_HL, league=league)
            if "B365_ll" not in rs:
                continue
            d = rs["model_ll_vs_B365"] - rs["B365_ll"]
            print(f"  {s:>7} {rs['n_B365']:>5} {rs['model_ll_vs_B365']:>9.4f} "
                  f"{rs['B365_ll']:>9.4f} {d:>+8.4f} "
                  f"{rs['model_rps_vs_B365']:>10.4f} {rs['B365_rps']:>9.4f}")

        # ---------- 4. calibration ----------
        print(f"\n  CALIBRATION ({league}) - are our probabilities honest?")
        print("  Overconfidence shows as actual < predicted at the high end.")
        cal = calibration_table(probs, y)
        print("  " + cal.round(4).to_string().replace("\n", "\n  "))
        big = cal[(cal["n"] > 100) & (cal["z"].abs() > 3)]
        print(f"\n  bins with n>100 and |z|>3: {len(big)}"
              f"{' -> ' + ', '.join(map(str, big.index.tolist())) if len(big) else ' (none)'}")
