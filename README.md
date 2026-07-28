# Premier League Prediction System

A from-scratch football forecasting system for the English Premier League: an Elo
power rating, a Dixon-Coles style goal model, and a rolling out-of-sample backtest
scored against bookmaker odds.

Built to make genuine predictions for the 2026-27 season, not just to demonstrate
technique — so every component is validated against something external rather than
declared correct.

---

## Headline result

Rolling backtest, **9 held-out seasons** (2017-18 → 2025-26, **3,420 Premier League
matches**). Hyperparameters were selected on earlier seasons only and frozen before
these were scored.

| Predictor | Log loss | RPS |
|---|---:|---:|
| Uniform (⅓ each) | 1.0986 | 0.2388 |
| Base rates (from training data only) | 1.0682 | 0.2333 |
| **This model** | **0.9868** | **0.2050** |
| Bookmaker market (Bet365 **opening**, de-vigged) | 0.9578 | 0.1955 |

**The model beats the base-rate baseline by 0.081 log loss.** It does not beat the
market in any of the nine seasons — which is the expected result, and the reassuring
one. A goals-only model with no injury, lineup or expected-goals data should not
outprice a bookmaker; a result claiming otherwise would be evidence of a bug, not of
skill.

### Measured against the closing line

Opening odds are a softer benchmark than they look. The **closing** line has absorbed
team news, weather and the market's own money right up to kick-off, and is the
recognised standard. Closing prices are available from 2019-20, so on those 7 seasons:

| Benchmark | Log loss | Our gap |
|---|---:|---:|
| Bet365 opening | 0.9689 | +0.0322 |
| Bet365 **closing** | 0.9640 | **+0.0370** |
| Market-average **closing** (sharpest) | 0.9639 | **+0.0371** |

**So the honest figure is 0.037 behind the closing line, not 0.029.** The closing
line is ~0.005 log loss sharper than the open, which is itself a small validation
that the de-vigging and comparison machinery are behaving sensibly.

**The probabilities are calibrated, not just accurate on average.** Binning
predicted probability against realised frequency, no bin with n>100 deviates by
more than |z| = 2. A model can post a respectable log loss while being
systematically overconfident, so this is checked explicitly.

---

## What it does

Three levels, two of them built:

- **Match level** — win/draw/loss probabilities, expected goals, likeliest scoreline
- **Team level** — Elo power ratings across two divisions and 26 seasons
- **Season level** — Monte Carlo simulation for title/top-four/relegation *(planned)*

---

## Architecture

```
football-data.co.uk CSVs  ─┐
                           ├─→  load_results.py  ─→  matches_combined.csv
teams.csv (name mapping)  ─┘                            24,232 matches
                                                        2000-2026, 2 divisions
                                     ┌──────────────────────┴───────────┐
                                     ↓                                  ↓
                          elo.py / build_elo.py              dixon_coles.py
                          Elo + cross-division                Poisson goal model
                          re-anchoring                        joint two-division fit
                                     │                                  │
                                     ↓                                  ↓
                          benchmark_elo.py                      backtest.py
                          validated vs ClubElo                  rolling, nested,
                          (Spearman 0.955)                      vs market baseline
```

### The Elo engine

Standard Elo with home advantage and a goal-difference multiplier, replayed over all
24,232 matches. The hard part is that the Premier League and Championship pools drift
apart over 25 years — they swap only three teams a season and never play each other,
so promoted sides arrived looking like top-six Premier League teams.

The fix is **per-season league re-anchoring**: the entire Championship pool slides as
a rigid block so its mean sits a fixed distance below the Premier League's, preserving
within-league order while removing the inter-league drift.

That distance was **measured, not assumed** — 232 Elo points, derived from the
points-per-game change across 75 promotions and 75 relegations.

**Validation:** Spearman rank correlation **0.955** against
[ClubElo](http://clubelo.com), an established independent rating system, across 43
shared clubs.

### The goal model

Attack and defence ratings per team by weighted maximum likelihood:

```
λ_home = exp(c + attack_home − defence_away + γ)
λ_away = exp(c + attack_away − defence_home)
```

fitted jointly across both divisions with exponential time decay, plus the
Dixon-Coles low-score correction term ρ.

**The interesting problem here is cross-division identification.** Adding a constant
to every Championship team's attack *and* defence leaves every within-Championship
prediction unchanged — so the relative scale of the two divisions is a flat direction
in the likelihood. Since league fixtures are never cross-division, the *only* thing
that identifies it is teams which played in both divisions inside the training window.

Measured: a one-season training window contains **zero** such teams and is formally
unidentified. Three seasons has 7, five has 14. Hence a five-season default, and a
deliberately gentle time decay — an aggressive half-life strips out precisely the
older-division matches carrying the linking information.

**The result cross-validates the Elo work.** The goal model *learns* a division gap
of 267 Elo-equivalent points; the Elo engine *independently derives* 232 from
points-per-game data by a completely different route. Two methods, one estimated and
one measured, landing in the same place.

---

## Things that turned out to be wrong

Kept here deliberately — the debugging is the part worth reading.

**The Dixon-Coles correction does nothing on modern data.** ρ is fitted rather than
hardcoded, and comes out at ~+0.006, i.e. nil. The classic 1997 motivation —
that Poisson under-predicts low-scoring draws — does not replicate on 2020s football:
Poisson predicts the overall draw rate to within 0.3%. The residual is also the wrong
*shape* for the correction (0-0 over-predicted, 1-1 under-predicted, pulling ρ in
opposite directions). Hardcoding the textbook −0.05 would have made predictions
worse.

**A ridge penalty silently destroyed the main result.** Shrinking attack/defence
toward zero also shrinks the *gap between divisions*, because Championship teams sit
below zero and Premier League teams above — it biased the exact quantity the joint
fit exists to estimate, collapsing the learned gap from 266 to 73 Elo-equivalent and
making promoted teams look far stronger than they are.

**A passing sanity check hid a misspecified model.** An earlier fit recovered a
home advantage of 0.33, which looked right and was accepted. It was inflated: the
model had no intercept term, so the home-advantage parameter was absorbing the
overall goal level as well. The tells were only visible in checks that could fail
*structurally* — away goals under-predicted by 5.3%, and the parameter drifting with
window length when a stable parameter shouldn't care. The corrected value is ~0.21.

---

## Honest limitations

- **The Championship model is weak.** Log loss 1.0658 against a base rate of 1.0777 —
  it adds very little, and is measurably overconfident in the 0.5–0.6 probability
  band (predicted 0.541, realised 0.481, z = −4.05). Reported rather than buried.
- **The gap to the market is widening**, and the cause is now identified: it is almost
  entirely **promoted teams**. On held-out seasons the deficit is +0.047 on matches
  involving a promoted side versus +0.022 on established sides only, and the widening
  trend is significant for the former (slope +0.011/season, p=0.014) but not the
  latter (p=0.48). In 2025-26 the split was +0.141 against +0.012.

  Notably, the market's own absolute performance is **flat** over the same period
  (p=0.30), so this is not "bookmakers got better" — it is our Championship-derived
  ratings failing to price newly promoted squads, which are reshaped by transfer
  spending the model cannot see.
- **No expected-goals, injury, lineup or transfer data yet.** Goals only. This is the
  main reason the model trails the market.
- **The hyperparameter surface is nearly flat** (0.9796–0.9841 across all twelve
  combinations tried), so the selected window and half-life shouldn't be read as
  meaningful — they're within noise of each other.

---

## Running it

```powershell
py -m venv venv
venv\Scripts\activate
py -m pip install -r requirements.txt

py src\load_results.py    # build the clean match table from raw CSVs
py src\build_elo.py       # replay all matches, produce Elo ratings
py src\dixon_coles.py     # fit the goal model, division-gap diagnostics
py src\backtest.py        # the full rolling backtest (the headline number)
py src\benchmark_elo.py   # validate Elo against ClubElo (requires network)
```

Everything except `benchmark_elo.py` runs offline from data in the repo.

---

## Roadmap

| Phase | Status |
|---|---|
| 1. Data sourcing & catalogue | ✅ |
| 2. Clean results loader (24,232 matches, table-verified) | ✅ |
| 3. Elo engine (validated vs ClubElo, 0.955) | ✅ |
| 4. Dixon-Coles model + backtest | ✅ |
| 5. Market blend (model + odds in logit space) | planned |
| 6. Season simulator (Monte Carlo) | planned |
| 7. Live 2026-27 harness, weekly re-runs with snapshots | planned |
| 8. Expected-goals layer (Understat), A/B tested on the backtest | planned |
| 9. Player-level model (minutes, goals, assists) | planned |

---

## Data sources & attribution

- **[football-data.co.uk](https://www.football-data.co.uk/)** — results, match stats
  and bookmaker odds, 2000-2026. The historical spine of the project.
- **[ClubElo](http://clubelo.com)** — independent Elo ratings, used as an external
  benchmark.
- **[Understat](https://understat.com)** — expected goals *(Phase 8)*.
- **[Transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)** —
  transfers and market values. The ~195 MB DuckDB file is **not** committed; download
  it from the upstream project.
- Accessed via [`soccerdata`](https://github.com/probberechts/soccerdata).

Odds are de-vigged by proportional normalisation. Residual miscalibration is
consistent with the well-documented favourite–longshot bias rather than an error in
the de-vigging: longshots are over-priced and favourites under-priced, with the sign
flipping cleanly across the probability range.

---

## Licence

MIT — see [LICENSE](LICENSE). Third-party data remains subject to the terms of its
original providers.
