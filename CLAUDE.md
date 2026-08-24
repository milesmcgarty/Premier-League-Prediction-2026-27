# Premier League Prediction Project — Context & Handoff

This file catches you up on the whole project: what it is, how it's built, every
significant decision and *why* it was made, what's done, what's next, and the
gotchas that will bite if you don't know about them. Read it fully before making
changes.

---

## 1. What this project is

A personal "supercomputer"-style prediction system for the English Premier League,
built by Miles — a recent BSc Computer Science graduate (First Class, Reading)
job-hunting in sports/data analytics. It's a portfolio piece AND intended to make
genuinely usable real-world predictions for the 2026-27 season.

The goal is to predict, at three levels:
- **Team level**: full-season simulation — title / top-4 / relegation probabilities,
  final points and position distributions.
- **Match level**: win/draw/loss probabilities, expected scorelines, per fixture.
- **Player level** (later phase): goals, assists, minutes, availability.

And critically: to **update through the live 2026-27 season** — feed in real results
each week, re-run, watch the predictions evolve.

The architectural target is Opta's real methodology (which Miles researched): an
Elo-derived power rating + market odds blend + Monte Carlo simulation. Not a
black-box neural net. We match that shape and extend it.

---

## 2. Working style — IMPORTANT, read this

Miles wants to **work incrementally and understand every step**, not have big blocks
of code dumped on him. The established rhythm across the whole project:

1. Explain the concept/maths first, plainly, before writing code.
2. Build in **small pieces** — one function or one idea at a time.
3. He runs each piece himself (Windows, PowerShell, `py` command) and pastes output.
4. We verify the output is correct *before* moving on — every phase has a concrete
   correctness test, not "looks reasonable."
5. Then the next small piece.

He values **honesty over cheerleading**. When something's wrong, say so plainly.
When a plan is over-engineered, push back. When a result looks good but has a
caveat, flag the caveat. Several times this project, checking a result rather than
trusting it caught a real bug. Keep that discipline.

He also wants things to be **impressive to sports-analytics employers** — which we've
consistently interpreted as "rigorous, tested, and defensible," not "maximally
complex." Five features that provably improve a backtest beat twenty that don't.

**Environment note:** Windows, VS Code, PowerShell. Python launched via `py` (not
`python` — the `python` alias is broken on his machine). Virtual env at `venv/`.
Watch for Windows path gotchas (backslash escaping — always use `pathlib.Path` or
forward slashes; `\r`, `\t` in string paths caused real errors early on).

---

## 3. Repository structure

```
Premier League Prediction Project/
├── data/
│   ├── raw/
│   │   ├── results/              # 52 football-data.co.uk CSVs (Prem + Champ, 2000-2026)
│   │   │                         #   named PremYYYY.csv / ChampYYYY.csv (e.g. Prem2526.csv)
│   │   ├── exploration/          # throwaway recon scripts (kept for provenance)
│   │   └── transfermarkt-datasets.duckdb   # transfer/player/market-value DB (GITIGNORED, ~195 MB)
│   ├── reference/
│   │   └── teams.csv             # THE join spine — curated, permanent
│   └── processed/
│       ├── matches_combined.csv  # THE clean match table — 24,232 matches, canonical names
│       ├── elo_ratings.csv       # final Elo per team (71 teams)
│       └── elo_history.csv       # every team's rating after every match (~48k rows)
├── src/
│   ├── paths.py                  # single source of truth for paths + shared helpers
│   ├── odds.py                   # Phase 4: de-vigging + the comparison-set guard
│   ├── backtest.py               # Phase 4: THE backtest — rolling, nested, calibrated
│   ├── load_results.py           # Phase 2: builds matches_combined.csv
│   ├── elo.py                    # Phase 3: the EloEngine class (core maths)
│   ├── build_elo.py              # Phase 3: replays all matches, builds+saves ratings
│   ├── benchmark_elo.py          # Phase 3: validates against ClubElo (rank correlation)
│   ├── measure_gap.py            # Phase 3: measured the cross-division gap empirically
│   ├── convert_gap.py            # Phase 3: converted that gap to Elo points (=232)
│   ├── dixon_coles.py            # Phase 4: fits attack/defence, predicts matches
│   ├── blend.py                  # Phase 5: market blend - TESTED, DOES NOT HELP
│   ├── simulate.py               # Phase 6: Monte Carlo season simulation
│   ├── fixtures.py               # Phase 7: 2026-27 fixtures+results from the FPL API
│   ├── harness.py                # Phase 7: THE weekly run - snapshot the forecast
│   └── validate_harness.py       # Phase 7: dry-run the harness over a done season
├── data/fixtures/2627.csv        # the live fixture+results table
├── data/snapshots/2627/<date>/   # dated, immutable weekly forecasts
└── venv/                         # GITIGNORED
```

**Paths: import them, don't retype them.** `src/paths.py` holds every project path
(`TEAMS_CSV`, `MATCHES_CSV`, `RESULTS_DIR`, …) plus two helpers worth knowing:
- `load_matches()` — reads matches_combined.csv with `dtype={"season": str}` and
  parsed dates already applied. **Use this instead of a bare `pd.read_csv`** so the
  "0001" → int 1 gotcha can't be forgotten.
- `active_teams(matches, season=, league=)` — the set of teams that actually played.
  Filter Elo ratings through this before ranking; dormant clubs have stale ratings.

The project is a git repo (`main`), pushed to
https://github.com/milesmcgarty/Premier-League-Prediction-2026-27

---

## 4. Data sources (all reconnaissance done, Phase 1 complete)

| Source | Access | Gives us | Key facts |
|---|---|---|---|
| **football-data.co.uk** | CSV downloads (have) | results, goals, shots, cards, referee, bookmaker odds | 2000-2026, Prem+Champ, the historical spine |
| **Transfermarkt-datasets** | DuckDB file (have) | transfers, fees, market values, player appearances | PL id = `GB1`, Champ = `GB2`; appearances from 2012; current to May 2026 |
| **Understat** (via `soccerdata`) | scraped (browser automation) | **xG**, non-penalty xG, PPDA, deep completions — match level | **PRIMARY xG source**. PL from 2014-15. See decision below. |
| **FBref** (via `soccerdata`) | scraped | shots, goals, cards, minutes | **Does NOT expose xG** in soccerdata 1.9.0 — only 5 stat types. See decision. |
| **FPL API** | free JSON, no key | live player prices, form, minutes, **injury/availability flags**, fixtures | Already on 26-27 season. Team names differ again (Man Utd, Spurs, Nott'm Forest) |
| **ClubElo** (via `soccerdata`) | free API | independent Elo ratings, all English tiers | Our benchmark; also `level` col = division. Used to validate our Elo (0.955 corr) |
| **StatsBomb open data** | GitHub, free | full event data (passes/shots/carries w/ coords) | Not current — 2015-16 PL + 2003-04 only. Future xT-model lab (Phase 10). |

**KEY DATA DECISION — xG source:** We planned to get xG from FBref. During recon we
discovered soccerdata 1.9.0's FBref reader only exposes 5 stat types (standard,
keeper, shooting, playing_time, misc) and **none contain xG**. Understat, however,
delivers clean match-level xG + npxG + PPDA, going back further (2014-15 vs FBref's
2017-18). So **Understat is promoted to primary xG source; FBref demoted to
shots/goals/cards backup.** This is a net positive — Understat's xG is more
established anyway and is match-level (better for our match-by-match loop).

**The join spine — `teams.csv`:** Every source spells team names differently
("Man United" / "Manchester United" / "Manchester Utd" / "Man Utd"). `teams.csv`
maps all of them to one `canonical_name` + a `team_id`. Columns: team_id,
canonical_name, football_data, transfermarkt, fbref, understat, clubelo, fpl.
71 clubs (every team appearing in Prem or Champ 2000-2026). **Every cross-source
join MUST route through this table.** It was validated: all names from all sources
map with zero unmapped stragglers.

---

## 5. Phase-by-phase progress

### Phase 1 — Data sourcing & catalogue — ✅ DONE
Full reconnaissance of all sources above. Confirmed each works, understood what each
provides, resolved the xG surprise, built `teams.csv`. No modelling code.

### Phase 2 — Clean results loader — ✅ DONE
`src/load_results.py` builds `data/processed/matches_combined.csv`:
- Loads all 52 result CSVs, handles encoding fallbacks (utf-8-sig → latin1) and
  ragged rows (some older files have trailing junk columns — use `usecols` on the
  header, NOT `on_bad_lines="skip"` which silently drops good matches).
- Keeps ~31 useful columns: match facts (date, teams, goals, result, half-time),
  match stats (shots, SoT, corners, fouls, cards, referee), and odds (B365, WH, Avg
  — kept where present; no single bookmaker spans all seasons, so it's permissive).
- Maps team names → canonical via teams.csv. If any name fails to map it raises
  `ValueError` naming the offenders. (It previously only *printed* a warning —
  `.map()` then turned unmapped names into NaN and silently deleted the club.
  Fixed 2026-07-28; this matters most in Phase 7 when a newly promoted side
  arrives with a spelling teams.csv hasn't seen.)
- Result: **24,232 matches, 26 seasons, both leagues, 2000-08-12 to 2026-05-24.**

**CORRECTNESS TEST (passed):** Reconstructed historical league tables from the clean
data and checked against reality — Man City 93pts (21-22), Leicester 81 (15-16),
Man City 91 (23-24), Liverpool 84 (24-25), all EXACT, full tables correct
top-to-bottom. This proves no matches are duplicated/dropped/mis-scored.

**Bugs caught & fixed here:** (a) 3 older files silently lost 45+45+32 matches to
`on_bad_lines="skip"` — fixed via header-based `usecols`. (b) Season codes like
"0001" were being read as int 1 — fixed with `.zfill(4)` / `dtype={"season": str}`.
Always read matches_combined.csv with `dtype={"season": str}`.

### Phase 3 — Elo engine — ✅ DONE (externally validated)
`src/elo.py` (EloEngine class) + `src/build_elo.py` (replay + save).

The engine: standard Elo + home advantage (fitted ~70 pts) + goal-difference
multiplier (bigger wins move ratings more, standard football-Elo formula). Core
maths unit-tested (equal teams → +10 on a win; strong beating weak → tiny gain).

**THE HARD PROBLEM — cross-division calibration.** Running Elo across both leagues,
promoted teams arrived *over-rated* (Coventry/Ipswich appeared top-6 Premier League).
Cause: the two league pools drift apart over 25 years because they're only weakly
connected (3 teams swap per year, never play directly). The Championship pool floats
up relative to the Premier League.

We tried and REJECTED a per-team promotion/relegation "jolt" (−232 on promotion) —
it didn't work, because it patches one team at the crossing moment but can't undo
*pool-level* drift.

The FIX that worked: **per-season league re-anchoring** (`reanchor_leagues` in
build_elo.py). At each season boundary, slide the entire Championship pool as a rigid
block so its mean sits exactly 232 points below the Premier League mean. Every Champ
team moves by the same amount → within-league order/gaps preserved, only the
inter-league offset corrected. Drift can't accumulate.

**The 232 number was measured empirically, not guessed** (`measure_gap.py` +
`convert_gap.py`): across 75 promotions and 75 relegations, promoted teams lose
~0.94 PPG going up, relegated teams gain ~0.81 (combined 0.87 PPG gap ≈ 33 pts over
a season ≈ 232 Elo points). This empirical derivation is a portfolio highlight.

Two more fixes: (a) **new-entrant floor** — teams appearing for the first time
(e.g. Wrexham arriving from un-tracked League One) start at the *bottom* of their
league's pool, not the 1500 default, so they don't inherit an inflated rating.
(b) **final recenter to 1500** — cosmetic uniform shift so numbers sit on the
familiar scale.

**CORRECTNESS TEST (passed):** Man United trajectory rises through the Ferguson era
(peak ~1794 in 2007-08), declines after 2013, bottoms 2024-25 — proving Elo is a
"current strength" running value, not a career average. AND **benchmarked against
ClubElo: Spearman rank correlation 0.955** across 43 shared teams. That's the
external, objective validation — our independent engine ranks teams the same way an
established professional system does.

### Phase 4 — Dixon-Coles match prediction — ✅ DONE (backtested 2026-07-28)
`src/dixon_coles.py` (rebuilt 2026-07-28), `src/odds.py`, `src/backtest.py`.

**THE HEADLINE NUMBER.** Rolling backtest, 9 held-out seasons (1718 → 2526),
3,420 Premier League matches, hyperparameters frozen beforehand:

| Predictor | Log loss | RPS |
|---|---|---|
| Uniform (1/3 each) | 1.0986 | 0.2388 |
| Base rates (training only) | 1.0682 | 0.2333 |
| **Our model** | **0.9868** | **0.2050** |
| Market (B365, de-vigged) | 0.9578 | 0.1955 |

Beats base rates by 0.081 log loss; trails the market by 0.0289. **Never beats the
market in any of the 9 seasons** — the expected, reassuring outcome. Championship:
1.0658 vs base rate 1.0777 (only 0.0119 better — genuinely weak, see limitations).

Calibration is sound in the PL: **no bin with n>100 deviates by more than |z|=2.**
The Championship is NOT: the 0.5–0.6 bin predicts 0.541 and realises 0.481
(n=1119, z=−4.05), i.e. overconfident on favourites.

**`fit_dixon_coles`** fits, by weighted maximum likelihood over BOTH divisions and
multiple seasons:

```
lambda_home = exp(c + atk_home − def_away + gamma)
lambda_away = exp(c + atk_away − def_home)
```

attack and defence per team (both constrained to sum to zero), a global intercept
`c`, home advantage `gamma`, and the Dixon-Coles `rho` — all in one likelihood.

**The cross-division connectivity problem (the hard part, and it mirrors Elo's).**
Adding a constant to every Championship team's attack *and* defence leaves every
within-Championship lambda unchanged, so the relative scale of the two divisions is
a **flat direction in the likelihood**. League fixtures are never cross-division, so
the ONLY thing identifying it is teams that played in both divisions inside the
training window. Measured: a **1-season window has zero such teams** and is formally
unidentified; 3 seasons has 7, 5 seasons has 14. Hence a 5-season default.
An explicit division-offset parameter was considered and rejected — it is exactly
collinear with the attack/defence of any team that never crosses, so it adds no
identifying information and only destabilises the optimiser. The gap is read off
post-hoc by `division_gap()` instead.

**Time decay** is `exp(−ξ·days)` with a 1-year half-life, deliberately gentle:
aggressive decay strips out the older-division matches that carry all the linking
information (a 3-month half-life leaves only **1.5%** of fitting weight doing so).
This tension between decay and connectivity is real — watch it when tuning ξ.

**RESULT — the division gap is now LEARNED, not imposed:** 301 / 267 / 255
Elo-equivalent across 3 / 5 / 8-season windows. The Elo engine independently
derives 232 by a completely different route (PPG drops across promotions). Two
methods, one estimated and one measured, land in the same place. That is genuine
corroboration and a portfolio highlight.

#### ⚠️ CORRECTION — home advantage is ~0.21, NOT 0.33

This file previously recorded 0.33 as a **passed correctness test**. That was
wrong, and the way it was wrong is the point:

The old single-season fit had **no intercept term**. With only `gamma` available,
it had to absorb both the overall goal level *and* the home effect, so it settled
too high. The tells were all there once looked for: away goals under-predicted by
5.3% in *both* divisions, home/away goal ratio inflated to 1.303 against an actual
1.213, `gamma` drifting 0.315 → 0.264 → 0.221 as the window lengthened (a stable
parameter should not care), and the 8-season fit failing to converge at all.

Adding the intercept fixed all four simultaneously. `gamma` is now stable at
**0.209–0.213** regardless of window, every window converges, and predicted goals
match the time-weighted actuals to three decimal places.

**The lesson is worth more than the number.** The original 0.33 *passed* its sanity
check — "the model recovers a sensible home advantage where the naive average said
0.06" — and the reasoning was even correct as far as it went. But a sanity check
can pass while the model underneath is misspecified, because the check only tested
whether the number looked plausible, not whether the model could represent the data
generating process. Prefer checks that can *fail structurally*: parameter stability
across specifications, and predicted-vs-actual on a quantity the fit did not target.

#### ⚠️ CORRECTED FINDING — rho DRIFTS; it is not nil, and it is not fixed

`rho` is **fitted inside the likelihood** rather than hardcoded at −0.05. An
earlier entry here recorded it as "~+0.006, i.e. nil, the classic correction does
not replicate on modern data". That was true of the window tested at the time and
**wrong as a general claim**. Measured across test seasons (5-season windows):

| test season | 1819 | 1920 | 2021 | 2122 | 2223 | 2324 | 2425 | 2526 | LIVE 2627 |
|---|---|---|---|---|---|---|---|---|---|
| rho | −0.058 | −0.037 | −0.018 | −0.010 | +0.000 | +0.012 | +0.015 | +0.007 | **−0.053** |

The correction was real in older data, faded to zero around 2022–2025, and has
**returned** on the live window. Cause: a surge in 1-1 draws — per-season rate
0.105 (2324) → 0.131 (2425) → 0.140 (2526), with the overall draw rate rising
from ~0.24 to ~0.268. On the live window Poisson under-predicts 1-1 by 13.7% and
over-predicts 1-0 and 0-1 by ~10% each, which is a textbook rho<0 signature; in
the previous window those cells pulled in opposite directions and the MLE
correctly sat near zero.

**The durable lesson is stronger than the original finding.** Hardcoding −0.05
would have been wrong in 2024; hardcoding 0 would be wrong now. Fitting rho is
what lets the model track a genuine change in how football is scored. The
backtest is unaffected — it refits rho for every window, so it adapted throughout.

#### ⚠️ Ridge regularisation biases the division gap — default is 0

An L2 penalty shrinks attack/defence toward zero. Championship teams sit below zero
and Premier League teams above, so **it shrinks the division gap itself** — it
biases the exact quantity the joint fit exists to estimate. Measured on a 5-season
window: ridge 0 → 266 Elo-equivalent, ridge 1 → 206, ridge 10 → **73**, with
promoted teams wildly overrated as a result (Burnley–Liverpool priced 35/27/38).
If thin-data teams ever need stabilising, shrink toward each team's **own division
mean**, which leaves the between-division gap untouched. Not toward zero.

**Newcomer prior.** Teams with zero training data (League One arrivals) get the mean
fitted rating of comparable arrivals in the window (~14 of them). These occur **only
in the Championship** — 29 team-seasons across 26 years, never in the Premier
League, since the only route into the tracked pool lands you in the Championship.
So the PL backtest is unaffected by this entirely.

**Odds hygiene (`src/odds.py`).** De-vigging is multiplicative (1/odds normalised to
sum to 1). Verified correct: probability conserved to 10 d.p., aggregate H/D/A
within 0.4pp of actual. Residual mis-calibration is **favourite–longshot bias**, not
a bug — longshots over-priced, favourites under-priced, signs flipping cleanly
across the range, affecting 0.44% of predictions. Shin/power de-vig would relax the
proportional-margin assumption; noted for Phase 5.

#### The backtest (`src/backtest.py`) — how it avoids lying to us

- **Rolling**, not a single split. 380 matches cannot resolve a 0.01 log-loss gap.
- **Nested selection of BOTH ξ and the window length** on TUNE seasons (0809→1617),
  frozen for REPORT (1718→2526). Choosing the *window* after seeing test results is
  the same leakage as tuning ξ on them — worth stating because it is easy to miss.
- **Identical match sets**, counts always printed. Missing odds are not random; they
  cluster on obscure fixtures, which are the hard ones, so a shrunken market set
  would hand the market an easier exam.
- **Per-season** breakdown (stable edge vs a lucky year) and **per-division**.
- **Calibration table**, because log loss can look fine while the model is
  systematically overconfident. This is what exposed the Championship weakness.

Other findings:
- The hyperparameter surface is nearly **flat** (0.9796–0.9841 across all 12 combos).
  Do not over-claim the 5-season/365-day choice — it is within noise. 365d did win
  at every window length, which is at least consistent.
- **B365 and Avg price almost identically** on a common set (0.9689 vs 0.9677), so
  the deficit is against the market in general, not one book's quirk.

#### ⚠️ OPENING vs CLOSING odds — the benchmark was too soft

football-data.co.uk ships BOTH: plain columns (`B365H`) are the **opening** price,
`*C*` columns (`B365CH`) are the **closing** price. We originally loaded only the
opening ones. Closing is the recognised benchmark — it has absorbed team news and
the market's own money up to kick-off — and it is measurably sharper. Both are now
loaded (closing exists from `1920` onward only). On the 7 overlapping seasons:

| Benchmark | Log loss | our gap |
|---|---|---|
| B365 opening | 0.9689 | +0.0322 |
| **B365 closing** | 0.9640 | **+0.0370** |
| **AvgC closing** (sharpest) | 0.9639 | **+0.0371** |

**Quote 0.037 vs the closing line, not 0.029 vs the open.** The ~0.005 open→close
sharpening is also a small sanity check that the de-vigging behaves correctly.

#### ⚠️ RESOLVED — the widening deficit is PROMOTED TEAMS, not a smarter market

The obvious hypothesis (bookmakers now price with xG/tracking/injury feeds, so the
frontier moved) is **not supported**. Market absolute log loss is FLAT across the
held-out seasons (slope +0.006/season, p=0.30). Decomposing the deficit instead:

| | deficit | trend slope | p |
|---|---|---|---|
| matches involving a **promoted** team | **+0.0468** | **+0.0112** | **0.014** |
| established teams only | +0.0219 | +0.0021 | 0.48 (ns) |

The drift lives *entirely* in promoted-team fixtures; established fixtures show no
significant trend. 2025-26 is extreme: **+0.1409 vs +0.0122**. Recent promoted
cohorts have been historically unusual (2425: all three relegated, a first; 2526:
Sunderland 7th), and the market prices transfer spend and squad overhaul that a
goals-only model cannot see.

**This unifies two open items**: the "Championship weakness" and the "widening
deficit" are the same defect. Fixing promoted-team ratings addresses both.

CAVEAT when quoting these: TUNE-season deficit averages +0.0143 vs REPORT +0.0289.
Part is genuine hyperparameter optimism, part is the real trend — they cannot be
cleanly separated, so do not attribute a number to either.

**PERFORMANCE NOTE:** `fit_dixon_coles` supplies an **analytic gradient**. Without
it, L-BFGS-B finite-differences ~143 parameters at ~144 evaluations per step and a
fit takes ~40s; with it, ~0.1s. Verified against numerical differentiation (relative
error 2e-7 to 2e-6) and reproduces the same estimates. If you touch the likelihood,
**re-verify the gradient** — a wrong gradient fails silently.

### Phase 8 — promoted-team ratings — ✅ DONE (market priors)
`src/market_prior.py`.

**The defect.** Promoted teams' ratings come from Championship form, and
Championship form does NOT predict Premier League performance — corr **+0.004**
across 75 promotions. Sunderland: 96.3% relegation, finished 7th.

**What was blocked.** Transfermarkt has **no Championship data at all** (0 rows of
GB2 in every table). The Phase 1 note "PL = GB1, Champ = GB2" was wrong and had
never been tested. Coventry is absent entirely; Hull's record stops at 2016.

**What was null.** Parachute payments (p=0.35, wrong sign), auto-vs-playoff
(p=0.47), Championship points (corr +0.004). Combined R² = **0.024**.

**What worked — market-implied priors.** Market's first-6-fixtures implied ppg
correlates **+0.382 (p=0.002)** with final points vs +0.147 for our own rating.
Fit a scalar strength offset per promoted team by minimising KL(market ‖ model)
over their priced fixtures, shrink by 0.75 (tuned on TUNE). Held out, with
prior-building fixtures EXCLUDED from scoring: all fixtures 0.9884 → 0.9829;
promoted fixtures 0.9668 → **0.9446**; gap to market on promoted fixtures halved
from +0.0462 to +0.0240.

**Why this is not the failed blend.** Phase 5 combined a prediction with the SAME
match's price and gained nothing. This moves information from the priced set to
the UNPRICED set (May fixtures), which the market cannot reach. Re-running the
blend test confirms the weight stays at 0 — the prior imports market information
rather than creating new information. Say that plainly; it is the honest framing.

#### ⚠️ The dispersion MUST be tuned with the prior active

Tuned without it, the tuner picks sd=0.55 to compensate for a bad point estimate.
Once the prior fixes the estimate, that spread is noise: it over-covers at 96%,
gives promoted sides a **9% top-six chance** (historical: 0 of 75) and a 75-point
upper bound (historical max 59). `tune_promoted_sd(apply_prior=True)` is now the
default and the harness uses it.

#### ⚠️ Promoted-team predictability is NON-STATIONARY — use the rolling tuner

No fixed dispersion works. On TUNE seasons (0809–1617) promoted teams need little
extra spread; on REPORT (1718–2526) they need a lot. A value tuned on TUNE and
frozen UNDERPERFORMS out of sample. The rolling tuner (re-select on the six
preceding seasons) is the answer, and what it picks climbs steadily:
0.15 ×5 → 0.25 → 0.35 ×3 across the nine held-out seasons.

Held-out comparison, promoted teams (n=27):

| approach | 80% cover | PIT p | P(top6) |
|---|---|---|---|
| no prior, no dispersion fix | 48.1% | **0.006** | 2.2% |
| fixed 0.55/0.35 (tuned w/o prior) | 96.3% | 0.121 | 9.0% |
| fixed 0.15/0.70 (tuned w/ prior) | 55.6% | 0.050 | 1.3% |
| **rolling + prior** | **70.4%** | **0.269** | **3.3%** |

Still short of the nominal 80% (z=−1.25, not significant). Promoted teams remain
the least predictable part of the league and that is a real limit, not a bug.

#### ⚠️ The prior needs ODDS on the fixture table

The FPL fixture feed carries no odds, so the prior silently did nothing on the
first live run. `fixtures.py` now joins football-data's upcoming-fixture odds,
and `market_prior` prints a warning instead of no-opping in silence. Caught by
reading the written meta.json rather than trusting the pipeline. The prior is
weak early (only ~1 fixture per team priced in mid-August) and sharpens as the
season is priced.

### Phase 9 — expected-goals layer — ✅ DONE (small but real gain)
`src/xg.py` builds `data/processed/xg_matches.csv`; `dixon_coles.XG_WEIGHT`
controls how much of it is used.

**Recon re-verified, not trusted.** The Phase 1 note about Transfermarkt/GB2 was
wrong and blocked Phase 8, so both xG claims from that same recon were retested.
Both were CORRECT: Understat works and is Premier League only
(`available_leagues()` returns just `ENG-Premier League`), and FBref genuinely
exposes no xG in soccerdata 1.9.0 (four stat types, none containing it).

**Data.** Understat `read_schedule()` gives 380 matches/season with `home_xg`,
`away_xg`, for 2014-15 → 2025-26. 4,560 matches. Verified: 100% join onto
matches_combined, and **0 goal-value mismatches across all 4,560 rows**, which
proves the join matches the right fixtures rather than merely the right count.
Mean xG [1.580, 1.271] sits alongside mean goals [1.543, 1.258].

**The premise, tested before modelling.** Does first-half xG predict second-half
GOALS better than first-half goals do? Across 240 team-seasons:

| predictor of 2nd-half … | goals-based r | xG-based r |
|---|---|---|
| goals scored | +0.666 | **+0.719** |
| goals conceded | +0.519 | **+0.601** |

Yes — clearly, and more so on defence.

**How it enters the model.** The MODEL is unchanged: goals are still Poisson
with the low-score correction. Only the ESTIMATION target changes:
`y = κ·xG + (1−κ)·goals`, a quasi-Poisson (score equations stay consistent for
the mean with non-integer y; `log(y!)` → `lgamma(y+1)`). `tau` still uses the
actual integer scoreline, since it describes discrete structure, not the rate.
κ=0 recovers the goals-only model EXACTLY (verified: home_adv 0.2096, rho
0.0070 unchanged), so the A/B is built into the parameterisation.

**Result.** κ selected on xG-covered TUNE seasons (1415-1617), held out on
1718-2526 over 3,420 matches:

| | log loss | RPS | division gap |
|---|---|---|---|
| goals only (κ=0) | 0.9868 | 0.2050 | 204 |
| **blended (κ=0.25)** | **0.9858** | 0.2047 | 199 |
| xG only (κ=1) | 0.9890 | 0.2057 | **183** |

#### ⚠️ Two findings that matter more than the +0.0010

**Pure xG is WORSE than pure goals** — in the joint fit (0.9890 vs 0.9868) and in
a PL-only fit (0.9802 vs 0.9792). Finishing quality is not purely noise;
throwing goals away throws real signal away. Do not raise κ toward 1.

**Understat's PL-only coverage costs about half the benefit.** Championship
matches always fall back to goals, so at high κ the two divisions are measured
with different instruments — and the learned division gap slides 204 → 183 as
κ goes 0 → 1. A PL-ONLY fit, where every match has xG, gains **+0.0021** against
+0.0010 for the joint fit, and its κ curve is a clean inverted-U peaking at 0.5
rather than the flat joint curve. That PL-only figure is a DIAGNOSTIC of the
mechanism, not a validated setting — it was found by looking at held-out data.
The shipped κ=0.25 came from proper nested tuning.

Championship xG would fix this, and no free source has it. That is the single
thing that would make this layer worth more.

### Phase 10 — key-player availability — ✅ DONE, and it WORKS (2026-08-24)
`src/availability.py`.

The same conversation that produced the rejected squad-value idea produced this
one, and the contrast is the lesson. **Squad value restates what the model
already knows; availability is information about the FUTURE.**

**Held out, with the market prior already applied:**

    model + market prior                  0.9790
    model + market prior + availability   0.9760     (+0.0030)

Positive in **8 of 9 seasons**, t = 2.97, **p = 0.018**. It also helps without
the prior (0.9858 → 0.9830), so the two are complementary, not rival.

**Why it survives where squad value did not.** The market prior is fitted ONCE on
a season's opening fixtures. It cannot know that a key player limped off in
November. Availability is transient and match-specific, so the prior has no
chance to absorb it.

**The historical proxy.** For each team-match, weight every player by their share
of the team's minutes over the previous 8 games, then measure how much of that
weight actually appeared in the team's MOST RECENT match. Only prior information
— no leakage. Built from Transfermarkt appearances (GB1, 2012-13 on): 10,415
team-match rows, mean 0.781. Sanity: availability differential correlates with
the model's goal-difference residual at +0.067 (p<0.001).

#### ⚠️ The two sources are on DIFFERENT scales — standardise

Historical availability averages **0.781** (squads rotate); the live FPL injury
measure averages **0.839** (most players are simply fit). Centering live values
on the historical mean would hand every team a positive offset. `offsets(...,
standardise=True)` z-scores against the source's own cross-sectional mean and
spread, then rescales onto the historical spread the coefficient was tuned on.
For the historical series this is algebraically identical, so the validated
behaviour is unchanged.

#### ⚠️ A bug worth remembering: adjustments that accumulate

The first version of this test reassigned `fit.adjustments` INSIDE the per-match
loop, so offsets compounded across all 380 matches of a season. A coefficient
worth ±0.01 appeared to move log loss by 0.008 and then diverge to NaN — and the
tuner "correctly" chose zero, which would have buried a real +0.0030 effect. It
was caught only because the magnitude was implausible for the perturbation.
Snapshot `fit.adjustments` once per season and rebuild per match.

---

### Squad-value features — TESTED AND REJECTED (2026-08-24)

Prompted by a good observation: the model rates Newcastle on last season's
results and cannot see that the squad was dismantled over the summer. The
proposal was to add player market value, form, stats and FPL price as team
strength context. It was tested thoroughly and **does not work**. Do not rebuild
it without reading this.

**The premise is real.** Squad value (top-11 Transfermarkt, within-season z,
271 team-seasons over 14 seasons) correlates with final points at **r = +0.762**,
and adds to a regression that already contains the model's own expected points:
R² 0.634 → 0.652, **+5.09 points per SD, p = 2.6e-04**. So squad value genuinely
knows something about a SEASON that the results-based model does not.

**It does not survive contact with match prediction.** Held out:

    model only                            0.9858
    model + squad value (gamma 0.04)      0.9856   (+0.0002, nothing)
    model + market prior                  0.9790
    model + market prior + squad value    0.9797   (WORSE)

The bookmakers already price squad quality — it is their job — so once the market
prior is applied, squad value is redundant and slightly harmful. Same shape as
the Phase 5 blend result.

**It does not help even when odds coverage is thin**, which was the obvious
escape hatch (in mid-August only one fixture per team is priced, so the prior is
shrunk to 1/6). Simulating that regime, squad value hurts monotonically:
0.9927 → 0.9939 → 0.9964 → 1.0004 as gamma rises 0 → 0.02 → 0.04 → 0.06.

**Squad value CHANGE — the sharper version, and the actual Newcastle case — has
no signal whatsoever.** A team that was strong and just got weaker is precisely
what the level cannot capture and the change should. Over 208 team-seasons:
correlation with change in points **r = -0.008 (p = 0.91)**; controlling for last
season's points, R² 0.460 → 0.460, coefficient **-0.11 pts per SD (p = 0.90)**.
Likely because Transfermarkt valuations are partly REACTIVE — they follow
results rather than lead them — and selling a star to reinvest is often net
neutral.

**Why the level correlates so strongly yet adds nothing:** the model's ratings
already encode squad quality implicitly, because good squads have been winning.
The increment shows up at season level (predicting a points total) but not at
match level, which is what the model is scored on.

**What WOULD change this:** the market prior is the thing that works, and it
depends on odds existing. For a competition with no betting market, or for
fixtures priced far in advance, squad value would be the fallback rather than a
redundant extra. Also untested here: FPL injury/availability flags, which are
information about the FUTURE rather than a restatement of the past — though the
market prices those too.

---

## 6. Roadmap beyond Phase 4 (planned, not built)

- **Phase 5 — market blend — ✅ DONE, and it FAILED.** `src/blend.py`. Log-opinion
  pool, weight tuned on TUNE and reported on REPORT. Result: PL blend w=0.10 scores
  0.9583 vs market-only 0.9578 — **worse**. Championship picked **w=0.00**,
  discarding the model outright. The TUNE curve is flat across w=0.00–0.20, so the
  weight was arbitrary within noise. **Our model carries essentially no information
  the market lacks.** Keep the code and re-run it after the promoted-team/transfer
  work; if the model ever adds something, w will move off zero.
  (Could not be tuned against the CLOSING line — closing odds start 2019-20, which
  is entirely inside REPORT, so there are no closing TUNE seasons.)

  **What this implies for the project's value:** match predictions where odds exist
  should just use the market. The real contribution is **season-level distributions**
  (Phase 6), which the market does not publish and which *cannot* be blended,
  because August simulation requires predicting unpriced May fixtures. Consequently
  the simulator runs on **model-only** probabilities all season — so model quality
  drives every title/relegation number, with no market to lean on. This RAISES the
  value of the promoted-team fix rather than lowering it.
- **Phase 6 — season simulator — ✅ DONE.** `src/simulate.py`. Samples a full
  SCORELINE per remaining fixture (not just W/D/L) so goal difference and the
  tie-breaks that decide titles come out right. 10,000 seasons in <1s.
  **Correctness:** reproduces real historical tables exactly; with every match
  played the forecast IS the final table (points and position, all 20 teams);
  simulated mean points match analytic expected points to 0.13.
  **Calibration — the important bit.** Independent per-match draws were badly
  overconfident: only 63.9% of actual points fell in the predicted 10–90% band
  (target 80%, z=−5.40), and the PIT rejected calibration at p=0.012. Cause:
  independence assumes a team's strength IS its rating, but that error persists all
  season, so a side better than rated is better in all 38 games at once. Fix:
  `STRENGTH_SD`, a per-team season-long offset drawn once per scenario, selected on
  TUNE by minimising PIT KS distance. Both divisions chose **0.15**. Held out:
  coverage 75.0%, PIT p=0.646. Still marginally narrow (z=−1.68) — propagating the
  fit's real parameter covariance instead of one scalar is the next refinement.
- **Phase 7 — live 26-27 harness — ✅ DONE.** `src/fixtures.py` + `src/harness.py`.
  The FPL API supplies all 380 fixtures (football-data's fixtures.csv carries only
  the next few days, so it cannot support August season simulation — but it IS the
  Championship source, which FPL does not cover). Weekly: refresh, re-fit on history
  PLUS this season's results, re-simulate, write a **dated immutable snapshot** with
  `meta.json` recording window/half-life/dispersion/git commit, so a mid-season
  model change is visible in the history rather than silently rewriting it.
  `snapshot_history()` reassembles the week-by-week series.
  **Two guards worth knowing:**
  (a) STALENESS — a fixture whose kick-off has passed with no result is flagged
  loudly; without it the harness would quietly SIMULATE already-played matches.
  (b) `historical_matches()` EXCLUDES the current season from matches_combined,
  because the fixture table is the source of truth for it. The dry run caught this
  as a real double-count (760 matches in a 380-match season, all points doubled) —
  it would have gone live the moment load_results.py was re-run mid-season.
  **Validated** by `validate_harness.py` replaying 2025-26: the champion's title
  probability climbs 29% → 100%, and the final snapshot reproduces the real table
  exactly.
- **Phase 8 — promoted-team ratings — ✅ DONE via MARKET PRIORS, not transfer data.**
  `src/market_prior.py`. See section 5 for the full account. Short version: the
  Transfermarkt route is BLOCKED (dataset is Premier League only — CLAUDE.md's
  "Champ = GB2" was wrong), features we already had are null (combined R²=0.024),
  and the thing that worked was bookmaker odds. Held out: promoted-fixture log
  loss 0.9668 → 0.9446, halving the gap to the market on those fixtures.
- ~~**Phase 8 — promoted-team ratings from transfer/squad-value data**~~ (superseded)
  The largest remaining gap, and both the "Championship weakness" and the "widening
  market deficit" reduce to it. The Transfermarkt DuckDB is already on disk.
  **The integration point already exists**: `DixonColesFit.adjustments`, a
  `{team: (d_attack, d_defence)}` dict applied inside `_rating()`, so every
  prediction path picks it up and it can be A/B tested on the existing backtest
  without touching the model. `blend.py` is the instrument that will say whether it
  worked: if the model gains real information, the blend weight moves off zero.
  Live evidence of the problem: the 2026-27 opening forecast puts **Hull City at
  21.8 xPts and 93.6% relegation** — the same shape of error as Sunderland last
  season (predicted 21.3 xPts / 96.3% relegation, actually finished 7th on 54).
- **Phase 9 — xG layer — ✅ DONE (small gain).** `src/xg.py`, and `XG_WEIGHT`
  in dixon_coles. See section 5. Short version: the premise holds but the payoff
  is modest — held-out log loss 0.9868 → 0.9858. Pure xG is WORSE than pure
  goals, and Understat's Premier-League-only coverage costs roughly half the
  benefit.
- ~~**Phase 9 — xG layer**~~ (superseded): refit team strength on Understat xG (less noisy than raw
  goals). Strict A/B against the goals-only model on the backtest — only keep it if
  log loss improves. Also: consider an xG-based Elo variant here.
- **Phase 10 — player layer**: minutes model + empirical-Bayes shrinkage on per-90
  goal/assist rates, coupled to the team model's fixture xG, with FPL availability.
  Deliberately NOT XGBoost first — shrinkage baseline first, ML only if it beats it.
- **Phase 11 — event-data lab (optional)**: build an Expected Threat model on
  StatsBomb's free 2015-16 PL season. Portfolio value; no live use.
- **Phase 12 — front end (optional)**: Streamlit over the snapshot history.

---

## 7. Known issues / gotchas (don't get caught out)

- **Stale team ratings**: teams that left the data window years ago (Tranmere,
  Wimbledon, etc.) have meaningless ratings clustered near the mean — they haven't
  played in the data since ~2001 and drifted. Confirmed still present: 22 dormant
  clubs currently outrank the worst actual PL side (Tranmere sits at 1581, which
  would be ~7th in the Premier League). HARMLESS **provided** you filter through
  `paths.active_teams()`. Never rank against the full 71.
- ~~**teams.csv location**~~: RESOLVED 2026-07-28 — now at `data/reference/teams.csv`,
  path centralised in `src/paths.py`.
- **FPL column is season-specific**: the `fpl` column in teams.csv reflects the
  current squad list; promoted/relegated clubs' FPL names shift yearly.
- **Championship xG doesn't exist** in Understat (top division only) — the xG layer
  will be Prem-only. Champ teams get goals-based strength until promoted.
- **soccerdata scraping**: uses browser automation (Selenium/chromedriver) for
  FBref/Understat/ClubElo; caches locally. Respect rate limits (FBref: 10 req/min).
  ClubElo's live API occasionally times out — just retry, or use a cached date.
- **Date-parse warning**: load_results.py throws a cosmetic "Could not infer format"
  UserWarning per file (football-data changed date formats over the years). Harmless;
  silence it if it bothers you by specifying formats per-era.
- **Always** read matches_combined.csv with `dtype={"season": str}` or "0001"-style
  season codes become integers. Easiest: just use `paths.load_matches()`.
- **Season codes are YYnn of the START year**: `"1920"` = 2019-20, `"0001"` = 2000-01,
  `"2526"` = 2025-26. Easy to misread `"1920"` as 1919-20 — I did exactly that when
  first assessing bookmaker-odds coverage.
- **Bookmaker odds coverage is NOT uniform** — check before using as a baseline:
  `B365` from `0203` (2002-03) at ~100%; `WH` from `0001` but **0% in `2526`** and
  80% in `2425`; `Avg` only from `1920` (2019-20), then 100%. Two `B365H` values were
  a corrupt `0.0` (Blackpool-Derby 2013-04-27, Brentford-Blackburn 2019-02-02);
  `1/0` = inf implied probability, which would turn any log loss into inf/NaN.
  RESOLVED 2026-07-28: `odds.clean_odds()` runs at load time and nulls impossible or
  partial triplets while **keeping the match** — the results are fine, only the price
  was bad. Use `odds.has_odds()` / `odds.comparison_set()` rather than testing the
  raw columns yourself.

---

## 7b. Open findings — deliberately DEFERRED, not forgotten

Found during the 2026-07-28 code review. All three are real, none block the
backtest, and each is a good interview talking point precisely because it is a
critique of our own work.

### (a) The 232 division gap is probably inflated by regression to the mean

`measure_gap.py` derives 232 from the PPG drop of promoted teams and the PPG gain
of relegated ones. But teams are promoted partly by **over**-performing and
relegated partly by **under**-performing, so both groups regress toward their true
level the following season regardless of division quality. That regression is baked
into the measured drop — and critically, **averaging the two directions does not
cancel the bias, it compounds it**, because both point the same way.

Evidence it is real. Z-scoring each PL team's PPG and its start-of-season Elo
within each season, then taking the residual:

```
          count   mean    std
promoted     75  +0.191  0.662     <- outperform their rating
stayed      425  −0.034  0.616
t = 2.50, p = 0.0146, n = 75
```

Promoted sides beat their Elo by ~0.19 SD ≈ **3 points a season**. Visible live:
Sunderland finished **7th** in 2025-26 but ranked **16th** by Elo.

Why deferred: the implied correction is only ~20 Elo points, and Dixon-Coles does
not use Elo at all, so it cannot touch the backtest. Revisit in Phase 5.
Note the Dixon-Coles fit **learns** 267 rather than assuming — see Phase 4.

Related smaller point: `convert_gap.py` maps PPG to Elo via `PPG_GAP / 3.0`, which
is exact only if the promoted teams' **draw rate is unchanged** (points weight a
draw 1/3, Elo weights it 1/2). Defensible, but currently an unstated assumption and
cheap to test from data we already have. `PPG_GAP = 0.874` is also hardcoded rather
than read from `measure_gap.py`, so the two scripts do not actually connect.

### (b) Elo has been validated for RANKING, never for CALIBRATION

Spearman 0.955 vs ClubElo says the *ordering* agrees. It says nothing about whether
a 100-point gap implies the right win probability. Our PL spread is sd 94 / range
374, somewhat tighter than ClubElo's ~110/450 — likely the 25%-per-season regression
compressing it.

Harmless today because Dixon-Coles ignores Elo entirely. **Must be checked before
Phase 5**, the moment an Elo rating feeds a probability or a blend weight.

### (c) The two leagues share literally zero information in the Elo engine

Verified: the end-of-season PL−Champ mean gap is *exactly* 232.0. That is not a
coincidence — Elo is zero-sum and every match is within-league, so each pool's mean
is **mathematically conserved** across a season. The re-anchor sets the gap and
nothing can ever move it. The 232 is a pure prior that no amount of data can update.

This makes the cup-data upgrade in §8 more valuable than it first reads: it is the
only thing that would let real results inform the cross-division offset in Elo.
(Dixon-Coles already sidesteps this — it learns the gap from crossing teams.)

---

## 8. High-value future upgrade worth knowing about

The single most impressive upgrade to the strength engine would be to build a
**properly connected pool** the way Opta actually does — bring in FA Cup, League Cup,
and European matches so the Premier League and Championship are linked by *real*
cross-league results rather than our imposed 232-point offset. soccerdata can pull
cup data. It's noted as a deliberate future project (not a quick win — knockout
quirks like extra time, penalties, two legs, rotated squads make cup data messier,
and cups add relatively few genuine cross-league games). Logged here so it isn't
forgotten. It would narrow the gap with ClubElo's globally-connected system.

---

## 9. Immediate next action

**Phases 1-7 are complete.** The model is backtested and calibrated, the season
simulator is calibrated, and the live harness ships dated weekly snapshots for
2026-27. The first live snapshot is in `data/snapshots/2627/`.

**Weekly operation** (the only routine job during the season):

```
py src/fixtures.py     # refresh results from the FPL API
py src/harness.py      # re-fit, re-simulate, write a dated snapshot
git add data/ && git commit     # the snapshot history IS the deliverable
```

Heed the staleness warning: if it reports fixtures past kick-off with no result,
the feed has not updated and those matches are being simulated rather than counted.

**Next substantive work — Phase 8, promoted-team ratings from transfer data.**
Everything points at it:
1. The entire widening market deficit lives in promoted-team fixtures (+0.047 vs
   +0.022, with the trend significant only for the promoted group, p=0.014).
2. We capture only ~33% of the market's available Championship edge.
3. The season simulator runs model-only all year, so this drives every title and
   relegation number with no market to lean on.
4. Live proof: Hull City is forecast at 21.8 xPts / 93.6% relegation for 2026-27 —
   the same error shape as Sunderland (21.3 xPts / 96.3%, finished 7th on 54).

The hook is already in place (`DixonColesFit.adjustments`), the data is on disk,
and `blend.py` is the test of whether it worked.

**A note on working method, earned the hard way.** Nearly every real defect in this
project was found by *checking a number*, never by reading code: the ridge
compressing the division gap, the missing intercept inflating home advantage, the
unmapped-name check that only printed, the model/market comparison run on two
different match sets, the overconfident season intervals, and the harness
double-count. Several were introduced by suggestions that sounded principled.
Prefer verification that can fail structurally — parameter stability across
specifications, byte-identical output after a refactor, an analytic gradient
checked against finite differences, a simulator reproducing a known final table
exactly, predicted-vs-actual on something the fit did not target — over "does this
look sensible".

**And re-check findings as data arrives.** `rho` was recorded here as "nil on
modern data"; one further season flipped it to −0.053. The claim was true of its
window and wrong as a general statement. Date your findings and re-test them.
