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
│   └── dixon_coles.py            # Phase 4: fits attack/defence, predicts matches
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

#### ⚠️ FINDING — the Dixon-Coles correction does nothing on modern data

`rho` is now **fitted inside the likelihood** rather than hardcoded at −0.05. It
comes out at **~+0.006, i.e. nil**. The classic 1997 motivation does not replicate
on 2020–2025 data: Poisson predicts the overall draw rate to within **0.3%**, and
the residual pattern is the wrong *shape* for the correction — 0-0 is
over-predicted (wants rho > 0) while 1-1 is under-predicted (wants rho < 0), so the
single parameter is pulled both ways and the MLE correctly lands at zero.

Consequence: hardcoding −0.05 would have **actively hurt**, inflating an already
over-predicted 0-0. Keep the parameter (it is free, and demonstrating it is ~0 is
stronger than assuming a value), but describe the model honestly — it is closer to
a time-weighted joint-division Poisson than to textbook Dixon-Coles.

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
- **The deficit to the market widens over time**: +0.0036 (1920) → +0.0488 (2526).
  Cause unknown. Worth investigating in Phase 5 — is the market improving, or is the
  model degrading as squads turn over faster?
- **B365 and Avg price almost identically** on a common set (0.9689 vs 0.9677), so
  the deficit is against the market in general, not one book's quirk.

**PERFORMANCE NOTE:** `fit_dixon_coles` supplies an **analytic gradient**. Without
it, L-BFGS-B finite-differences ~143 parameters at ~144 evaluations per step and a
fit takes ~40s; with it, ~0.1s. Verified against numerical differentiation (relative
error 2e-7 to 2e-6) and reproduces the same estimates. If you touch the likelihood,
**re-verify the gradient** — a wrong gradient fails silently.

---

## 6. Roadmap beyond Phase 4 (planned, not built)

- **Phase 5 — market blend**: combine model probabilities with bookmaker-implied
  probabilities (remove overround first; blend in logit space; tune the weight
  out-of-sample). Honest expectation: optimal weight may lean heavily on the market
  for match outcomes — that's fine, the model provides season-level distributions
  the market doesn't publish.
- **Phase 6 — season simulator**: Monte Carlo the remaining fixtures thousands of
  times → title/top-4/relegation probabilities + full points distributions.
- **Phase 7 — live 26-27 harness**: one fixtures table (all 380, results filled in
  weekly), a weekly re-run that updates Elo (incremental) + refits Dixon-Coles
  (fast) + re-simulates, and **snapshots every output with a date** so you build a
  week-by-week history of how the forecast evolved. Ships before 21 Aug 2026.
- **Phase 8 — xG layer**: refit team strength on Understat xG (less noisy than raw
  goals). Strict A/B against the goals-only model on the backtest — only keep it if
  log loss improves. Also: consider an xG-based Elo variant here.
- **Phase 9 — player layer**: minutes model + empirical-Bayes shrinkage on per-90
  goal/assist rates, coupled to the team model's fixture xG, with FPL availability.
  Deliberately NOT XGBoost first — shrinkage baseline first, ML only if it beats it.
- **Phase 10 — event-data lab (optional)**: build an Expected Threat model on
  StatsBomb's free 2015-16 PL season. Portfolio value; no live use.
- **Phase 11 — front end (optional)**: Streamlit over the snapshot history.

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

**Phase 4 is complete and backtested.** `README.md`, `LICENSE` and
`requirements.txt` are in place, so the repo is presentable to employers.

Next is **Phase 5 — market blend**. The backtest gives the honest starting point:
the model trails B365 by 0.0289 log loss, so the optimal blend weight will likely
lean heavily on the market. That is fine and expected — the value the model adds is
season-level distributions the market does not publish, which is Phase 6.

Three things the backtest surfaced that Phase 5 should address, in priority order:

1. **The Championship model is weak** — only 0.0119 log loss better than base rates,
   and overconfident at z=−4.05 in the 0.5–0.6 band. It is also the only place the
   newcomer prior is exercised. Fixing this matters for the season simulator, since
   promoted teams' ratings come from Championship form.
2. **The widening deficit to the market** (+0.0036 in 1920 → +0.0488 in 2526).
   Establish whether the market improved or the model is degrading.
3. **Elo calibration is still unchecked** (§7b(b)) — this becomes load-bearing the
   moment an Elo rating feeds a blend weight or probability.

**A note on working method, earned the hard way on 2026-07-28.** Several real
defects that day were found by *checking a number*, never by reading the code: the
ridge compressing the division gap, the missing intercept inflating home advantage,
the unmapped-team-name check that only printed, and the model/market comparison
initially being run on two different match sets. Two of them were introduced by
suggestions that sounded principled. Prefer verification that can fail structurally
— parameter stability across specifications, byte-identical output after a refactor,
an analytic gradient checked against finite differences, predicted-vs-actual on
something the fit did not target — over "does this look sensible".

**A note on working method, earned the hard way on 2026-07-28.** Three real defects
that day were found by *checking a number*, never by reading the code: the ridge
compressing the division gap, the missing intercept inflating home advantage, and
the unmapped-team-name check that only printed. Two of them were introduced by
suggestions that sounded principled. Prefer verification that can fail structurally
— parameter stability across specifications, byte-identical output after a refactor,
predicted-vs-actual on something the fit did not target — over "does this look
sensible".
