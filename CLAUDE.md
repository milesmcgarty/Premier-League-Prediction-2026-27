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
│   │   ├── exploration/          # throwaway recon scripts + teams.csv (see note below)
│   │   └── transfermarkt-datasets.duckdb   # transfer/player/market-value DB
│   └── processed/
│       ├── matches_combined.csv  # THE clean match table — 24,232 matches, canonical names
│       ├── elo_ratings.csv       # final Elo per team (71 teams)
│       └── elo_history.csv       # every team's rating after every match (~48k rows)
├── src/
│   ├── load_results.py           # Phase 2: builds matches_combined.csv
│   ├── elo.py                    # Phase 3: the EloEngine class (core maths)
│   ├── build_elo.py              # Phase 3: replays all matches, builds+saves ratings
│   ├── benchmark_elo.py          # Phase 3: validates against ClubElo (rank correlation)
│   ├── measure_gap.py            # Phase 3: measured the cross-division gap empirically
│   ├── convert_gap.py            # Phase 3: converted that gap to Elo points (=232)
│   └── dixon_coles.py            # Phase 4: fits attack/defence, predicts matches
└── venv/

NOTE: teams.csv currently lives in data/raw/exploration/ but SHOULD be moved to
data/reference/teams.csv — it's a permanent curated asset, not throwaway recon.
When you move it, update REFERENCE_DIR in load_results.py accordingly.
```

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
- Maps team names → canonical via teams.csv, with a safety check that errors loudly
  if any name fails to map.
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

### Phase 4 — Dixon-Coles match prediction — 🔶 IN PROGRESS
`src/dixon_coles.py`.

DONE so far:
- **`fit_poisson`** — fits attack + defence rating per team by maximum likelihood
  (scipy L-BFGS-B). Expected goals: home ~ exp(atk_home − def_away + home_adv),
  away ~ exp(atk_away − def_home). Both attack and defence constrained to sum to
  zero (identifiability). Needs `maxiter=1000` — at 200 it under-converged and gave
  a wrong home advantage (0.06 instead of the correct 0.33). Fitted ratings verified
  correct against 24-25 reality (Liverpool best attack, Arsenal best defence).
- **`dc_correction`** — the Dixon-Coles low-score adjustment for 0-0/1-0/0-1/1-1
  (rho, small negative ~−0.05). This is what makes it Dixon-Coles not plain Poisson;
  it fixes the known under-prediction of low-scoring draws.
- **`predict_match`** — builds the scoreline probability grid, applies the
  correction, returns W/D/L probabilities + expected goals + likeliest score.
  Verified: sensible favourites, draws in the realistic 22-31% range, and the
  home-advantage flip works (Arsenal-vs-City reverses correctly when venue swaps).

**A subtle lesson worth remembering:** the *raw* home/away goal split in 24-25 was
tiny (1.51 vs 1.42 → implied 0.06), but the fitted model correctly recovered 0.33
because it controls for *who played whom* at home. Naive averages mislead; the model
corrects. Good illustration of why we fit rather than eyeball.

STILL TO DO in Phase 4 — **THE BACKTEST** (this is the next task):
- Fit on past seasons, predict a held-out season the model never saw.
- Score with **log loss** (punishes confident-wrong predictions) and ideally
  **RPS** (ranked probability score, the standard for ordered W/D/L outcomes).
- Compare against baselines: (a) naive "always home win", (b) the **bookmaker odds**
  already stored in matches_combined.csv (B365/WH/Avg columns). Beating the naive
  baseline is required; landing close to the market is the real, honest measure.
- Also add **time-decay weighting** to the fit (recent matches matter more —
  exp(−ξ·days_ago), ξ tuned by out-of-sample log loss) and fit across BOTH divisions
  jointly with a division offset (mirrors the Elo cross-division handling).
- This backtest is the number that proves the predictions are genuinely good, not
  just plausible-looking. It's the most important deliverable of the phase.

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
  played in the data since ~2001 and drifted. HARMLESS as long as you filter to
  *recently-active* teams when using ratings. Never rank against the full 71.
- **teams.csv location**: still in `data/raw/exploration/` — move to
  `data/reference/` and update `REFERENCE_DIR` in load_results.py.
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
  season codes become integers.

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

Build the Phase 4 backtest (Section 5, "STILL TO DO"). Explain the train/test split
and the metrics (log loss, RPS, vs baselines) before coding. Keep the incremental,
verify-each-step rhythm. The bookmaker odds for the baseline comparison are already
in matches_combined.csv (B365H/D/A, WHH/D/A, AvgH/D/A columns).
