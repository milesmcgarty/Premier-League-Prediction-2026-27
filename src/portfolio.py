"""Render the project case-study page: what was built, and what it was graded on.

Separate from dashboard.py, which is the operational weekly page. This one is
the story: the architecture, the held-out numbers, and the ideas that were
tested and thrown away.

Every live figure is read from the newest snapshot. The historical validation
numbers are frozen constants with the date they were measured, because their
backtests take minutes and they only move when the model does.

    py src/portfolio.py
"""
from datetime import datetime

from dashboard import esc, gather, pc
from paths import ROOT

OUT = ROOT / "portfolio.html"
REPO = "https://github.com/milesmcgarty/Premier-League-Prediction-2026-27"

SCALE = {"matches": 24232, "seasons": 26, "clubs": 71, "modules": 26,
         "lines": 5923, "held_out_seasons": 9, "held_out_matches": 3420}

LADDER = [("Uniform, 1/3 each", 1.0986, "base"),
          ("Historical base rates", 1.0682, "base"),
          ("Dixon-Coles, goals only", 0.9868, "step"),
          ("+ expected-goals layer", 0.9858, "step"),
          ("+ market prior", 0.9790, "step"),
          ("+ key-player availability", 0.9760, "ship"),
          ("Bookmaker closing line", 0.9640, "market")]

SEASON_LEVEL = [("Title", 0.0767, 0.1985, 0.61),
                ("Top four", 0.2590, 0.5004, 0.48),
                ("Relegation", 0.3452, 0.4227, 0.18)]

# Each layer, with the evidence that kept it in or threw it out. The kept/cut
# split is the real structure of this project, so it is the page's structure.
LAYERS = [
    ("kept", "Elo with cross-division re-anchoring",
     "Promoted clubs arrived over-rated because two league pools drift apart "
     "when only three teams swap a year. Fixed by sliding the whole "
     "Championship pool to sit a measured 232 points below the Premier League.",
     "Spearman 0.955 against ClubElo across 43 shared clubs. The 232 was "
     "measured from 75 promotions and 75 relegations, not chosen."),
    ("kept", "Dixon-Coles fitted jointly across both divisions",
     "Attack and defence per club, a global intercept, home advantage and the "
     "low-score correction rho, all in one weighted likelihood with a "
     "one-year half-life.",
     "Learns the division gap at 267 Elo-equivalent by a completely different "
     "route from the Elo engine's measured 232. Analytic gradient verified "
     "against finite differences to 2e-07; a fit takes 0.1s instead of 40s."),
    ("kept", "Market priors for promoted clubs",
     "Championship form does not predict Premier League performance at all "
     "(correlation +0.004 over 75 promotions). Bookmakers price the summer "
     "rebuild we cannot see, so a strength offset is read off their opening "
     "prices by minimising KL divergence.",
     "Promoted-fixture log loss 0.9668 to 0.9446, halving the gap to the "
     "market on exactly the fixtures where it was widest."),
    ("kept", "Expected goals as an estimation target",
     "A quasi-Poisson: y = 0.25 xG + 0.75 goals. The model is unchanged, only "
     "what it is fitted to. Weight zero recovers the goals-only model exactly, "
     "so the A/B is built into the parameterisation.",
     "0.9868 to 0.9858. Small, and pure xG is WORSE than pure goals, so "
     "finishing quality is not noise."),
    ("kept", "Key-player availability",
     "Weight each player by their share of recent minutes, then measure how "
     "much of that weight actually appeared in the last match. Live values "
     "come from the FPL injury flags.",
     "+0.0030 on top of the market prior, positive in 8 of 9 seasons, "
     "p = 0.018. It survives because it is information about the future, "
     "which a prior fitted in August cannot contain."),
    ("kept", "Monte Carlo season simulation",
     "Samples a full scoreline for every remaining fixture, so goal difference "
     "and the tie-breaks that decide titles come out right. 20,000 seasons.",
     "Reproduces historical tables exactly. Independent per-match draws were "
     "badly overconfident at 63.9% interval coverage; a season-long strength "
     "offset drawn once per scenario fixed it to 75.0%, PIT p = 0.646."),
    ("cut", "Blending model probabilities with market odds",
     "The obvious move: pool the model's match probabilities with the "
     "bookmaker's.",
     "The tuner chose weight 0.10 for the Premier League, scoring 0.9583 "
     "against the market's own 0.9578, and weight 0.00 for the Championship. "
     "The model carries essentially no information the market lacks."),
    ("cut", "Squad market value",
     "Newcastle lost most of their key players over one summer and the model "
     "could not see it. Squad value should have caught that.",
     "Correlates +0.762 with final points on its own, but once the market "
     "prior is applied it makes match prediction WORSE (0.9790 to 0.9797). "
     "Bookmakers already price squad quality."),
    ("cut", "Squad value CHANGE, the sharper version",
     "A club that was strong and just got weaker is precisely what the level "
     "cannot capture.",
     "No signal whatsoever across 208 club-seasons: r = -0.008, p = 0.91. "
     "Transfermarkt valuations are partly reactive, following results rather "
     "than leading them."),
    ("cut", "Weighting the current season more heavily",
     "Four gameweeks in, the season in progress is only 3.3% of the fitting "
     "weight. Weighting it up should help.",
     "It does not. A nested sweep over eight multipliers picked 1.0 as a "
     "genuine interior optimum. Exponential time decay already weights the "
     "season about right."),
]

ERRATA = [
    ("A ridge penalty silently destroyed the main result",
     "Shrinking attack and defence toward zero also shrinks the gap BETWEEN "
     "divisions, because Championship clubs sit below zero and Premier League "
     "clubs above. It biased the exact quantity the joint fit exists to "
     "estimate: the learned gap collapsed from 266 to 73.",
     "It was my own suggestion, and it sounded principled."),
    ("A passing sanity check hid a misspecified model",
     "An early fit recovered a home advantage of 0.33, which looked right and "
     "was accepted. The model had no intercept, so home advantage was "
     "absorbing the overall goal level too. The real value is 0.21.",
     "The check tested whether a number looked plausible, not whether the "
     "model could represent the data. Prefer checks that fail structurally."),
    ("A finding that was true of its window and wrong as a claim",
     "The low-score correction rho was recorded as nil on modern data. One "
     "further season flipped it to -0.053, driven by 1-1 draws rising from "
     "10.5% to 14.0% of matches in three years.",
     "Findings now carry the date they were measured."),
    ("The weekly harness was double-counting a whole season",
     "The fixture table and the historical table both held the current "
     "season, so a dry run produced 760 matches in a 380-match season with "
     "every point doubled.",
     "Caught by replaying a completed season rather than by reading code."),
    ("The outright fitter was solving a different problem",
     "It discarded the injury and market-prior adjustments before fitting, "
     "then simulated all 380 fixtures from scratch while production simulates "
     "from the real table. Offsets fitted in one world, applied in another.",
     "Caught because the published table looked wrong to a human eye, and the "
     "bookmaker agreed with the human. Residual: 0.553 to 0.169."),
]


def _stat(v, k):
    return (f'<div class="stat"><span class="n">{v}</span>'
            f'<span class="k">{esc(k)}</span></div>')


def _ladder():
    lo, hi = 0.955, 1.105
    out = []
    for name, v, kind in LADDER:
        w = 100 * (hi - v) / (hi - lo)
        cls = {"base": "pale", "step": "mid", "ship": "kept",
               "market": "mkt"}[kind]
        out.append(
            f'<div class="lrow"><span class="lname">{esc(name)}</span>'
            f'<span class="ltrack"><i class="{cls}" style="width:{w:.1f}%"></i>'
            f'</span><span class="lval {cls}">{v:.4f}</span></div>')
    return '<div class="ladder">' + "".join(out) + "</div>"


def _table_now(d):
    """The league table as it stands. Results only, no model output."""
    rows = full_table(d)
    head = "".join(f"<th>{c}</th>" for c in
                   ("P", "W", "D", "L", "GF", "GA", "GD", "Pts"))
    body = []
    for r in rows:
        i = r["pos"]
        cls = "eur" if i <= 4 else "rel" if i >= 18 else ""
        body.append(
            f'<tr class="{cls}"><td class="num">{i}</td>'
            f'<td class="l club">{esc(r["team"])}</td>'
            f'<td>{r["P"]}</td><td>{r["W"]}</td><td>{r["D"]}</td>'
            f'<td>{r["L"]}</td><td>{r["GF"]}</td><td>{r["GA"]}</td>'
            f'<td>{r["GD"]:+d}</td><td class="pts">{r["Pts"]}</td></tr>')
    return ('<div class="tscroll"><table class="now"><thead><tr><th></th>'
            '<th class="l">Club</th>' + head + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def _table_final(d):
    """The projected FINAL table: expected points, the spread around them, and
    the three published markets. Ordered by expected points, which is the
    model's actual ranking rather than the current one."""
    rows = sorted(full_table(d), key=lambda r: -r["xpts"])
    lo = min(r["p10"] for r in rows)
    hi = max(r["p90"] for r in rows)
    span = max(hi - lo, 1.0)

    def x(v):
        return 100 * (v - lo) / span

    body = []
    for i, r in enumerate(rows, 1):
        cls = "eur" if i <= 4 else "rel" if i >= 18 else ""
        bar = (f'<span class="dist" title="10th-90th percentile '
               f'{r["p10"]:.0f} to {r["p90"]:.0f} points">'
               f'<i class="w" style="left:{x(r["p10"]):.1f}%;'
               f'width:{x(r["p90"])-x(r["p10"]):.1f}%"></i>'
               f'<i class="q" style="left:{x(r["p25"]):.1f}%;'
               f'width:{x(r["p75"])-x(r["p25"]):.1f}%"></i>'
               f'<b style="left:{x(r["p50"]):.1f}%"></b></span>')
        body.append(
            f'<tr class="{cls}"><td class="num">{i}</td>'
            f'<td class="l club">{esc(r["team"])}</td>'
            f'<td class="xp">{r["xpts"]:.1f}</td>'
            f'<td class="dim">{r["p10"]:.0f}</td>'
            f'<td>{r["p25"]:.0f}</td><td class="pts">{r["p50"]:.0f}</td>'
            f'<td>{r["p75"]:.0f}</td><td class="dim">{r["p90"]:.0f}</td>'
            f'<td class="l">{bar}</td>'
            f'<td>{pc(r["title"])}</td><td>{pc(r["top4"])}</td>'
            f'<td>{pc(r["releg"])}</td></tr>')
    return ('<div class="tscroll"><table class="final"><thead><tr><th></th>'
            '<th class="l">Club</th><th>xPts</th>'
            '<th class="dim">10th</th><th>25th</th><th>Median</th>'
            '<th>75th</th><th class="dim">90th</th>'
            '<th class="l">Points distribution</th>'
            '<th>Title</th><th>Top 4</th><th>Rel</th>'
            '</tr></thead><tbody>' + "".join(body) + "</tbody></table></div>")


def _layers():
    out = []
    for state, title, what, evidence in LAYERS:
        tag = "Kept" if state == "kept" else "Cut"
        out.append(
            f'<article class="layer {state}">'
            f'<span class="tag">{tag}</span>'
            f'<h3>{esc(title)}</h3>'
            f'<p class="what">{esc(what)}</p>'
            f'<p class="ev"><span class="evl">Evidence</span> {esc(evidence)}</p>'
            f'</article>')
    return out


def _errata():
    return "".join(
        f'<article class="err"><h3>{esc(t)}</h3>'
        f'<p>{esc(body)}</p><p class="lesson">{esc(lesson)}</p></article>'
        for t, body, lesson in ERRATA)


def full_table(d, season="2627", league="Prem"):
    """League table joined to the forecast. gather() carries points and goal
    difference but not the W/D/L/GF/GA breakdown a real table shows."""
    import pandas as pd
    import simulate as S
    from fixtures import load_fixtures, played_matches
    tab = S.results_table(played_matches(load_fixtures(season))).set_index("team")
    fc = {r["team"]: r for r in d["rows"]}
    rows = []
    for t, c in tab.iterrows():
        f = fc.get(t)
        if f is None:
            continue
        rows.append({"team": t, **{k: int(c[k]) for k in
                     ("pos", "P", "W", "D", "L", "GF", "GA", "GD", "Pts")},
                     "xpts": f["xpts"], "title": f["title"],
                     "top4": f["top4"], "releg": f["releg"],
                     **{f"p{q}": f[f"pts_{q}"] for q in (10, 25, 50, 75, 90)}})
    rows.sort(key=lambda r: r["pos"])
    return rows


CSS = """
:root{
  --paper:#ECEFEA; --surface:#FFFFFF; --raise:#F5F7F3;
  --ink:#121714; --body:#333D37; --muted:#5E6B63; --faint:#8B968E;
  --line:#D2D9D2; --hair:#E2E7E1;
  --kept:#1D6B45; --kept-soft:#DCEBE2;
  --cut:#A8452F; --cut-soft:#F4DED8;
  --mkt:#8A6A1B; --mkt-soft:#EFE6CF;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0C110E; --surface:#141A16; --raise:#1B221D;
    --ink:#E7EDE8; --body:#C0CAC3; --muted:#8C9891; --faint:#68746D;
    --line:#28322B; --hair:#1F2721;
    --kept:#54C08D; --kept-soft:#12301F;
    --cut:#E4886E; --cut-soft:#361B14;
    --mkt:#D6AA4C; --mkt-soft:#332711;
  }
}
:root[data-theme="dark"]{
  --paper:#0C110E; --surface:#141A16; --raise:#1B221D;
  --ink:#E7EDE8; --body:#C0CAC3; --muted:#8C9891; --faint:#68746D;
  --line:#28322B; --hair:#1F2721;
  --kept:#54C08D; --kept-soft:#12301F;
  --cut:#E4886E; --cut-soft:#361B14;
  --mkt:#D6AA4C; --mkt-soft:#332711;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--body);margin:0;
  font-family:'IBM Plex Sans',system-ui,-apple-system,Segoe UI,sans-serif;
  font-size:15.5px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:820px;margin:0 auto;padding:0 22px;padding-block:0 84px}
h1,h2,h3{color:var(--ink);margin:0;text-wrap:balance}
h1{font-family:Fraunces,Georgia,serif;font-size:clamp(38px,7.2vw,68px);
  font-weight:600;line-height:1.0;letter-spacing:-.022em;
  font-variation-settings:'SOFT' 0,'WONK' 1}
h2{font-family:Fraunces,Georgia,serif;font-size:clamp(24px,3.6vw,31px);
  font-weight:600;line-height:1.14;letter-spacing:-.014em}
h3{font-size:16.5px;font-weight:600;line-height:1.35;letter-spacing:-.008em}
p{margin:0}
a{color:var(--kept);text-decoration-thickness:1px;text-underline-offset:2px}
.lab{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:10.5px;
  letter-spacing:.15em;text-transform:uppercase;color:var(--faint);font-weight:500}
.mono{font-family:'IBM Plex Mono',ui-monospace,monospace;
  font-variant-numeric:tabular-nums}

header.hero{padding-block:70px 42px;display:flex;flex-direction:column;gap:20px;
  border-bottom:1px solid var(--line)}
.thesis{font-family:Fraunces,Georgia,serif;font-size:clamp(18px,2.5vw,21.5px);
  line-height:1.5;color:var(--body);max-width:33ch;font-weight:400}
.byline{display:flex;gap:8px 20px;flex-wrap:wrap;align-items:baseline}

.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
  background:var(--line);border-block:1px solid var(--line);margin-top:6px}
.stat{background:var(--paper);padding:15px 4px 14px;display:flex;
  flex-direction:column;gap:3px;min-width:0}
.stat .n{font-family:'IBM Plex Mono',monospace;font-size:23px;font-weight:600;
  color:var(--ink);line-height:1;font-variant-numeric:tabular-nums;
  letter-spacing:-.03em}
.stat .k{font-size:11.5px;color:var(--muted);line-height:1.3}

section{padding-block:52px 0}
.shead{display:flex;flex-direction:column;gap:9px;margin-bottom:26px}
.lede{font-size:16px;color:var(--muted);max-width:62ch}
.prose{max-width:64ch}
.prose + .prose{margin-top:14px}
.prose b{color:var(--ink);font-weight:600}

/* log-loss ladder */
.ladder{display:flex;flex-direction:column;gap:7px;
  font-family:'IBM Plex Mono',monospace}
.lrow{display:grid;grid-template-columns:210px 1fr 62px;gap:12px;
  align-items:center}
.lname{font-family:'IBM Plex Sans',sans-serif;font-size:13.5px;
  text-align:right;color:var(--body)}
.ltrack{height:17px;background:var(--hair);border-radius:2px;overflow:hidden}
.ltrack i{display:block;height:100%;border-radius:2px}
.lval{font-size:12.5px;font-variant-numeric:tabular-nums;font-weight:600}
i.pale{background:var(--faint)} .lval.pale{color:var(--muted)}
i.mid{background:var(--kept);opacity:.42} .lval.mid{color:var(--muted)}
i.kept{background:var(--kept)} .lval.kept{color:var(--kept)}
i.mkt{background:var(--mkt)} .lval.mkt{color:var(--mkt)}

/* league table */
.tscroll{overflow-x:auto;border:1px solid var(--line);border-radius:3px;
  background:var(--surface)}
table{border-collapse:collapse;width:100%;min-width:720px;font-size:13px}
th{font-family:'IBM Plex Mono',monospace;font-size:9.5px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--faint);font-weight:500;padding:10px 7px;
  text-align:right;background:var(--raise);border-bottom:1px solid var(--line);
  white-space:nowrap}
th.l,td.l{text-align:left}
td{padding:7px;text-align:right;border-bottom:1px solid var(--hair);
  font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums;
  font-size:12.5px;color:var(--body)}
tbody tr:last-child td{border-bottom:0}
td.club{font-family:'IBM Plex Sans',sans-serif;font-size:13.5px;font-weight:600;
  color:var(--ink);white-space:nowrap;border-left:3px solid transparent}
tr.eur td.club{border-left-color:var(--kept)}
tr.rel td.club{border-left-color:var(--cut)}
td.num{color:var(--faint)}
td.pts{color:var(--ink);font-weight:600}
td.xp{color:var(--kept);font-weight:600}

td.dim{color:var(--faint)}
th.dim{color:var(--faint);opacity:.75}
.dist{position:relative;display:inline-block;width:150px;height:11px;
  vertical-align:middle}
.dist i{position:absolute;height:100%;border-radius:1px;display:block}
.dist i.w{background:var(--kept);opacity:.18}
.dist i.q{background:var(--kept);opacity:.42}
.dist b{position:absolute;width:2px;height:100%;background:var(--kept)}
table.final{min-width:930px}

/* kept / cut layers */
.layers{display:flex;flex-direction:column;gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:3px;overflow:hidden}
.layer{background:var(--surface);padding:19px 20px;display:grid;
  grid-template-columns:64px 1fr;gap:6px 18px;align-items:start}
.layer h3{grid-column:2}
.layer .what,.layer .ev{grid-column:2;font-size:14.5px}
.layer .what{color:var(--body)}
.layer .ev{color:var(--muted);font-size:13.5px;padding-top:2px}
.evl{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.12em;
  text-transform:uppercase;margin-right:7px}
.tag{grid-row:1/2;grid-column:1;font-family:'IBM Plex Mono',monospace;
  font-size:10px;letter-spacing:.11em;text-transform:uppercase;font-weight:600;
  padding:3px 0;border-top:2px solid;margin-top:4px}
.layer.kept .tag{color:var(--kept);border-color:var(--kept)}
.layer.cut .tag{color:var(--cut);border-color:var(--cut)}
.layer.kept .evl{color:var(--kept)}
.layer.cut .evl{color:var(--cut)}

/* errata */
.errata{display:flex;flex-direction:column;gap:20px}
.err{border-left:2px solid var(--cut);padding:2px 0 2px 18px}
.err p{font-size:14.5px;color:var(--body);max-width:62ch;margin-top:5px}
.err .lesson{color:var(--cut);font-size:13.5px;margin-top:7px}

.grid2{display:grid;grid-template-columns:1fr 1fr;gap:30px}
.mini{width:100%;border-collapse:collapse;min-width:0}
.mini th{background:transparent;padding:6px 5px}
.mini td{padding:6px 5px}

.pull{border-block:1px solid var(--line);padding-block:22px;margin-block:34px;
  font-family:Fraunces,Georgia,serif;font-size:clamp(19px,2.7vw,23px);
  line-height:1.42;color:var(--ink);max-width:36ch;font-weight:500}

footer{margin-top:60px;padding-top:20px;border-top:1px solid var(--line);
  display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;
  font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--faint)}
@media (max-width:700px){
  .stats{grid-template-columns:1fr 1fr}
  .grid2{grid-template-columns:1fr}
  .lrow{grid-template-columns:132px 1fr 54px;gap:9px}
  .lname{font-size:12px}
  .layer{grid-template-columns:1fr;gap:5px}
  .layer h3,.layer .what,.layer .ev{grid-column:1}
  .tag{grid-row:auto;grid-column:1;justify-self:start;padding-right:14px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;
  transition:none!important}}
"""

FONTS = ("https://fonts.googleapis.com/css2?"
         "family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&"
         "family=IBM+Plex+Sans:wght@400;600&"
         "family=IBM+Plex+Mono:wght@400;500;600&display=swap")


def render(d):
    meta = d["meta"]
    m = meta["model"]
    rows = full_table(d)
    n_kept = sum(1 for s, *_ in LAYERS if s == "kept")
    n_cut = len(LAYERS) - n_kept
    lead = max(rows, key=lambda r: r["xpts"])

    season_rows = "".join(
        f'<tr><td class="l club">{esc(n)}</td><td>{ll:.4f}</td>'
        f'<td>{base:.4f}</td><td class="xp">{sk:.0%}</td></tr>'
        for n, ll, base, sk in SEASON_LEVEL)

    return f"""<title>Twenty-Six Seasons</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{CSS}</style>

<div class="wrap">

<header class="hero">
  <span class="lab">Premier League forecasting &middot; 2000&ndash;2026</span>
  <h1>Twenty-six seasons,<br>and the things that<br>did not work</h1>
  <p class="thesis">A football prediction system built from scratch: an Elo
  power rating, a Dixon-Coles goal model fitted jointly across two divisions,
  and a Monte Carlo season simulator now forecasting 2026-27 live. Roughly half
  the ideas tried were thrown away, and those are recorded too.</p>
  <div class="byline">
    <span class="lab">Miles McGarty</span>
    <span class="lab"><a href="{REPO}">github.com/milesmcgarty</a></span>
  </div>
</header>

<div class="stats">
  {_stat(f'{SCALE["matches"]:,}', 'matches, both divisions, cleaned and name-mapped')}
  {_stat(SCALE["seasons"], 'seasons, 2000-01 to 2025-26')}
  {_stat(SCALE["held_out_seasons"], 'seasons held out for scoring')}
  {_stat(f'{n_kept}/{n_kept + n_cut}', 'components that survived testing')}
</div>

<section>
  <div class="shead">
    <span class="lab">The result</span>
    <h2>It beats a base rate by a clear margin, and has never beaten the
    bookmakers</h2>
    <p class="lede">Rolling backtest over {SCALE["held_out_seasons"]} held-out
    seasons and {SCALE["held_out_matches"]:,} Premier League matches.
    Hyperparameters were selected on earlier seasons and frozen before these
    were scored. Log loss, lower is better.</p>
  </div>
  {_ladder()}
  <p class="prose" style="margin-top:24px">The gap to the closing line is
  <b>0.0120</b>. That the model never beats the market in any of the nine
  seasons is the reassuring outcome, not the disappointing one: a system with no
  team-news feed and no tracking data should not outprice a bookmaker, and a
  result claiming otherwise would be evidence of a bug. The market is the
  benchmark, not the competitor.</p>
  <p class="prose">The probabilities are also <b>calibrated</b>, not merely
  accurate on average. Binning predicted probability against realised frequency,
  no bin with more than 100 matches deviates by more than two standard errors.
  A model can post a respectable log loss while being systematically
  overconfident, so this is checked separately.</p>
</section>

<section>
  <div class="shead">
    <span class="lab">Live &middot; snapshot {esc(meta["as_of"][:10])}</span>
    <h2>The 2026-27 table, and where it finishes</h2>
    <p class="lede"><b>{d["played"]} of 380 league fixtures</b> played &mdash;
    {d["played"] // 10} of 38 matches per club. The remaining
    {d["remaining"]} are simulated {meta["n_sims"]:,} times each. Every figure
    here is read from a dated snapshot on disk rather than typed.</p>
  </div>
  <h3 style="margin-bottom:12px">As it stands</h3>
  {_table_now(d)}
  <h3 style="margin:34px 0 12px">Projected final table</h3>
  <p class="prose" style="margin-bottom:14px;font-size:14.5px">Ordered by
  expected points, so this is the model's ranking rather than today's. The
  percentiles are the spread of final points across
  {meta["n_sims"]:,} simulated seasons: the pale bar spans the 10th to 90th,
  the solid block the 25th to 75th, and the line marks the median. A wide bar
  is a club the model genuinely cannot place.</p>
  {_table_final(d)}
  <p class="prose" style="margin-top:18px">{esc(lead["team"])} lead the forecast
  on {lead["xpts"]:.1f} expected points at {pc(lead["title"])} for the title.
  The model is re-fitted and re-simulated every week, and each run is written to
  its own dated folder recording the exact configuration and git commit that
  produced it, so a mid-season change to the model shows up in the history
  instead of quietly rewriting it.</p>
</section>

<section>
  <div class="shead">
    <span class="lab">Season-level scoring</span>
    <h2>Title and relegation probabilities, graded separately</h2>
    <p class="lede">Match log loss says almost nothing about whether season
    probabilities are any good &mdash; they aggregate 380 correlated fixtures
    and fail differently. Scored against a historical base rate over the same
    nine held-out seasons.</p>
  </div>
  <div class="grid2">
    <div>
      <div class="tscroll"><table class="mini"><thead><tr>
        <th class="l">Market</th><th>Model</th><th>Base rate</th><th>Skill</th>
      </tr></thead><tbody>{season_rows}
      <tr><td class="l club">Final position (RPS)</td><td>0.1068</td>
      <td>0.1750</td><td class="xp">39%</td></tr>
      </tbody></table></div>
    </div>
    <div>
      <p class="prose">Relegation is the weakest of the three, and promoted
      clubs are the reason: Championship form correlates <b>+0.004</b> with
      Premier League points across 75 promotions. It is the single largest
      remaining defect, and it is also where the market prior earns most of its
      keep.</p>
      <p class="prose">The season product has never been scored against a
      bookmaker, because outright prices were not archived until September 2026.
      Every weekly run now freezes them, so that comparison switches on by
      itself once this season finishes.</p>
    </div>
  </div>
</section>

<div class="pull">Nearly every real defect in this project was found by checking
a number, never by reading code.</div>

<section>
  <div class="shead">
    <span class="lab">Architecture</span>
    <h2>What went in, and what came back out</h2>
    <p class="lede">Each component had to improve a rolling backtest with its
    hyperparameters chosen on earlier seasons. {n_cut} of {n_kept + n_cut} did
    not, and the code for those is still in the repository.</p>
  </div>
  <div class="layers">{"".join(_layers())}</div>
</section>

<section>
  <div class="shead">
    <span class="lab">Errata</span>
    <h2>Five bugs worth more than the features</h2>
    <p class="lede">Kept deliberately. Each one passed casual inspection and was
    caught by a check that could fail structurally.</p>
  </div>
  <div class="errata">{_errata()}</div>
</section>

<section>
  <div class="shead">
    <span class="lab">Method</span>
    <h2>How it is kept honest</h2>
  </div>
  <p class="prose"><b>Nested selection.</b> Both the time-decay half-life and
  the training-window length are chosen on tuning seasons and frozen before the
  reporting seasons are touched. Choosing the window after seeing test results
  is the same leakage as tuning the decay on them, and is easy to miss.</p>
  <p class="prose"><b>Identical comparison sets.</b> Missing odds are not
  randomly distributed &mdash; they cluster on obscure fixtures, which are the
  hard ones. Scoring the market on a shrunken set while the model is scored on
  everything would hand the market an easier exam.</p>
  <p class="prose"><b>Checks that can fail structurally.</b> An analytic
  gradient verified against finite differences. A simulator that reproduces a
  known final table exactly. A parameter that must stay stable as the window
  length changes. Byte-identical output when a new knob is set to its neutral
  value. Not "does this look sensible".</p>
  <p class="prose"><b>Dated findings.</b> Every recorded result carries the date
  it was measured, because at least one of them was true of its window and wrong
  as a general claim.</p>
</section>

<footer>
  <span>Python, NumPy, SciPy, pandas &middot; {SCALE["modules"]} modules,
  {SCALE["lines"]:,} lines &middot; no notebooks</span>
  <span>Built {datetime.now().strftime("%B %Y")} &middot;
  window {m["window_seasons"]}yr &middot; half-life {m["half_life_days"]}d
  &middot; home {m["home_adv"]:.3f}</span>
</footer>
</div>
"""


def build(out=OUT):
    d = gather()
    out.write_text(render(d), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    return out


if __name__ == "__main__":
    build()
