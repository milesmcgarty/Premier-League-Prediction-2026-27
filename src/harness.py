"""Phase 7: the live weekly harness for the 2026-27 season.

Two commands, run weekly:

    py src\\fixtures.py     # refresh results from the FPL API
    py src\\harness.py      # re-fit, re-predict, re-simulate, snapshot

Every run writes a DATED, immutable snapshot. The point is not only the current
forecast but the HISTORY of forecasts -- how the title race, the top four and the
relegation fight looked week by week, and how the model reacted to results. That
series is the actual deliverable, and it can only be built by recording it as the
season happens; it cannot be reconstructed afterwards.

Each snapshot records the model that produced it (window, half-life, dispersion,
git commit). A mid-season model change is therefore visible in the history rather
than silently rewriting it -- if the transfer-value work lands in October, every
snapshot before and after says which model produced it.
"""
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

import dixon_coles as dc
import availability as AVAIL
import market_prior as MP
import simulate as S
from fixtures import CURRENT_SEASON, load_fixtures, played_matches
from paths import ROOT, load_matches

SNAPSHOT_DIR = ROOT / "data" / "snapshots"
N_SIMS = 20000
UPCOMING_DAYS = 10          # how far ahead to publish match probabilities


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _with_xg(m):
    """Attach match-level xG where we have it (Premier League, 2014-15 on).

    Missing rows keep NaN and fall back to goals inside the fit, so this is safe
    when the xG table is absent entirely.
    """
    try:
        import xg as _X
        return _X.attach_xg(m, _X.load_xg())
    except Exception as e:
        print(f"  NOTE: xG unavailable ({type(e).__name__}); fitting on goals only.")
        return m


def historical_matches(season):
    """Completed matches EXCLUDING `season`.

    The fixture table is the single source of truth for the current season. If
    matches_combined.csv also contains it -- which happens as soon as
    load_results.py is re-run mid-season with refreshed football-data CSVs --
    concatenating both would double-count every played match, doubling points
    and games. The dry run over 2025-26 caught exactly this.
    """
    m = load_matches().dropna(subset=["home_goals", "away_goals"])
    return _with_xg(m[m["season"] != season])


def build_history(season=CURRENT_SEASON):
    """Historical matches plus this season's results so far, in one frame."""
    hist = historical_matches(season)
    fx = load_fixtures(season)
    done = played_matches(fx)
    if len(done):
        hist = pd.concat([hist, done], ignore_index=True)
    return hist, fx


def run_snapshot(season=CURRENT_SEASON, as_of=None, n_sims=N_SIMS,
                 league="Prem", write=True, seed=1):
    """Fit, predict, simulate and snapshot. The whole weekly job."""
    hist, fx = build_history(season)
    as_of = (pd.Timestamp(as_of) if as_of is not None
             else pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)))

    done = played_matches(fx)
    done = done[done["date"] < as_of]

    # STALENESS GUARD. A fixture whose kick-off has passed but which carries no
    # result is either genuinely postponed or -- far more likely -- a sign the
    # results feed has not been refreshed. Without this the harness would quietly
    # SIMULATE matches that have already been played, and the weekly snapshot
    # would be wrong in a way nothing else would reveal.
    stale = fx[(fx["date"] < as_of) & fx["home_goals"].isna()]
    if len(stale):
        print(f"WARNING: {len(stale)} fixture(s) kicked off before "
              f"{as_of.date()} but have no result recorded.")
        print("         Run  py src/fixtures.py  to refresh, or accept")
        print("         postponed. They are being treated as UNPLAYED and will be")
        print("         simulated. Earliest: "
              f"{stale['date'].min().date()}; latest: {stale['date'].max().date()}")
        for _, r in stale.head(5).iterrows():
            print(f"           {r['date'].date()}  {r['home_team']} v {r['away_team']}")
        if len(stale) > 5:
            print(f"           ... and {len(stale)-5} more")
        print()

    # Fit on history PLUS this season's results so far. Those carry the newest
    # dates, so time decay already gives them the heaviest weight.
    fit = dc.fit_for_league(hist, season, league, extra=done, cutoff=as_of)

    # MARKET PRIOR for promoted teams. Their rating comes from Championship
    # form, which does not predict Premier League performance (corr +0.004 over
    # 75 promotions). Bookmakers price their opening fixtures knowing the summer
    # transfers and squad overhaul we cannot see, so we read a strength offset
    # off those prices. Held out this cuts promoted-fixture log loss from 0.9668
    # to 0.9446. Applied ONLY to promoted sides: for established teams the model
    # is already well calibrated and there is nothing to correct.
    _all = pd.concat([historical_matches(season), fx.assign(season=season)],
                     ignore_index=True)
    promoted = sorted(S.promoted_teams(_all, season, league))
    season_teams = sorted(set(fx["home_team"]) | set(fx["away_team"]))
    prior_info = {}
    if season_teams:
        fit, _prior_used = MP.apply_market_prior(fit, fx, season_teams)
        prior_info = getattr(fit, "market_prior_info", {})

    # KEY-PLAYER AVAILABILITY. Held out this is worth +0.0030 log loss ON TOP of
    # the market prior (positive in 8 of 9 seasons, p=0.018) -- unlike squad
    # value, which the prior absorbs entirely. The prior is fitted once on the
    # opening fixtures, so it cannot know who is injured in November; this can.
    # Live numbers come from FPL's injury flags, standardised onto the scale the
    # coefficient was tuned on, since the two sources centre differently.
    avail_info = {}
    try:
        live = AVAIL.live_from_fpl()
        off = AVAIL.offsets(live, standardise=True)
        merged = dict(fit.adjustments)
        for t, (da, dd) in off.items():
            d0 = merged.get(t, (0.0, 0.0))
            merged[t] = (d0[0] + da, d0[1] + dd)
        fit.adjustments = merged
        avail_info = {t: round(v, 4) for t, v in live.items()}
    except Exception as e:
        print(f"  NOTE: availability unavailable ({type(e).__name__}); "
              "injuries are invisible to this snapshot.")

    # Season simulation needs the full fixture list with results where played.
    combined = pd.concat(
        [historical_matches(season), fx.assign(season=season)], ignore_index=True)

    # Promoted-team dispersion is re-tuned each season on the six preceding ones,
    # because how predictable promoted sides are has been changing: the selected
    # value has risen from 0.15 to 0.45 over the last decade. A fixed value left
    # them at 44% coverage of their nominal 80% band.
    # Tuned WITH the prior active, on the seasons before this one. Both matter:
    # tuning without the prior compensates for an error the prior already fixed
    # (it selects 0.55, over-covers at 96% and gives promoted sides a 9% top-six
    # chance against a historical 0 in 75), and a single fixed value cannot work
    # because promoted-team predictability is non-stationary -- the rolling pick
    # climbs 0.15 -> 0.35 across the last nine seasons.
    sd_promoted, up_promoted = S.tune_promoted_sd(
        historical_matches(season), season, league, apply_prior=True)

    sim = S.simulate_season(combined, season, league, n_sims=n_sims,
                            as_of=as_of, fit=fit, seed=seed,
                            strength_sd_promoted=sd_promoted,
                            promoted_up_ratio=up_promoted)
    forecast = S.summarise(sim, league)

    # --- upcoming match probabilities ---
    upcoming = fx[(fx["date"] >= as_of) &
                  (fx["date"] < as_of + pd.Timedelta(days=UPCOMING_DAYS))].copy()
    if len(upcoming):
        preds = [fit.predict(r["home_team"], r["away_team"])
                 for _, r in upcoming.iterrows()]
        upcoming["prob_H"] = [p["home_win"] for p in preds]
        upcoming["prob_D"] = [p["draw"] for p in preds]
        upcoming["prob_A"] = [p["away_win"] for p in preds]
        upcoming["exp_home_goals"] = [p["exp_home_goals"] for p in preds]
        upcoming["exp_away_goals"] = [p["exp_away_goals"] for p in preds]
        upcoming["likely_score"] = [
            str(p["likely_score"][0]) + "-" + str(p["likely_score"][1])
            for p in preds]
        # The market blend was tested and did NOT beat the market (see blend.py),
        # so match predictions are published model-only. Tagged explicitly: a
        # snapshot history mixing blended and unblended numbers would be
        # uninterpretable later.
        upcoming["source"] = "model_only"

    meta = {
        "as_of": as_of.isoformat(timespec="seconds"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "league": league,
        "n_sims": n_sims,
        "matches_played": int(sim["n_played"]),
        "matches_remaining": int(sim["n_remaining"]),
        "model": {
            "window_seasons": dc.LEAGUE_PARAMS[league]["window"],
            "half_life_days": dc.LEAGUE_PARAMS[league]["half_life"],
            "strength_sd": S.STRENGTH_SD,
            "strength_sd_promoted": sd_promoted,
            "promoted_up_ratio": up_promoted,
            "promoted_teams": promoted,
            "market_prior_shrink": MP.SHRINK,
            "market_prior": {k: round(v["delta"], 4) for k, v in prior_info.items()},
            "market_prior_scope": "all teams",
            "availability_gamma": AVAIL.AVAIL_GAMMA,
            "availability": avail_info,
            "ridge": dc.RIDGE,
            "xg_weight": fit.xg_weight,
            "xg_training_matches": fit.n_xg_matches,
            "intercept": round(fit.intercept, 5),
            "home_adv": round(fit.home_adv, 5),
            "rho": round(fit.rho, 5),
            "training_matches": int(fit.n_matches),
            "effective_sample_size": round(fit.ess, 1),
            "converged": bool(fit.converged),
            "newcomer_prior": getattr(fit, "prior_source", None),
        },
        "git_commit": _git_commit(),
        "match_prediction_source": "model_only_plus_market_prior_for_promoted",
        "stale_fixtures": int(len(stale)),
    }

    if write:
        d = SNAPSHOT_DIR / season / as_of.strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(d / ("season_forecast_" + league + ".csv"), index=False)
        S.position_matrix(sim).to_csv(d / ("position_matrix_" + league + ".csv"))
        if len(upcoming):
            cols = ["date", "gameweek", "home_team", "away_team",
                    "prob_H", "prob_D", "prob_A", "exp_home_goals",
                    "exp_away_goals", "likely_score", "source"]
            upcoming[cols].to_csv(
                d / ("match_predictions_" + league + ".csv"), index=False)
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        meta["snapshot_dir"] = str(d.relative_to(ROOT))

    return {"forecast": forecast, "upcoming": upcoming, "meta": meta, "sim": sim}


def snapshot_history(season=CURRENT_SEASON, league="Prem", metric="title"):
    """Every snapshot's forecast for one metric, as a date x team table.

    This is what makes the week-by-week evolution plottable, and it is the
    reason snapshots are immutable and dated rather than overwritten.
    """
    root = SNAPSHOT_DIR / season
    if not root.exists():
        return pd.DataFrame()
    rows = {}
    for d in sorted(root.iterdir()):
        f = d / ("season_forecast_" + league + ".csv")
        if f.exists():
            s = pd.read_csv(f).set_index("team")
            if metric in s.columns:
                rows[d.name] = s[metric]
    return pd.DataFrame(rows).T.sort_index()


if __name__ == "__main__":
    import sys

    as_of = sys.argv[1] if len(sys.argv) > 1 else None
    out = run_snapshot(as_of=as_of)
    f, meta = out["forecast"], out["meta"]

    print("=" * 78)
    print("SNAPSHOT  " + meta["season"] + "  " + meta["league"]
          + "   as of " + meta["as_of"][:10])
    print("=" * 78)
    print("played " + str(meta["matches_played"])
          + " / remaining " + str(meta["matches_remaining"])
          + "   |  " + str(meta["n_sims"]) + " simulations"
          + "   |  model " + meta["git_commit"])
    print("training: " + str(meta["model"]["training_matches"]) + " matches, ESS "
          + str(meta["model"]["effective_sample_size"]) + ", home_adv "
          + str(meta["model"]["home_adv"]) + "\n")

    print(f"{'team':<24}{'xPts':>6}{'80% range':>12}{'title':>8}{'top4':>8}{'releg':>8}")
    print("-" * 66)
    for _, r in f.iterrows():
        rng = f"{r.pts_10:.0f}-{r.pts_90:.0f}"
        print(f"{r['team']:<24}{r['exp_pts']:>6.1f}{rng:>12}"
              f"{r['title']:>8.1%}{r['top4']:>8.1%}{r['releg']:>8.1%}")

    up = out["upcoming"]
    if len(up):
        print(f"\nNext {UPCOMING_DAYS} days ({len(up)} fixtures):")
        for _, r in up.iterrows():
            print(f"  {r['date'].strftime('%a %d %b')}  {r['home_team']:<22}"
                  f"{r['prob_H']:>6.1%}{r['prob_D']:>6.1%}{r['prob_A']:>6.1%}"
                  f"   {r['away_team']:<22} {r['likely_score']}")
    print("\nsnapshot written to " + str(meta.get("snapshot_dir")))
