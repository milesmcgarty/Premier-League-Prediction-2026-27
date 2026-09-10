"""Schema contracts for anything this project publishes.

A published probability file is the product. Until now nothing checked that the
numbers in it were even internally coherent: a title column that failed to sum
to 1, a points band with the bounds the wrong way round, or a club missing
entirely would all have been written out and committed without complaint.

These assertions run BEFORE a snapshot is written. A snapshot that fails one is
not written at all, on the principle that no forecast is better than a quietly
malformed one.
"""
import numpy as np
import pandas as pd

TOL_TITLE = 0.005
TOL_PLACES = 0.02


class ContractError(AssertionError):
    """Raised when a published artefact violates its schema."""


def check_season_forecast(df, teams=None, league="Prem"):
    """Validate a season_forecast_<league>.csv before it is written."""
    errs = []
    required = ["team", "exp_pts", "pts_10", "pts_90", "title", "releg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ContractError(f"missing columns: {missing}")

    if teams is not None and set(df["team"]) != set(teams):
        extra = set(df["team"]) - set(teams)
        absent = set(teams) - set(df["team"])
        errs.append(f"team set mismatch (extra={sorted(extra)}, "
                    f"missing={sorted(absent)})")

    if df["team"].duplicated().any():
        errs.append("duplicate teams")

    for col in ("title", "releg") + (("top4",) if "top4" in df.columns else ()):
        v = df[col].to_numpy(float)
        if not np.isfinite(v).all():
            errs.append(f"{col}: non-finite values")
        elif (v < -1e-9).any() or (v > 1 + 1e-9).any():
            errs.append(f"{col}: outside [0, 1]")

    # a market must sum to the number of places it fills
    if abs(df["title"].sum() - 1.0) > TOL_TITLE:
        errs.append(f"title sums to {df['title'].sum():.4f}, expected 1.0")
    n_rel = 3 if league == "Prem" else 3
    if abs(df["releg"].sum() - n_rel) > TOL_PLACES:
        errs.append(f"releg sums to {df['releg'].sum():.4f}, expected {n_rel}")
    if "top4" in df.columns and abs(df["top4"].sum() - 4.0) > TOL_PLACES:
        errs.append(f"top4 sums to {df['top4'].sum():.4f}, expected 4.0")

    bad = df[(df["pts_10"] > df["exp_pts"]) | (df["exp_pts"] > df["pts_90"])]
    if len(bad):
        errs.append(f"{len(bad)} row(s) where pts_10 <= exp_pts <= pts_90 fails: "
                    f"{bad['team'].tolist()[:5]}")

    n_games = 38 if league == "Prem" else 46
    if (df["exp_pts"] < 0).any() or (df["exp_pts"] > 3 * n_games).any():
        errs.append("exp_pts outside [0, 3*games]")

    if errs:
        raise ContractError("season forecast failed its contract:\n  - "
                            + "\n  - ".join(errs))
    return True


def check_match_predictions(df):
    """Validate a match_predictions_<league>.csv before it is written."""
    errs = []
    for c in ("prob_H", "prob_D", "prob_A", "home_team", "away_team"):
        if c not in df.columns:
            raise ContractError(f"missing column: {c}")
    p = df[["prob_H", "prob_D", "prob_A"]].to_numpy(float)
    if not np.isfinite(p).all():
        errs.append("non-finite probabilities")
    elif np.abs(p.sum(axis=1) - 1.0).max() > 1e-6:
        errs.append(f"rows not summing to 1 (max deviation "
                    f"{np.abs(p.sum(axis=1) - 1.0).max():.2e})")
    if (df["home_team"] == df["away_team"]).any():
        errs.append("a team plays itself")
    if errs:
        raise ContractError("match predictions failed contract:\n  - "
                            + "\n  - ".join(errs))
    return True
