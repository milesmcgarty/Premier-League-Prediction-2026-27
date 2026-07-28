import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson

ROOT = Path(__file__).parent.parent
MATCHES = ROOT / "data" / "processed" / "matches_combined.csv"


def fit_poisson(matches):
    """
    Fit attack + defence ratings per team by maximum likelihood.
    Expected goals:
      home ~ exp(attack_home - defence_away + home_advantage)
      away ~ exp(attack_away - defence_home)
    Find the ratings that make the ACTUAL goals most likely (Poisson).
    """
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    home_idx = matches["home_team"].map(idx).to_numpy()
    away_idx = matches["away_team"].map(idx).to_numpy()
    home_goals = matches["home_goals"].to_numpy()
    away_goals = matches["away_goals"].to_numpy()

    init = np.concatenate([np.zeros(n), np.zeros(n), [0.25]])

    def neg_log_likelihood(params):
        attack = params[:n]
        defence = params[n:2 * n]
        home_adv = params[2 * n]

        lambda_home = np.exp(attack[home_idx] - defence[away_idx] + home_adv)
        lambda_away = np.exp(attack[away_idx] - defence[home_idx])

        ll = poisson.logpmf(home_goals, lambda_home) + poisson.logpmf(away_goals, lambda_away)
        penalty = 1000 * (attack.sum() ** 2) + 1000 * (defence.sum() ** 2)
        return -ll.sum() + penalty

    result = minimize(neg_log_likelihood, init, method="L-BFGS-B",
                      options={"maxiter": 1000})
    if not result.success:
        print(f"  [optimiser note] {result.message}")

    attack = result.x[:n]
    defence = result.x[n:2 * n]
    home_adv = result.x[2 * n]

    ratings = pd.DataFrame({"team": teams, "attack": attack, "defence": defence})
    return ratings, home_adv


def dc_correction(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Dixon-Coles low-score adjustment for 0-0, 1-0, 0-1, 1-1."""
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_home * lambda_away * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_home * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_away * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0


def predict_match(ratings, home_adv, home_team, away_team, rho=-0.05, max_goals=8):
    """Predict one match: win/draw/loss probs, expected goals, likely scoreline."""
    r = ratings.set_index("team")
    lambda_home = np.exp(r.loc[home_team, "attack"] - r.loc[away_team, "defence"] + home_adv)
    lambda_away = np.exp(r.loc[away_team, "attack"] - r.loc[home_team, "defence"])

    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    grid = np.outer(home_probs, away_probs)

    for hg in range(2):
        for ag in range(2):
            grid[hg, ag] *= dc_correction(hg, ag, lambda_home, lambda_away, rho)
    grid /= grid.sum()

    home_win = np.tril(grid, -1).sum()
    draw = np.trace(grid)
    away_win = np.triu(grid, 1).sum()
    hg, ag = np.unravel_index(grid.argmax(), grid.shape)

    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "exp_home_goals": lambda_home, "exp_away_goals": lambda_away,
        "likely_score": (int(hg), int(ag)),
    }


if __name__ == "__main__":
    df = pd.read_csv(MATCHES, dtype={"season": str})
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    season = df[(df["season"] == "2425") & (df["league"] == "Prem")]
    ratings, home_adv = fit_poisson(season)
    print(f"Fitted home advantage: {home_adv:.3f}\n")

    fixtures = [
        ("Liverpool", "Everton"),
        ("Arsenal", "Manchester City"),
        ("Manchester City", "Arsenal"),
        ("Chelsea", "Brighton"),
    ]

    for home, away in fixtures:
        p = predict_match(ratings, home_adv, home, away)
        print(f"{home} vs {away}")
        print(f"  {home} win: {p['home_win']*100:4.1f}%  |  "
              f"Draw: {p['draw']*100:4.1f}%  |  {away} win: {p['away_win']*100:4.1f}%")
        print(f"  Expected goals: {p['exp_home_goals']:.2f} - {p['exp_away_goals']:.2f}"
              f"   Likely score: {p['likely_score'][0]}-{p['likely_score'][1]}\n")
