import math

import pandas as pd

from paths import MATCHES_CSV as MATCHES

df = pd.read_csv(MATCHES, dtype={"season": str})
df = df.dropna(subset=["home_goals", "away_goals"])

# Average PPG in each league (sanity anchor)
def league_avg_ppg(league):
    m = df[df["league"] == league]
    # every match awards 2 or 3 points total; avg per team per game:
    total_points = ((m["home_goals"] > m["away_goals"]).sum() * 3 +
                    (m["away_goals"] > m["home_goals"]).sum() * 3 +
                    (m["home_goals"] == m["away_goals"]).sum() * 2)
    total_team_games = len(m) * 2
    return total_points / total_team_games

prem_ppg = league_avg_ppg("Prem")
champ_ppg = league_avg_ppg("Champ")
print(f"Avg PPG — Prem: {prem_ppg:.3f}, Champ: {champ_ppg:.3f}")
print(f"(Both should be near 1.35-1.40; leagues average out similarly internally)")

# The measured cross-division gap from the previous step
PPG_GAP = 0.874

# Convert PPG gap to Elo. A team scoring `PPG_GAP` fewer points/game is
# winning far less often. We map PPG to an expected match "score" (0-1):
# roughly, expected_score ≈ PPG / 3 for the win-heavy approximation, but
# a cleaner map uses points as fraction of max. We solve for the Elo gap D
# such that the S-curve expected score matches the observed performance drop.

# Observed: promoted team goes from ~top-of-Champ to ~bottom-of-PL.
# Express the gap as a shift in expected score per game:
# 0.874 PPG out of 3 = 0.291 shift in expected points-fraction.
exp_shift = PPG_GAP / 3.0
print(f"\nPPG gap as expected-score shift: {exp_shift:.3f}")

# Elo: expected_score = 1 / (1 + 10^(-D/400))
# We want the D that moves expected score from 0.5 by roughly this shift.
# Solve: 0.5 + exp_shift = 1/(1+10^(-D/400))
target = 0.5 + exp_shift
D = -400 * math.log10((1 / target) - 1)
print(f"\n>>> Implied Elo gap between divisions: {D:.0f} points <<<")
print(f"\nThis is the adjustment to apply when a team changes division:")
print(f"  Promoted (Champ->PL): subtract ~{D:.0f} from their rating")
print(f"  Relegated (PL->Champ): add ~{D:.0f} to their rating")