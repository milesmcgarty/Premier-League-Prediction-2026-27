import warnings
warnings.simplefilter("ignore", FutureWarning)

import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import soccerdata as sd
from build_elo import build_ratings

ROOT = Path(__file__).parent.parent

# 1. Build YOUR ratings
print("Building your Elo ratings...")
my_elo, _ = build_ratings()

# 2. Pull ClubElo's CURRENT ratings for English clubs
print("Fetching ClubElo ratings...")
elo_reader = sd.ClubElo()
clubelo = elo_reader.read_by_date()   # today's ratings
english = clubelo[clubelo["country"] == "ENG"].copy()
clubelo_ratings = english["elo"].to_dict()   # {clubelo_name: rating}

# 3. Map ClubElo names to YOUR canonical names via teams.csv
teams = pd.read_csv(ROOT / "data" / "raw" / "exploration" / "teams.csv")
clubelo_to_canon = dict(zip(teams["clubelo"], teams["canonical_name"]))

# 4. Build a comparison table: teams present in BOTH systems
rows = []
for clubelo_name, ce_rating in clubelo_ratings.items():
    canon = clubelo_to_canon.get(clubelo_name)
    if canon and canon in my_elo.ratings:
        rows.append({
            "team": canon,
            "my_elo": round(my_elo.ratings[canon]),
            "clubelo": round(ce_rating),
        })

comp = pd.DataFrame(rows).sort_values("my_elo", ascending=False).reset_index(drop=True)

print("\n" + "="*55)
print(f"COMPARISON ({len(comp)} teams in both systems)")
print("="*55)
print(comp.to_string(index=False))

# 5. The actual test: rank correlation
rho, pval = spearmanr(comp["my_elo"], comp["clubelo"])
print("\n" + "="*55)
print(f"SPEARMAN RANK CORRELATION: {rho:.3f}")
print("="*55)
print("  0.9+  = excellent agreement (your engine is sound)")
print("  0.7-0.9 = good agreement (minor differences, expected)")
print("  <0.7  = something worth investigating")