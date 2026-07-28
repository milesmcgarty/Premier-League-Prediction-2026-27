import warnings
warnings.simplefilter("ignore", FutureWarning)

import soccerdata as sd
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

understat = sd.Understat(leagues="ENG-Premier League", seasons="2025-2026")

# Team-level season stats — this is where Understat's xG should live
team_stats = understat.read_team_match_stats()
print("SHAPE:", team_stats.shape)
print("\nCOLUMNS:")
for col in team_stats.columns:
    print("   ", col)
print("\nFIRST FEW ROWS:")
print(team_stats.head())