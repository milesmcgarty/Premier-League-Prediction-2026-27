import warnings
warnings.simplefilter("ignore", FutureWarning)

import soccerdata as sd
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

fbref = sd.FBref(leagues="ENG-Premier League", seasons="2024-2025")

# Step 1: get the real list of valid stat_types from the library itself
print("="*50)
print("VALID STAT TYPES")
print("="*50)
try:
    fbref.read_team_season_stats(stat_type="__invalid__")
except ValueError as e:
    print(e)

# Step 2: loop over the ACTUAL valid names and search each for xG
print("\n" + "="*50)
print("SEARCHING EACH STAT TYPE FOR xG")
print("="*50)

# These are the standard FBref stat_type names — if the list above differs,
# we'll correct this after seeing it
stat_types = ["standard", "keeper", "keeper_adv", "shooting", "passing",
              "passing_types", "goal_shot_creation", "defense", "possession",
              "playing_time", "misc"]

for st in stat_types:
    try:
        df = fbref.read_team_season_stats(stat_type=st)
        hits = [col for col in df.columns
                if any("xg" in str(level).lower() or "expected" in str(level).lower()
                       or "npxg" in str(level).lower()
                       for level in (col if isinstance(col, tuple) else (col,)))]
        if hits:
            print(f"  [{st}] HAS xG: {hits}")
        else:
            print(f"  [{st}] no xG")
    except Exception as e:
        print(f"  [{st}] skipped ({str(e)[:40]})")