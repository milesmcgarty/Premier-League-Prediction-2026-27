import requests
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# The main FPL data dump — no key, no auth needed
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
data = requests.get(url).json()

# What are the top-level sections?
print("TOP-LEVEL KEYS:", list(data.keys()))

# The 20 PL teams, with FPL's own IDs and short names
teams = pd.DataFrame(data["teams"])
print("\nTEAMS:")
print(teams[["id", "name", "short_name"]])

# Players — huge table. Look at the columns, then a few examples
players = pd.DataFrame(data["elements"])
print(f"\nPLAYERS: {len(players)} rows, {len(players.columns)} columns")

# The injury/availability fields — the whole reason FPL is in your stack
print("\nAVAILABILITY SAMPLE (players who are flagged):")
flagged = players[players["status"] != "a"]  # 'a' = available
print(flagged[["web_name", "status", "chance_of_playing_next_round", "news"]].head(10))