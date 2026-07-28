import warnings
warnings.simplefilter("ignore", FutureWarning)

import soccerdata as sd
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

elo = sd.ClubElo()

# Today's full ranking of every club Elo tracks
ratings = elo.read_by_date()   # no date = today
print("SHAPE:", ratings.shape)
print("\nCOLUMNS:", list(ratings.columns))

# Filter to English clubs so we can read the spellings that matter to us.
# ClubElo tags country as 'ENG' for English teams.
english = ratings[ratings["country"] == "ENG"]
print(f"\nENGLISH CLUBS ({len(english)}):")
print(english[["rank", "country", "level", "elo"]].sort_values("elo", ascending=False).head(40))