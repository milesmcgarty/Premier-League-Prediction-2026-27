"""
Stage 1 of building teams.csv: gather every distinct team name from all six
data sources and lay them out for manual alignment.
"""

import warnings
warnings.simplefilter("ignore", FutureWarning)

from pathlib import Path
import pandas as pd
import duckdb
import soccerdata as sd

# Anchor all paths to the project root (two levels up from this scratch file)
# This script is at data/raw/exploration/ — go up to the folder that CONTAINS data/
HERE = Path(__file__).resolve()
RAW_DIR = HERE.parent.parent          # data/raw/
RESULTS_DIR = RAW_DIR / "results"
DUCKDB_PATH = RAW_DIR / "transfermarkt-datasets.duckdb"

name_lists = {}

# --- Source 1: football-data.co.uk CSVs (both Prem and Champ) ---
print("Reading football-data CSVs...")
fd_names = set()
for csv in sorted(RESULTS_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(csv, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(csv, encoding="latin1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    fd_names.update(df["HomeTeam"].unique())
    fd_names.update(df["AwayTeam"].unique())
name_lists["football_data"] = sorted(fd_names)
print(f"  {len(fd_names)} distinct names")

# --- Source 2: Transfermarkt DuckDB (English clubs only) ---
print("Reading Transfermarkt clubs...")
con = duckdb.connect(str(DUCKDB_PATH))
tm = con.execute("""
    SELECT DISTINCT name
    FROM clubs
    WHERE domestic_competition_id IN ('GB1', 'GB2')
    ORDER BY name
""").df()
name_lists["transfermarkt"] = sorted(tm["name"].dropna().tolist())
print(f"  {len(name_lists['transfermarkt'])} distinct names")
con.close()

# --- Sources 3-5: soccerdata (FBref, Understat, ClubElo) ---
print("Reading FBref names...")
try:
    fbref = sd.FBref(leagues="ENG-Premier League", seasons="2024-2025")
    fb = fbref.read_team_season_stats(stat_type="standard")
    fb_names = sorted(fb.index.get_level_values("team").unique())
    name_lists["fbref"] = fb_names
    print(f"  {len(fb_names)} distinct names")
except Exception as e:
    print(f"  FBref failed: {e}")
    name_lists["fbref"] = []

print("Reading Understat names...")
try:
    us = sd.Understat(leagues="ENG-Premier League", seasons="2024-2025")
    us_df = us.read_team_match_stats()
    us_names = sorted(set(us_df["home_team"]) | set(us_df["away_team"]))
    name_lists["understat"] = us_names
    print(f"  {len(us_names)} distinct names")
except Exception as e:
    print(f"  Understat failed: {e}")
    name_lists["understat"] = []

print("Reading ClubElo names (English clubs)...")
try:
    elo = sd.ClubElo()
    ratings = elo.read_by_date()
    eng = ratings[ratings["country"] == "ENG"]
    elo_names = sorted(eng.index.get_level_values("team").unique())
    name_lists["clubelo"] = elo_names
    print(f"  {len(elo_names)} distinct names")
except Exception as e:
    print(f"  ClubElo failed: {e}")
    name_lists["clubelo"] = []

# --- Lay each list out as its own column, padded to equal length ---
max_len = max(len(v) for v in name_lists.values())
padded = {k: v + [""] * (max_len - len(v)) for k, v in name_lists.items()}
out = pd.DataFrame(padded)

output_dir = Path(__file__).parent

csv_path = output_dir / "team_names_by_source.csv"

csv_path = output_dir / "team_names_by_source.csv"
out.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

try:
    xlsx_path = output_dir / "team_names_by_source.xlsx"
    out.to_excel(xlsx_path, index=False)
    print(f"Saved: {xlsx_path}")
except Exception:
    print("(Excel export needs openpyxl - CSV is enough to work from.)")

print("\nEach column is one source's distinct team names, sorted.")
print("Open it and build teams.csv by lining up the same club across columns.")