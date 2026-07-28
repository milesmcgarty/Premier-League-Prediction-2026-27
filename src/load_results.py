import pandas as pd

from odds import clean_odds
from paths import PROCESSED_DIR, RESULTS_DIR, TEAMS_CSV

OUTPUT_DIR = PROCESSED_DIR

# --- The columns we want, grouped by type ---
MATCH_FACTS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"]
MATCH_STATS = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "Referee"]
ODDS        = ["B365H", "B365D", "B365A", "WHH", "WHD", "WHA", "AvgH", "AvgD", "AvgA"]
WANTED      = MATCH_FACTS + MATCH_STATS + ODDS

# Clean, human-readable names for the ones we rename
RENAME = {
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HTHG": "ht_home_goals", "HTAG": "ht_away_goals", "HTR": "ht_result",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls",
    "HY": "home_yellow", "AY": "away_yellow",
    "HR": "home_red", "AR": "away_red",
    "Referee": "referee",
}


def load_one_file(filepath):
    """Load one football-data CSV: keep wanted columns (where present), tag season+league."""
    # First, find the real header so we know the legit column names
    for enc in ["utf-8-sig", "latin1"]:
        try:
            header = pd.read_csv(filepath, encoding=enc, nrows=0).columns.tolist()
            break
        except UnicodeDecodeError:
            continue
    else:
        # Without this, `header` stays undefined and the next read dies with a
        # confusing NameError instead of naming the file that actually failed.
        raise RuntimeError(
            f"Could not decode {filepath.name} as utf-8-sig or latin1"
        )

    # Read using ONLY the declared header columns -- extra trailing junk on some
    # rows is ignored rather than crashing the parse or dropping the whole row.
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig",
                         usecols=lambda c: c in header,
                         on_bad_lines="warn")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin1",
                         usecols=lambda c: c in header,
                         on_bad_lines="warn")

    # Keep only wanted columns that actually exist in THIS file (permissive)
    available = [c for c in WANTED if c in df.columns]
    df = df[available].copy()

    # Drop junk/blank rows (older files have trailing empties)
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])

    # Tag season + league from filename, e.g. "Prem2526"
    stem = filepath.stem
    df["league"] = "Prem" if stem.startswith("Prem") else "Champ"
    df["season"] = str(stem[-4:]).zfill(4)   # keep "0001" as text, not int 1

    # Parse date (football-data uses day-first: 15/08/2025)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Rename to clean column names
    df = df.rename(columns=RENAME)
    df = df.rename(columns={"Date": "date"})

    return df


def apply_team_mapping(df):
    """Replace football-data team names with canonical names from teams.csv."""
    teams = pd.read_csv(TEAMS_CSV)
    mapping = dict(zip(teams["football_data"], teams["canonical_name"]))

    # Find any names in the data that AREN'T in the mapping, before replacing.
    # This MUST raise, not warn: .map() turns an unmapped name into NaN, which
    # silently deletes that club from the dataset. A newly promoted side whose
    # football-data spelling we haven't seen would vanish without a trace.
    all_names = set(df["home_team"]) | set(df["away_team"])
    unmapped = sorted(n for n in all_names if n not in mapping)
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} team name(s) in the results data are missing from "
            f"{TEAMS_CSV.name}: {unmapped}. Add them (with their canonical_name) "
            "before trusting the output -- unmapped names become NaN and the "
            "club is dropped from every downstream table."
        )
    print("\n  All team names mapped successfully.")

    df["home_team"] = df["home_team"].map(mapping)
    df["away_team"] = df["away_team"].map(mapping)
    return df


def load_all():
    """Load and combine every Prem + Champ file into one clean table."""
    files = sorted(RESULTS_DIR.glob("Prem*.csv")) + sorted(RESULTS_DIR.glob("Champ*.csv"))
    print(f"Found {len(files)} files to load")

    all_dfs = []
    for f in files:
        df = load_one_file(f)
        all_dfs.append(df)
        print(f"  {f.stem}: {len(df)} matches, {df.shape[1]} cols")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = apply_team_mapping(combined)
    print("\nOdds hygiene:")
    combined = clean_odds(combined)
    return combined


if __name__ == "__main__":
    combined = load_all()

    print(f"\n{'='*50}")
    print(f"COMBINED: {len(combined)} total matches")
    print(f"Seasons: {combined['season'].nunique()}, Leagues: {list(combined['league'].unique())}")
    print(f"Date range: {combined['date'].min().date()} -> {combined['date'].max().date()}")
    print(f"Columns ({combined.shape[1]}): {list(combined.columns)}")

    # Save it
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "matches_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")