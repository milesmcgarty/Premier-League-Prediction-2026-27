import duckdb
from pathlib import Path

db_path = Path(__file__).parent.parent / "transfermarkt-datasets.duckdb"
con = duckdb.connect(str(db_path))

# 1. Which release is this? (note it down for future-you)
print("--- VERSION ---")
print(con.execute("SELECT * FROM version").df())

# 2. Confirm the Premier League's competition ID
print("\n--- PREMIER LEAGUE ID ---")
print(con.execute("""
    SELECT competition_id, name, country_name
    FROM competitions
    WHERE name ILIKE '%premier%'
""").df())

# 3. What columns does the transfers table actually have?
print("\n--- TRANSFERS COLUMNS ---")
print(con.execute("DESCRIBE transfers").df())

# 4. And the appearances table (player-per-match: minutes, goals, assists)
print("\n--- APPEARANCES COLUMNS ---")
print(con.execute("DESCRIBE appearances").df())

# Sanity-check: do we actually have PL data, and does it look real?
print("\n--- PL APPEARANCES SAMPLE ---")
print(con.execute("""
    SELECT player_name, date, minutes_played, goals, assists
    FROM appearances
    WHERE competition_id = 'GB1'
    ORDER BY date DESC
    LIMIT 10
""").df())

print("\n--- HOW MANY PL APPEARANCES, AND DATE RANGE? ---")
print(con.execute("""
    SELECT COUNT(*) AS n_appearances,
           MIN(date) AS earliest,
           MAX(date) AS latest
    FROM appearances
    WHERE competition_id = 'GB1'
""").df())