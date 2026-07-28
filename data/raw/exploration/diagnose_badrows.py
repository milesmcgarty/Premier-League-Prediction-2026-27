import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"

for name in ["Prem0304.csv", "Prem0405.csv", "Champ0203.csv"]:
    path = RESULTS / name
    print(f"\n{'='*50}\n{name}")

    # Count raw lines in the file
    for enc in ["utf-8-sig", "latin1"]:
        try:
            with open(path, encoding=enc) as f:
                lines = f.readlines()
            print(f"  Raw line count ({enc}): {len(lines)}")
            break
        except Exception as e:
            print(f"  {enc} failed: {e}")

    # Try loading WITHOUT skipping, catch the exact error
    try:
        df = pd.read_csv(path, encoding="latin1")
        print(f"  Loaded OK with latin1 (no skip): {len(df)} rows")
    except Exception as e:
        print(f"  latin1 no-skip ERROR: {str(e)[:100]}")

    # How many columns does the header expect?
    df_head = pd.read_csv(path, encoding="latin1", nrows=1)
    print(f"  Header column count: {df_head.shape[1]}")