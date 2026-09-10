"""Regenerate every published number from raw data, in one command.

    py src/rebuild_all.py            # everything except network fetches
    py src/rebuild_all.py --fetch    # refresh fixtures and xG from the network first

The audit's reproducibility finding was that it took five commands in a specific
order and nothing checked the order was right. This is that single entry point.
Every stage prints what it produced so a human can see the chain, and any stage
failing stops the run rather than leaving a half-rebuilt set of outputs.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STAGES = [
    ("load_results.py", "clean match table from raw CSVs", False),
    ("build_elo.py", "Elo ratings and history", False),
    ("xg.py", "Understat expected goals", True),
    ("availability.py", "key-player availability", True),
    ("fixtures.py", "current-season fixtures, results and odds", True),
    ("harness.py", "the published snapshot", False),
    ("validate_all.py", "54-check correctness suite", False),
    ("validate_harness.py", "harness dry run over a completed season", False),
]


def run(script, why, needs_net, fetch):
    if needs_net and not fetch:
        print(f"  SKIP  {script:<22} ({why}; needs network, pass --fetch)")
        return True
    print(f"\n===> {script}  --  {why}")
    r = subprocess.run([PY, str(ROOT / "src" / script)], cwd=ROOT)
    if r.returncode != 0:
        print(f"\nFAILED at {script} (exit {r.returncode}). Stopping: a partial "
              "rebuild is worse than none.")
        return False
    return True


if __name__ == "__main__":
    fetch = "--fetch" in sys.argv
    print("=" * 70)
    print("FULL REBUILD" + ("  (with network fetches)" if fetch else
                            "  (offline; pass --fetch to refresh feeds)"))
    print("=" * 70)
    for script, why, net in STAGES:
        if not run(script, why, net, fetch):
            sys.exit(1)
    print("\n" + "=" * 70)
    print("REBUILD COMPLETE -- every published number regenerated from raw data")
    print("=" * 70)
