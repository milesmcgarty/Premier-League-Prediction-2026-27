"""Dry-run the live harness over a COMPLETED season.

This is the harness's correctness test, and it is not decorative: it caught a
real double-counting bug. run_snapshot combined matches_combined.csv with the
fixture table for the same season, so every played match was counted twice --
760 matches in a 380-match season, all points doubled. That would not have shown
up in the 2026-27 live run until load_results.py was next re-run mid-season with
refreshed CSVs, at which point every weekly snapshot would have been silently
wrong.

Checks, each able to fail structurally:
  1. match counts always sum to the true season length
  2. the forecast converges: the eventual champion's title probability rises
     toward 100% and the relegated sides toward 100% as results come in
  3. with every match played the forecast IS the final table -- expected points
     and expected position equal the real ones exactly, for every team
"""
import sys
import warnings

import pandas as pd

import simulate as S
from harness import run_snapshot
from paths import load_matches

warnings.filterwarnings("ignore")

SEASON = "2526"
LEAGUE = "Prem"
DATES = ["2025-08-01", "2025-10-01", "2025-12-01",
         "2026-02-01", "2026-04-01", "2026-06-01"]


def main(season=SEASON, league=LEAGUE, n_sims=4000):
    m = load_matches()
    played = m[(m["season"] == season) & (m["league"] == league)]
    actual = S.results_table(played).set_index("team")
    champion = actual.index[0]
    relegated = list(actual.index[-3:])
    total = len(played)

    print("=" * 84)
    print(f"HARNESS DRY RUN - {league} {season}")
    print("=" * 84)
    print(f"actual champion: {champion}")
    print(f"actual relegated: {', '.join(relegated)}")
    print(f"season length: {total} matches\n")

    print(f"{'as_of':>12}{'played':>8}{'left':>6}{'sum':>6}"
          f"{'  P(title) champ':>17}{'  P(rel) actual 3':>18}{'  top pick':>20}")
    print("-" * 84)

    rows = []
    count_ok = True
    for d in DATES:
        out = run_snapshot(season=season, as_of=d, n_sims=n_sims,
                           league=league, write=False, seed=5)
        f = out["forecast"].set_index("team")
        meta = out["meta"]
        tot = meta["matches_played"] + meta["matches_remaining"]
        count_ok &= (tot == total)
        rows.append({"as_of": d, "played": meta["matches_played"],
                     "title": f.loc[champion, "title"],
                     "rel": f.loc[relegated, "releg"].mean()})
        print(f"{d:>12}{meta['matches_played']:>8}{meta['matches_remaining']:>6}"
              f"{tot:>6}{f.loc[champion, 'title']:>17.1%}"
              f"{f.loc[relegated, 'releg'].mean():>18.1%}"
              f"{f['title'].idxmax()[:19]:>20}")

    R = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print("CHECKS")
    print("=" * 84)

    print(f"  1. played + remaining == {total} at every snapshot: "
          f"{'PASS' if count_ok else 'FAIL'}")

    rose = R["title"].iloc[-1] > R["title"].iloc[0]
    ended = R["title"].iloc[-1] > 0.999 and R["rel"].iloc[-1] > 0.999
    print(f"  2. forecast converges to reality: "
          f"champion {R['title'].iloc[0]:.1%} -> {R['title'].iloc[-1]:.1%}, "
          f"relegated {R['rel'].iloc[0]:.1%} -> {R['rel'].iloc[-1]:.1%}  "
          f"{'PASS' if rose and ended else 'FAIL'}")

    out = run_snapshot(season=season, as_of="2099-01-01", n_sims=500,
                       league=league, write=False, seed=5)
    f = out["forecast"].set_index("team")
    pts_ok = all(abs(float(f.loc[t, "exp_pts"]) - int(actual.loc[t, "Pts"])) < 1e-9
                 for t in actual.index)
    pos_ok = all(int(f.loc[t, "exp_pos"]) == int(actual.loc[t, "pos"])
                 for t in actual.index)
    print(f"  3. with all matches played the forecast IS the table: "
          f"points {'PASS' if pts_ok else 'FAIL'}, "
          f"positions {'PASS' if pos_ok else 'FAIL'}")

    ok = count_ok and rose and ended and pts_ok and pos_ok
    print("\n" + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
