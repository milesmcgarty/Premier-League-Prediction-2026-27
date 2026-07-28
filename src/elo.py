class EloEngine:
    def __init__(self, k_factor=20, base_rating=1500, home_advantage=70):
        self.k = k_factor
        self.base = base_rating
        self.home_adv = home_advantage
        self.ratings = {}   # team_name -> current rating

    def get(self, team):
        """Current rating for a team (base rating if unseen before)."""
        return self.ratings.get(team, self.base)

    def expected_score(self, rating_a, rating_b):
        """Probability that team A beats team B, per the Elo S-curve."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    @staticmethod
    def _gd_multiplier(goal_diff):
        """Bigger winning margins move ratings more. Standard football-Elo formula."""
        gd = abs(goal_diff)
        if gd <= 1:
            return 1.0
        elif gd == 2:
            return 1.5
        else:                      # 3+ goal margin
            return (11 + gd) / 8   # 3 goals -> 1.75, 4 -> 1.875, etc.

    def update_one_match(self, home, away, home_goals, away_goals):
        """Update both teams' ratings after a single match. Returns the rating change."""
        r_home = self.get(home)
        r_away = self.get(away)

        # Home advantage applied ONLY to the prediction, not stored in the rating
        exp_home = self.expected_score(r_home + self.home_adv, r_away)

        # Actual result for the home team: win=1, draw=0.5, loss=0
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals < away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5

        # Scale the update by the goal-difference multiplier
        mult = self._gd_multiplier(home_goals - away_goals)
        change = self.k * mult * (actual_home - exp_home)

        self.ratings[home] = r_home + change
        self.ratings[away] = r_away - change   # zero-sum: away loses what home gains
        return change


# --- Sanity tests ---
if __name__ == "__main__":
    # Test 1: home advantage should make an even home win LESS rewarded
    elo = EloEngine(k_factor=20, home_advantage=70)
    exp = elo.expected_score(1500 + 70, 1500)
    print(f"Even teams, home adv applied: expected home score = {exp:.3f}")  # >0.5, ~0.60
    change = elo.update_one_match("Home", "Away", 1, 0)
    print(f"  1-0 home win, change: {change:+.2f}")   # smaller than +10 now (~+8)

    # Test 2: goal-difference multiplier -- a thrashing moves more
    print("\nSame 1500 vs 1500 (no home adv for clarity), different margins:")
    for hg, ag in [(1, 0), (2, 0), (4, 0)]:
        e = EloEngine(k_factor=20, home_advantage=0)
        c = e.update_one_match("H", "A", hg, ag)
        print(f"  {hg}-{ag}: change {c:+.2f}  (multiplier {e._gd_multiplier(hg - ag)})")