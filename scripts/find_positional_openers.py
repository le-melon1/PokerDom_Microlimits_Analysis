"""Is there a real player who opens (raises an unopened pot) near-100% of
the time from ONE SPECIFIC position, but plays much tighter from every
other position -- a genuinely positional leak, distinct from a generic
Maniac who's just loose everywhere (already covered by the existing
archetype/adaptation work)?

For every preflop action, determines whether the pot was still unopened
at that point (no raise yet this hand) -- vectorized via a cumulative
raise count within each hand, same "row order = chronological order"
assumption every other sequential-action script in this project relies on.
Open rate by (player, position) = share of those unopened-pot opportunities
where the player raised.

RESULT (2026-08-11): only 6 (player, position) pairs qualify, ALL at BTN
(open rate 0.86-1.00 at BTN, n=20-90, vs <=0.42 everywhere else with a
real sample) -- makes sense, BTN is the one position where opening very
wide is close to standard theory anyway, so "near-100% at BTN, tight
elsewhere" is a modest, unsurprising leak, not a big one, and the small
per-player sample sizes (n=20-90) mean each individual flag is itself
noisy. Real, but narrow: this is a per-PLAYER exploit (only usable via
the player-profile bots, which already model individual real players by
profile_id -- see backend/bots/player_profile_bots.py in the sibling
Practice_App repo), not a population-level rule the static ABC strategy
could use (it has no per-opponent-identity model beyond archetype).
None of the 6 flagged players happen to be in the 35-player profile pool
selected for that system as of this run.

Usage: python3 scripts/find_positional_openers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

MIN_OPPORTUNITIES_AT_FLAGGED_POSITION = 20
MIN_OPPORTUNITIES_AT_OTHER_POSITIONS = 20
FLAGGED_OPEN_RATE_THRESHOLD = 0.85
OTHER_POSITIONS_OPEN_RATE_CEILING = 0.50


def main():
    print("loading actions.parquet (preflop only)...")
    actions = pd.read_parquet(
        "data/processed/actions.parquet", columns=["hand_id", "player", "position", "street", "action"]
    )
    preflop = actions[actions["street"] == "preflop"].copy()
    del actions

    preflop["is_raise"] = (preflop["action"] == "raises").astype(int)
    preflop["raises_before"] = preflop.groupby("hand_id", sort=False)["is_raise"].cumsum() - preflop["is_raise"]
    had_opportunity = preflop[preflop["raises_before"] == 0].copy()
    had_opportunity["opened"] = had_opportunity["action"] == "raises"
    print(f"{len(had_opportunity)} unopened-pot preflop opportunities")

    by_player_position = had_opportunity.groupby(["player", "position"], observed=True)["opened"].agg(["mean", "count"])
    by_player_position = by_player_position.rename(columns={"mean": "open_rate", "count": "n"})

    flagged_rows = []
    for (player, position), row in by_player_position.iterrows():
        if row["n"] < MIN_OPPORTUNITIES_AT_FLAGGED_POSITION or row["open_rate"] < FLAGGED_OPEN_RATE_THRESHOLD:
            continue
        others = by_player_position.loc[player].drop(index=position, errors="ignore")
        others = others[others["n"] >= MIN_OPPORTUNITIES_AT_OTHER_POSITIONS]
        if others.empty:
            continue
        if others["open_rate"].max() >= OTHER_POSITIONS_OPEN_RATE_CEILING:
            continue  # not positional -- opens wide elsewhere too (a generic Maniac, already covered)
        flagged_rows.append({
            "player": player,
            "flagged_position": position,
            "open_rate_at_position": row["open_rate"],
            "n_at_position": int(row["n"]),
            "max_open_rate_elsewhere": others["open_rate"].max(),
            "n_positions_with_sample_elsewhere": len(others),
        })

    flagged = pd.DataFrame(flagged_rows).sort_values("open_rate_at_position", ascending=False)
    print(f"\n{len(flagged)} (player, position) pairs look genuinely positional "
          f"(open>={FLAGGED_OPEN_RATE_THRESHOLD:.0%} at n>={MIN_OPPORTUNITIES_AT_FLAGGED_POSITION}, "
          f"<{OTHER_POSITIONS_OPEN_RATE_CEILING:.0%} everywhere else at n>={MIN_OPPORTUNITIES_AT_OTHER_POSITIONS}):")
    print(flagged.to_string(index=False))

    flagged.to_csv("data/reference/positional_openers.csv", index=False)
    print("\nsaved data/reference/positional_openers.csv")


if __name__ == "__main__":
    main()
