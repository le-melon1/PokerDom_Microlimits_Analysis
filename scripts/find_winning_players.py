"""2026-08-11: who are the real, biggest winners in this dataset, and what
do they actually do differently? Grounds the "build a fuller strategy"
effort in real winning behavior instead of guessing from generic published
advice.

Net winnings aren't in the raw data (Public HandHQ-obfuscated files zero
out the `winnings` field -- see phh_parser.py's module docstring), so this
reconstructs each player's per-hand net from contributed amounts + who won:

  total_contributed_bb(player, hand) = sum(actions.amount_bb for that
  player-hand) + their blind (small_blind_bb if they were SB, 1.0 if BB,
  0 otherwise -- blinds are NOT logged as their own action rows, only the
  first real decision's amount_bb is already net of the blind already in,
  per the parser's `increment = new_total - contributed` logic).

  net_bb(player, hand) = share_of_pot_if_winner - total_contributed_bb,
  where share = (pot_bb - rake_bb) / n_winners for winners, 0 otherwise.

KNOWN GAP #1, disclosed rather than silently patched: if a player wins a
hand totally uncontested with ZERO actions logged (folded to preflop, so
no decision point was ever reached -- e.g. BB when everyone including SB
folds), their (hand_id, player, position) triple never appears in
actions_df, so their blind can't be attributed to them by this method.
Rather than guess, every hand is validated via pot conservation (sum of
reconstructed contributions must equal hands.parquet's real pot, in bb,
within float tolerance) and any hand that fails this check is EXCLUDED
from every player's stats entirely -- not estimated, not patched. Prints
what fraction of hands that is, so the exclusion's size is visible, not
hidden.

KNOWN GAP #2, a real bug caught before shipping (not by inspection --
the first run put literally every high-volume player at -10 to -21
bb/100, which is the kind of "too clean to be a real finding" result
this project's whole history says to distrust): 11.3% of hands
(401,915/3,564,757) have `outcome_known=False` in hand_outcomes.parquet
-- see extract_showdowns.py's docstring, a genuine data limit (a
showdown participant mucked without revealing, so who actually won is
unknowable from this data, not just unparsed). The first version of this
script still counted those hands' CONTRIBUTIONS (money going out of
every player's stack) while crediting the pot to nobody (since no winner
is known) -- a systematic downward bias hitting every player, which is
exactly why the "best" players all still showed negative. Fixed: hands
with outcome_known=False are excluded from every player's stats
entirely, same as the pot-conservation failures above.

Usage: python3 scripts/find_winning_players.py [min_hands] [top_n]
"""

import sys
import time

import numpy as np
import pandas as pd


def main():
    min_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    t0 = time.monotonic()
    actions = pd.read_parquet(
        "data/processed/actions.parquet",
        columns=["hand_id", "player", "position", "street", "action", "amount_bb"],
    )
    hands = pd.read_parquet("data/processed/hands.parquet", columns=["hand_id", "small_blind", "big_blind", "pot", "rake"])
    outcomes = pd.read_parquet(
        "data/processed/hand_outcomes.parquet", columns=["hand_id", "outcome_known", "winners", "n_winners"]
    )
    print(f"loaded in {time.monotonic() - t0:.1f}s")

    hands = hands.set_index("hand_id")
    pot_bb = hands["pot"] / hands["big_blind"]
    rake_bb = hands["rake"] / hands["big_blind"]
    small_blind_bb = hands["small_blind"] / hands["big_blind"]

    # 1. contribution from logged actions (already net of the blind already
    #    in, per the parser's increment logic)
    action_contrib = actions.groupby(["hand_id", "player"])["amount_bb"].sum()

    # 2. blinds, attributed via the position seen on any of that player's
    #    logged action rows this hand (misses the "won fully uncontested,
    #    zero actions logged" edge case -- caught by the pot-conservation
    #    check below, not guessed around)
    blind_rows = actions.loc[actions["position"].isin(["SB", "BB"]), ["hand_id", "player", "position"]].drop_duplicates()
    blind_rows = blind_rows.join(small_blind_bb.rename("sb_bb"), on="hand_id")
    blind_rows["blind_bb"] = np.where(blind_rows["position"] == "BB", 1.0, blind_rows["sb_bb"])
    blind_contrib = blind_rows.set_index(["hand_id", "player"])["blind_bb"]

    total_contrib = action_contrib.add(blind_contrib, fill_value=0.0)

    # 3. pot-conservation validation, per hand -- exclude any hand that
    #    doesn't balance instead of guessing at the missing piece
    hand_totals = total_contrib.groupby("hand_id").sum()
    reconstructed_vs_real = hand_totals.reindex(pot_bb.index) - pot_bb
    balances = reconstructed_vs_real.abs() < 1e-6
    n_total_hands = len(pot_bb)
    n_balances = int(balances.sum())
    print(f"pot-conservation check: {n_balances}/{n_total_hands} hands balance exactly "
          f"({n_balances / n_total_hands:.2%})")

    known_outcome_ids = set(outcomes.loc[outcomes["outcome_known"], "hand_id"])
    n_known = len(known_outcome_ids)
    print(f"known-winner check: {n_known}/{n_total_hands} hands have a determinable winner "
          f"({n_known / n_total_hands:.2%}) -- see KNOWN GAP #2 in the module docstring")

    clean_hand_ids = set(balances[balances].index) & known_outcome_ids
    print(f"using {len(clean_hand_ids)}/{n_total_hands} hands that pass BOTH checks "
          f"({len(clean_hand_ids) / n_total_hands:.2%})")
    total_contrib = total_contrib[total_contrib.index.get_level_values("hand_id").isin(clean_hand_ids)]

    # 4. winnings: explode the winners list, one row per (hand_id, winner)
    outcomes = outcomes[outcomes["hand_id"].isin(clean_hand_ids)]
    won = outcomes.explode("winners").dropna(subset=["winners"])
    won = won.join((pot_bb - rake_bb).rename("net_pot_bb"), on="hand_id")
    won["share_bb"] = won["net_pot_bb"] / won["n_winners"]
    win_share = won.set_index(["hand_id", "winners"])["share_bb"]
    win_share.index.set_names(["hand_id", "player"], inplace=True)

    # (win_share where present, else 0) - total_contrib (always present for
    # anyone who acted or posted a blind this hand)
    net = win_share.reindex(total_contrib.index, fill_value=0.0) - total_contrib

    per_player = net.groupby("player").agg(["sum", "count"])
    per_player.columns = ["net_bb", "hands"]
    per_player["bb_per_100"] = 100 * per_player["net_bb"] / per_player["hands"]
    # rough SE assuming per-hand net_bb has population std comparable to the
    # sample std observed here -- reported so small samples aren't overclaimed,
    # not a rigorous derivation
    per_hand_std = net.groupby("player").std()
    per_player["se_bb_per_100"] = 100 * per_hand_std / np.sqrt(per_player["hands"])

    reliable = per_player[per_player["hands"] >= min_hands].sort_values("bb_per_100", ascending=False)
    print(f"\n{len(reliable)} players with >= {min_hands} hands (clean-hand basis)\n")
    print(f"top {top_n} by bb/100:")
    top = reliable.head(top_n)
    for player, row in top.iterrows():
        print(f"  {player[:20]:22s} hands={row['hands']:>6.0f}  bb/100={row['bb_per_100']:>7.2f}  "
              f"(95% CI +/-{1.96 * row['se_bb_per_100']:.2f})")

    top.to_csv("data/reference/top_winning_players.csv")
    print("\nwrote data/reference/top_winning_players.csv")


if __name__ == "__main__":
    main()
