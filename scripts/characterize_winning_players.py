"""What do the real winners (data/reference/top_winning_players.csv, built by
find_winning_players.py) actually do differently from everyone else? Rather
than eyeball the top 30, this computes standard per-player stats (VPIP, PFR,
postflop aggression factor, preflop 3-bet%, average open size) for every
player in the reliable set (>=min_hands, same set find_winning_players.py
ranked) and correlates each against real bb_per_100 -- so the answer is
"what predicts winning across the whole reliable population," not a
just-so story built from eyeballing a handful of names.

Usage: python3 scripts/characterize_winning_players.py [min_hands]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

WINRATE_PATH = "data/reference/top_winning_players.csv"


def main():
    min_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    t0 = time.monotonic()
    # Single read with every column this script needs (position included) --
    # the first version read actions.parquet TWICE (once here, once again
    # further down just for `position`), holding both in memory at once at
    # peak. Also downcasts `player` to category right after loading: it's a
    # plain string column with ~91.7k distinct values repeated across 34.5M
    # rows, which is exactly the shape category dtype is for (real, measured
    # effect -- this alone was the dominant contributor to a single run of
    # this script's original version showing multiple GB of real memory use
    # in Activity Monitor, confirmed the hard way when two concurrent copies
    # of it pushed an 8GB machine into 19.86GB of swap and hung the machine).
    actions = pd.read_parquet(
        "data/processed/actions.parquet",
        columns=["hand_id", "player", "position", "street", "action", "amount_bb"],
    )
    actions["player"] = actions["player"].astype("category")
    print(f"loaded actions in {time.monotonic() - t0:.1f}s")

    # find_winning_players.py already wrote the top N -- but we want stats
    # for the FULL reliable population (>=min_hands), not just the top 30
    # it printed, so hands-played and bb_per_100 are recomputed here the
    # same way rather than re-reading a truncated CSV. Cheap relative to
    # the actions.parquet load above, and keeps this script runnable on
    # its own.
    hands = pd.read_parquet("data/processed/hands.parquet", columns=["hand_id", "small_blind", "big_blind", "pot", "rake"]).set_index("hand_id")
    outcomes = pd.read_parquet("data/processed/hand_outcomes.parquet", columns=["hand_id", "outcome_known", "winners", "n_winners"])
    pot_bb = hands["pot"] / hands["big_blind"]
    small_blind_bb = hands["small_blind"] / hands["big_blind"]

    action_contrib = actions.groupby(["hand_id", "player"], observed=True)["amount_bb"].sum()
    positions = actions[["hand_id", "player", "position"]].drop_duplicates(["hand_id", "player"])
    blind_rows = positions[positions["position"].isin(["SB", "BB"])].copy()
    blind_rows = blind_rows.join(small_blind_bb.rename("sb_bb"), on="hand_id")
    blind_rows["blind_bb"] = np.where(blind_rows["position"] == "BB", 1.0, blind_rows["sb_bb"])
    blind_contrib = blind_rows.set_index(["hand_id", "player"])["blind_bb"]
    total_contrib = action_contrib.add(blind_contrib, fill_value=0.0)

    hand_totals = total_contrib.groupby("hand_id").sum()
    balances = (hand_totals.reindex(pot_bb.index) - pot_bb).abs() < 1e-6
    known_ids = set(outcomes.loc[outcomes["outcome_known"], "hand_id"])
    clean_ids = set(balances[balances].index) & known_ids
    total_contrib = total_contrib[total_contrib.index.get_level_values("hand_id").isin(clean_ids)]

    outcomes = outcomes[outcomes["hand_id"].isin(clean_ids)]
    won = outcomes.explode("winners").dropna(subset=["winners"])
    won = won.join(pot_bb.rename("pot_bb"), on="hand_id")
    won["share_bb"] = won["pot_bb"] / won["n_winners"]
    win_share = won.set_index(["hand_id", "winners"])["share_bb"]
    win_share.index.set_names(["hand_id", "player"], inplace=True)
    net = win_share.reindex(total_contrib.index, fill_value=0.0) - total_contrib
    per_player = net.groupby("player", observed=True).agg(["sum", "count"])
    per_player.columns = ["net_bb", "clean_hands"]
    per_player["bb_per_100"] = 100 * per_player["net_bb"] / per_player["clean_hands"]

    # descriptive stats from the FULL (unfiltered) action history -- these
    # don't need outcome-known/pot-balance filtering, they're just action
    # frequencies, not money reconstruction
    preflop = actions[actions["street"] == "preflop"].copy()
    hands_seen = preflop.groupby("player", observed=True)["hand_id"].nunique()

    vpip_hands = preflop[preflop["action"].isin(["calls", "bets", "raises"])].groupby("player", observed=True)["hand_id"].nunique()
    pfr_hands = preflop[preflop["action"] == "raises"].groupby("player", observed=True)["hand_id"].nunique()

    postflop = actions[actions["street"] != "preflop"]
    postflop_aggr = postflop[postflop["action"].isin(["bets", "raises"])].groupby("player", observed=True)["hand_id"].count()
    postflop_calls = postflop[postflop["action"] == "calls"].groupby("player", observed=True)["hand_id"].count()

    # raises_before = how many raises happened strictly BEFORE this row
    # within the hand (same technique as find_positional_openers.py's
    # raises_before, subtracting the row's own raise indicator from the
    # inclusive cumsum so a row where the player themselves raises doesn't
    # count itself).
    is_raise = (preflop["action"] == "raises").astype(int)
    preflop["raises_before"] = preflop.groupby("hand_id", sort=False)["action"].transform(lambda s: (s == "raises").cumsum()) - is_raise

    facing_one_raise = preflop[(preflop["raises_before"] == 1) & (preflop["action"] != "raises")]
    threebet_hands = preflop[(preflop["raises_before"] == 1) & (preflop["action"] == "raises")].groupby("player", observed=True)["hand_id"].nunique()
    faced_one_raise_hands = facing_one_raise.groupby("player", observed=True)["hand_id"].nunique()

    opens = preflop[(preflop["action"] == "raises") & (preflop["raises_before"] == 0)]
    avg_open_bb = opens.groupby("player", observed=True)["amount_bb"].mean()

    stats_df = pd.DataFrame({"hands_seen": hands_seen})
    stats_df["vpip"] = (vpip_hands / stats_df["hands_seen"]).fillna(0.0)
    stats_df["pfr"] = (pfr_hands / stats_df["hands_seen"]).fillna(0.0)
    stats_df["postflop_af"] = (postflop_aggr / postflop_calls.replace(0, np.nan))
    stats_df["threebet_pct"] = (threebet_hands / faced_one_raise_hands.replace(0, np.nan))
    stats_df["avg_open_bb"] = avg_open_bb

    merged = stats_df.join(per_player, how="inner")
    reliable = merged[(merged["hands_seen"] >= min_hands) & (merged["clean_hands"] >= min_hands * 0.5)].copy()
    reliable = reliable.dropna(subset=["postflop_af", "threebet_pct"])
    print(f"\n{len(reliable)} players with >= {min_hands} hands seen and a resolvable postflop_af/3bet%\n")

    print("Spearman correlation of each stat with real bb_per_100:")
    for col in ["vpip", "pfr", "postflop_af", "threebet_pct", "avg_open_bb"]:
        sub = reliable[[col, "bb_per_100"]].replace([np.inf, -np.inf], np.nan).dropna()
        corr, pvalue = stats.spearmanr(sub[col], sub["bb_per_100"])
        print(f"  {col:14s} rho={corr:+.4f}  p={pvalue:.6f}  (n={len(sub)})")

    print("\ntop decile (by bb_per_100) vs bottom decile, mean stats:")
    reliable_sorted = reliable.sort_values("bb_per_100")
    decile = max(1, len(reliable_sorted) // 10)
    bottom = reliable_sorted.iloc[:decile]
    top = reliable_sorted.iloc[-decile:]
    for col in ["vpip", "pfr", "postflop_af", "threebet_pct", "avg_open_bb", "bb_per_100"]:
        print(f"  {col:14s} bottom={bottom[col].mean():.4f}   top={top[col].mean():.4f}")

    reliable.to_csv("data/reference/player_stats_vs_winrate.csv")
    print("\nwrote data/reference/player_stats_vs_winrate.csv")


if __name__ == "__main__":
    main()
