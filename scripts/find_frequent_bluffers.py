"""Identify players who bet/raised the river and then LOST at a real,
outcome-known showdown -- the standard proxy for "this was likely a bluff
that got caught" when hole cards for non-showdown folds are masked (which is
most folds -- we can only ever see a bluff that got TO showdown and lost;
successful bluffs that won the pot via a fold are invisible in this metric).

Real, disclosed limitation: this conflates "pure air bluff" with "a value
bet/thin bluff that happened to lose" -- there's no way to distinguish those
without the actual hand strength at bet time, which would need full-board
range reconstruction. Treat "bluff_rate" as "river-aggression-that-lost
rate," not a literal bluff-detector.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

MIN_RIVER_SHOWDOWNS = 40


def main():
    print("loading actions.parquet (river bets/raises only)...")
    actions = pd.read_parquet("data/processed/actions.parquet", columns=["hand_id", "player", "street", "action"])
    river_agg = actions[(actions["street"] == "river") & (actions["action"].isin(["bets", "raises"]))]
    # last river aggressive action per hand per player (a player could bet then get raised and call, etc;
    # we want whoever ended up as the LAST aggressor on the river, i.e. the one "showing strength" last)
    last_river_agg = river_agg.groupby("hand_id").tail(1)[["hand_id", "player"]].rename(columns={"player": "river_aggressor"})
    print(f"hands with a river aggressor: {len(last_river_agg)}")

    print("loading showdowns.parquet...")
    showdowns = pd.read_parquet("data/processed/showdowns.parquet", columns=["hand_id", "player", "outcome_known", "is_winner"])
    showdowns = showdowns[showdowns["outcome_known"]]

    merged = last_river_agg.merge(showdowns, left_on=["hand_id", "river_aggressor"], right_on=["hand_id", "player"], how="inner")
    print(f"river-aggressor-reached-real-showdown rows: {len(merged)}")

    population_loss_rate = 1 - merged["is_winner"].mean()
    print(f"population baseline: river aggressors lose at real showdown {population_loss_rate:.3f} of the time")
    print("(NOT a bug -- a known selection effect: someone who calls your river bet/raise usually")
    print("has a hand worth calling with, so the aggressor being behind more than half the time is expected)")

    g = merged.groupby("river_aggressor")
    n = g.size()
    losses = g["is_winner"].apply(lambda s: (~s).sum())
    # empirical-Bayes shrinkage toward the population rate -- ranking by raw
    # rate at small n is dominated by noise (a handful of coin-flip-lucky/
    # unlucky players at the minimum sample size), not genuine signal.
    PRIOR_WEIGHT = 30  # pseudo-observations pulling small samples toward the population mean
    shrunk_rate = (losses + PRIOR_WEIGHT * population_loss_rate) / (n + PRIOR_WEIGHT)

    bluff_table = pd.DataFrame({
        "n_river_showdowns": n,
        "raw_bluff_rate": losses / n,
        "shrunk_bluff_rate": shrunk_rate,
    }).reset_index().rename(columns={"river_aggressor": "player"})
    bluff_table = bluff_table[bluff_table["n_river_showdowns"] >= MIN_RIVER_SHOWDOWNS]
    bluff_table = bluff_table.sort_values("shrunk_bluff_rate", ascending=False)

    bluff_table.to_csv("data/reference/frequent_bluffers.csv", index=False)
    print(f"\nsaved {len(bluff_table)} players (>= {MIN_RIVER_SHOWDOWNS} river showdowns) to data/reference/frequent_bluffers.csv")
    print()
    print("distribution of shrunk_bluff_rate:")
    print(bluff_table["shrunk_bluff_rate"].describe())
    print()
    print("top 20 by shrinkage-adjusted bluff rate (most reliable signal, not just highest raw rate):")
    print(bluff_table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
