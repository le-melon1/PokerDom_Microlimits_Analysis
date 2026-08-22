"""Compare two candidate definitions for a per-opponent "bluff frequency"
signal, in response to find_frequent_bluffers.py's coverage problem: its
MIN_RIVER_SHOWDOWNS=40 bar leaves only 49/26,797 players (0.2%) with a
reliable individual estimate -- too thin to build a population-tier
distribution the way archetypes.py's postflop_freq_tier could.

Variant A: same definition as find_frequent_bluffers.py (last RIVER
aggressor who reached a real showdown and lost), just a lower reliability
bar (15 showdowns instead of 40) -- trades some individual-estimate
precision for ~16x more covered players.

Variant C: broader proxy -- aggressive (bet/raise) on ANY street that hand,
reached a real showdown, lost. Not restricted to being specifically the
LAST river aggressor -- captures preflop/flop/turn aggression too, so a
given player accumulates qualifying hands much faster. Different, less
precise concept ("showed aggression this hand and it didn't hold up" vs
"specifically led out on the river and got caught"), but might have enough
coverage to be worth building even at a stricter reliability bar.

Both use the same empirical-Bayes shrinkage as find_frequent_bluffers.py
(PRIOR_WEIGHT=30) to control small-n noise.

This script does NOT build any pipeline/retrain anything -- it only
reports coverage and distribution stats so a real choice can be made
before committing to either one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

PRIOR_WEIGHT = 30
THRESHOLDS = [10, 15, 20, 25, 30, 40]


def log(msg):
    print(msg, flush=True)


def _shrunk_table(qualifying: pd.DataFrame, key_col: str, name: str) -> pd.DataFrame:
    """qualifying: one row per (hand_id, player) that counts as an
    'aggression event' for this variant, with an is_winner column."""
    population_loss_rate = 1 - qualifying["is_winner"].mean()
    log(f"[{name}] population baseline: loses at real showdown {population_loss_rate:.3f} of the time")

    g = qualifying.groupby(key_col)
    n = g.size()
    losses = g["is_winner"].apply(lambda s: (~s).sum())
    shrunk_rate = (losses + PRIOR_WEIGHT * population_loss_rate) / (n + PRIOR_WEIGHT)

    table = pd.DataFrame({"n_events": n, "raw_rate": losses / n, "shrunk_rate": shrunk_rate}).reset_index()
    table = table.rename(columns={key_col: "player"})
    return table


def main():
    log("loading actions.parquet...")
    actions = pd.read_parquet("data/processed/actions.parquet", columns=["hand_id", "player", "street", "action"])

    log("loading showdowns.parquet...")
    showdowns = pd.read_parquet("data/processed/showdowns.parquet", columns=["hand_id", "player", "outcome_known", "is_winner"])
    showdowns = showdowns[showdowns["outcome_known"]]

    # ---------- Variant A: last river aggressor, reached showdown, lost ----------
    log("\n=== Variant A: last river aggressor (find_frequent_bluffers.py's own definition) ===")
    river_agg = actions[(actions["street"] == "river") & (actions["action"].isin(["bets", "raises"]))]
    last_river_agg = river_agg.groupby("hand_id").tail(1)[["hand_id", "player"]].rename(columns={"player": "river_aggressor"})
    variant_a = last_river_agg.merge(
        showdowns, left_on=["hand_id", "river_aggressor"], right_on=["hand_id", "player"], how="inner"
    )
    log(f"qualifying (hand,player) rows: {len(variant_a):,}")
    table_a = _shrunk_table(variant_a, "river_aggressor", "A")
    for t in THRESHOLDS:
        log(f"  threshold {t}: {int((table_a['n_events'] >= t).sum()):,} players")
    log("distribution at threshold=15:")
    log(str(table_a[table_a["n_events"] >= 15]["shrunk_rate"].describe()))

    # ---------- Variant C: any-street aggressor, reached showdown, lost ----------
    log("\n=== Variant C: any-street aggressor (broader proxy) ===")
    any_agg = actions[actions["action"].isin(["bets", "raises"])][["hand_id", "player"]].drop_duplicates()
    variant_c = any_agg.merge(showdowns, on=["hand_id", "player"], how="inner")
    log(f"qualifying (hand,player) rows: {len(variant_c):,}")
    table_c = _shrunk_table(variant_c, "player", "C")
    for t in THRESHOLDS:
        log(f"  threshold {t}: {int((table_c['n_events'] >= t).sum()):,} players")
    log("distribution at threshold=40 (same bar as variant A's original):")
    log(str(table_c[table_c["n_events"] >= 40]["shrunk_rate"].describe()))
    log("distribution at threshold=100:")
    log(str(table_c[table_c["n_events"] >= 100]["shrunk_rate"].describe()))

    # ---------- population context ----------
    total_players_seen = actions["player"].nunique()
    log(f"\ntotal distinct players in actions.parquet: {total_players_seen:,}")
    log("(compare against the 26,797 players archetype-labeled with >=100 hands)")

    table_a.to_csv("data/reference/bluff_frequency_variant_a.csv", index=False)
    table_c.to_csv("data/reference/bluff_frequency_variant_c.csv", index=False)
    log("\nsaved data/reference/bluff_frequency_variant_a.csv and _variant_c.csv")


if __name__ == "__main__":
    main()
