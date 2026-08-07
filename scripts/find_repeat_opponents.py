"""Find the player pairs with the most shared hands, as candidates for
checking whether real players adapt their strategy to a specific repeat
opponent over time.

hand_id is a real PokerStars hand ID (assigned sequentially site-wide), so
sorting by it gives a valid chronological ordering across the whole dataset,
not just within one table -- used later to split each pair's shared history
into "earlier" vs "later" halves.
"""

import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

OUT_DIR = Path("data/reference")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("loading actions.parquet (hand_id, player only)...")
    df = pd.read_parquet("data/processed/actions.parquet", columns=["hand_id", "player"])
    log(f"loaded {len(df)} action rows")

    pair_counts: Counter = Counter()
    n_hands = 0

    for hand_id, grp in df.groupby("hand_id", sort=False):
        players = grp["player"].unique()
        n_hands += 1
        if len(players) < 2:
            continue
        for a, b in combinations(sorted(players), 2):
            pair_counts[(a, b)] += 1
        if n_hands % 500_000 == 0:
            log(f"  processed {n_hands} hands, {len(pair_counts)} distinct pairs so far")

    log(f"done: {n_hands} hands, {len(pair_counts)} distinct pairs")

    rows = [(a, b, n) for (a, b), n in pair_counts.items()]
    pairs_df = pd.DataFrame(rows, columns=["player_a", "player_b", "shared_hands"])
    pairs_df = pairs_df.sort_values("shared_hands", ascending=False)
    pairs_df.to_csv(OUT_DIR / "repeat_opponent_pairs.csv", index=False)

    log(f"saved {len(pairs_df)} pairs to {OUT_DIR / 'repeat_opponent_pairs.csv'}")
    log("top 20 by shared hands:")
    print(pairs_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
