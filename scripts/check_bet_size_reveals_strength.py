"""Does a bigger preflop raise (open OR 3bet+) correlate with a stronger
REAL starting hand at real showdown? If real players size their preflop
aggression by hand strength even a little, that's a genuine, exploitable
signal for postflop: a big preflop raise implies a narrower, stronger
range, so a hero facing that raiser who misses the flop can lean toward
folding more readily than against a min-raise, and vice versa.

Uses real showdown hole cards (data/processed/showdowns.parquet) joined to
real preflop raise/3bet actions (data/processed/actions.parquet) by
(hand_id, player) -- the same selection-bias caveat as every other
showdown-conditioned check in this project applies (only reaches showdown,
only sees hands that got shown/couldn't be mucked), disclosed, not hidden.

RESULT (2026-08-11, 396,208 showdown-reaching preflop raises): real,
statistically robust, but MODEST. Opens: rho=0.199 (p~0), mean real
equity 0.633 at the smallest raise-size quartile vs 0.667-0.668 at the
two largest -- about a 3.4pp gap. 3-bets: rho=0.085 (p~0, weaker),
0.690 at the smallest size bucket vs 0.709-0.717 at the larger three --
about a 2-2.7pp gap. Direction confirms the textbook intuition (bigger
raise -> somewhat stronger range), significance is real at this sample
size, but the actual equity gap between "smallest" and "largest" bet-
size buckets is small -- this is a real, minor lean, not a strong
exploit. A strategy rule built on this (e.g. "fold more to a big raise,
call more vs a small one") should weight it accordingly: real, but weak.

Usage: python3 scripts/check_bet_size_reveals_strength.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.hand_rankings import compute_hand_rankings

RANKS = "23456789TJQKA"


def _notation(hole_cards: str) -> str | None:
    if not isinstance(hole_cards, str) or len(hole_cards) != 4:
        return None
    r1, r2, s1, s2 = hole_cards[0], hole_cards[2], hole_cards[1], hole_cards[3]
    if r1 == r2:
        return r1 + r2
    hi, lo = (r1, r2) if RANKS.index(r1) > RANKS.index(r2) else (r2, r1)
    return f"{hi}{lo}{'s' if s1 == s2 else 'o'}"


def main():
    print("loading actions.parquet (preflop raises only)...")
    actions = pd.read_parquet(
        "data/processed/actions.parquet", columns=["hand_id", "player", "street", "action", "amount_bb"]
    )
    preflop_raises = actions[(actions["street"] == "preflop") & (actions["action"] == "raises")].copy()
    preflop_raises = preflop_raises[np.isfinite(preflop_raises["amount_bb"])]
    preflop_raises["raise_order"] = preflop_raises.groupby("hand_id").cumcount()
    # order 0 = opening raise, order >=1 = a 3bet-or-deeper re-raise this hand
    preflop_raises["is_reraise"] = preflop_raises["raise_order"] > 0
    raises = preflop_raises[["hand_id", "player", "amount_bb", "is_reraise"]]
    print(f"{len(raises)} finite-sized preflop raises ({(~raises['is_reraise']).sum()} opens, {raises['is_reraise'].sum()} 3bet+)")

    print("loading showdowns.parquet...")
    showdowns = pd.read_parquet("data/processed/showdowns.parquet", columns=["hand_id", "player", "hole_cards"])

    merged = raises.merge(showdowns, on=["hand_id", "player"], how="inner")
    print(f"raises that reached a real showdown: {len(merged)}")

    merged["notation"] = merged["hole_cards"].apply(_notation)
    merged = merged.dropna(subset=["notation"])

    rankings = compute_hand_rankings().set_index("hand")["equity"]
    merged["equity"] = merged["notation"].map(rankings)
    merged = merged.dropna(subset=["equity"])
    print(f"rows with a resolved equity: {len(merged)}")

    for label, sub in (("OPENS", merged[~merged["is_reraise"]]), ("3BET+", merged[merged["is_reraise"]])):
        print(f"\n--- {label} (n={len(sub)}) ---")
        corr, pvalue = stats.spearmanr(sub["amount_bb"], sub["equity"])
        print(f"Spearman correlation (bet size vs real equity): rho={corr:.4f}, p={pvalue:.6f}")

        # bucket by bet size quartile within this group, report mean equity per bucket
        sub = sub.copy()
        try:
            sub["size_bucket"] = pd.qcut(sub["amount_bb"], 4, duplicates="drop")
        except ValueError:
            print("  (not enough distinct sizes to bucket)")
            continue
        summary = sub.groupby("size_bucket", observed=True)["equity"].agg(["mean", "count"])
        print(summary)


if __name__ == "__main__":
    main()
