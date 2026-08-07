"""Rank the 169 canonical starting hands by all-in equity vs one random hand.

This is the standard "how strong is this hand in isolation" percentile table
that lets a VPIP% be translated into an implied range without ever seeing a
player's actual cards (see implied_range.py). Monte Carlo, not exact --
adjacent close hands can swap rank slightly run to run, which is fine for a
percentile cut, not for hand-vs-hand precision claims.
"""

from pathlib import Path

import pandas as pd

from src.engine.cards import monte_carlo_equity

RANKS = "AKQJT98765432"
CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "hand_rankings.csv"


def _canonical_hands() -> list[str]:
    hands = []
    for i, r1 in enumerate(RANKS):
        for r2 in RANKS[i:]:
            if r1 == r2:
                hands.append(f"{r1}{r2}")
            else:
                hands.append(f"{r1}{r2}s")
                hands.append(f"{r1}{r2}o")
    return hands


def _sample_cards(hand_notation: str) -> list[str]:
    if len(hand_notation) == 2:
        r1, r2 = hand_notation[0], hand_notation[1]
        return [f"{r1}s", f"{r2}h"]
    r1, r2, kind = hand_notation[0], hand_notation[1], hand_notation[2]
    if kind == "s":
        return [f"{r1}s", f"{r2}s"]
    return [f"{r1}s", f"{r2}h"]


def compute_hand_rankings(trials: int = 800, force: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not force:
        return pd.read_csv(CACHE_PATH)

    rows = []
    for hand in _canonical_hands():
        cards = _sample_cards(hand)
        win, tie = monte_carlo_equity(cards, [], n_opponents=1, trials=trials)
        rows.append({"hand": hand, "equity": win + tie / 2})

    df = pd.DataFrame(rows).sort_values("equity", ascending=False).reset_index(drop=True)
    df["percentile"] = (df.index + 1) / len(df)  # 1/169 = strongest hand's cutoff, ... 169/169 = all hands
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df
