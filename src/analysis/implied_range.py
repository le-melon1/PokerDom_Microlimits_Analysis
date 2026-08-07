"""Map a VPIP% to an implied opening range, using the equity-percentile table
from hand_rankings.py. No hole cards needed -- this is the standard technique
HUD range-widgets use: a player who plays X% of hands is playing (approximately)
the top X% of hands by raw equity vs a random hand.

This is an approximation, not a measurement: real players don't open exactly
the top-N%-by-raw-equity set (e.g. suited connectors get played wider than raw
equity alone suggests, offsuit broadways narrower) -- but it's the right order
of magnitude and the standard starting point absent observed cards.
"""

import pandas as pd

from src.analysis.hand_rankings import compute_hand_rankings


def implied_range(vpip: float, rankings: pd.DataFrame | None = None) -> list[str]:
    if rankings is None:
        rankings = compute_hand_rankings()
    n = max(1, round(vpip * len(rankings)))
    return rankings.iloc[:n]["hand"].tolist()


def format_range_compact(hands: list[str]) -> str:
    """Collapse a hand list into 'AA-77, AKs-ATs, AKo+, ...'-style notation.

    Groups by first rank + suitedness and shows the second-rank span within
    each group (min..max present) -- an approximation of standard shorthand,
    not a guarantee every rank in the span is included (Monte Carlo ranking
    noise can leave small gaps).
    """
    RANKS = "AKQJT98765432"
    rank_idx = {r: i for i, r in enumerate(RANKS)}

    pairs = sorted([h for h in hands if len(h) == 2], key=lambda h: rank_idx[h[0]])
    suited = [h for h in hands if h.endswith("s")]
    offsuit = [h for h in hands if h.endswith("o")]

    def collapse_by_first_rank(group: list[str], suffix: str) -> list[str]:
        by_first: dict[str, list[str]] = {}
        for h in group:
            by_first.setdefault(h[0], []).append(h[1])
        out = []
        for first in sorted(by_first, key=lambda r: rank_idx[r]):
            seconds_sorted = sorted(by_first[first], key=lambda r: rank_idx[r])
            hi, lo = seconds_sorted[0], seconds_sorted[-1]
            out.append(f"{first}{hi}{suffix}" if hi == lo else f"{first}{hi}{suffix}-{first}{lo}{suffix}")
        return out

    parts = []
    if pairs:
        parts.append(f"{pairs[-1]}-{pairs[0]}" if len(pairs) > 1 else pairs[0])
    parts += collapse_by_first_rank(suited, "s")
    parts += collapse_by_first_rank(offsuit, "o")
    return ", ".join(parts)
