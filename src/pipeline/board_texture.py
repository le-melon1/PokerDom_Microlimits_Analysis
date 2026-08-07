"""Board texture features used by the fold-equity model (dry vs wet boards)."""

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}


def texture_features(board: list[str]) -> dict:
    if not board:
        return {
            "board_paired": False,
            "board_monotone": False,
            "board_two_tone": False,
            "board_max_suit_count": 0,
            "board_connectedness": 0,
            "board_high_card": 0,
        }

    ranks = [c[0] for c in board]
    suits = [c[1] for c in board]
    values = sorted(RANK_VALUE[r] for r in ranks)

    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit_count = max(suit_counts.values())

    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    connectedness = sum(1 for g in gaps if g <= 2)

    return {
        "board_paired": len(set(ranks)) < len(ranks),
        "board_monotone": max_suit_count == len(board),
        "board_two_tone": max_suit_count == 2,
        "board_max_suit_count": max_suit_count,
        "board_connectedness": connectedness,
        "board_high_card": max(values),
    }
