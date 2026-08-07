"""Range-vs-range Monte Carlo equity, extending src/engine/cards.py (which
only supports one hero hand vs random opponents). Needed to compare two
*implied* ranges (see analysis/implied_range.py) against each other -- e.g.
"this Nit's UTG open range" vs "this Station's BB calling range".

Combos within a hand notation are weighted equally (AKs has 4 real combos,
AKo has 12) -- naive notation-level weighting would overstate suited hands.
"""

import random

from src.engine.cards import RANKS, SUITS, Card, evaluate_7cards


def expand_hand_to_combos(notation: str) -> list[tuple[str, str]]:
    if len(notation) == 2:
        r = notation[0]
        return [(r + s1, r + s2) for i, s1 in enumerate(SUITS) for s2 in SUITS[i + 1 :]]

    r1, r2, kind = notation[0], notation[1], notation[2]
    if kind == "s":
        return [(r1 + s, r2 + s) for s in SUITS]
    return [(r1 + s1, r2 + s2) for s1 in SUITS for s2 in SUITS if s1 != s2]


def _expand_range(hands: list[str]) -> list[tuple[str, str]]:
    combos = []
    for h in hands:
        combos.extend(expand_hand_to_combos(h))
    return combos


def filter_combos_for_board(combos: list[tuple[str, str]], board: list[str]) -> list[tuple[str, str]]:
    """Drop combos that use a card already on the board."""
    board_set = set(board)
    return [c for c in combos if c[0] not in board_set and c[1] not in board_set]


def rank_combos_on_board(combos: list[tuple[str, str]], board: list[str]) -> list[tuple[str, str]]:
    """Sort combos by realized hand strength on a *complete* (5-card) board,
    best first. No Monte Carlo needed there -- nothing more will be dealt, so
    current strength IS final strength. Only correct for a full board; use
    `rank_combos_by_forward_equity` for flop/turn (see its docstring for why).
    """
    board_cards = [Card(c) for c in board]
    scored = [
        (evaluate_7cards([Card(combo[0]), Card(combo[1])] + board_cards), combo) for combo in combos
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [combo for _, combo in scored]


def rank_combos_by_forward_equity(
    combos: list[tuple[str, str]], board: list[str], trials_per_combo: int = 150
) -> list[tuple[str, str]]:
    """Sort combos by equity vs a random hand, completing the board randomly
    each trial. Needed for flop/turn (board not yet complete): ranking by
    *current* made-hand strength there systematically undervalues draws (a
    flush draw is just "high card" until it completes) and would make the
    narrowed "continuing range" collapse toward made hands only, excluding
    the draws that real players actually continue with. Slower than
    rank_combos_on_board (Monte Carlo per combo) -- only pay that cost where
    it's needed, i.e. when the board isn't complete yet.
    """
    full_deck = [r + s for r in RANKS for s in SUITS]
    board_cards = [Card(c) for c in board]
    n_needed = 5 - len(board)

    scored = []
    for combo in combos:
        used = set(combo) | set(board)
        deck_left = [c for c in full_deck if c not in used]
        wins = ties = 0
        for _ in range(trials_per_combo):
            drawn = random.sample(deck_left, n_needed + 2)
            board_fill, opp_hole = drawn[:n_needed], drawn[n_needed:]
            full_board_cards = board_cards + [Card(c) for c in board_fill]
            my_best = evaluate_7cards([Card(combo[0]), Card(combo[1])] + full_board_cards)
            opp_best = evaluate_7cards([Card(opp_hole[0]), Card(opp_hole[1])] + full_board_cards)
            if my_best > opp_best:
                wins += 1
            elif my_best == opp_best:
                ties += 1
        equity = (wins + ties / 2) / trials_per_combo
        scored.append((equity, combo))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [combo for _, combo in scored]


def narrow_range_by_board(
    hands_or_combos: list, board: list[str], keep_fraction: float, forward_equity_trials: int = 150
) -> list[tuple[str, str]]:
    """Approximate range narrowing: rank a range's combos by how well they
    connect with the board, keep the top `keep_fraction`. Uses forward
    (Monte Carlo) equity when the board isn't complete (flop/turn) and exact
    made-hand strength on a complete (river) board -- see the two ranking
    functions' docstrings for why that split matters. This stands in for
    "which hands would realistically still be continuing" without needing
    real opponent cards -- a real range also narrows by blockers and
    line-consistency, not just raw strength/equity, so treat this as
    directional, not exact.
    """
    if hands_or_combos and isinstance(hands_or_combos[0], str):
        combos = _expand_range(hands_or_combos)
    else:
        combos = list(hands_or_combos)

    combos = filter_combos_for_board(combos, board)
    if not combos:
        return []

    if len(board) >= 5:
        ranked = rank_combos_on_board(combos, board)
    else:
        ranked = rank_combos_by_forward_equity(combos, board, trials_per_combo=forward_equity_trials)

    n_keep = max(1, round(keep_fraction * len(ranked)))
    return ranked[:n_keep]


def _combos_vs_combos_equity(
    combos_a: list[tuple[str, str]],
    combos_b: list[tuple[str, str]],
    board: list[str] | None = None,
    trials: int = 3000,
) -> tuple[float, float]:
    """Core Monte Carlo loop, working on concrete combos rather than notations
    so it serves both the no-board (preflop) and fixed-board (postflop) cases.
    `board` may be 0/3/4/5 known cards -- the remaining cards up to 5 are
    randomized per trial.
    """
    if not combos_a or not combos_b:
        return 0.5, 0.5

    board = board or []
    n_board_needed = 5 - len(board)
    full_deck = [r + s for r in RANKS for s in SUITS]

    wins_a = wins_b = ties = 0
    trial = 0
    attempts = 0
    max_attempts = trials * 20

    while trial < trials and attempts < max_attempts:
        attempts += 1
        hand_a = random.choice(combos_a)
        hand_b = random.choice(combos_b)
        used = set(hand_a) | set(hand_b) | set(board)
        if len(used) < 4 + len(board):
            continue  # card conflict, resample

        if n_board_needed > 0:
            remaining = [c for c in full_deck if c not in used]
            full_board = board + random.sample(remaining, n_board_needed)
        else:
            full_board = board

        board_cards = [Card(c) for c in full_board]
        best_a = evaluate_7cards([Card(hand_a[0]), Card(hand_a[1])] + board_cards)
        best_b = evaluate_7cards([Card(hand_b[0]), Card(hand_b[1])] + board_cards)

        if best_a > best_b:
            wins_a += 1
        elif best_b > best_a:
            wins_b += 1
        else:
            ties += 1
        trial += 1

    if trial == 0:
        return 0.5, 0.5

    equity_a = (wins_a + ties / 2) / trial
    return equity_a, 1 - equity_a


def range_vs_range_equity(range_a: list[str], range_b: list[str], trials: int = 3000) -> tuple[float, float]:
    """Preflop, notation-based entry point (no board). Returns
    (range_a_equity, range_b_equity), summing to 1 (ties split the pot).
    """
    return _combos_vs_combos_equity(_expand_range(range_a), _expand_range(range_b), board=None, trials=trials)


def combos_vs_range_equity_on_board(
    combos_a: list[tuple[str, str]],
    range_b: list,
    board: list[str],
    trials: int = 2000,
) -> tuple[float, float]:
    """Postflop entry point: `combos_a` is already a concrete (narrowed)
    combo list, `range_b` may be notations or combos; `board` is the known
    board (3/4/5 cards) shared by both hands.
    """
    combos_b = _expand_range(range_b) if range_b and isinstance(range_b[0], str) else list(range_b)
    combos_b = filter_combos_for_board(combos_b, board)
    return _combos_vs_combos_equity(combos_a, combos_b, board=board, trials=trials)
