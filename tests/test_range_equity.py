import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.range_equity import (
    combos_vs_multiple_ranges_equity_on_board,
    combos_vs_range_equity_on_board,
)


def test_falls_back_to_half_with_no_opponent_ranges():
    assert combos_vs_multiple_ranges_equity_on_board([("As", "Ah")], [], board=[]) == 0.5


def test_falls_back_to_half_with_an_empty_opponent_range():
    assert combos_vs_multiple_ranges_equity_on_board([("As", "Ah")], [[]], board=[]) == 0.5


def test_near_chop_against_the_other_two_aces():
    # Hero holds As/Ah; the ONLY legal combo for a single "AA" opponent given
    # those two cards are blocked is Ac/Ad -- both players always have a pair
    # of aces off the SAME 5-card board, so it's a tie almost every trial
    # (the rare exception: the board runs out enough spades/hearts for hero's
    # specific suits, or clubs/diamonds for villain's, to complete a flush
    # the other can't match -- hole-card RANKS are symmetric here but suits
    # aren't). Also exercises the conflict-resample path directly: 5 of AA's
    # 6 notation combos collide with hero's own cards.
    equity = combos_vs_multiple_ranges_equity_on_board([("As", "Ah")], [["AA"]], board=[], trials=2000)
    assert abs(equity - 0.5) < 0.05


def test_multiway_equity_is_lower_than_two_way_equity_against_the_same_range():
    # Same opponent range, once as a single opponent and once duplicated as
    # two independent opponents on a fixed complete board (so the only
    # randomness is which villain combo gets sampled each trial, not board
    # dealing) -- beating two independently-drawn opponents from a live
    # range must be at least as hard as beating one, so equity should drop.
    board = ["2c", "7d", "9h", "Jc", "4s"]
    hero = [("Ks", "Kd")]  # a strong but not lock-of-the-pot hand on this board
    wide_range = ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "KQs", "QJs", "JTs", "T9s"]

    random.seed(1234)
    one_opponent = combos_vs_multiple_ranges_equity_on_board(hero, [wide_range], board=board, trials=4000)
    random.seed(1234)
    two_opponents = combos_vs_multiple_ranges_equity_on_board(hero, [wide_range, wide_range], board=board, trials=4000)

    assert two_opponents < one_opponent


def test_roughly_matches_the_pairwise_function_for_a_single_opponent():
    board = ["2c", "7d", "9h"]
    hero = [("Ks", "Kd")]
    villain_range = ["AA", "QQ", "JJ", "AKs", "AKo", "AQo", "KQs", "JTs", "98s"]

    random.seed(42)
    multiway_equity = combos_vs_multiple_ranges_equity_on_board(hero, [villain_range], board=board, trials=6000)
    random.seed(42)
    pairwise_equity, _ = combos_vs_range_equity_on_board(hero, villain_range, board=board, trials=6000)

    assert abs(multiway_equity - pairwise_equity) < 0.03
