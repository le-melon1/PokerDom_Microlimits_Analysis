"""Minimal hand evaluator + Monte Carlo equity, vendored from the sibling
Poker_Engine project so this project stays self-contained. Used here only to
build the 169-starting-hand equity ranking that backs implied-range estimates
(src/analysis/implied_range.py) -- not a full solver.
"""

import random
from collections import Counter
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_TO_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}


class Card:
    def __init__(self, s):
        self.rank = s[0]
        self.suit = s[1]
        self.value = RANK_TO_VALUE[self.rank]

    def __repr__(self):
        return f"{self.rank}{self.suit}"


class Deck:
    def __init__(self):
        self.cards = [Card(r + s) for r in RANKS for s in SUITS]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, n=1):
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def remove(self, cards):
        to_rem = {(c[0], c[1]) for c in cards}
        self.cards = [c for c in self.cards if (c.rank, c.suit) not in to_rem]


def evaluate_7cards(cards):
    best = None
    for combo in combinations(cards, 5):
        val = evaluate_5cards(combo)
        if best is None or val > best:
            best = val
    return best


def evaluate_5cards(cards):
    vals = sorted([c.value for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    counts = sorted(Counter(vals).items(), key=lambda x: (-x[1], -x[0]))
    is_flush = len(set(suits)) == 1
    unique_vals = sorted(set(vals), reverse=True)

    def straight_high(vals_list):
        if len(vals_list) < 5:
            return None
        for i in range(len(vals_list) - 4):
            window = vals_list[i : i + 5]
            if window[0] - window[4] == 4 and len(set(window)) == 5:
                return window[0]
        if {14, 5, 4, 3, 2}.issubset(set(vals_list)):
            return 5
        return None

    s_high = straight_high(unique_vals)

    if is_flush:
        suited_vals = sorted([c.value for c in cards if c.suit == suits[0]], reverse=True)
        sf_high = straight_high(sorted(set(suited_vals), reverse=True))
        if sf_high:
            return (9, sf_high)

    if counts[0][1] == 4:
        four_rank = counts[0][0]
        kicker = max(v for v in vals if v != four_rank)
        return (8, four_rank, kicker)

    if counts[0][1] == 3 and counts[1][1] >= 2:
        return (7, counts[0][0], counts[1][0])

    if is_flush:
        return (6, tuple(vals))

    if s_high:
        return (5, s_high)

    if counts[0][1] == 3:
        trips = counts[0][0]
        kickers = sorted([v for v in vals if v != trips], reverse=True)[:2]
        return (4, trips, tuple(kickers))

    if counts[0][1] == 2 and counts[1][1] == 2:
        high_pair, low_pair = counts[0][0], counts[1][0]
        kicker = max(v for v in vals if v not in (high_pair, low_pair))
        return (3, high_pair, low_pair, kicker)

    if counts[0][1] == 2:
        pair = counts[0][0]
        kickers = sorted([v for v in vals if v != pair], reverse=True)[:3]
        return (2, pair, tuple(kickers))

    return (1, tuple(vals))


def monte_carlo_equity(hole_cards, board_cards=None, n_opponents=1, trials=2000):
    if board_cards is None:
        board_cards = []
    my_hole = [Card(c) for c in hole_cards]
    known_board = [Card(c) for c in board_cards]
    wins = ties = 0
    for _ in range(trials):
        deck = Deck()
        deck.remove(hole_cards + board_cards)
        deck.shuffle()

        opps = [deck.draw(2) for _ in range(n_opponents)]
        board = known_board.copy()
        board += deck.draw(5 - len(board))

        my_best = evaluate_7cards(my_hole + board)
        opp_bests = [evaluate_7cards(o + board) for o in opps]

        better = sum(1 for ob in opp_bests if ob > my_best)
        equal = sum(1 for ob in opp_bests if ob == my_best)
        if better == 0:
            if equal == 0:
                wins += 1
            else:
                ties += 1

    return wins / trials, ties / trials
