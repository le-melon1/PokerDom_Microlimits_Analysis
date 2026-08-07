"""Regex-based parser for plain-text poker hand history logs.

Patterns are written against the PokerStars-style hand history format, which is
the common denominator most datamining tools (KingsHands, HHDealer) and the
Zenodo iPoker dataset export to, or are close enough to require only pattern
tweaks. Once real PokerDom-adjacent samples are on hand, adjust the regexes
below rather than the calling code in pipeline/preprocess.py.
"""

import re

from src.parser.models import Action, Hand, Seat

HAND_SPLIT_RE = re.compile(r"\n\s*\n(?=\S)")

HEADER_RE = re.compile(
    r"Hand #(?P<hand_id>\S+):.*?\(\D*(?P<sb>[\d.]+)\s*/\s*\D*(?P<bb>[\d.]+)\s*\w*\)"
)
TABLE_RE = re.compile(
    r"Table '(?P<table_name>[^']+)' (?P<max_seats>\d+)-max"
    r"(?:.*?Seat #(?P<button_seat>\d+) is the button)?"
)
SEAT_RE = re.compile(
    r"Seat (?P<seat_no>\d+): (?P<player>.+?) \(\D*(?P<stack>[\d.]+) in chips\)"
)
DEALT_RE = re.compile(r"Dealt to (?P<player>.+?) \[(?P<cards>[2-9TJQKA][cdhs] [2-9TJQKA][cdhs])\]")
ACTION_RE = re.compile(
    r"(?P<player>.+?): (?P<action>folds|checks|calls|bets|raises)"
    r"(?:\s+\D*(?P<amount1>[\d.]+))?(?:\s+to\s+\D*(?P<amount2>[\d.]+))?"
)
STREET_MARKERS = {
    "*** FLOP ***": "flop",
    "*** TURN ***": "turn",
    "*** RIVER ***": "river",
    "*** SHOW DOWN ***": "showdown",
    "*** SUMMARY ***": "summary",
}
# TURN/RIVER lines show two bracket groups, e.g. "*** TURN *** [2h 7c Jd] [4s]"
# (existing board, then the new card) -- match every bracket group on the line
# and pull card tokens out of all of them, rather than just the first.
BOARD_MARKER_RE = re.compile(r"\*\*\* (?:FLOP|TURN|RIVER) \*\*\*((?:\s*\[[^\]]+\])+)")
CARD_TOKEN_RE = re.compile(r"[2-9TJQKA][cdhs]")
SUMMARY_POT_RE = re.compile(r"Total pot \D*(?P<pot>[\d.]+)\s*(?:\|\s*Rake \D*(?P<rake>[\d.]+))?")
WINNER_RE = re.compile(r"(?P<player>.+?) collected \D*(?P<amount>[\d.]+) from pot")


def split_hands(raw_text: str) -> list[str]:
    return [block.strip() for block in HAND_SPLIT_RE.split(raw_text) if block.strip()]


def parse_hand(block: str) -> Hand | None:
    header = HEADER_RE.search(block)
    table = TABLE_RE.search(block)
    if not header or not table:
        return None

    hand = Hand(
        hand_id=header.group("hand_id"),
        table_name=table.group("table_name"),
        max_seats=int(table.group("max_seats")),
        small_blind=float(header.group("sb")),
        big_blind=float(header.group("bb")),
        button_seat=int(table.group("button_seat")) if table.group("button_seat") else 1,
    )

    for m in SEAT_RE.finditer(block):
        hand.seats.append(
            Seat(seat_no=int(m.group("seat_no")), player=m.group("player"), stack=float(m.group("stack")))
        )

    dealt = DEALT_RE.search(block)
    if dealt:
        hand.hole_cards[dealt.group("player")] = dealt.group("cards").split()

    current_street = "preflop"
    for line in block.splitlines():
        stripped = line.strip()
        matched_marker = next((s for s in STREET_MARKERS if stripped.startswith(s)), None)
        if matched_marker:
            current_street = STREET_MARKERS[matched_marker]
            board_match = BOARD_MARKER_RE.match(stripped)
            if board_match:
                hand.board = CARD_TOKEN_RE.findall(board_match.group(1))
            continue
        if current_street == "summary":
            continue

        action_match = ACTION_RE.match(stripped)
        if action_match:
            amount = action_match.group("amount2") or action_match.group("amount1") or 0.0
            hand.actions.append(
                Action(
                    street=current_street,
                    player=action_match.group("player"),
                    action=action_match.group("action"),
                    amount=float(amount),
                )
            )

    pot_match = SUMMARY_POT_RE.search(block)
    if pot_match:
        hand.pot = float(pot_match.group("pot"))
        hand.rake = float(pot_match.group("rake")) if pot_match.group("rake") else 0.0

    for m in WINNER_RE.finditer(block):
        hand.winners[m.group("player")] = hand.winners.get(m.group("player"), 0.0) + float(m.group("amount"))

    return hand


def parse_file(path: str) -> list[Hand]:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        raw_text = fh.read()
    hands = []
    for block in split_hands(raw_text):
        hand = parse_hand(block)
        if hand:
            hands.append(hand)
    return hands
