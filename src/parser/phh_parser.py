"""Parser for the real PHH (Poker Hand History) format used by the Zenodo
"A Dataset of Poker Hand Histories" / HandHQ mirror (uoftcprg/phh-dataset).

Each .phhs file is a single TOML document: top-level tables keyed "1", "2", ...,
one per hand. Player i (1-indexed, "p1".."pN") posts blinds_or_straddles[i-1];
by convention p1=SB, p2=BB, ..., pN=BTN, with preflop action starting at p3 and
wrapping to p1/p2 last -- confirmed against real sample files, not guessed.

Only the no-limit hold'em variant ('NT') is handled; other variants are skipped.
Public HandHQ-obfuscated files zero out winnings/stacks and mask non-hero
showdown cards, so `pot`/`rake` here are reconstructed from action amounts,
not read from a `winnings` field (which is always 0 in the obfuscated data).
"""

import tomllib

from src.parser.models import Action, Hand, Seat
from src.parser.positions import _TABLES_BY_SEATS, _collapse_labels

SUPPORTED_VARIANT = "NT"


def _split_cards(token: str) -> list[str]:
    return [token[i : i + 2] for i in range(0, len(token), 2)]


def _p1_to_pn_labels(n: int) -> list[str]:
    """BTN-first templates in positions.py are ordered (BTN, SB, BB, ...);
    PHH orders players (p1=SB, p2=BB, ..., pN=BTN) -- rotate BTN to the end.
    """
    btn_first = _TABLES_BY_SEATS.get(n) or _collapse_labels(n)
    return list(btn_first[1:]) + [btn_first[0]]


def _street_for_board_len(n_cards: int) -> str:
    return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n_cards, "river")


def _parse_one_hand(fields: dict) -> Hand | None:
    if fields.get("variant") != SUPPORTED_VARIANT:
        return None

    players = fields["players"]
    n = len(players)
    if n < 2:
        return None

    position_labels = _p1_to_pn_labels(n)
    blinds = fields["blinds_or_straddles"]
    stacks = fields.get("starting_stacks", [float("nan")] * n)

    seats = [
        Seat(
            seat_no=i + 1,
            player=players[i],
            stack=stacks[i] if stacks[i] not in (float("inf"),) else float("nan"),
            position=position_labels[i],
        )
        for i in range(n)
    ]

    hand = Hand(
        hand_id=str(fields.get("hand", "")),
        table_name=fields.get("table", ""),
        max_seats=n,
        small_blind=float(blinds[0]),
        big_blind=float(blinds[1]) if n > 1 else float(blinds[0]),
        button_seat=n,  # pN is always BTN by convention (see module docstring)
        seats=seats,
    )

    street_contributed = {players[0]: hand.small_blind, players[1]: hand.big_blind}
    current_street_bet = hand.big_blind
    current_street = "preflop"
    total_pot = hand.small_blind + hand.big_blind

    for raw in fields.get("actions", []):
        parts = raw.split()
        if not parts:
            continue

        if parts[0] == "d":
            if parts[1] == "dh":
                player, cards = players[int(parts[2][1:]) - 1], parts[3]
                if cards != "????":
                    hand.hole_cards[player] = _split_cards(cards)
            elif parts[1] == "db":
                hand.board.extend(_split_cards(parts[2]))
                current_street = _street_for_board_len(len(hand.board))
                street_contributed = {}
                current_street_bet = 0.0
            continue

        player = players[int(parts[0][1:]) - 1]
        verb = parts[1]

        if verb == "f":
            hand.actions.append(Action(street=current_street, player=player, action="folds"))
        elif verb == "cc":
            contributed = street_contributed.get(player, 0.0)
            increment = current_street_bet - contributed
            action_name = "calls" if increment > 0 else "checks"
            hand.actions.append(Action(current_street, player, action_name, max(increment, 0.0)))
            street_contributed[player] = current_street_bet
            total_pot += max(increment, 0.0)
        elif verb == "cbr":
            new_total = float(parts[2])
            contributed = street_contributed.get(player, 0.0)
            increment = new_total - contributed
            action_name = "bets" if current_street_bet == 0 else "raises"
            hand.actions.append(Action(current_street, player, action_name, increment))
            street_contributed[player] = new_total
            current_street_bet = new_total
            total_pot += increment
        elif verb == "sm":
            cards = parts[2] if len(parts) > 2 else "????"
            if cards != "????":
                hand.showdown[player] = _split_cards(cards)
        # other verbs (e.g. 'sd' show/discard variants) intentionally skipped

    hand.pot = total_pot
    return hand


def parse_phhs_file(path: str) -> list[Hand]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    hands = []
    for fields in data.values():
        try:
            hand = _parse_one_hand(fields)
        except (KeyError, IndexError, ValueError):
            continue
        if hand:
            hands.append(hand)
    return hands
