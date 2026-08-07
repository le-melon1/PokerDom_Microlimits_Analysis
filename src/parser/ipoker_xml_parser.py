"""Parser for the real iPoker Network XML hand-history format.

iPoker is the network PokerDom itself runs on, so this is the most directly
relevant format available -- unlike the PokerStars-style guess in
hand_history_parser.py or the Zenodo/HandHQ TOML format in phh_parser.py.

Verified against real sample files from HHSmithy/PokerHandHistoryParser
(an open-source parser project) and its C# source
(HandHistories.Parser/Parsers/FastParser/IPoker/IPokerFastParserImpl.cs),
which gave the authoritative action-type-code mapping. The RAISE-vs-ALL_IN
`sum` semantics below were reverse-engineered from a real 3-bet-shove hand
(HandActionTests/3BetHand.txt) by checking the arithmetic against the
players' final `bet` totals -- not guessed:
  - RAISE (type 23): `sum` is the cumulative total for the street ("raise to X").
  - ALL_IN (type 6/7): `sum` is the incremental amount added (remaining stack).
  - CALL/BET/blinds/antes: `sum` is already incremental.
"""

import re
import xml.etree.ElementTree as ET

from src.parser.models import Action, Hand, Seat
from src.parser.positions import _TABLES_BY_SEATS, _collapse_labels

ACTION_TYPE_FOLD = 0
ACTION_TYPE_SMALL_BLIND = 1
ACTION_TYPE_BIG_BLIND = 2
ACTION_TYPE_CALL = 3
ACTION_TYPE_CHECK = 4
ACTION_TYPE_BET = 5
ACTION_TYPE_ALLIN = (6, 7)
ACTION_TYPE_SITTING_OUT = (8, 9)
ACTION_TYPE_ANTE = 15
ACTION_TYPE_RAISE = 23

ROUND_STREET = {0: "preflop", 1: "preflop", 2: "flop", 3: "turn", 4: "river"}

CARD_RE = re.compile(r"([cdhs])(10|[2-9TJQKA])", re.IGNORECASE)


def _convert_card(token: str) -> str:
    """iPoker cards are '<suit><rank>' (e.g. 'c10', 'hK') -- flip to rank+suit
    ('Tc', 'Kh') to match the convention used by the other two parsers."""
    m = CARD_RE.match(token.strip())
    if not m:
        return token
    suit, rank = m.group(1).lower(), m.group(2).upper()
    rank = "T" if rank == "10" else rank
    return f"{rank}{suit}"


def _parse_cards_text(text: str) -> list[str]:
    text = (text or "").strip()
    if not text or "X" in text.upper():
        return []
    return [_convert_card(tok) for tok in text.split()]


def _money(text: str) -> float:
    return float(re.sub(r"[^\d.]", "", text or "0") or 0.0)


def _seat_order_labels(n: int) -> list[str]:
    return list(_TABLES_BY_SEATS.get(n) or _collapse_labels(n))


def _parse_one_game(game_el: ET.Element) -> Hand | None:
    players_el = game_el.find("./general/players")
    if players_el is None:
        return None
    player_els = players_el.findall("player")
    if len(player_els) < 2:
        return None

    seats_by_no = {}
    button_seat_no = None
    for p in player_els:
        seat_no = int(p.get("seat"))
        seats_by_no[seat_no] = p.get("name")
        if p.get("dealer") == "1":
            button_seat_no = seat_no

    ordered_seat_nos = sorted(seats_by_no)
    n = len(ordered_seat_nos)
    if button_seat_no is None:
        button_seat_no = ordered_seat_nos[0]

    btn_idx = ordered_seat_nos.index(button_seat_no)
    rotated = ordered_seat_nos[btn_idx:] + ordered_seat_nos[:btn_idx]
    labels = _seat_order_labels(n)

    stacks = {p.get("name"): _money(p.get("chips")) for p in player_els}
    seats = [
        Seat(seat_no=seat_no, player=seats_by_no[seat_no], stack=stacks[seats_by_no[seat_no]],
             position=labels[rotated.index(seat_no)])
        for seat_no in ordered_seat_nos
    ]

    hand = Hand(
        hand_id=game_el.get("gamecode", ""),
        table_name="",
        max_seats=n,
        small_blind=0.0,
        big_blind=0.0,
        button_seat=button_seat_no,
        seats=seats,
    )

    street_contributed: dict[str, float] = {}
    current_street_bet = 0.0
    total_pot = 0.0
    blinds_seen = []

    for round_el in game_el.findall("round"):
        round_no = int(round_el.get("no"))
        street = ROUND_STREET.get(round_no, "river")
        if round_no in (2, 3, 4):
            street_contributed = {}
            current_street_bet = 0.0

        for child in round_el:
            if child.tag == "cards":
                card_type = child.get("type")
                player = child.get("player")
                cards = _parse_cards_text(child.text)
                if card_type == "Pocket" and player and cards:
                    hand.hole_cards[player] = cards
                elif card_type in ("Flop", "Turn", "River"):
                    hand.board.extend(cards)
                continue

            if child.tag != "action":
                continue

            player = child.get("player")
            action_type = int(child.get("type"))
            sum_ = _money(child.get("sum"))

            if action_type in ACTION_TYPE_SITTING_OUT:
                continue

            if action_type == ACTION_TYPE_SMALL_BLIND:
                hand.small_blind = sum_
                street_contributed[player] = street_contributed.get(player, 0.0) + sum_
                current_street_bet = max(current_street_bet, street_contributed[player])
                total_pot += sum_
                blinds_seen.append(sum_)
                continue
            if action_type == ACTION_TYPE_BIG_BLIND:
                hand.big_blind = sum_
                street_contributed[player] = street_contributed.get(player, 0.0) + sum_
                current_street_bet = max(current_street_bet, street_contributed[player])
                total_pot += sum_
                blinds_seen.append(sum_)
                continue
            if action_type == ACTION_TYPE_ANTE:
                total_pot += sum_
                continue

            if action_type == ACTION_TYPE_FOLD:
                hand.actions.append(Action(street, player, "folds"))
            elif action_type == ACTION_TYPE_CHECK:
                hand.actions.append(Action(street, player, "checks"))
            elif action_type == ACTION_TYPE_CALL:
                hand.actions.append(Action(street, player, "calls", sum_))
                street_contributed[player] = street_contributed.get(player, 0.0) + sum_
                total_pot += sum_
            elif action_type == ACTION_TYPE_BET:
                hand.actions.append(Action(street, player, "bets", sum_))
                street_contributed[player] = street_contributed.get(player, 0.0) + sum_
                current_street_bet = max(current_street_bet, street_contributed[player])
                total_pot += sum_
            elif action_type == ACTION_TYPE_RAISE:
                contributed = street_contributed.get(player, 0.0)
                increment = sum_ - contributed
                name = "bets" if current_street_bet == 0 else "raises"
                hand.actions.append(Action(street, player, name, increment))
                street_contributed[player] = sum_
                current_street_bet = max(current_street_bet, sum_)
                total_pot += increment
            elif action_type in ACTION_TYPE_ALLIN:
                contributed = street_contributed.get(player, 0.0)
                new_total = contributed + sum_
                if current_street_bet == 0:
                    name = "bets"
                elif new_total > current_street_bet:
                    name = "raises"
                else:
                    name = "calls"
                hand.actions.append(Action(street, player, name, sum_))
                street_contributed[player] = new_total
                current_street_bet = max(current_street_bet, new_total)
                total_pot += sum_

    if hand.big_blind == 0.0 and blinds_seen:
        hand.big_blind = max(blinds_seen)
    if hand.small_blind == 0.0 and len(blinds_seen) > 1:
        hand.small_blind = min(blinds_seen)

    hand.pot = total_pot
    return hand


def parse_ipoker_xml_file(path: str) -> list[Hand]:
    with open(path, "rb") as fh:
        raw = fh.read()

    text = raw.decode("utf-8-sig", errors="ignore")
    text = re.sub(r"<\?xml[^>]*\?>", "", text).strip()

    if text.count("<session") > 1:
        text = f"<root>{text}</root>"

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    sessions = [root] if root.tag == "session" else list(root.findall("session"))

    hands = []
    for session_el in sessions:
        for game_el in session_el.findall("game"):
            try:
                hand = _parse_one_game(game_el)
            except (TypeError, ValueError, KeyError, IndexError):
                # e.g. 9/10-handed tables outside the project's 6-8max scope
                # (position templates only cover up to 8-max) -- skip, don't
                # let one malformed/out-of-scope hand kill the whole batch.
                continue
            if hand:
                hands.append(hand)
    return hands
