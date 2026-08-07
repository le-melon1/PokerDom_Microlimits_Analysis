from pathlib import Path

from src.parser.hand_history_parser import parse_file
from src.parser.positions import assign_positions

FIXTURE = Path(__file__).parent / "fixtures" / "sample_hands.txt"


def test_parses_expected_number_of_hands():
    hands = parse_file(str(FIXTURE))
    assert len(hands) == 2


def test_first_hand_header_and_seats():
    hands = parse_file(str(FIXTURE))
    hand = hands[0]
    assert hand.hand_id == "100000000001"
    assert hand.small_blind == 1.0
    assert hand.big_blind == 2.0
    assert hand.max_seats == 6
    assert len(hand.seats) == 6
    assert hand.hole_cards["Hero"] == ["Ah", "Kd"]


def test_first_hand_actions_and_board():
    hands = parse_file(str(FIXTURE))
    hand = hands[0]
    assert hand.board == ["2h", "7c", "Jd"]
    assert hand.pot == 14.0
    assert hand.rake == 1.0

    streets = [a.street for a in hand.actions]
    assert "preflop" in streets
    assert "flop" in streets

    last_action = hand.actions[-1]
    assert last_action.player == "Villain3"
    assert last_action.action == "folds"


def test_second_hand_turn_board_has_four_cards():
    hands = parse_file(str(FIXTURE))
    hand = hands[1]
    assert hand.board == ["7h", "8c", "2d", "4s"]
    assert hand.pot == 92.0


def test_positions_assigned_button_first():
    hands = parse_file(str(FIXTURE))
    hand = hands[0]
    assign_positions(hand)
    button_seat = next(s for s in hand.seats if s.seat_no == hand.button_seat)
    assert button_seat.position == "BTN"
    positions = {s.position for s in hand.seats}
    assert "BTN" in positions and "SB" in positions and "BB" in positions
