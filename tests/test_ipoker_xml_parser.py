"""Fixtures are real, single-hand samples from the HHSmithy/PokerHandHistoryParser
open-source project (MIT-licensed), used here to validate parsing logic against
a known-real format rather than a guessed one. Pot totals were independently
verified by hand against the players' final `bet` totals in the raw XML."""

from pathlib import Path

from src.parser.ipoker_xml_parser import parse_ipoker_xml_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_3bet_shove_heads_up_pot_and_positions():
    hands = parse_ipoker_xml_file(str(FIXTURES / "ipoker_3bet_sample.txt"))
    assert len(hands) == 1
    hand = hands[0]

    assert hand.small_blind == 0.05
    assert hand.big_blind == 0.10
    assert round(hand.pot, 2) == 19.22  # verified: 12.12 (Amalfitano1) + 7.10 (killAA007)

    positions = {s.player: s.position for s in hand.seats}
    assert positions["Amalfitano1"] == "BTN"
    assert positions["killAA007"] == "BB"  # button posts SB itself heads-up

    assert hand.hole_cards["Amalfitano1"] == ["Jc", "Js"]
    assert hand.board == ["5d", "3s", "2c", "9s", "Qd"]


def test_3bet_shove_action_sequence():
    hands = parse_ipoker_xml_file(str(FIXTURES / "ipoker_3bet_sample.txt"))
    hand = hands[0]
    actions = [(a.player, a.action, round(a.amount, 2)) for a in hand.actions]
    assert actions == [
        ("Amalfitano1", "raises", 0.35),   # opens to 0.40 total (0.05 already in)
        ("Amalfitano1", "raises", 11.70),  # 4-bet shove to 12.10 total
        ("killAA007", "raises", 1.10),     # 3-bets to 1.20 total (0.10 already in)
        ("killAA007", "calls", 5.88),      # calls all-in for remaining stack
    ]


def test_ante_hand_pot_matches_manual_reconstruction():
    hands = parse_ipoker_xml_file(str(FIXTURES / "ipoker_ante_sample.txt"))
    assert len(hands) == 1
    hand = hands[0]
    # 3x $0.02 ante + preflop (SB completes to 0.10, BB checks) + river bet 0.10
    assert round(hand.pot, 2) == 0.36
