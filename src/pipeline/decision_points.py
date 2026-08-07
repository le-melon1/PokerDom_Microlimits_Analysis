"""Build a (bet sizing -> did villain fold) dataset for the fold-equity model.

`pot_before` is approximated as the sum of chips already committed earlier in
the hand (blinds aren't captured as actions by the current parser). Good
enough to rank sizings relative to each other; recalibrate once blind-posting
lines are parsed from real sample hands.
"""

import pandas as pd

from src.parser.models import Hand
from src.parser.positions import assign_positions
from src.pipeline.board_texture import texture_features

STREET_BOARD_LEN = {"flop": 3, "turn": 4, "river": 5}


def build_fold_equity_dataset(hands: list[Hand]) -> pd.DataFrame:
    rows = []

    for hand in hands:
        if hand.big_blind <= 0:
            continue  # rare malformed/missed-blind hand; can't size bets in bb terms
        assign_positions(hand)
        position_by_player = {s.player: s.position for s in hand.seats}
        actions = hand.actions

        committed_so_far = 0.0
        for i, action in enumerate(actions):
            if action.action not in ("bets", "raises"):
                committed_so_far += action.amount
                continue

            board_len = STREET_BOARD_LEN.get(action.street)
            if board_len is None:
                committed_so_far += action.amount
                continue

            pot_before = max(committed_so_far, hand.small_blind + hand.big_blind)
            response = next((a for a in actions[i + 1 :] if a.player != action.player), None)
            committed_so_far += action.amount
            if response is None or response.street != action.street:
                continue

            row = {
                "hand_id": hand.hand_id,
                "street": action.street,
                "position": position_by_player.get(action.player, "UNKNOWN"),
                "opponent_position": position_by_player.get(response.player, "UNKNOWN"),
                "bet_size_bb": action.amount / hand.big_blind,
                "pot_fraction": action.amount / pot_before,
                "villain_folded": int(response.action == "folds"),
            }
            row.update(texture_features(hand.board[:board_len]))
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ("street", "position", "opponent_position"):
            df[col] = df[col].astype("category")
    return df


def build_postflop_response_dataset(hands: list[Hand]) -> pd.DataFrame:
    """Same walk as build_fold_equity_dataset, but keeps player identity and
    the responder's raw action (fold/call/raise) instead of a fold-only flag
    -- needed to join archetype labels and get fold/call/raise splits per
    archetype rather than just a binary fold rate.

    Also tags each postflop bet/raise with `bettor_had_initiative`: was this
    player the last preflop raiser in this hand? False means it's a donk
    bet/lead (betting into the preflop aggressor without having raised
    yourself) rather than a continuation bet -- population-level fold-equity
    tables haven't previously split on this, so it's not yet known whether
    donks and cbets get folded to at different rates here.
    """
    rows = []

    for hand in hands:
        if hand.big_blind <= 0:
            continue
        assign_positions(hand)
        position_by_player = {s.player: s.position for s in hand.seats}
        actions = hand.actions
        last_preflop_raiser = next(
            (a.player for a in reversed(actions) if a.street == "preflop" and a.action == "raises"), None
        )

        committed_so_far = 0.0
        for i, action in enumerate(actions):
            if action.action not in ("bets", "raises"):
                committed_so_far += action.amount
                continue

            board_len = STREET_BOARD_LEN.get(action.street)
            if board_len is None:
                committed_so_far += action.amount
                continue

            pot_before = max(committed_so_far, hand.small_blind + hand.big_blind)
            response = next((a for a in actions[i + 1 :] if a.player != action.player), None)
            committed_so_far += action.amount
            if response is None or response.street != action.street:
                continue

            rows.append(
                {
                    "hand_id": hand.hand_id,
                    "street": action.street,
                    "bettor": action.player,
                    "responder": response.player,
                    "bettor_position": position_by_player.get(action.player, "UNKNOWN"),
                    "responder_position": position_by_player.get(response.player, "UNKNOWN"),
                    "bet_size_bb": action.amount / hand.big_blind,
                    "pot_fraction": action.amount / pot_before,
                    "response": response.action,
                    "bettor_had_initiative": action.player == last_preflop_raiser,
                }
            )

    return pd.DataFrame(rows)


def breakeven_fold_frequency(pot_fraction: "pd.Series | float"):
    """Minimum fold% needed for a pure bluff to break even: bet / (pot + bet)."""
    return pot_fraction / (1 + pot_fraction)
