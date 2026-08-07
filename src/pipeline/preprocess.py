"""Flatten parsed Hand objects into pandas DataFrames and compute core stats."""

import pandas as pd

from src.parser.models import Hand
from src.parser.positions import assign_positions

ACTION_CATEGORY_DTYPE = pd.CategoricalDtype(
    categories=["folds", "checks", "calls", "bets", "raises"], ordered=False
)


def hands_to_frames(hands: list[Hand]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (hands_df, actions_df) — one row per hand and one row per action."""
    hand_rows = []
    action_rows = []

    for hand in hands:
        assign_positions(hand)
        position_by_player = {s.player: s.position for s in hand.seats}
        stack_by_player = {s.player: s.stack for s in hand.seats}

        hand_rows.append(
            {
                "hand_id": hand.hand_id,
                "table_name": hand.table_name,
                "max_seats": hand.max_seats,
                "n_seats_active": len(hand.seats),
                "small_blind": hand.small_blind,
                "big_blind": hand.big_blind,
                "pot": hand.pot,
                "rake": hand.rake,
                "board": " ".join(hand.board),
                "n_board_cards": len(hand.board),
            }
        )

        for action in hand.actions:
            action_rows.append(
                {
                    "hand_id": hand.hand_id,
                    "player": action.player,
                    "position": position_by_player.get(action.player, "UNKNOWN"),
                    "stack": stack_by_player.get(action.player, float("nan")),
                    "street": action.street,
                    "action": action.action,
                    "amount": action.amount,
                    "big_blind": hand.big_blind,
                }
            )

    hands_df = pd.DataFrame(hand_rows)
    actions_df = pd.DataFrame(action_rows)
    if not actions_df.empty:
        actions_df["action"] = actions_df["action"].astype(ACTION_CATEGORY_DTYPE)
        actions_df["position"] = actions_df["position"].astype("category")
        actions_df["street"] = actions_df["street"].astype("category")
        actions_df["amount_bb"] = actions_df["amount"] / actions_df["big_blind"]

    return hands_df, actions_df


def player_stats(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Core per-player frequencies: VPIP, PFR, 3-bet, aggression factor, hands seen."""
    preflop = actions_df[actions_df["street"] == "preflop"]

    hands_seen = actions_df.groupby("player")["hand_id"].nunique().rename("hands_seen")

    vpip_hands = preflop[preflop["action"].isin(["calls", "bets", "raises"])].groupby("player")[
        "hand_id"
    ].nunique()
    pfr_hands = preflop[preflop["action"] == "raises"].groupby("player")["hand_id"].nunique()

    postflop = actions_df[actions_df["street"] != "preflop"]
    aggressive = postflop[postflop["action"].isin(["bets", "raises"])].groupby("player").size()
    passive = postflop[postflop["action"].isin(["calls"])].groupby("player").size()

    stats = pd.DataFrame(hands_seen)
    stats["vpip"] = (vpip_hands / stats["hands_seen"]).fillna(0.0)
    stats["pfr"] = (pfr_hands / stats["hands_seen"]).fillna(0.0)
    stats["aggression_factor"] = (aggressive / passive.replace(0, pd.NA)).fillna(0.0)
    return stats.reset_index()


def player_position_stats(actions_df: pd.DataFrame, min_hands_per_position: int = 30) -> pd.DataFrame:
    """VPIP/PFR broken down by (player, position) -- needed for implied-range
    estimates, since opening range varies enormously by position and a single
    blended VPIP hides that. Positions below `min_hands_per_position` for a
    given player are dropped as too small to trust.
    """
    preflop = actions_df[actions_df["street"] == "preflop"]

    hands_seen = preflop.groupby(["player", "position"], observed=True)["hand_id"].nunique().rename(
        "hands_seen"
    )
    vpip_hands = (
        preflop[preflop["action"].isin(["calls", "bets", "raises"])]
        .groupby(["player", "position"], observed=True)["hand_id"]
        .nunique()
    )
    pfr_hands = (
        preflop[preflop["action"] == "raises"]
        .groupby(["player", "position"], observed=True)["hand_id"]
        .nunique()
    )

    stats = pd.DataFrame(hands_seen)
    stats["vpip"] = (vpip_hands / stats["hands_seen"]).fillna(0.0)
    stats["pfr"] = (pfr_hands / stats["hands_seen"]).fillna(0.0)
    stats = stats.reset_index()
    return stats[stats["hands_seen"] >= min_hands_per_position]


def winrate_bb_per_100(hands_df: pd.DataFrame, actions_df: pd.DataFrame, winnings_by_player: dict) -> pd.DataFrame:
    """winnings_by_player: {player: net_bb_won_across_all_hands}. Kept as an explicit
    input rather than re-derived here, since net winnings require matching contributed
    vs collected amounts per hand -- do that once in the caller and pass the result in.
    """
    hands_seen = actions_df.groupby("player")["hand_id"].nunique()
    rows = []
    for player, net_bb in winnings_by_player.items():
        n = hands_seen.get(player, 0)
        if n == 0:
            continue
        rows.append({"player": player, "hands": n, "winrate_bb_per_100": 100 * net_bb / n})
    return pd.DataFrame(rows)
