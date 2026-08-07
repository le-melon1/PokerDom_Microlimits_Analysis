"""Per-player fold/call/3-bet frequency when facing an opening raise.

Needed alongside VPIP-implied opening ranges (analysis/implied_range.py) to
also estimate a player's *defending* range -- opening range and calling/3-bet
range are different populations of hands, so a single VPIP number isn't
enough to model a confrontation between two players.
"""

import pandas as pd


def facing_raise_stats(actions_df: pd.DataFrame, min_hands: int = 20) -> pd.DataFrame:
    preflop = actions_df[actions_df["street"] == "preflop"][
        ["hand_id", "player", "position", "action"]
    ]

    rows = []
    for _, grp in preflop.groupby("hand_id", sort=False):
        actions = list(grp.itertuples(index=False))
        open_i = next((i for i, a in enumerate(actions) if a.action in ("bets", "raises")), None)
        if open_i is None:
            continue
        opener = actions[open_i].player

        seen = set()
        for a in actions[open_i + 1 :]:
            if a.player == opener or a.player in seen:
                continue
            seen.add(a.player)
            rows.append((a.player, a.position, a.action))

    resp_df = pd.DataFrame(rows, columns=["player", "position", "response"])
    if resp_df.empty:
        return resp_df

    grouped = resp_df.groupby(["player", "position"], observed=True)
    hands_faced = grouped.size().rename("hands_faced")
    fold_pct = grouped["response"].apply(lambda s: (s == "folds").mean()).rename("fold_pct")
    call_pct = grouped["response"].apply(lambda s: (s == "calls").mean()).rename("call_pct")
    threebet_pct = grouped["response"].apply(lambda s: (s == "raises").mean()).rename("threebet_pct")

    out = pd.concat([hands_faced, fold_pct, call_pct, threebet_pct], axis=1).reset_index()
    return out[out["hands_faced"] >= min_hands]
