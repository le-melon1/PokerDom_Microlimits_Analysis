"""Do real players change how they respond to a SPECIFIC repeat opponent's
aggression over the course of playing many hands against them?

For each of the top-N most-repeated player pairs (see find_repeat_opponents.py),
extract every "player X faces a bet/raise directly from player Y" event across
their shared history (all streets, using the same "immediate next responder"
convention as vs_raise_stats.py/decision_points.py), order chronologically by
hand_id (a real sequentially-assigned PokerStars ID), split into first half vs
second half, and compare fold/call/raise rates between the two halves.

This only tests ONE narrow, well-defined kind of adaptation (response to
direct aggression from one specific opponent) -- it doesn't test bluffing
frequency, bet sizing choices, or hand selection against that opponent. A null
result here doesn't rule out adaptation elsewhere; a positive result is real
evidence adaptation exists in at least this dimension.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy import stats

TOP_N_PAIRS = 30
MIN_EVENTS_PER_DIRECTION = 40  # need enough on each side of the split to say anything


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_facing_events(df: pd.DataFrame) -> pd.DataFrame:
    """Every action by a non-aggressor player counts as 'facing' whoever the
    current street aggressor is (updated as raises happen) -- broader than
    decision_points.py/vs_raise_stats.py's "immediate next responder only"
    convention, needed here because a specific pair's *direct* face-offs are
    a thin subset of their shared hands (usually other players act between
    them at a 6-8max table).
    """
    rows = []
    for hand_id, grp in df.groupby("hand_id", sort=False):
        current_aggressor = None
        current_street = None
        for a in grp.itertuples(index=False):
            if a.street != current_street:
                current_street = a.street
                current_aggressor = None
            if current_aggressor is not None and a.player != current_aggressor:
                rows.append(
                    {"hand_id": hand_id, "bettor": current_aggressor, "responder": a.player, "response": a.action}
                )
            if a.action in ("bets", "raises"):
                current_aggressor = a.player
    return pd.DataFrame(rows)


def main():
    log("loading repeat_opponent_pairs.csv...")
    pairs_df = pd.read_csv("data/reference/repeat_opponent_pairs.csv").head(TOP_N_PAIRS)
    target_players = set(pairs_df["player_a"]) | set(pairs_df["player_b"])
    log(f"top {TOP_N_PAIRS} pairs -> {len(target_players)} distinct players")

    log("loading actions.parquet...")
    df = pd.read_parquet("data/processed/actions.parquet", columns=["hand_id", "player", "street", "action"])

    log("finding hands where 2+ target players are present...")
    is_target = df["player"].isin(target_players)
    target_hand_ids = df.loc[is_target].groupby("hand_id")["player"].nunique()
    relevant_hand_ids = set(target_hand_ids[target_hand_ids >= 2].index)
    log(f"{len(relevant_hand_ids)} hands involve 2+ target players")

    df_relevant = df[df["hand_id"].isin(relevant_hand_ids)]
    log(f"extracting facing-aggression events from {len(df_relevant)} action rows...")
    events = build_facing_events(df_relevant)
    log(f"{len(events)} total facing-aggression events extracted")
    events["hand_id_num"] = events["hand_id"].astype("int64")

    results = []
    for _, prow in pairs_df.iterrows():
        a, b = prow["player_a"], prow["player_b"]
        for bettor, responder in [(a, b), (b, a)]:
            sub = events[(events["bettor"] == bettor) & (events["responder"] == responder)]
            sub = sub.sort_values("hand_id_num")
            n = len(sub)
            if n < MIN_EVENTS_PER_DIRECTION:
                continue

            mid = n // 2
            first_half, second_half = sub.iloc[:mid], sub.iloc[mid:]

            fold1, fold2 = (first_half["response"] == "folds").mean(), (second_half["response"] == "folds").mean()
            call1, call2 = (first_half["response"] == "calls").mean(), (second_half["response"] == "calls").mean()
            raise1, raise2 = (first_half["response"] == "raises").mean(), (second_half["response"] == "raises").mean()

            n_fold1 = (first_half["response"] == "folds").sum()
            n_fold2 = (second_half["response"] == "folds").sum()
            try:
                _, pvalue = stats.chi2_contingency(
                    [[n_fold1, mid - n_fold1], [n_fold2, (n - mid) - n_fold2]]
                )[:2]
            except ValueError:
                # a zero marginal (e.g. 0 folds in both halves) makes the test
                # undefined, not "no difference" -- flag rather than fabricate a p-value
                pvalue = float("nan")

            results.append(
                {
                    "responder": responder,
                    "bettor": bettor,
                    "n_events": n,
                    "fold_pct_first_half": fold1,
                    "fold_pct_second_half": fold2,
                    "fold_pct_delta": fold2 - fold1,
                    "call_pct_first_half": call1,
                    "call_pct_second_half": call2,
                    "raise_pct_first_half": raise1,
                    "raise_pct_second_half": raise2,
                    "fold_shift_pvalue": pvalue,
                }
            )

    results_df = pd.DataFrame(results).sort_values("fold_shift_pvalue")
    results_df.to_csv("data/reference/pairwise_adaptation_check.csv", index=False)
    log(f"saved {len(results_df)} directional pair results")

    sig = results_df[results_df["fold_shift_pvalue"] < 0.05]
    log(f"{len(sig)} / {len(results_df)} directions show p<0.05 fold-rate shift between halves")
    print(results_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
