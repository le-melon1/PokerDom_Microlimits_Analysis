"""Does the within-session fold-rate shift (see check_within_session_adaptation.py:
-1.19pp, p=0.026, pooled) look different when the BETTOR being responded to is
an obviously distinctive player (Maniac or Nit -- the two extreme archetypes,
easy to notice within a session) vs a more average-looking one (TAG/LAG/
Station/Loose-passive, which blend into "normal" play more)?

Same session-detection (real timestamp gaps) and pooling logic as
check_within_session_adaptation.py, split by the bettor's archetype label.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy import stats

from src.pipeline.archetypes import label_archetypes
from src.pipeline.preprocess import player_stats

TOP_N_PAIRS = 300
SESSION_GAP_MINUTES = 45
MIN_EVENTS_PER_SESSION = 6
EXTREME_ARCHETYPES = {"Maniac", "Nit"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_facing_events(df: pd.DataFrame) -> pd.DataFrame:
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


def assign_sessions(sub: pd.DataFrame) -> pd.Series:
    gap = sub["timestamp"].diff().dt.total_seconds().fillna(0)
    return (gap > (SESSION_GAP_MINUTES * 60)).cumsum()


def pooled_shift(half_col_df: pd.DataFrame, label: str):
    first_half = half_col_df[half_col_df["half"] == "first"]
    second_half = half_col_df[half_col_df["half"] == "second"]
    n1, n2 = len(first_half), len(second_half)
    if n1 == 0 or n2 == 0:
        print(f"{label}: insufficient data (n1={n1}, n2={n2})")
        return
    f1, f2 = (first_half["response"] == "folds").sum(), (second_half["response"] == "folds").sum()
    try:
        _, pvalue = stats.chi2_contingency([[f1, n1 - f1], [f2, n2 - f2]])[:2]
    except ValueError:
        pvalue = float("nan")
    print(
        f"{label}: n_sessions_contrib unknown-here, n1={n1}, n2={n2}, "
        f"fold%% first={f1/n1:.4f}, fold%% second={f2/n2:.4f}, delta={f2/n2 - f1/n1:+.4f}, p={pvalue:.4f}"
    )


def main():
    log("rebuilding per-player archetype labels from actions.parquet (the /tmp cache from the earlier session is gone)...")
    actions_for_archetypes = pd.read_parquet("data/processed/actions.parquet")
    archetype_stats = label_archetypes(player_stats(actions_for_archetypes))
    archetype_by_player = dict(zip(archetype_stats["player"], archetype_stats["archetype"]))
    del actions_for_archetypes
    log(f"{len(archetype_by_player)} players with archetype labels")

    log("loading repeat_opponent_pairs.csv...")
    pairs_df = pd.read_csv("data/reference/repeat_opponent_pairs.csv").head(TOP_N_PAIRS)
    target_players = set(pairs_df["player_a"]) | set(pairs_df["player_b"])

    log("loading actions.parquet...")
    df = pd.read_parquet("data/processed/actions.parquet", columns=["hand_id", "player", "street", "action"])
    is_target = df["player"].isin(target_players)
    target_hand_ids = df.loc[is_target].groupby("hand_id")["player"].nunique()
    relevant_hand_ids = set(target_hand_ids[target_hand_ids >= 2].index)

    df_relevant = df[df["hand_id"].isin(relevant_hand_ids)]
    log("extracting facing-aggression events...")
    events = build_facing_events(df_relevant)
    log(f"{len(events)} events extracted")

    log("merging timestamps...")
    ts = pd.read_parquet("data/processed/hand_timestamps.parquet")
    ts["hand_id"] = ts["hand_id"].astype(str)
    events["hand_id"] = events["hand_id"].astype(str)
    events = events.merge(ts[["hand_id", "timestamp"]], on="hand_id", how="inner")

    pair_set = set(zip(pairs_df["player_a"], pairs_df["player_b"])) | set(
        zip(pairs_df["player_b"], pairs_df["player_a"])
    )
    mask = pd.MultiIndex.from_arrays([events["bettor"], events["responder"]]).isin(pair_set)
    events = events[mask]
    log(f"{len(events)} events within target pairs")

    events["bettor_archetype"] = events["bettor"].map(archetype_by_player)
    events = events[~events["bettor_archetype"].isna()]
    log(f"{len(events)} events with a known bettor archetype")
    print(events["bettor_archetype"].value_counts())

    halved_rows = []
    for (bettor, responder), sub in events.groupby(["bettor", "responder"], sort=False):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        sub["session_id"] = assign_sessions(sub)
        for _, session in sub.groupby("session_id"):
            if len(session) < MIN_EVENTS_PER_SESSION:
                continue
            mid = len(session) // 2
            session = session.copy()
            session["half"] = ["first"] * mid + ["second"] * (len(session) - mid)
            halved_rows.append(session)

    halved = pd.concat(halved_rows)
    log(f"{halved['bettor'].astype(str).add(halved['responder'].astype(str)).nunique()} pair-instances contributed sessions")
    halved.to_csv("data/reference/within_session_events_by_archetype.csv", index=False)

    print("\n=== ALL bettors pooled ===")
    pooled_shift(halved, "all")

    print("\n=== EXTREME bettor archetypes (Maniac + Nit) ===")
    pooled_shift(halved[halved["bettor_archetype"].isin(EXTREME_ARCHETYPES)], "extreme (Maniac/Nit)")

    print("\n=== MODERATE bettor archetypes (TAG/LAG/Station/Loose-passive) ===")
    pooled_shift(halved[~halved["bettor_archetype"].isin(EXTREME_ARCHETYPES)], "moderate")

    print("\n=== per-archetype breakdown ===")
    for arch in halved["bettor_archetype"].unique():
        pooled_shift(halved[halved["bettor_archetype"] == arch], arch)


if __name__ == "__main__":
    main()
