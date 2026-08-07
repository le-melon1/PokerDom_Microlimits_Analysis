"""Within-session pairwise adaptation check -- the corrected version of
check_pairwise_adaptation.py's question, per user feedback: that script
compared "earlier vs later across the pair's entire ~3-week co-occurrence
history," which can miss adaptation that happens instantly within a single
sitting (ceiling effect) or gets diluted by the days/weeks of hands against
OTHER opponents in between.

This one detects actual continuous sessions (real timestamps, gap >
SESSION_GAP_MINUTES ends a session) for many pairs, then POOLS every
session's early-half vs late-half facing-events together -- instead of
needing one huge session per pair (individual sessions between two specific
players are short), aggregate power comes from pooling across many session
instances. Tests: does the pooled fold rate differ between the first and
second half of a session, for events between two specific repeat players?
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy import stats

TOP_N_PAIRS = 300
SESSION_GAP_MINUTES = 45
MIN_EVENTS_PER_SESSION = 6  # need >=3 per half to be worth splitting


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_facing_events(df: pd.DataFrame) -> pd.DataFrame:
    """Same broadened 'facing the current street aggressor' convention as
    check_pairwise_adaptation.py -- see that file for why."""
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
    """sub must be sorted by timestamp. Returns a session-id Series: a new
    id starts whenever the gap since the previous event exceeds the
    session-break threshold."""
    gap = sub["timestamp"].diff().dt.total_seconds().fillna(0)
    new_session = gap > (SESSION_GAP_MINUTES * 60)
    return new_session.cumsum()


def main():
    log("loading repeat_opponent_pairs.csv...")
    pairs_df = pd.read_csv("data/reference/repeat_opponent_pairs.csv").head(TOP_N_PAIRS)
    target_players = set(pairs_df["player_a"]) | set(pairs_df["player_b"])
    log(f"top {TOP_N_PAIRS} pairs -> {len(target_players)} distinct players")

    log("loading actions.parquet...")
    df = pd.read_parquet("data/processed/actions.parquet", columns=["hand_id", "player", "street", "action"])
    is_target = df["player"].isin(target_players)
    target_hand_ids = df.loc[is_target].groupby("hand_id")["player"].nunique()
    relevant_hand_ids = set(target_hand_ids[target_hand_ids >= 2].index)
    log(f"{len(relevant_hand_ids)} hands involve 2+ target players")

    df_relevant = df[df["hand_id"].isin(relevant_hand_ids)]
    log("extracting facing-aggression events...")
    events = build_facing_events(df_relevant)
    log(f"{len(events)} total facing-aggression events extracted")

    log("loading hand_timestamps.parquet and merging...")
    ts = pd.read_parquet("data/processed/hand_timestamps.parquet")
    ts["hand_id"] = ts["hand_id"].astype(str)
    events["hand_id"] = events["hand_id"].astype(str)
    events = events.merge(ts[["hand_id", "timestamp"]], on="hand_id", how="inner")
    log(f"{len(events)} events after timestamp merge")

    pair_set = set(zip(pairs_df["player_a"], pairs_df["player_b"])) | set(
        zip(pairs_df["player_b"], pairs_df["player_a"])
    )
    mask = pd.MultiIndex.from_arrays([events["bettor"], events["responder"]]).isin(pair_set)
    events = events[mask]
    log(f"{len(events)} events are within our target pairs specifically")

    first_half_rows = []
    second_half_rows = []
    n_sessions_used = 0

    for (bettor, responder), sub in events.groupby(["bettor", "responder"], sort=False):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        sub["session_id"] = assign_sessions(sub)

        for _, session in sub.groupby("session_id"):
            if len(session) < MIN_EVENTS_PER_SESSION:
                continue
            mid = len(session) // 2
            first_half_rows.append(session.iloc[:mid])
            second_half_rows.append(session.iloc[mid:])
            n_sessions_used += 1

    first_half = pd.concat(first_half_rows) if first_half_rows else pd.DataFrame(columns=events.columns)
    second_half = pd.concat(second_half_rows) if second_half_rows else pd.DataFrame(columns=events.columns)

    log(f"{n_sessions_used} usable sessions (>= {MIN_EVENTS_PER_SESSION} events, gap > {SESSION_GAP_MINUTES}min ends a session)")
    log(f"first-half pooled events: {len(first_half)}, second-half pooled events: {len(second_half)}")

    for label, half in [("first half of session", first_half), ("second half of session", second_half)]:
        if len(half) == 0:
            continue
        print(f"\n{label} (n={len(half)}):")
        print(f"  fold%  = {(half['response'] == 'folds').mean():.4f}")
        print(f"  call%  = {(half['response'] == 'calls').mean():.4f}")
        print(f"  raise% = {(half['response'] == 'raises').mean():.4f}")

    if len(first_half) > 0 and len(second_half) > 0:
        n1, n2 = len(first_half), len(second_half)
        f1, f2 = (first_half["response"] == "folds").sum(), (second_half["response"] == "folds").sum()
        try:
            _, pvalue = stats.chi2_contingency([[f1, n1 - f1], [f2, n2 - f2]])[:2]
        except ValueError:
            pvalue = float("nan")
        print(f"\nfold-rate shift (early-session vs late-session), pooled across {n_sessions_used} sessions:")
        print(f"  delta = {f2 / n2 - f1 / n1:+.4f}, chi2 p-value = {pvalue:.4f}")

    all_sessions = pd.concat(first_half_rows + second_half_rows) if first_half_rows else pd.DataFrame()
    all_sessions.to_csv("data/reference/within_session_events.csv", index=False)
    log("saved data/reference/within_session_events.csv")


if __name__ == "__main__":
    main()
