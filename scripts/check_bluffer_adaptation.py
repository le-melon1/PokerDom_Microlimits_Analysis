"""Do real players adapt WITHIN A SESSION to a specific opponent identified
as a frequent bluffer (see find_frequent_bluffers.py -- river aggressors who
lose at real showdown meaningfully more than the 75.3% population baseline)?

Same session-detection (real timestamp gaps) and early-half/late-half pooling
methodology as check_within_session_adaptation.py, but the bettor pool is now
"identified frequent bluffers" instead of "top repeat-opponent pairs" -- a
different, smaller population (49 players), so pairs aren't pre-filtered to
repeat opponents; any responder facing one of these 49 bettors counts.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy import stats

SESSION_GAP_MINUTES = 45
MIN_EVENTS_PER_SESSION = 6


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


def main():
    log("loading frequent_bluffers.csv...")
    bluffers = pd.read_csv("data/reference/frequent_bluffers.csv")
    bluffer_players = set(bluffers["player"])
    log(f"{len(bluffer_players)} identified frequent bluffers")

    log("loading actions.parquet...")
    df = pd.read_parquet("data/processed/actions.parquet", columns=["hand_id", "player", "street", "action"])
    is_bluffer = df["player"].isin(bluffer_players)
    relevant_hand_ids = set(df.loc[is_bluffer, "hand_id"].unique())
    log(f"{len(relevant_hand_ids)} hands involve a known frequent bluffer")

    df_relevant = df[df["hand_id"].isin(relevant_hand_ids)]
    log("extracting facing-aggression events...")
    events = build_facing_events(df_relevant)
    events = events[events["bettor"].isin(bluffer_players)]
    log(f"{len(events)} events where the bettor is a known frequent bluffer")

    log("merging timestamps...")
    ts = pd.read_parquet("data/processed/hand_timestamps.parquet")
    ts["hand_id"] = ts["hand_id"].astype(str)
    events["hand_id"] = events["hand_id"].astype(str)
    events = events.merge(ts[["hand_id", "timestamp"]], on="hand_id", how="inner")
    log(f"{len(events)} events after timestamp merge")

    halved_rows = []
    n_sessions = 0
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
            n_sessions += 1

    if not halved_rows:
        log("no usable sessions found -- try a larger bettor pool or lower MIN_EVENTS_PER_SESSION")
        return

    halved = pd.concat(halved_rows)
    log(f"{n_sessions} usable sessions, {len(halved)} pooled events")

    first_half = halved[halved["half"] == "first"]
    second_half = halved[halved["half"] == "second"]
    n1, n2 = len(first_half), len(second_half)
    f1, f2 = (first_half["response"] == "folds").sum(), (second_half["response"] == "folds").sum()

    print(f"\nfirst half of session (n={n1}): fold%={f1/n1:.4f}, call%={(first_half['response']=='calls').mean():.4f}, raise%={(first_half['response']=='raises').mean():.4f}")
    print(f"second half of session (n={n2}): fold%={f2/n2:.4f}, call%={(second_half['response']=='calls').mean():.4f}, raise%={(second_half['response']=='raises').mean():.4f}")
    print(f"delta (fold%): {f2/n2 - f1/n1:+.4f}")

    try:
        _, pvalue = stats.chi2_contingency([[f1, n1 - f1], [f2, n2 - f2]])[:2]
        print(f"chi2 p-value: {pvalue:.4f}")
    except ValueError:
        print("chi2 test undefined (a cell is zero)")

    halved.to_csv("data/reference/bluffer_adaptation_events.csv", index=False)
    log("saved data/reference/bluffer_adaptation_events.csv")


if __name__ == "__main__":
    main()
