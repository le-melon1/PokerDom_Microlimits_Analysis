"""Select ~20 real, well-observed players to serve as the seed set for
"real player" behavior-clone bots (PokerDom_Practice_App), covering the six
archetype buckets in roughly the same proportion as the real population
(ARCHETYPE_POPULATION_WEIGHTS in the sibling repo's live_dynamics.py), so the
sample isn't accidentally all TAGs or all Maniacs.

Within each archetype, picks the players with the MOST observed hands (more
data -> a more reliable, more idiosyncratic profile, and more real sessions
to learn within-session dynamics from), among players who also have a
reasonable number of distinct real sessions (>= MIN_SESSIONS) so the
session-dynamics half of the project has something to learn from per player
-- a player with 2000 hands in one uninterrupted sitting doesn't tell us
anything about session-position effects.

Outputs data/reference/player_profile_seeds.csv: one row per selected
player with their real aggregate stats, archetype, hand count, and session
count -- the starting point for the richer per-player feature extraction
(scripts/build_player_profiles.py, not written yet).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.pipeline.archetypes import UNKNOWN_LABEL, label_archetypes
from src.pipeline.preprocess import player_stats

N_PROFILES = 20
MIN_HANDS = 500
MIN_SESSIONS = 5
SESSION_GAP_MINUTES = 45

# Same real counts already used for ARCHETYPE_POPULATION_WEIGHTS in
# PokerDom_Practice_App/backend/sessions/live_dynamics.py (3.56M-hand
# dataset, >=100-hand reliable labels) -- kept in sync manually, see that
# file's own comment for provenance.
ARCHETYPE_POPULATION_WEIGHTS = {
    "Loose-passive": 8007,
    "Station": 7001,
    "LAG": 3899,
    "Maniac": 3352,
    "TAG": 2547,
    "Nit": 1991,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def largest_remainder_allocation(weights: dict, total: int) -> dict:
    """Standard largest-remainder (Hamilton) apportionment: split `total`
    discrete slots across `weights` proportionally, without ties/rounding
    ever dropping the total below/above the requested count."""
    keys = list(weights.keys())
    weight_sum = sum(weights.values())
    raw = {k: total * weights[k] / weight_sum for k in keys}
    base = {k: int(raw[k]) for k in keys}
    remainder = total - sum(base.values())
    order = sorted(keys, key=lambda k: raw[k] - base[k], reverse=True)
    for k in order[:remainder]:
        base[k] += 1
    return base


def n_sessions_for(timestamps: pd.Series) -> int:
    ts = timestamps.sort_values()
    gap = ts.diff().dt.total_seconds().fillna(0)
    return int((gap > SESSION_GAP_MINUTES * 60).sum()) + 1


def main():
    log("loading actions.parquet (player, hand_id, street, action)...")
    actions = pd.read_parquet("data/processed/actions.parquet", columns=["player", "hand_id", "street", "action"])

    log("computing player_stats...")
    stats = player_stats(actions)
    del actions

    log("labeling archetypes...")
    labeled = label_archetypes(stats)
    reliable = labeled[(labeled["archetype"] != UNKNOWN_LABEL) & (labeled["hands_seen"] >= MIN_HANDS)].copy()
    log(f"{len(reliable)} players with hands_seen >= {MIN_HANDS} and a reliable archetype label")

    log("loading hand_timestamps.parquet to count real sessions per candidate...")
    timestamps = pd.read_parquet("data/processed/hand_timestamps.parquet")
    log(f"timestamps columns: {list(timestamps.columns)}")
    # actions.parquet has (player, hand_id); timestamps has (hand_id, timestamp).
    # Rebuild a slim (player, hand_id) map restricted to candidate players only,
    # to avoid re-loading the full 225M actions.parquet a second time.
    actions_slim = pd.read_parquet("data/processed/actions.parquet", columns=["player", "hand_id"])
    actions_slim = actions_slim[actions_slim["player"].isin(set(reliable["player"]))].drop_duplicates()
    merged = actions_slim.merge(timestamps[["hand_id", "timestamp"]], on="hand_id", how="inner")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    log(f"{len(merged)} (player, hand) rows with timestamps for {merged['player'].nunique()} candidate players")

    log("counting sessions per candidate player...")
    session_counts = merged.groupby("player")["timestamp"].apply(n_sessions_for).rename("n_sessions")
    reliable = reliable.merge(session_counts, on="player", how="left")
    reliable["n_sessions"] = reliable["n_sessions"].fillna(0).astype(int)

    eligible = reliable[reliable["n_sessions"] >= MIN_SESSIONS].copy()
    log(f"{len(eligible)} players also have >= {MIN_SESSIONS} distinct real sessions")

    allocation = largest_remainder_allocation(ARCHETYPE_POPULATION_WEIGHTS, N_PROFILES)
    log(f"target allocation for {N_PROFILES} profiles: {allocation}")

    chosen_rows = []
    for archetype, n in allocation.items():
        pool = eligible[eligible["archetype"] == archetype].sort_values("hands_seen", ascending=False)
        picked = pool.head(n)
        if len(picked) < n:
            log(f"WARNING: only {len(picked)}/{n} eligible {archetype} players available (of {len(pool)} candidates)")
        chosen_rows.append(picked)

    chosen = pd.concat(chosen_rows, ignore_index=True)
    chosen = chosen.sort_values(["archetype", "hands_seen"], ascending=[True, False])
    chosen["profile_id"] = [f"real_{i+1:02d}" for i in range(len(chosen))]

    out_path = Path("data/reference/player_profile_seeds.csv")
    chosen.to_csv(out_path, index=False)
    log(f"wrote {len(chosen)} profiles to {out_path}")
    print(chosen[["profile_id", "player", "archetype", "hands_seen", "n_sessions", "vpip", "pfr", "aggression_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
