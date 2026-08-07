"""One-time (per dataset refresh) build of archetype-level lookup tables used
by the overnight multi-street EV batch:

1. archetype_position_vpip.csv   -- avg VPIP by (archetype, position)
2. archetype_vs_raise.csv        -- avg fold/call/3bet facing a raise, by (archetype, position)
3. archetype_facing_bet.csv      -- avg fold/call/raise facing a postflop bet,
                                     by (archetype, street, pot-size bucket)

2026-07-30 rewrite: the original version re-parsed all raw .phhs files into
one big `hands` list (every Hand object, with nested Seat/Action objects,
held in memory at once) before doing anything else. That's the exact pattern
that OOM-crashed main.py tonight after the dataset grew from 1000 to 4379
PokerStars files (~3.6M hands) on this 8GB-RAM machine. Fixed two ways:
  1. actions_df/hands_df are no longer re-derived from scratch here -- they're
     read from data/processed/actions.parquet, which rebuild_processed_data.py
     already builds via memory-safe streaming. Avoids a second full re-parse.
  2. build_postflop_response_dataset still needs real Hand objects (it does a
     street-aware sequential walk that flat actions_df can't support), but is
     now called in FILE BATCHES, accumulating only its (much smaller) output
     rows across batches instead of holding all ~3.6M hands at once.
"""

import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.parser.phh_parser import parse_phhs_file
from src.pipeline.archetypes import label_archetypes
from src.pipeline.decision_points import build_postflop_response_dataset
from src.pipeline.preprocess import player_position_stats, player_stats
from src.pipeline.vs_raise_stats import facing_raise_stats

OUT_DIR = "data/reference"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
BATCH_SIZE_FILES = 200


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("loading actions.parquet (already built by rebuild_processed_data.py)...")
    actions_df = pd.read_parquet(PROCESSED_DIR / "actions.parquet")
    log(f"loaded {len(actions_df)} actions")

    stats = label_archetypes(player_stats(actions_df))
    archetype_by_player = dict(zip(stats["player"], stats["archetype"]))
    log(f"archetype labels built ({len(stats)} players)")

    # 1. VPIP by archetype x position
    pos_stats = player_position_stats(actions_df, min_hands_per_position=30)
    pos_stats["archetype"] = pos_stats["player"].map(archetype_by_player)
    pos_stats = pos_stats[~pos_stats["archetype"].isin([None, "Insufficient sample"])]
    vpip_table = (
        pos_stats.groupby(["archetype", "position"], observed=True)
        .agg(vpip=("vpip", "mean"), n_players=("player", "nunique"))
        .reset_index()
    )
    vpip_table.to_csv(f"{OUT_DIR}/archetype_position_vpip.csv", index=False)
    log(f"saved archetype_position_vpip.csv ({len(vpip_table)} rows)")

    # 2. vs-raise stats by archetype x position
    vs_raise = facing_raise_stats(actions_df, min_hands=20)
    vs_raise["archetype"] = vs_raise["player"].map(archetype_by_player)
    vs_raise = vs_raise[~vs_raise["archetype"].isin([None, "Insufficient sample"])]
    vs_raise_table = (
        vs_raise.groupby(["archetype", "position"], observed=True)
        .agg(
            fold_pct=("fold_pct", "mean"),
            call_pct=("call_pct", "mean"),
            threebet_pct=("threebet_pct", "mean"),
            n_players=("player", "nunique"),
        )
        .reset_index()
    )
    vs_raise_table.to_csv(f"{OUT_DIR}/archetype_vs_raise.csv", index=False)
    log(f"saved archetype_vs_raise.csv ({len(vs_raise_table)} rows)")

    # 3. postflop facing-bet stats by archetype x street x pot-size bucket --
    # the slow, memory-sensitive part. Re-parses raw files (needed for the
    # street-aware walk), but in bounded batches, keeping only the much
    # smaller per-response-event rows across batches, not the raw hands.
    files = sorted(glob.glob("data/raw/ps_nl25/*.phhs"))
    log(f"building postflop response dataset from {len(files)} files, in batches of {BATCH_SIZE_FILES}...")
    resp_fragments = []
    n_hands = 0
    for batch_start in range(0, len(files), BATCH_SIZE_FILES):
        batch_files = files[batch_start : batch_start + BATCH_SIZE_FILES]
        batch_hands = []
        for f in batch_files:
            batch_hands.extend(parse_phhs_file(f))
        n_hands += len(batch_hands)
        frag = build_postflop_response_dataset(batch_hands)
        if not frag.empty:
            resp_fragments.append(frag)
        files_done = min(batch_start + BATCH_SIZE_FILES, len(files))
        log(f"  {files_done}/{len(files)} files, {n_hands} hands, {sum(len(f) for f in resp_fragments)} response rows so far")

    resp_df = pd.concat(resp_fragments, ignore_index=True) if resp_fragments else pd.DataFrame()
    log(f"postflop response rows: {len(resp_df)}")

    resp_df["archetype"] = resp_df["responder"].map(archetype_by_player)
    resp_df = resp_df[~resp_df["archetype"].isin([None, "Insufficient sample"])]

    bins = [0, 0.4, 0.7, float("inf")]
    labels = ["small", "medium", "large"]
    resp_df["pot_bucket"] = pd.cut(resp_df["pot_fraction"], bins=bins, labels=labels)

    g = resp_df.groupby(["archetype", "street", "pot_bucket"], observed=True)
    facing_bet_table = pd.DataFrame(
        {
            "n": g.size(),
            "fold_pct": g["response"].apply(lambda s: (s == "folds").mean()),
            "call_pct": g["response"].apply(lambda s: (s == "calls").mean()),
            "raise_pct": g["response"].apply(lambda s: (s == "raises").mean()),
        }
    ).reset_index()
    facing_bet_table = facing_bet_table[facing_bet_table["n"] >= 30]
    facing_bet_table.to_csv(f"{OUT_DIR}/archetype_facing_bet.csv", index=False)
    log(f"saved archetype_facing_bet.csv ({len(facing_bet_table)} rows)")

    # C2: does fold% to a donk/lead (bettor_had_initiative=False) differ from
    # fold% to a cbet/continuation bet (=True)? Collapsed across street and
    # pot_bucket (not split by them, unlike the table above) -- this is a new,
    # coarser question asked for the first time, and splitting by all four
    # dimensions at once would leave most cells too thin to trust. If this
    # shows a real difference, a follow-up pass can narrow to street/sizing.
    gi = resp_df.groupby(["archetype", "bettor_had_initiative"], observed=True)
    initiative_table = pd.DataFrame(
        {
            "n": gi.size(),
            "fold_pct": gi["response"].apply(lambda s: (s == "folds").mean()),
            "call_pct": gi["response"].apply(lambda s: (s == "calls").mean()),
            "raise_pct": gi["response"].apply(lambda s: (s == "raises").mean()),
        }
    ).reset_index()
    initiative_table = initiative_table[initiative_table["n"] >= 30]
    initiative_table.to_csv(f"{OUT_DIR}/archetype_facing_bet_by_initiative.csv", index=False)
    log(f"saved archetype_facing_bet_by_initiative.csv ({len(initiative_table)} rows)")
    print(initiative_table.to_string(index=False))

    log("ALL TABLES BUILT SUCCESSFULLY")


if __name__ == "__main__":
    main()
