"""Memory-safe, STREAMING rebuild of data/processed/hands.parquet +
actions.parquet.

main.py's `load_all_hands()` parses every raw file into one giant list of
Hand objects (each with nested Seat/Action objects) BEFORE calling
hands_to_frames() a single time at the end. That's fine at ~1000 files/841k
hands, but after the 2026-07-30 dataset expansion (1000 -> 4379 PokerStars
files, ~3.56M hands / tens of millions of actions total across all raw
sources) it silently exhausted memory on this machine (8GB total RAM):
main.py printed "Parsed 3,564,757 hands" and the process just disappeared --
no traceback, no parquet update -- almost certainly an OOM kill that a
`| head` / `| tee` pipe masked (the pipeline's reported exit code reflects
the pipe's LAST command, not python3's real one).

An earlier attempt at a "lighter" rewrite (flat dicts instead of full Hand
objects, but still ALL of them accumulated in Python lists before building
one big DataFrame) was calculated to still need on the order of ~28GB+ for
the actions alone -- still far more than this machine has. The actual fix:
never hold more than one BATCH of files' worth of rows in memory. Process
files in batches, build a small DataFrame per batch, and stream each batch
into the output parquet files via pyarrow's ParquetWriter (which appends row
groups to a growing file without needing the whole dataset in memory at once).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config
from src.parser.hand_history_parser import parse_file
from src.parser.ipoker_xml_parser import parse_ipoker_xml_file
from src.parser.phh_parser import parse_phhs_file
from src.parser.positions import assign_positions
from src.pipeline.preprocess import ACTION_CATEGORY_DTYPE

BATCH_SIZE_FILES = 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _is_ipoker_xml(txt_file: Path) -> bool:
    with open(txt_file, "rb") as fh:
        head = fh.read(200).decode("utf-8-sig", errors="ignore").lstrip()
    return head.startswith("<?xml") or head.startswith("<session")


def _hand_to_rows(hand):
    assign_positions(hand)
    position_by_player = {s.player: s.position for s in hand.seats}
    stack_by_player = {s.player: s.stack for s in hand.seats}

    hand_row = {
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
    action_rows = [
        {
            "hand_id": hand.hand_id,
            "player": a.player,
            "position": position_by_player.get(a.player, "UNKNOWN"),
            "stack": stack_by_player.get(a.player, float("nan")),
            "street": a.street,
            "action": a.action,
            "amount": a.amount,
            "big_blind": hand.big_blind,
        }
        for a in hand.actions
    ]
    return hand_row, action_rows


def _parse_file_to_hands(f: Path):
    if f.suffix == ".phhs":
        return parse_phhs_file(str(f))
    if _is_ipoker_xml(f):
        return parse_ipoker_xml_file(str(f))
    return parse_file(str(f))


def main():
    raw_dir = config.RAW_DATA_DIR
    files = sorted(raw_dir.rglob("*.txt")) + sorted(raw_dir.rglob("*.phhs"))
    total_files = len(files)
    log(f"{total_files} raw files to process, in batches of {BATCH_SIZE_FILES}")

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    hands_writer = None
    actions_writer = None
    n_hands_total = 0
    n_actions_total = 0

    try:
        for batch_start in range(0, total_files, BATCH_SIZE_FILES):
            batch_files = files[batch_start : batch_start + BATCH_SIZE_FILES]
            hand_rows = []
            action_rows = []
            for f in batch_files:
                for hand in _parse_file_to_hands(f):
                    hr, ars = _hand_to_rows(hand)
                    hand_rows.append(hr)
                    action_rows.extend(ars)

            if hand_rows:
                hands_df = pd.DataFrame(hand_rows)
                actions_df = pd.DataFrame(action_rows)
                if not actions_df.empty:
                    actions_df["action"] = actions_df["action"].astype(ACTION_CATEGORY_DTYPE)
                    actions_df["position"] = actions_df["position"].astype("category")
                    actions_df["street"] = actions_df["street"].astype("category")
                    actions_df["amount_bb"] = actions_df["amount"] / actions_df["big_blind"]
                    # categories must be consistent across batches for a single
                    # ParquetWriter schema -- cast back to plain strings on write,
                    # the categorical dtype was only needed for the astype above
                    actions_df["action"] = actions_df["action"].astype(str)
                    actions_df["position"] = actions_df["position"].astype(str)
                    actions_df["street"] = actions_df["street"].astype(str)

                hands_table = pa.Table.from_pandas(hands_df, preserve_index=False)
                actions_table = pa.Table.from_pandas(actions_df, preserve_index=False)

                if hands_writer is None:
                    hands_writer = pq.ParquetWriter(
                        str(config.PROCESSED_DATA_DIR / "hands.parquet"), hands_table.schema
                    )
                    actions_writer = pq.ParquetWriter(
                        str(config.PROCESSED_DATA_DIR / "actions.parquet"), actions_table.schema
                    )
                hands_writer.write_table(hands_table)
                actions_writer.write_table(actions_table)

                n_hands_total += len(hand_rows)
                n_actions_total += len(action_rows)

            files_done = min(batch_start + BATCH_SIZE_FILES, total_files)
            log(f"  {files_done}/{total_files} files, {n_hands_total} hands, {n_actions_total} actions so far")
    finally:
        if hands_writer is not None:
            hands_writer.close()
        if actions_writer is not None:
            actions_writer.close()

    log(f"ALL DONE: {n_hands_total} hands, {n_actions_total} actions written")


if __name__ == "__main__":
    main()
