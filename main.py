"""Pipeline: raw hand history logs -> stats -> fold-equity model -> Excel report.

Usage:
    python main.py --raw-dir data/raw --out reports/analysis.xlsx

2026-07-30 rewrite: the original version parsed every raw file into one big
`hands` list (every Hand object, with nested Seat/Action objects, held in
memory at once) before doing anything else. That's exactly the pattern that
OOM-crashed silently on this 8GB-RAM machine once the dataset grew to ~3.6M
hands (4379 PokerStars files) -- the process vanished mid-run with no
traceback, only visible via `ps aux` and unwritten output files. Fixed with
the same batched-fragment pattern already used in scripts/rebuild_processed_data.py,
scripts/extract_showdowns.py and scripts/build_archetype_tables.py: parse
files in bounded batches, immediately reduce each batch's raw Hand objects to
small DataFrame fragments (hands_df/actions_df/sizing_df rows), discard the
raw hands, and only concat the (much smaller) fragments once at the end.
"""

import argparse
import glob
import time
from pathlib import Path

import pandas as pd

from src import config
from src.modeling.fold_equity_model import find_profitable_sizings, train_fold_equity_model
from src.parser.hand_history_parser import parse_file
from src.parser.ipoker_xml_parser import parse_ipoker_xml_file
from src.parser.phh_parser import parse_phhs_file
from src.pipeline.archetypes import label_archetypes
from src.pipeline.decision_points import build_fold_equity_dataset
from src.pipeline.preprocess import hands_to_frames, player_stats
from src.reporting.excel_report import generate_report

BATCH_SIZE_FILES = 200


def _is_ipoker_xml(txt_file: Path) -> bool:
    with open(txt_file, "rb") as fh:
        head = fh.read(200).decode("utf-8-sig", errors="ignore").lstrip()
    return head.startswith("<?xml") or head.startswith("<session")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _list_raw_files(raw_dir: Path) -> list[Path]:
    txt_files = sorted(raw_dir.rglob("*.txt"))
    phhs_files = sorted(raw_dir.rglob("*.phhs"))
    return txt_files + phhs_files


def _parse_one(raw_file: Path) -> list:
    if raw_file.suffix == ".phhs":
        return parse_phhs_file(str(raw_file))
    if _is_ipoker_xml(raw_file):
        return parse_ipoker_xml_file(str(raw_file))
    return parse_file(str(raw_file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(config.RAW_DATA_DIR))
    parser.add_argument("--out", default=str(config.REPORTS_DIR / "analysis.xlsx"))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_files = _list_raw_files(raw_dir)
    if not raw_files:
        print(f"No .txt/.phhs hand histories found in {raw_dir}. Nothing to analyze.")
        return

    _log(f"found {len(raw_files)} raw files under {raw_dir}, processing in batches of {BATCH_SIZE_FILES}...")

    hands_fragments = []
    actions_fragments = []
    sizing_fragments = []
    n_hands = 0
    for batch_start in range(0, len(raw_files), BATCH_SIZE_FILES):
        batch_files = raw_files[batch_start : batch_start + BATCH_SIZE_FILES]
        batch_hands = []
        for f in batch_files:
            batch_hands.extend(_parse_one(f))
        n_hands += len(batch_hands)

        hands_frag, actions_frag = hands_to_frames(batch_hands)
        hands_fragments.append(hands_frag)
        actions_fragments.append(actions_frag)

        sizing_frag = build_fold_equity_dataset(batch_hands)
        if not sizing_frag.empty:
            sizing_fragments.append(sizing_frag)

        files_done = min(batch_start + BATCH_SIZE_FILES, len(raw_files))
        _log(f"  {files_done}/{len(raw_files)} files, {n_hands} hands so far")

    print(f"Parsed {n_hands} hands from {raw_dir}")

    hands_df = pd.concat(hands_fragments, ignore_index=True)
    actions_df = pd.concat(actions_fragments, ignore_index=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    hands_df.to_parquet(config.PROCESSED_DATA_DIR / "hands.parquet")
    actions_df.to_parquet(config.PROCESSED_DATA_DIR / "actions.parquet")

    stats_df = label_archetypes(player_stats(actions_df))

    sizing_df = pd.concat(sizing_fragments, ignore_index=True) if sizing_fragments else pd.DataFrame()
    sizing_edge_df = pd.DataFrame()
    if len(sizing_df) >= 50 and sizing_df["villain_folded"].nunique() == 2:
        model = train_fold_equity_model(sizing_df)
        sizing_edge_df = find_profitable_sizings(model, sizing_df)
    else:
        print("Not enough bet/raise decision points yet to train the fold-equity model.")

    generate_report(args.out, player_stats_df=stats_df, sizing_edge_df=sizing_edge_df)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
