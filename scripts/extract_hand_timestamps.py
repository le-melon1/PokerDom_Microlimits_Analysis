"""Lightweight extraction of (hand_id, table, timestamp) from the raw .phhs
files -- NOT the full Hand-object parse, since we only need these three
fields per hand to detect continuous sessions (same table, no big time gap)
for the within-session pairwise-adaptation check. Full parser + Hand objects
would be far more memory/CPU than this needs.

Batched + streaming write (see pokerdom_project_status memory: this machine
OOMs if a raw-file pass accumulates everything in one Python list before
writing) -- write each batch straight to parquet via ParquetWriter.
"""

import glob
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BATCH_SIZE = 200
OUT_PATH = "data/processed/hand_timestamps.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_from_file(path: str) -> list[dict]:
    rows = []
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    for fields in data.values():
        if fields.get("variant") != "NT":
            continue
        try:
            ts = datetime(
                fields["year"], fields["month"], fields["day"],
                fields["time"].hour, fields["time"].minute, fields["time"].second,
            )
        except (KeyError, ValueError, TypeError):
            continue
        rows.append({"hand_id": str(fields.get("hand", "")), "table": fields.get("table"), "timestamp": ts})
    return rows


def main():
    files = sorted(glob.glob("data/raw/ps_nl25/*.phhs"))
    log(f"{len(files)} files to process")

    writer = None
    total = 0
    batch_rows: list[dict] = []

    for i, f in enumerate(files):
        batch_rows.extend(extract_from_file(f))

        if (i + 1) % BATCH_SIZE == 0 or i == len(files) - 1:
            df = pd.DataFrame(batch_rows)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT_PATH, table.schema)
            writer.write_table(table)
            total += len(batch_rows)
            log(f"  processed {i + 1}/{len(files)} files, {total} rows written so far")
            batch_rows = []

    if writer is not None:
        writer.close()
    log(f"DONE: {total} rows written to {OUT_PATH}")


if __name__ == "__main__":
    main()
