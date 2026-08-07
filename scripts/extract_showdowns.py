"""Extracts REAL revealed hole cards from real showdowns in the PHH dataset,
PLUS a per-hand "is the winner actually known" determination.

This corrects a stale claim in an earlier session's notes ("0% of hole cards
revealed... verified by grep") -- that check grepped for a PokerStars-text
"shows" keyword against files that use the real PHH format's `sm` (show)
verb, and simply never matched. Re-checked directly: across all ps_nl25 files
there are real 'sm' showdown actions with genuine unmasked cards, sitting
parsed-and-then-discarded the whole time: `phh_parser.py` already fills
`Hand.showdown[player]` correctly, but `preprocess.hands_to_frames()` never
persists it into the cached dataset. As of the 2026-07-30 dataset expansion
(1000 -> 4379 PokerStars NL25 files, downloaded from the full Zenodo archive
at DOI 10.5281/zenodo.10796885, ~3.6M hands total), this covers a much
bigger sample.

Important caveat about the SHOWN HANDS themselves, disclosed rather than
ignored -- this is a SELECTED sample, not a random one, in two ways:
  1. Showdown-only: only hands that reach showdown appear at all. Most raises
     win uncontested or get folded to postflop aggression before showdown,
     so this cannot tell you "what hands does this population open," only
     "what hands does this population show down, conditional on getting
     there" -- a real hand-strength selection bias (weak bluffs that don't
     improve rarely reach showdown).
  2. Show-or-muck: at showdown a player can still muck instead of showing,
     and losing a chip-losing hand face-up is a bigger ego cost than showing
     a winner -- so even within "reached showdown," real cards are biased
     toward stronger/winning hands versus the true distribution of hands
     that were live at showdown.

A SEPARATE, related caveat about WINNER determination (flagged directly by
the user, not something to gloss over): "reached showdown" does NOT mean "we
know who won." A hand can go to showdown with 3 live players where only 2
show real cards and the 3rd mucks unrevealed ('????') -- if that 3rd hand
could plausibly have been the best hand, we cannot safely say who won just
from the two we can see. `outcome_known` below is True only when:
  (a) the hand ended with exactly one live (non-folded) player (an
      uncontested win -- no cards needed to know who won), OR
  (b) the hand reached a genuine showdown AND every single live player's
      cards are unmasked (so the winner can be computed exactly via
      evaluate_7cards, ties handled as a split pot).
Any hand not meeting (a) or (b) has `outcome_known = False` and `winners`
is left empty -- do not infer a winner for those from partial information.

Streams output in batches via pyarrow's ParquetWriter (same fix as
rebuild_processed_data.py) rather than accumulating all ~3.6M hands' outcome
rows in memory at once -- this machine has 8GB total RAM, and main.py's
naive "hold everything in one big list" approach already OOM-crashed once
tonight at this dataset size.
"""

import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.engine.cards import Card, evaluate_7cards
from src.parser.phh_parser import parse_phhs_file
from src.parser.positions import assign_positions

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "showdowns.parquet"
OUTCOMES_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "hand_outcomes.parquet"
RANKS = "AKQJT98765432"
BATCH_SIZE_FILES = 200


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _notation(cards: list[str]) -> str:
    r1, r2 = cards[0][0], cards[1][0]
    suited = cards[0][1] == cards[1][1]
    order = {r: i for i, r in enumerate(RANKS)}
    if r1 == r2:
        return r1 + r2
    hi, lo = (r1, r2) if order[r1] < order[r2] else (r2, r1)
    return f"{hi}{lo}{'s' if suited else 'o'}"


def _determine_outcome(hand) -> tuple[bool, list[str]]:
    """Returns (outcome_known, winners). See module docstring for exactly
    when outcome_known is True -- deliberately conservative."""
    all_players = {s.player for s in hand.seats}
    folded = {a.player for a in hand.actions if a.action == "folds"}
    live = all_players - folded

    if len(live) == 1:
        return True, list(live)

    if len(hand.board) != 5:
        return False, []  # didn't reach a real river showdown

    if not live.issubset(hand.showdown.keys()):
        return False, []  # at least one live player mucked without showing

    scores = {}
    for player in live:
        cards = hand.showdown[player]
        if len(cards) != 2:
            return False, []
        scores[player] = evaluate_7cards([Card(c) for c in cards] + [Card(c) for c in hand.board])
    best = max(scores.values())
    winners = [p for p, v in scores.items() if v == best]
    return True, winners


def _process_hand(hand, rows, outcome_rows):
    assign_positions(hand)
    outcome_known, winners = _determine_outcome(hand)
    outcome_rows.append(
        {
            "hand_id": hand.hand_id,
            "table_name": hand.table_name,
            "outcome_known": outcome_known,
            "winners": winners,
            "n_winners": len(winners),
        }
    )

    if not hand.showdown:
        return
    position_by_player = {s.player: s.position for s in hand.seats}

    preflop_actions = [a for a in hand.actions if a.street == "preflop"]
    open_i = next((j for j, a in enumerate(preflop_actions) if a.action in ("bets", "raises")), None)
    opener = preflop_actions[open_i].player if open_i is not None else None
    n_preflop_raises = sum(1 for a in preflop_actions if a.action == "raises") + (
        1 if open_i is not None and preflop_actions[open_i].action == "bets" else 0
    )

    for player, cards in hand.showdown.items():
        if len(cards) != 2:
            continue
        rows.append(
            {
                "hand_id": hand.hand_id,
                "table_name": hand.table_name,
                "player": player,
                "position": position_by_player.get(player, "UNKNOWN"),
                "hole_cards": "".join(cards),
                "notation": _notation(cards),
                "was_preflop_opener": player == opener,
                "n_preflop_raises": n_preflop_raises,
                "n_board_cards": len(hand.board),
                "outcome_known": outcome_known,
                "is_winner": player in winners,
            }
        )


def main():
    files = sorted(glob.glob("data/raw/ps_nl25/*.phhs"))
    total_files = len(files)
    log(f"parsing {total_files} files in batches of {BATCH_SIZE_FILES}...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    showdown_writer = None
    outcomes_writer = None
    n_hands_total = 0
    n_rows_total = 0
    n_known_total = 0

    try:
        for batch_start in range(0, total_files, BATCH_SIZE_FILES):
            batch_files = files[batch_start : batch_start + BATCH_SIZE_FILES]
            rows = []
            outcome_rows = []
            for f in batch_files:
                for hand in parse_phhs_file(f):
                    n_hands_total += 1
                    _process_hand(hand, rows, outcome_rows)

            if outcome_rows:
                outcomes_df = pd.DataFrame(outcome_rows)
                outcomes_table = pa.Table.from_pandas(outcomes_df, preserve_index=False)
                if outcomes_writer is None:
                    outcomes_writer = pq.ParquetWriter(str(OUTCOMES_PATH), outcomes_table.schema)
                outcomes_writer.write_table(outcomes_table)
                n_known_total += int(outcomes_df["outcome_known"].sum())

            if rows:
                df = pd.DataFrame(rows)
                showdown_table = pa.Table.from_pandas(df, preserve_index=False)
                if showdown_writer is None:
                    showdown_writer = pq.ParquetWriter(str(OUT_PATH), showdown_table.schema)
                showdown_writer.write_table(showdown_table)
                n_rows_total += len(rows)

            files_done = min(batch_start + BATCH_SIZE_FILES, total_files)
            log(f"  {files_done}/{total_files} files, {n_hands_total} hands, {n_rows_total} showdown reveals so far")
    finally:
        if showdown_writer is not None:
            showdown_writer.close()
        if outcomes_writer is not None:
            outcomes_writer.close()

    log(f"done: {n_hands_total} hands, {n_rows_total} real showdown hole-card reveals")
    log(
        f"saved {OUTCOMES_PATH}: {n_hands_total} hands, "
        f"{n_known_total} ({n_known_total / n_hands_total * 100:.1f}%) with a fully known winner"
    )
    log(f"saved {OUT_PATH} ({n_rows_total} rows)")


if __name__ == "__main__":
    main()
