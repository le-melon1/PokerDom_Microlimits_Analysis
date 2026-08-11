"""Range-reading theory: does betting MULTIPLE consecutive streets (barrels)
correlate with a genuinely stronger real hand at showdown, compared to a
single bet? This is the "what did they do on previous streets, and does that
narrow what they could be holding" axis -- distinct from check_bet_size_
reveals_strength.py (single-bet SIZE) and find_positional_openers.py
(per-player positional leaks). Neither the live EV panel nor abc_bot.py
currently track a player's action SEQUENCE across streets within one hand at
all -- every decision re-derives the opponent's range from scratch (position
+ archetype/dossier + board texture + a single generic continue-frequency
stat for the CURRENT bet), with no memory of "they also bet the last two
streets." If barrel count real-and-meaningfully predicts hand strength, that
justifies building it; if not, that assumption isn't worth the engineering
cost.

Method: for every hand, find the aggressor (last bettor/raiser) on each of
flop/turn/river via actions.parquet (vectorized, no row-by-row hand replay).
For every REAL SHOWDOWN river bettor (data/processed/showdowns.parquet),
count consecutive barrels ending at the river (river-only=1, turn+river=2,
flop+turn+river=3 -- broken streaks, e.g. bet flop/checked turn/bet river,
count from the most recent unbroken run backward). Compare real 7-card hand
category (evaluate_7cards, 0=high card .. 9=straight flush) across the three
groups.

RESULT (2026-08-11, 269,760 real-showdown river bettors -- 147,324 one-
barrel / 60,508 two-barrel / 61,928 three-barrel): the theory did NOT
hold, and not just as noise -- it went the WRONG direction. Spearman
rho=-0.0639 (p~0, huge n) between barrel count and real hand category.
Mean category: 1 barrel=3.71, 2 barrels=3.63, 3 barrels=3.38 (0=high
card..9=straight flush) -- three-barrel bettors show up SLIGHTLY WEAKER
at showdown, not stronger. Fraction >= two pair stays flat-to-slightly-up
across all three groups (93.7% / 94.9% / 94.7%), so this isn't about
weak hands reaching showdown more with more barrels either -- almost
everyone who bets three streets and shows up at showdown has at least
two pair regardless of barrel count; the DIFFERENCE is in how much
stronger, and that difference doesn't grow with barrels.

Likely explanation, not confirmed further: the same show-or-mock
selection bias flagged throughout this project (see extract_showdowns.py)
probably cuts harder here than for a single bet. A player who barrels
three streets as a bluff and gets caught has a real incentive to muck
quietly rather than show an embarrassing multi-street bluff -- so
three-barrel BLUFFS that got caught are plausibly underrepresented in
the shown sample specifically, while three-barrel VALUE hands (typically
proud to show) are overrepresented -- but that selection effect would
bias the result the OTHER way (stronger-looking), not explain this
negative finding. Left as an honest, reported null/negative result
rather than a resolved mechanism -- worth a real look if revisited, not
worth guessing at further here.

CONCLUSION: do not build a "more barrels -> narrow the range upward"
rule into abc_bot.py or live_ev.py on this data. The one range-reading
axis this project has actually tested (bet SIZE, check_bet_size_
reveals_strength.py) found a real, if modest, positive signal; this one
(bet SEQUENCE across streets) found no signal in the expected direction.

Usage: python3 scripts/check_multi_barrel_range.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy import stats

from src.engine.cards import Card, evaluate_7cards


def main():
    t0 = time.monotonic()
    actions = pd.read_parquet(
        "data/processed/actions.parquet", columns=["hand_id", "player", "street", "action"]
    )
    print(f"loaded actions in {time.monotonic()-t0:.1f}s")

    aggr = actions[actions["action"].isin(["bets", "raises"])]
    # last aggressive action per (hand_id, street) -- row order is
    # chronological within a hand (same assumption every sequential-action
    # script in this project relies on), so .groupby(...).tail(1) via idxmax
    # on the implicit row position gives the FINAL bettor/raiser of the street.
    last_aggr_idx = aggr.groupby(["hand_id", "street"], observed=True, sort=False).tail(1)
    pivot = last_aggr_idx.pivot(index="hand_id", columns="street", values="player")
    for col in ("flop", "turn", "river"):
        if col not in pivot.columns:
            pivot[col] = None

    river_bettors = pivot[pivot["river"].notna()][["flop", "turn", "river"]].copy()

    def barrel_count(row) -> int:
        if row["turn"] != row["river"]:
            return 1
        if row["flop"] != row["turn"]:
            return 2
        return 3

    river_bettors["barrels"] = river_bettors.apply(barrel_count, axis=1)
    river_bettors["player"] = river_bettors["river"]
    print(f"{len(river_bettors)} hands with a real river bet")
    print(river_bettors["barrels"].value_counts().sort_index())

    print("\nloading showdowns + hands (board)...")
    showdowns = pd.read_parquet("data/processed/showdowns.parquet", columns=["hand_id", "player", "hole_cards"])
    hands = pd.read_parquet("data/processed/hands.parquet", columns=["hand_id", "board"]).set_index("hand_id")

    merged = river_bettors.reset_index().merge(showdowns, on=["hand_id", "player"], how="inner")
    merged = merged.join(hands["board"], on="hand_id")
    merged = merged[merged["board"].str.len() > 0]
    print(f"{len(merged)} river bettors with real showdown cards + a real board")

    def _cards(hole_cards: str) -> list[str] | None:
        if not isinstance(hole_cards, str) or len(hole_cards) != 4:
            return None
        return [hole_cards[0:2], hole_cards[2:4]]

    def _hand_category(row) -> int | None:
        hole = _cards(row["hole_cards"])
        if hole is None:
            return None
        board = row["board"].split()
        if len(board) != 5:
            return None
        try:
            score = evaluate_7cards([Card(c) for c in hole + board])
        except (KeyError, IndexError):
            return None
        return score[0]

    print("scoring real 7-card hands (this is the slow part)...")
    t0 = time.monotonic()
    merged["hand_category"] = merged.apply(_hand_category, axis=1)
    print(f"  done in {time.monotonic()-t0:.1f}s")
    merged = merged.dropna(subset=["hand_category"])

    print(f"\n{len(merged)} rows with a resolved hand category\n")
    corr, pvalue = stats.spearmanr(merged["barrels"], merged["hand_category"])
    print(f"Spearman correlation (barrel count vs real hand category): rho={corr:.4f}, p={pvalue:.6f}")

    summary = merged.groupby("barrels")["hand_category"].agg(["mean", "median", "count"])
    print("\nmean real hand category by barrel count (0=high card .. 9=straight flush):")
    print(summary)

    # also report: fraction that's at least two pair (category >= 2) by barrel count
    merged["two_pair_plus"] = merged["hand_category"] >= 2
    print("\nfraction >= two pair, by barrel count:")
    print(merged.groupby("barrels")["two_pair_plus"].mean())


if __name__ == "__main__":
    main()
