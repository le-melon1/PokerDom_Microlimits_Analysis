"""Test whether a bot should prefer raising to passive preflop entry.

The broad "increase PFR and 3-bet" finding is already strong.  This is a
narrower candidate rule: once two players are equally active before the flop,
does the one who enters pots with raises instead of calls also win more?

Uses the 614 reliable players in ``player_stats_vs_winrate.csv``.  The key
measure is ``passive_gap = VPIP - PFR``: the share of dealt hands voluntarily
played without raising.  It reports both a simple association and a partial
test after controlling for PFR, 3-bet%, postflop AF, and opening size.

RESULT (2026-08-14): the simple result is real but modest (Spearman
rho=-0.154, p=0.00012); holding the other four metrics fixed it is not
independently reliable (partial r=-0.052, p=0.196).  Do not add a standalone
"punish calls" bot rule.  Prefer a raise over a call only as a tie-breaker
when the existing range and opponent-specific rules consider both actions
reasonable.

Like the preceding winner-characterisation work, this is observational, not
a causal A/B test of a bot.

Usage: python3 scripts/check_raise_over_call_strategy.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


INPUT_PATH = Path("data/reference/player_stats_vs_winrate.csv")
OUTPUT_PATH = Path("data/reference/raise_vs_call_strategy_check.csv")
METRICS = ["vpip", "pfr", "threebet_pct", "postflop_af", "avg_open_bb", "bb_per_100"]
CONTROLS = ["pfr", "threebet_pct", "postflop_af", "avg_open_bb"]


def residuals(y: pd.Series, controls: pd.DataFrame) -> np.ndarray:
    """Residualise y on an intercept plus the supplied control variables."""
    design = np.column_stack([np.ones(len(controls)), controls.to_numpy(dtype=float)])
    coefficients, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
    return y.to_numpy(dtype=float) - design @ coefficients


def main() -> None:
    players = pd.read_csv(INPUT_PATH).replace([np.inf, -np.inf], np.nan)
    players = players.dropna(subset=METRICS)
    players = players[players["vpip"] > 0].copy()
    players["passive_gap"] = players["vpip"] - players["pfr"]
    players["raise_share"] = players["pfr"] / players["vpip"]

    gap_rho, gap_pvalue = stats.spearmanr(players["passive_gap"], players["bb_per_100"])
    share_rho, share_pvalue = stats.spearmanr(players["raise_share"], players["bb_per_100"])
    partial_r, partial_pvalue = stats.pearsonr(
        residuals(players["passive_gap"], players[CONTROLS]),
        residuals(players["bb_per_100"], players[CONTROLS]),
    )

    # Compare passive-gap quartiles *within* PFR quintiles.  This prevents the
    # descriptive table from merely rediscovering that a higher PFR wins more.
    players["pfr_quintile"] = pd.qcut(players["pfr"], 5, duplicates="drop")
    players["gap_quartile"] = players.groupby("pfr_quintile", observed=True)["passive_gap"].transform(
        lambda values: pd.qcut(values, 4, labels=False, duplicates="drop")
    )
    summary = (
        players.groupby("gap_quartile", observed=True)
        .agg(
            passive_gap=("passive_gap", "mean"),
            raise_share=("raise_share", "mean"),
            pfr=("pfr", "mean"),
            threebet_pct=("threebet_pct", "mean"),
            bb_per_100=("bb_per_100", "mean"),
            n_players=("player", "count"),
        )
        .reset_index()
    )
    summary["strategy_interpretation"] = np.where(
        summary["gap_quartile"] == 0,
        "most_raise_oriented",
        np.where(summary["gap_quartile"] == 3, "most_passive", "middle"),
    )
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"{len(players)} reliable players")
    print(f"passive gap vs bb/100: rho={gap_rho:+.4f}, p={gap_pvalue:.6g}")
    print(f"raise share vs bb/100: rho={share_rho:+.4f}, p={share_pvalue:.6g}")
    print(f"partial passive-gap result (controlling {', '.join(CONTROLS)}): r={partial_r:+.4f}, p={partial_pvalue:.6g}")
    print("\nWithin-PFR-quintile comparison:")
    print(summary.to_string(index=False))
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
