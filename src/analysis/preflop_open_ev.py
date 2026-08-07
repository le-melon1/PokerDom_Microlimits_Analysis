"""Approximate preflop-open EV: raiser's implied range vs a specific
opponent's implied *defending* range, weighted by that opponent's own
fold/call/3-bet frequency facing a raise.

Deliberately scoped to preflop only (v1) -- see project memory
`pokerdom_ev_model_followup` for what a full multi-street version would add.
Simplifications, disclosed rather than hidden:

- Models a clean heads-up confrontation (raiser vs one caller in the blinds);
  other players already folded. Dead money from a folded small blind is
  counted; if the raiser IS the small blind, pass small_blind=0.
- The 3-bet branch is NOT resolved (no further range narrowing or postflop
  play) -- it's scored as the raiser folding to the 3-bet, i.e. losing just
  their raise-sized investment. That's a conservative floor, not a true EV:
  real raisers sometimes continue and change the number.
- A called pot is resolved by raw preflop equity only, as if going straight
  to showdown -- no postflop betting, no equity realization/fold equity on
  later streets. That's the single biggest simplification versus a real EV.
- The defending range is approximated the same way as an opening range
  (top-X% by raw equity, X = call% + 3bet%) -- real defending ranges skew
  toward suited connectors/pairs for implied odds rather than raw equity
  percentile, so this is a rougher stand-in specifically for the defender.
"""

from dataclasses import dataclass

from src.analysis.hand_rankings import compute_hand_rankings
from src.analysis.implied_range import implied_range
from src.engine.range_equity import range_vs_range_equity


@dataclass
class OpenEVResult:
    raiser_range_size: int
    defender_range_size: int
    equity_raiser_if_called: float
    ev_bb: float
    ev_breakdown: dict


def estimate_open_ev(
    raiser_vpip_at_position: float,
    sizing_bb: float,
    opponent_fold_pct: float,
    opponent_call_pct: float,
    opponent_threebet_pct: float,
    small_blind_bb: float = 0.5,
    equity_trials: int = 3000,
) -> OpenEVResult:
    rankings = compute_hand_rankings()
    raiser_range = implied_range(raiser_vpip_at_position, rankings)

    defend_frac = opponent_call_pct + opponent_threebet_pct
    defender_range = implied_range(defend_frac, rankings) if defend_frac > 0 else []

    if defender_range:
        equity_raiser, _ = range_vs_range_equity(raiser_range, defender_range, trials=equity_trials)
    else:
        equity_raiser = 1.0  # no data on their continuing range; treat as always folding in this branch

    pot_if_called = small_blind_bb + 2 * sizing_bb
    ev_fold = small_blind_bb + 1.0  # wins the dead small blind + the big blind (defender's posted bb)
    ev_called = equity_raiser * pot_if_called - sizing_bb
    ev_threebet = -sizing_bb  # conservative: modeled as raiser folding to the 3-bet (see docstring)

    ev_total = (
        opponent_fold_pct * ev_fold
        + opponent_call_pct * ev_called
        + opponent_threebet_pct * ev_threebet
    )

    return OpenEVResult(
        raiser_range_size=len(raiser_range),
        defender_range_size=len(defender_range),
        equity_raiser_if_called=equity_raiser,
        ev_bb=ev_total,
        ev_breakdown={
            "fold_branch": opponent_fold_pct * ev_fold,
            "call_branch": opponent_call_pct * ev_called,
            "threebet_branch": opponent_threebet_pct * ev_threebet,
        },
    )
