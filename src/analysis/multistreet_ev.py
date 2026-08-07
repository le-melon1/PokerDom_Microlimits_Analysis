"""Multi-street preflop-to-river EV for one specific starting hand, opening
against one archetype's typical defending tendencies, on one sampled board.

Split into two steps because almost everything here does NOT depend on the
bettor's specific hand -- range narrowing, pot/invested bookkeeping, and every
fold/raise branch are identical for all 169 starting hands in a given
(matchup, board) pair. Only the final river showdown-equity check is
hand-specific. So:

  1. precompute_matchup(...)   -- expensive (Monte Carlo range narrowing),
                                   run ONCE per (archetype pair, board).
  2. estimate_hand_ev(...)     -- cheap (one equity call), run once per hand.

Doing the narrowing inside a per-hand loop (169x per matchup+board) would be
169x more Monte Carlo work than necessary -- this split is what makes an
overnight batch across many archetype pairs x hands x boards tractable.

Same family of disclosed simplifications as preflop_open_ev.py, plus ones
specific to going postflop:

- Each street's continuation (call) range is approximated by narrowing the
  defender's *entering* range to the top fraction (by forward equity vs a
  random hand on flop/turn, by exact made-hand strength on river -- see
  engine.range_equity for why that split matters) matching their empirical
  call+raise% for that street. Real ranges narrow by blockers and
  line-consistency too, not just equity -- directional, not a measurement.
- A raise on any street is scored as the bettor folding right there (a
  conservative floor, not a resolved subgame).
- Bet sizing per street is fixed by convention (preflop: a flat bb size;
  postflop: a pot-fraction) rather than solved for -- one sizing line's EV,
  not the best one.
- The bettor holds one concrete starting hand throughout; only the
  defender's range is narrowed by the betting line, not the bettor's own.
"""

from dataclasses import dataclass, field

from src.engine.range_equity import _expand_range, combos_vs_range_equity_on_board, narrow_range_by_board

STREET_BOARD_LEN = {"flop": 3, "turn": 4, "river": 5}


@dataclass
class MatchupContext:
    board: list[str]
    final_defender_range: list[tuple[str, str]]
    pot_at_river_call: float
    invested_at_river_call: float
    path_prob_to_river_call: float
    baseline_ev: float  # sum of every fold/raise branch across all streets (hand-independent)


@dataclass
class MultiStreetEVResult:
    ev_bb: float
    reached_probability: float
    branch_evs: dict = field(default_factory=dict)


def _bettor_combo(hand_notation: str) -> tuple[str, str]:
    return _expand_range([hand_notation])[0]


def precompute_matchup(
    defender_entering_range: list[str],
    preflop_fold_pct: float,
    preflop_call_pct: float,
    preflop_threebet_pct: float,
    postflop_facing_bet: dict,
    board: list[str],
    preflop_sizing_bb: float = 2.5,
    postflop_pot_fraction: float = 0.55,
    small_blind_bb: float = 0.5,
    forward_equity_trials: int = 150,
) -> MatchupContext:
    assert len(board) == 5, "pass a full 5-card board; streets slice into it"

    branch_evs: dict[str, float] = {
        "preflop_fold": preflop_fold_pct * (small_blind_bb + 1.0),
        "preflop_threebet": preflop_threebet_pct * (-preflop_sizing_bb),
    }

    if preflop_call_pct <= 0:
        return MatchupContext(board, [], 0.0, 0.0, 0.0, sum(branch_evs.values()))

    pot = small_blind_bb + 2 * preflop_sizing_bb
    invested = preflop_sizing_bb
    defender_range = defender_entering_range
    path_prob = preflop_call_pct

    for street in ("flop", "turn", "river"):
        board_so_far = board[: STREET_BOARD_LEN[street]]
        stats = postflop_facing_bet.get(street, {"fold_pct": 0.5, "call_pct": 0.5, "raise_pct": 0.0})
        fold_pct, call_pct, raise_pct = stats["fold_pct"], stats["call_pct"], stats.get("raise_pct", 0.0)

        bet_size = postflop_pot_fraction * pot
        branch_evs[f"{street}_fold"] = path_prob * fold_pct * pot
        branch_evs[f"{street}_raise"] = path_prob * raise_pct * (-invested - bet_size)

        if call_pct <= 0:
            path_prob = 0.0
            break

        continue_frac = min(1.0, call_pct + raise_pct)
        defender_range = narrow_range_by_board(
            defender_range, board_so_far, keep_fraction=continue_frac, forward_equity_trials=forward_equity_trials
        )

        pot = pot + 2 * bet_size
        invested = invested + bet_size
        path_prob = path_prob * call_pct

    return MatchupContext(
        board=board,
        final_defender_range=defender_range,
        pot_at_river_call=pot,
        invested_at_river_call=invested,
        path_prob_to_river_call=path_prob,
        baseline_ev=sum(branch_evs.values()),
    )


def estimate_hand_ev(
    bettor_hand: str, matchup: MatchupContext, equity_trials: int = 1200
) -> MultiStreetEVResult:
    if matchup.path_prob_to_river_call <= 0 or not matchup.final_defender_range:
        return MultiStreetEVResult(ev_bb=matchup.baseline_ev, reached_probability=0.0)

    bettor_combo = _bettor_combo(bettor_hand)
    equity, _ = combos_vs_range_equity_on_board(
        [bettor_combo], matchup.final_defender_range, matchup.board, trials=equity_trials
    )
    ev_call = equity * matchup.pot_at_river_call - matchup.invested_at_river_call
    river_call_ev = matchup.path_prob_to_river_call * ev_call

    return MultiStreetEVResult(
        ev_bb=matchup.baseline_ev + river_call_ev,
        reached_probability=matchup.path_prob_to_river_call,
        branch_evs={"baseline": matchup.baseline_ev, "river_call": river_call_ev, "equity": equity},
    )
