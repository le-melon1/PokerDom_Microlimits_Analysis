"""Overnight batch: for every ordered (bettor_archetype, defender_archetype)
pair, BTN opens vs BB defends, compute a per-169-hand EV table averaged over
several sampled boards.

Writes incrementally to data/reference/matchup_hand_ev.csv (append mode) so
partial progress survives an interruption -- check that file's contents to
see what's done; resume by editing DONE_PAIRS below if needed.
"""

import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analysis.hand_rankings import compute_hand_rankings
from src.analysis.implied_range import implied_range
from src.analysis.multistreet_ev import estimate_hand_ev, precompute_matchup
from src.engine.cards import RANKS, SUITS

OUT_PATH = Path("data/reference/matchup_hand_ev.csv")
ARCHETYPES = ["Nit", "TAG", "LAG", "Loose-passive", "Station", "Maniac"]
N_BOARDS = 8
RANDOM_SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sample_boards(n, seed):
    rng = random.Random(seed)
    full_deck = [r + s for r in RANKS for s in SUITS]
    return [rng.sample(full_deck, 5) for _ in range(n)]


def already_done_pairs():
    if not OUT_PATH.exists():
        return set()
    df = pd.read_csv(OUT_PATH)
    return set(zip(df["bettor_archetype"], df["defender_archetype"]))


def main():
    vpip_table = pd.read_csv("data/reference/archetype_position_vpip.csv")
    vs_raise_table = pd.read_csv("data/reference/archetype_vs_raise.csv")
    facing_bet_table = pd.read_csv("data/reference/archetype_facing_bet.csv")
    rankings = compute_hand_rankings()
    all_hands = rankings["hand"].tolist()

    boards = sample_boards(N_BOARDS, RANDOM_SEED)
    done = already_done_pairs()
    log(f"{len(done)} pairs already done, boards={N_BOARDS}")

    write_header = not OUT_PATH.exists()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(OUT_PATH, "a", newline="")
    writer = csv.writer(fh)
    if write_header:
        writer.writerow(
            ["bettor_archetype", "defender_archetype", "hand", "avg_ev_bb", "n_boards_used"]
        )

    total_pairs = len(ARCHETYPES) * len(ARCHETYPES)
    done_count = len(done)

    for bettor_arch in ARCHETYPES:
        vpip_row = vpip_table[(vpip_table.archetype == bettor_arch) & (vpip_table.position == "BTN")]
        if vpip_row.empty:
            log(f"skip {bettor_arch}: no BTN VPIP data")
            continue
        bettor_vpip = vpip_row.iloc[0]["vpip"]

        for defender_arch in ARCHETYPES:
            if (bettor_arch, defender_arch) in done:
                continue

            vs_raise_row = vs_raise_table[
                (vs_raise_table.archetype == defender_arch) & (vs_raise_table.position == "BB")
            ]
            if vs_raise_row.empty:
                log(f"skip {bettor_arch} vs {defender_arch}: no BB vs-raise data")
                continue
            vr = vs_raise_row.iloc[0]

            postflop_stats = {}
            ok = True
            for street in ("flop", "turn", "river"):
                row = facing_bet_table[
                    (facing_bet_table.archetype == defender_arch)
                    & (facing_bet_table.street == street)
                    & (facing_bet_table.pot_bucket == "medium")
                ]
                if row.empty:
                    ok = False
                    break
                postflop_stats[street] = {
                    "fold_pct": row.iloc[0]["fold_pct"],
                    "call_pct": row.iloc[0]["call_pct"],
                    "raise_pct": row.iloc[0]["raise_pct"],
                }
            if not ok:
                log(f"skip {bettor_arch} vs {defender_arch}: missing postflop data")
                continue

            t0 = time.time()
            defender_range = implied_range(vr["call_pct"] + vr["threebet_pct"], rankings)

            hand_evs = {h: [] for h in all_hands}
            for board in boards:
                matchup = precompute_matchup(
                    defender_range,
                    preflop_fold_pct=vr["fold_pct"],
                    preflop_call_pct=vr["call_pct"],
                    preflop_threebet_pct=vr["threebet_pct"],
                    postflop_facing_bet=postflop_stats,
                    board=board,
                    forward_equity_trials=150,
                )
                for hand in all_hands:
                    result = estimate_hand_ev(hand, matchup, equity_trials=1000)
                    hand_evs[hand].append(result.ev_bb)

            for hand in all_hands:
                vals = hand_evs[hand]
                writer.writerow([bettor_arch, defender_arch, hand, sum(vals) / len(vals), len(vals)])
            fh.flush()

            done_count += 1
            log(
                f"[{done_count}/{total_pairs}] {bettor_arch} opens BTN vs {defender_arch} BB "
                f"done in {time.time() - t0:.1f}s"
            )

    fh.close()
    log("ALL PAIRS DONE")


if __name__ == "__main__":
    main()
