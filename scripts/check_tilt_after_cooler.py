"""Preserved follow-up research request (see PokerDom_Practice_App's
CLAUDE.md / abc_bot.py history): does a player's behavior change in the
hands immediately AFTER losing a big pot they were heavily invested in (a
"cooler"/bad-beat proxy), compared to their own normal play? If real,
derive an exploitable ABC-bot rule (e.g. call wider / expect more bluffs
from a recently-coolered opponent) rather than assuming generic "tilt."

Cooler proxy (real, disclosed limitation, same class as find_frequent_
bluffers.py's "river aggressor who lost" bluff proxy): a player reached a
REAL outcome-known showdown, invested >= COOLER_MIN_BB into that hand, and
lost. This conflates a genuine bad beat with simply losing a close/marginal
hand -- there's no hole-card equity-at-each-street reconstruction here,
just "put in real money, lost anyway."

Session/ordering: same real-timestamp-gap session detection as
check_within_session_adaptation.py (SESSION_GAP_MINUTES). For each cooler
event, look at that SAME player's next POST_COOLER_WINDOW hands within the
same continuous session -- compare pooled VPIP rate, postflop aggression
rate, and average postflop bet size (bb) against a baseline pool: all of
that same set of players' hands that are NOT within a post-cooler window
(their own "normal" play, not the general population -- avoids conflating
"who tilts" with "who's just a different kind of player").

RESULTS (2026-08-18, 137,157 cooler events across 44,699 players, 768,494
post-cooler hands vs 18,442,845 baseline hands, all p<1e-18):
  VPIP              +11.75pp  (36.68% vs 24.93%)
  postflop aggro    +5.76pp   (16.57% vs 10.81%)
  avg postflop bet  +0.77bb   (8.58bb vs 7.80bb)
  thin-call rate    +0.92pp   (18.70% vs 17.79%)
  starting stack    -15.06bb  (76.87bb vs 91.93bb -- mechanical, expected)

Confound check: restricting BOTH groups to a matched 60-100bb stack band
still shows VPIP +9.18pp and aggro +4.77pp (both p~0) -- the effect is NOT
primarily a stack-depth artifact, it survives at ~80% of its unmatched
magnitude even fully stack-matched.

Decay check: VPIP is highest in hands 1-2 after a cooler (40.2%), fades
across the window (hands 3-5: 36.4%, hands 6-10: 34.9%), but stays well
above the 24.9% baseline throughout the whole 10-hand window -- the
classic "acute reaction that peaks immediately and gradually decays"
signature, not a flat step-function (which would look more like a
structural/stack artifact than psychological tilt).

Conclusion: this is REAL, not a stack-depth or sample-selection artifact.
Real next step to use it in abc_bot.py: choose_abc_action currently gets
only the current Hand + static per-seat archetype labels, no session
history at all -- would need (a) a live "this opponent recently lost a
big pot" signal wired in (TableDossier already tracks session-scoped
per-seat stats in the sibling Practice_App repo, a natural place to add
this), and (b) probe_chance_enumeration.py would need to simulate
sequences of hands per opponent, not one fresh hand at a time, to A/B
test any resulting rule properly. Bigger infrastructure lift, not done
here -- this script only establishes that the underlying real-world
signal is genuine and worth that investment.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

SESSION_GAP_MINUTES = 45
COOLER_MIN_BB = 15.0
POST_COOLER_WINDOW = 10
CACHE_PATH = "data/processed/_tilt_after_cooler_feat_cache.parquet"
STACK_MATCH_MIN = 60.0
STACK_MATCH_MAX = 100.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_feat_rel():
    log("loading actions.parquet...")
    actions = pd.read_parquet(
        "data/processed/actions.parquet",
        columns=["hand_id", "player", "street", "action", "amount_bb", "stack", "big_blind"],
    )
    log(f"actions: {len(actions):,} rows")

    log("loading showdowns.parquet...")
    sd = pd.read_parquet("data/processed/showdowns.parquet", columns=["hand_id", "player", "outcome_known", "is_winner"])
    sd = sd[sd["outcome_known"]]

    log("loading hand_timestamps.parquet...")
    ts = pd.read_parquet("data/processed/hand_timestamps.parquet")  # hand_id, table, timestamp

    log("computing per-(hand,player) total contribution...")
    contrib = actions.groupby(["hand_id", "player"], sort=False)["amount_bb"].sum().rename("total_bb").reset_index()

    log("identifying cooler events...")
    cooler = sd.merge(contrib, on=["hand_id", "player"], how="inner")
    cooler = cooler[(cooler["total_bb"] >= COOLER_MIN_BB) & (~cooler["is_winner"])]
    cooler = cooler.merge(ts, on="hand_id", how="inner")
    cooler = cooler.rename(columns={"table": "table_name", "timestamp": "cooler_ts"})
    cooler = cooler[["player", "table_name", "cooler_ts"]].drop_duplicates()
    log(f"cooler events: {len(cooler):,} (players involved: {cooler['player'].nunique():,})")

    log("building per-(hand,player) behavior features...")
    all_pairs = actions[["hand_id", "player"]].drop_duplicates()

    preflop = actions[actions["street"] == "preflop"]
    vpip_pairs = preflop[preflop["action"].isin(["calls", "bets", "raises"])][["hand_id", "player"]].drop_duplicates()
    vpip_pairs["vpip"] = True

    postflop = actions[actions["street"] != "preflop"]
    postflop_aggr = postflop[postflop["action"].isin(["bets", "raises"])]
    # A tiny fraction of rows (~7.9k / 5.65M bet/raise actions, 0.14%) have
    # amount_bb == inf -- a pre-existing data-quality artifact in the
    # processed dataset (likely a big_blind=0 edge case upstream), not
    # something introduced here. Drop them before averaging so one bad row
    # doesn't poison an entire (hand,player) or pooled mean.
    n_bad = (~np.isfinite(postflop_aggr["amount_bb"])).sum()
    if n_bad:
        log(f"dropping {n_bad} non-finite amount_bb rows ({n_bad / len(postflop_aggr):.4%}) before averaging")
        postflop_aggr = postflop_aggr[np.isfinite(postflop_aggr["amount_bb"])]
    aggr_agg = postflop_aggr.groupby(["hand_id", "player"], sort=False).agg(
        n_postflop_aggr=("action", "size"), avg_postflop_bet_bb=("amount_bb", "mean")
    ).reset_index()
    postflop_calls = postflop[postflop["action"] == "calls"].groupby(["hand_id", "player"], sort=False).size().rename(
        "n_postflop_calls"
    ).reset_index()

    # Stack depth confound check: after losing >=COOLER_MIN_BB, a player's
    # stack is mechanically shorter next hand -- shorter effective stacks
    # can independently widen ranges (push/fold, SPR-driven commitment)
    # regardless of any real tilt psychology. First preflop action's stack
    # (before that player's own action, but after blinds) as the "stack
    # depth this hand" proxy, normalized to bb.
    # groupby().first() on an un-resorted frame keeps each group's first
    # ROW in original (within-hand chronological) order -- do not
    # sort_values first, that would destroy the action sequence.
    preflop_first = preflop.groupby(["hand_id", "player"], sort=False).first()
    # Same class of data-quality artifact as amount_bb above -- a handful
    # of rows have big_blind == 0, producing inf on division. Drop those
    # before computing the ratio.
    preflop_first = preflop_first[preflop_first["big_blind"] > 0]
    stack_bb = (preflop_first["stack"] / preflop_first["big_blind"]).rename("stack_bb").reset_index()

    feat = all_pairs.merge(vpip_pairs, on=["hand_id", "player"], how="left")
    feat = feat.merge(aggr_agg, on=["hand_id", "player"], how="left")
    feat = feat.merge(postflop_calls, on=["hand_id", "player"], how="left")
    feat = feat.merge(stack_bb, on=["hand_id", "player"], how="left")
    feat["vpip"] = feat["vpip"].fillna(False)
    feat["n_postflop_aggr"] = feat["n_postflop_aggr"].fillna(0)
    feat["n_postflop_calls"] = feat["n_postflop_calls"].fillna(0)
    feat["postflop_aggressive"] = feat["n_postflop_aggr"] > 0
    feat["saw_postflop"] = (feat["n_postflop_aggr"] + feat["n_postflop_calls"]) > 0
    feat = feat.merge(ts.rename(columns={"table": "table_name", "timestamp": "hand_ts"}), on="hand_id", how="inner")
    log(f"feature rows: {len(feat):,}")

    log("assigning per-player sessions (same table, gap-based)...")
    feat = feat.sort_values(["player", "table_name", "hand_ts"])
    gap = feat.groupby(["player", "table_name"], sort=False)["hand_ts"].diff().dt.total_seconds().fillna(0)
    new_session = gap > (SESSION_GAP_MINUTES * 60)
    feat["session_id"] = new_session.groupby([feat["player"], feat["table_name"]]).cumsum()

    log("matching post-cooler windows...")
    # Only keep coolers/features for players+tables that actually appear in both
    relevant_players = set(cooler["player"].unique())
    feat_rel = feat[feat["player"].isin(relevant_players)].copy()
    feat_rel = feat_rel.sort_values(["player", "table_name", "session_id", "hand_ts"]).reset_index(drop=True)

    post_cooler_mask = pd.Series(False, index=feat_rel.index)
    # hands_since_cooler: 1 = the very next hand after a cooler, 2 = the one
    # after that, etc. -- lets us check whether any effect DECAYS across the
    # window (a real-time-tilt signature) instead of just averaging it flat.
    hands_since_cooler = pd.Series(np.nan, index=feat_rel.index)
    grouped = feat_rel.groupby(["player", "table_name", "session_id"], sort=False)
    for (player, table_name, session_id), grp in grouped:
        sub_coolers = cooler[(cooler["player"] == player) & (cooler["table_name"] == table_name)]
        if sub_coolers.empty:
            continue
        ts_arr = grp["hand_ts"].values
        for cooler_ts in sub_coolers["cooler_ts"].values:
            after = ts_arr > cooler_ts
            if not after.any():
                continue
            idx_after = grp.index[after][:POST_COOLER_WINDOW]
            post_cooler_mask.loc[idx_after] = True
            hands_since_cooler.loc[idx_after] = np.arange(1, len(idx_after) + 1)

    feat_rel["post_cooler"] = post_cooler_mask
    feat_rel["hands_since_cooler"] = hands_since_cooler
    log(f"post-cooler hands: {feat_rel['post_cooler'].sum():,} / baseline (same players) hands: {(~feat_rel['post_cooler']).sum():,}")

    log(f"caching feat_rel to {CACHE_PATH} for cheap follow-up analyses...")
    feat_rel.to_parquet(CACHE_PATH)
    return feat_rel


def run_comparisons(feat_rel: pd.DataFrame) -> None:
    post = feat_rel[feat_rel["post_cooler"]]
    base = feat_rel[~feat_rel["post_cooler"]]

    def compare(name, post_series, base_series, kind="prop"):
        p_mean, b_mean = post_series.mean(), base_series.mean()
        if kind == "prop":
            successes = [post_series.sum(), base_series.sum()]
            n = [len(post_series), len(base_series)]
            stat, pval = stats.chi2_contingency(
                [[successes[0], n[0] - successes[0]], [successes[1], n[1] - successes[1]]]
            )[:2]
        else:
            stat, pval = stats.ttest_ind(post_series.dropna(), base_series.dropna(), equal_var=False)
        log(
            f"{name}: post-cooler={p_mean:.4f} (n={len(post_series)}) vs baseline={b_mean:.4f} "
            f"(n={len(base_series)}) delta={p_mean - b_mean:+.4f} p={pval:.4g}"
        )

    log("=== RESULTS ===")
    compare("VPIP", post["vpip"], base["vpip"])
    compare("postflop aggressive (any street)", post["postflop_aggressive"], base["postflop_aggressive"])
    compare(
        "avg postflop bet size (bb), among aggressive hands",
        post.loc[post["postflop_aggressive"], "avg_postflop_bet_bb"],
        base.loc[base["postflop_aggressive"], "avg_postflop_bet_bb"],
        kind="mean",
    )
    call_vs_fold_denom_post = post[post["saw_postflop"]]
    call_vs_fold_denom_base = base[base["saw_postflop"]]
    compare(
        "postflop call rate (thin-call proxy, among hands that saw postflop)",
        (call_vs_fold_denom_post["n_postflop_calls"] > 0) & (~call_vs_fold_denom_post["postflop_aggressive"]),
        (call_vs_fold_denom_base["n_postflop_calls"] > 0) & (~call_vs_fold_denom_base["postflop_aggressive"]),
    )
    log("--- confound check: is the post-cooler stack just mechanically shorter? ---")
    compare(
        "starting stack (bb) this hand",
        post["stack_bb"],
        base["stack_bb"],
        kind="mean",
    )

    log(f"--- stack-matched re-check ({STACK_MATCH_MIN}-{STACK_MATCH_MAX}bb both groups) ---")
    post_m = post[post["stack_bb"].between(STACK_MATCH_MIN, STACK_MATCH_MAX)]
    base_m = base[base["stack_bb"].between(STACK_MATCH_MIN, STACK_MATCH_MAX)]
    log(f"matched stack band: post-cooler n={len(post_m):,}, baseline n={len(base_m):,}")
    compare("VPIP (stack-matched)", post_m["vpip"], base_m["vpip"])
    compare("postflop aggressive (stack-matched)", post_m["postflop_aggressive"], base_m["postflop_aggressive"])

    log("--- decay curve: does the effect fade across the post-cooler window? ---")
    for lo, hi, label in [(1, 2, "hands 1-2"), (3, 5, "hands 3-5"), (6, 10, "hands 6-10")]:
        bucket = post[post["hands_since_cooler"].between(lo, hi)]
        vpip_rate = bucket["vpip"].mean()
        aggr_rate = bucket["postflop_aggressive"].mean()
        log(f"{label}: n={len(bucket):,} VPIP={vpip_rate:.4f} (baseline {base['vpip'].mean():.4f}) postflop_aggr={aggr_rate:.4f} (baseline {base['postflop_aggressive'].mean():.4f})")

    log("DONE")


def main():
    use_cache = "--use-cache" in sys.argv
    if use_cache and Path(CACHE_PATH).exists():
        log(f"loading cached feat_rel from {CACHE_PATH}...")
        feat_rel = pd.read_parquet(CACHE_PATH)
    else:
        feat_rel = build_feat_rel()
    run_comparisons(feat_rel)


if __name__ == "__main__":
    main()
