"""Rule-based villain archetype labeling, gated by sample size.

Project brief point 4: labels like "Maniac" or "Station" are only trustworthy
once the sample clears MIN_HANDS_FOR_LABEL -- below that, VPIP/PFR/AF swing
too much on small samples to mean anything, so callers should treat
`reliable=False` rows as "insufficient data", not as a real read.

2026-08-18: added a SECOND, independent axis -- POSTFLOP_FREQ_TIER, a
3-way split of the same postflop aggression_factor already partially
folded into `_label_row`'s Maniac cutoff, but now exposed as its own
column instead of collapsed into one flat archetype label. User's
request: "мы делим игроков не просто по архетипам, а по архетипам на
префлопе, а потом ещё каждого из них мы делим на три типа частоты рейза
на постфлопе (как сделано на покердоме) -- редко/нормально/часто", so
every player ends up with a COMPOUND (archetype, postflop_freq_tier) read
instead of one flat label. Thresholds are literature-grounded (standard
poker-HUD AF convention: AF<2.0 passive, ~2.0-3.0 "solid regular"
range, AF>3.0 aggressive -- see e.g. BlackRain79's "What is a Good AF in
Poker?"), not fit to this dataset's own percentiles, though they happen
to land close to this population's natural quantiles too (0.40 quantile
~=1.88, 0.75 quantile =3.00) and produce a sane non-degenerate split here
(44% rare / 32% normal / 24% often among reliable-sample players).
Follow-up correction, same conversation: the archetype axis itself was
NOT purely preflop -- `_label_row`'s Maniac branch used postflop `af`,
mixing the two axes right where they're meant to be independent. Fixed:
Maniac is now `vpip > 0.45 and pfr_ratio >= 0.45` (an extreme-VPIP LAG,
same preflop-only signals as every other archetype), no `af` parameter
at all. This is a real, deliberate re-labeling, not a bugfix in the
"restores old intended behavior" sense -- some players who were Maniac
under the old af-gated rule (loose but with low PFR-ratio) now land in
Station or Loose-passive instead, and some who weren't (loose, high
PFR-ratio, but af<2.0 postflop) now DO count as Maniac. What used to be
"loose AND postflop-aggressive" is now correctly represented as the
compound label, e.g. `LAG (often)` or `Loose-passive (often)`, rather
than baked into the preflop bucket itself.

Purely ADDITIVE otherwise -- existing callers (reference-table builders,
the ML behavior-clone model's `archetype` feature, abc_bot.py's
archetype-gated rules) still read the same `archetype` column name and
same 6 label strings, just re-computed with a cleaner, preflop-only
definition. Anything gated on `archetype == "Maniac"` downstream will see
a real population shift and should be revisited before trusting old
confirmed numbers against it.
"""

import pandas as pd

from src.config import MIN_HANDS_FOR_LABEL

UNKNOWN_LABEL = "Insufficient sample"

POSTFLOP_FREQ_RARE_MAX = 2.0  # AF below this -> "rare"
POSTFLOP_FREQ_OFTEN_MIN = 3.0  # AF above this -> "often"; between the two -> "normal"


def _label_row(vpip: float, pfr: float) -> str:
    """Preflop-only -- no postflop stat (af) enters this function. See
    POSTFLOP_FREQ_TIER / `_postflop_freq_tier` below for the independent
    postflop axis."""
    pfr_ratio = pfr / vpip if vpip > 0 else 0.0

    if vpip < 0.15:
        return "Nit"
    if vpip > 0.45 and pfr_ratio >= 0.45:
        return "Maniac"
    if vpip > 0.35 and pfr_ratio < 0.35:
        return "Station"
    if pfr_ratio >= 0.6 and vpip <= 0.28:
        return "TAG"
    if pfr_ratio >= 0.45:
        return "LAG"
    return "Loose-passive"


def _postflop_freq_tier(af: float) -> str:
    if af < POSTFLOP_FREQ_RARE_MAX:
        return "rare"
    if af > POSTFLOP_FREQ_OFTEN_MIN:
        return "often"
    return "normal"


def label_archetypes(stats: pd.DataFrame) -> pd.DataFrame:
    """stats: output of pipeline.preprocess.player_stats (needs hands_seen, vpip, pfr,
    aggression_factor columns). Returns stats with `archetype`, `postflop_freq_tier`,
    `archetype_freq` (the compound "Archetype (tier)" label), and `reliable` columns.
    """
    out = stats.copy()
    out["reliable"] = out["hands_seen"] >= MIN_HANDS_FOR_LABEL
    out["archetype"] = out.apply(
        lambda r: _label_row(r["vpip"], r["pfr"]), axis=1
    )
    out["postflop_freq_tier"] = out["aggression_factor"].apply(_postflop_freq_tier)
    out.loc[~out["reliable"], "archetype"] = UNKNOWN_LABEL
    out.loc[~out["reliable"], "postflop_freq_tier"] = UNKNOWN_LABEL
    out["archetype_freq"] = out["archetype"] + " (" + out["postflop_freq_tier"] + ")"
    out.loc[~out["reliable"], "archetype_freq"] = UNKNOWN_LABEL
    return out
