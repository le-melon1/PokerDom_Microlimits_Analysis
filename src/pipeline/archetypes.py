"""Rule-based villain archetype labeling, gated by sample size.

Project brief point 4: labels like "Maniac" or "Station" are only trustworthy
once the sample clears MIN_HANDS_FOR_LABEL -- below that, VPIP/PFR/AF swing
too much on small samples to mean anything, so callers should treat
`reliable=False` rows as "insufficient data", not as a real read.
"""

import pandas as pd

from src.config import MIN_HANDS_FOR_LABEL

UNKNOWN_LABEL = "Insufficient sample"


def _label_row(vpip: float, pfr: float, af: float) -> str:
    pfr_ratio = pfr / vpip if vpip > 0 else 0.0

    if vpip < 0.15:
        return "Nit"
    if vpip > 0.45 and af >= 2.0:
        return "Maniac"
    if vpip > 0.35 and pfr_ratio < 0.35:
        return "Station"
    if pfr_ratio >= 0.6 and vpip <= 0.28:
        return "TAG"
    if pfr_ratio >= 0.45:
        return "LAG"
    return "Loose-passive"


def label_archetypes(stats: pd.DataFrame) -> pd.DataFrame:
    """stats: output of pipeline.preprocess.player_stats (needs hands_seen, vpip, pfr,
    aggression_factor columns). Returns stats with `archetype` and `reliable` columns.
    """
    out = stats.copy()
    out["reliable"] = out["hands_seen"] >= MIN_HANDS_FOR_LABEL
    out["archetype"] = out.apply(
        lambda r: _label_row(r["vpip"], r["pfr"], r["aggression_factor"]), axis=1
    )
    out.loc[~out["reliable"], "archetype"] = UNKNOWN_LABEL
    return out
