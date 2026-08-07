"""CatBoost model: P(villain folds | street, position, board texture, sizing).

Used to find sizings whose predicted fold frequency clears the breakeven fold
frequency by more than the field's noise floor -- i.e. sizings that show up as
"auto-profit" bluffs against the mined population (project brief, direction 1).
"""

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from src.pipeline.decision_points import breakeven_fold_frequency

CAT_FEATURES = ["street", "position", "opponent_position"]
FEATURE_COLUMNS = CAT_FEATURES + [
    "bet_size_bb",
    "pot_fraction",
    "board_paired",
    "board_monotone",
    "board_two_tone",
    "board_max_suit_count",
    "board_connectedness",
    "board_high_card",
]
TARGET_COLUMN = "villain_folded"


def train_fold_equity_model(df: pd.DataFrame, random_state: int = 42) -> CatBoostClassifier:
    train_df, eval_df = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df[TARGET_COLUMN]
    )
    train_pool = Pool(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], cat_features=CAT_FEATURES)
    eval_pool = Pool(eval_df[FEATURE_COLUMNS], eval_df[TARGET_COLUMN], cat_features=CAT_FEATURES)

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_state=random_state,
    )
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    return model


def find_profitable_sizings(model: CatBoostClassifier, df: pd.DataFrame) -> pd.DataFrame:
    """Flags rows where predicted fold-equity beats the breakeven bluff threshold."""
    preds = model.predict_proba(Pool(df[FEATURE_COLUMNS], cat_features=CAT_FEATURES))[:, 1]
    out = df.copy()
    out["predicted_fold_prob"] = preds
    out["breakeven_fold_freq"] = breakeven_fold_frequency(out["pot_fraction"])
    out["edge_over_breakeven"] = out["predicted_fold_prob"] - out["breakeven_fold_freq"]
    return out.sort_values("edge_over_breakeven", ascending=False)
