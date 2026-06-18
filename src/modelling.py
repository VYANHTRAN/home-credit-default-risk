from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import * 

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: list = field(default_factory=list)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_scores: list[float] = field(default_factory=list)

    @property
    def mean_cv_score(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std_cv_score(self) -> float:
        return float(np.std(self.fold_scores))


def train_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> ModelResult:
    """
    Train a LightGBM binary classifier with stratified k-fold cross-validation.
    """

    lgb_params = {**DEFAULT_PARAMS, "random_state": seed}

    if params:
        lgb_params.update(params)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(X))
    models: list[lgb.LGBMClassifier] = []
    fold_scores: list[float] = []
    importance_dfs: list[pd.DataFrame] = []

    print(f"LightGBM - {n_splits}-fold Stratified CV")
    print(f"Features : {X.shape[1]}")
    print(f"Samples : {X.shape[0]}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(VERBOSE_EVAL),
            ],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        from sklearn.metrics import roc_auc_score
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
        models.append(model)

        imp = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        })
        importance_dfs.append(imp)

        print(f"Fold {fold}/{n_splits}  |  best iter: {model.best_iteration_:>5}  |  val AUC: {fold_auc:.5f}")

    # Aggregate feature importances 
    feature_importance = (
        pd.concat(importance_dfs)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    result = ModelResult(
        oof_preds=oof_preds,
        models=models,
        feature_importance=feature_importance,
        fold_scores=fold_scores,
    )

    print(f"\nCV AUC : {result.mean_cv_score:.5f} ± {result.std_cv_score:.5f}")

    return result