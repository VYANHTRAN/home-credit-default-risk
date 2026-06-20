# Home Credit Default Risk — Pipeline Results Analysis

## Overview

Two parallel LightGBM pipelines were evaluated on the Home Credit Default Risk dataset using 5-fold stratified cross-validation and submitted to Kaggle for public leaderboard scoring.

| Pipeline | CV AUC (mean ± std) | OOF AUC | OOF Avg Precision | Features | Public LB AUC | Private LB AUC |
|---|---|---|---|---|---|---|
| **Raw** (baseline) | 0.78236 ± 0.00353 | 0.78224 | 0.27525 | 1,206 | 0.78209 | 0.77933 |
| **Processed** (engineered) | **0.78563 ± 0.00320** | **0.78548** | **0.27707** | 923 | **0.78634** | **0.78610** |
| Δ (Processed − Raw) | **+0.00327** | **+0.00324** | **+0.00182** | −283 | **+0.00425** | **+0.00677** |

Feature engineering improved across every metric. The processed pipeline achieves higher AUC with 283 fewer features, and its lower CV standard deviation (0.00320 vs 0.00353) indicates more stable fold-to-fold performance. The private leaderboard gain of +0.00677, which is larger than the public gain of +0.00425, confirms the engineered features generalise well rather than fitting noise in the public split.

---

## Feature Importance Analysis

### Raw pipeline 

In the raw pipeline, the model leans heavily on the three external bureau scores (`EXT_SOURCE_1/2/3`) as its primary discriminators, together accounting for the three largest importance values (440, 385, 365 respectively). The next tier consists of time-based stability signals: `DAYS_EMPLOYED`, `DAYS_BIRTH`, `AMT_ANNUITY`, and `AMT_CREDIT`.

This pattern tells a clear story: without derived features, the model is essentially doing a weighted combination of the credit bureau scores plus a few raw financial magnitudes. The bureau scores dominate because they already encode a large amount of credit risk information, but they saturate quickly as the model cannot squeeze much more signal from them once they're in.

### Processed pipeline 

Feature engineering substantially changes which signals the model relies on:

**`CREDIT_TERM_YEARS`** (importance ≈ 350) jumps to the top position. This is computed as `AMT_CREDIT / (AMT_ANNUITY × 12)` and represents the implicit loan term in years. A longer term correlates with higher repayment risk, and it synthesises two originally separate columns into a ratio the model can split on more efficiently.

**`EXT_SOURCE_MEAN`** (≈ 230) rises above the individual source scores by aggregating the three scores into a single, lower-noise statistic. The individual scores (`EXT_SOURCE_1`, `EXT_SOURCE_3`) remain in the top 5, but their individual importance drops because the mean already captures most of their shared variance.

**`ANNUITY_GOODS_RATIO`** and **`EXT_23_PRODUCT`** entering the top 10 show the model benefiting from interaction terms — combinations of signals that individually are weaker but jointly discriminate well.

**`EMPLOYED_FRACTION_OF_LIFE_LOG`** and **`AGE_YEARS`** replace raw `DAYS_EMPLOYED` and `DAYS_BIRTH`, indicating the model finds the normalised, ratio form more informative than the raw day counts.

---

## What the Engineering Actually Did

The 10 feature groups in `feature_engineering.py` each served a distinct purpose, and the importance chart validates most of them:

| Feature group | Representative features in top 30 | Validated? |
|---|---|---|
| EXT_SOURCE interactions | `EXT_SOURCE_MEAN`, `EXT_23_PRODUCT`, `EXT_SOURCE_MIN/MAX` | Strong |
| Credit affordability | `CREDIT_TERM_YEARS`, `ANNUITY_GOODS_RATIO`, `CREDIT_GOODS_DIFF/RATIO` | Strong |
| Employment stability | `EMPLOYED_FRACTION_OF_LIFE_LOG`, `EMPLOYMENT_YEARS` | Moderate |
| Age & life stage | `AGE_YEARS` | Moderate |
| Composite risk | `RISK_COMPOSITE` | Moderate |
| Bureau enquiry recency | Not in top 30 | Weak signal |
| Document / contact flags | Not in top 30 | Weak signal |
| Asset ownership | `CAR_AGE_TO_OWNER_AGE_LOG` | Marginal |

The preprocessing steps (log-transforming skewed columns, dropping high-missing/high-corr columns) also show their effect: several features appear in `_LOG` form (`AMT_ANNUITY_LOG`, `EMPLOYED_FRACTION_OF_LIFE_LOG`, `PREV_INSTAL_AMT_PAYMENT_MIN_MEAN_LOG`), suggesting the skewness correction helped the model find better split thresholds.

---

## Generalisation Quality

The CV-to-leaderboard gap tells a reassuring story. For the processed model, CV AUC (0.78563) and OOF AUC (0.78548) are nearly identical, indicating no significant data leakage between folds. The public leaderboard score (0.78634) is within 0.001 of both, confirming the cross-validation setup faithfully reflects real generalisation.

The raw model shows a more pronounced public-to-private drop (0.78209 → 0.77933, Δ = 0.00276) compared to the processed model (0.78634 → 0.78610, Δ = 0.00024). This is consistent with what domain-informed feature engineering should do: expressing credit risk explicitly as ratios and interactions gives the model less to discover from scratch, making its learned splits more stable across different data slices.