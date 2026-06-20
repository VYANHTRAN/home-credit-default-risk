import numpy as np
import pandas as pd

from src.config import * 

def handle_anomaly(df):
    df = df.copy()
    df.loc[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY, "DAYS_EMPLOYED"] = np.nan

    if "CODE_GENDER" in df.columns:
        valid_genders = df.loc[df["CODE_GENDER"] != "XNA", "CODE_GENDER"]
        if not valid_genders.empty:
            mode_gender = valid_genders.mode().iloc[0]
            df.loc[df["CODE_GENDER"] == "XNA", "CODE_GENDER"] = mode_gender

    return df

def encode_categorical(train, test=None):
    """
    The function is used to encode categorical columns. 
 
    If `test` is provided, the same mapping / dummy columns learned on
    `train` are applied to `test`, and the two frames are aligned to have
    identical columns (missing dummy columns in test are filled with 0).
    """

    train = train.copy()
    test = test.copy() if test is not None else None
 
    cat_cols = train.select_dtypes(include="object").columns.tolist()

    binary_cols = [c for c in cat_cols if train[c].nunique(dropna=False) <= 2]
    multi_cols = [c for c in cat_cols if c not in binary_cols]

    # Label encoding binary columns using a fitted mapping 
    label_maps = {}

    for col in binary_cols:
        categories = sorted(train[col].dropna().unique().tolist())
        mapping = {cat: i for i, cat in enumerate(categories)}
        label_maps[col] = mapping
 
        train[col] = train[col].map(mapping)

        if test is not None:
            test[col] = test[col].map(mapping)

    # One-hot encoding multi-class columns 
    if multi_cols:
        train = pd.get_dummies(train, columns=multi_cols, dummy_na=True)

        if test is not None:
            test = pd.get_dummies(test, columns=multi_cols, dummy_na=True)

    # Align train/test columns after encoding 
    if test is not None:
        train_cols = train.columns
        test_cols = test.columns
        missing_in_test = set(train_cols) - set(test_cols)

        for col in missing_in_test:
            if col == "TARGET":
                continue
            test[col] = 0
 
        missing_in_train = set(test_cols) - set(train_cols)
        for col in missing_in_train:
            train[col] = 0

        ordered_cols = [c for c in train.columns if c != "TARGET"]
        test = test[ordered_cols]
        train = train[ordered_cols + (["TARGET"] if "TARGET" in train.columns else [])]
 
    return train, test

def find_columns_to_drop(df, high_missing_threshold=HIGH_MISSING_THRESHOLD,
                          high_corr_threshold=HIGH_CORR_THRESHOLD,
                          exclude_cols=("TARGET", "SK_ID_CURR")):
    """
    Identify columns to drop that satisfies two conditions:
      1. Have a missing rate above high_missing_threshold
      2. Are highly correlated (|corr| > `high_corr_threshold`) with at
         least one other numeric column that has a LOWER missing rate.
    """
    exclude_cols = set(exclude_cols)
 
    # Missing rate per column
    missing_pct = (df.isnull().sum() / len(df) * 100)
 
    high_missing_cols = missing_pct[missing_pct > high_missing_threshold].index.tolist()
    high_missing_cols = [c for c in high_missing_cols if c not in exclude_cols]
 
    if not high_missing_cols:
        return []
 
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
 
    if len(numeric_cols) < 2:
        return []
 
    corr_matrix = df[numeric_cols].corr().abs()
 
    to_drop = []
    for col in high_missing_cols:
        if col not in corr_matrix.columns:
            continue
 
        # Correlation of `col` with every other numeric column
        correlated_with = corr_matrix[col].drop(col, errors="ignore")
        correlated_with = correlated_with[correlated_with > high_corr_threshold]
 
        for other_col in correlated_with.index:
            # Only drop `col` if `other_col` is more complete (lower missing %)
            if missing_pct[other_col] < missing_pct[col]:
                to_drop.append(col)
                break
 
    return sorted(set(to_drop))

def drop_redundant_columns(df, columns_to_drop):
    cols_present = [c for c in columns_to_drop if c in df.columns]
    return df.drop(columns=cols_present), cols_present

def find_skewed_columns(df, skew_threshold=SKEW_THRESHOLD, exclude_cols=("TARGET", "SK_ID_CURR")):
    """
    Identify positive-valued numeric columns whose absolute skewness exceeds
    `skew_threshold`.
    """
    exclude_cols = set(exclude_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    skewed_cols = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
 
        # Skip binary columns (only 0/1 values) and negative values
        unique_vals = series.unique()

        if series.min() < 0:
            continue
        if set(np.unique(unique_vals)).issubset({0, 1}):
            continue
 
        skew = series.skew()
        if abs(skew) > skew_threshold:
            skewed_cols.append(col)

    skewed_cols = [
        c for c in skewed_cols
        if not any(c.startswith(p) for p in ENGINEERED_PREFIXES)
    ]
 
    return skewed_cols
 
 
def log_transform_columns(df, columns):
    """
    Apply np.log1p to the given columns (in-place on a copy), creating new
    columns with a `_LOG` suffix and dropping the original columns.
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[f"{col}_LOG"] = np.log1p(df[col].clip(lower=0))
            df = df.drop(columns=[col])
            
    return df