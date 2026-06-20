from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

from src.config import * 
from src.data_loader import *
from src.aggregation import *
from src.preprocessing import *
from src.feature_engineering import *
from src.modelling import *
from src.metrics import *


def load_all_tables(data_dir):
    """
    Load every table referenced in config.TABLE_FILES.
    """
    tables = {}

    for key, file_name in TABLE_FILES.items():
        tables[key] = load_data(file_name, data_dir=data_dir)

    return tables

def build_merged_frames(tables):
    """
    Merge all auxiliary tables onto both train and test.
    """
    aux_tables = {
        "bureau": tables["bureau"],
        "bureau_bal": tables["bureau_bal"],
        "prev_app": tables["prev_app"],
        "pos_cash": tables["pos_cash"],
        "cc_bal": tables["cc_bal"],
        "installments": tables["installments"],
    }

    train = merge_all_aux_tables(tables["train"], aux_tables)
    test = merge_all_aux_tables(tables["test"], aux_tables)

    return train, test


def basic_clean_and_encode(train, test):
    train = handle_anomaly(train)
    test = handle_anomaly(test)

    train, test = encode_categorical(train, test)

    return train, test


# Pipeline A: raw features

def build_raw_dataset(train_merged, test_merged):
    train, test = basic_clean_and_encode(train_merged, test_merged)
    return train, test


# Pipeline B: engineered features + full preprocessing

def build_processed_dataset(train_merged, test_merged):
    train, test = basic_clean_and_encode(train_merged, test_merged)

    # Engineered features 
    train = add_application_features(train)
    test = add_application_features(test)

    # Drop redundant high-missing / high-corr columns, fit on train only
    cols_to_drop = find_columns_to_drop(train)
    train, dropped = drop_redundant_columns(train, cols_to_drop)
    test, _ = drop_redundant_columns(test, dropped)

    # Skew correction, fit on train only
    skewed_cols = find_skewed_columns(train)
    train = log_transform_columns(train, skewed_cols)
    test = log_transform_columns(test, skewed_cols)

    # Re-align columns 
    missing_in_test = [c for c in train.columns if c != TARGET_COL and c not in test.columns]
    for col in missing_in_test:
        test[col] = 0
    test = test[[c for c in train.columns if c != TARGET_COL]]

    return train, test


def sanitize_feature_names(df):
    new_cols = [re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)) for c in df.columns]

    seen = {}
    deduped = []

    for c in new_cols:
        if c not in seen:
            seen[c] = 0
            deduped.append(c)
        else:
            seen[c] += 1
            deduped.append(f"{c}_{seen[c]}")

    df = df.copy()
    df.columns = deduped

    return df


def get_feature_matrix(df):
    drop_cols = [c for c in (ID_COL, TARGET_COL) if c in df.columns]

    X = df.drop(columns=drop_cols)
    X = sanitize_feature_names(X)

    y = df[TARGET_COL]

    return X, y


def run_pipeline(label, train, params):
    X, y = get_feature_matrix(train)
    result = train_lgbm(X, y, params=params)
    evaluate(y, result.oof_preds, label=label)

    return result


def main():
    parser = argparse.ArgumentParser(description="Home Credit Default Risk pipeline")
    parser.add_argument("--data-dir", default="data", help="Directory containing the raw CSV files")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directory to save plots/results")
    parser.add_argument(
        "--pipeline",
        choices=["raw", "processed", "both"],
        default="both",
        help=(
            "Which pipeline(s) to run: "
            "'raw' (baseline, no feature engineering), "
            "'processed' (engineered features + preprocessing), "
            "or 'both' to run and compare (default: both)"
        ),
    )
    args = parser.parse_args()

    run_raw = args.pipeline in ("raw", "both")
    run_processed = args.pipeline in ("processed", "both")

    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Load all tables
    print("Loading all tables...")
    tables = load_all_tables(args.data_dir)

    # 2. Merge all auxiliary tables onto train/test
    print("\nMerging auxiliary tables onto application_train / application_test...")
    train_merged, test_merged = build_merged_frames(tables)
    print(f"Train shape after merge: {train_merged.shape}")
    print(f"Test shape after merge : {test_merged.shape}")

    # 3. Build requested datasets
    train_raw = test_raw = train_proc = test_proc = None

    if run_raw:
        print("\nBuilding raw dataset (no feature engineering)...")
        train_raw, test_raw = build_raw_dataset(train_merged, test_merged)
        print(f"Train (raw) shape: {train_raw.shape}")

    if run_processed:
        print("\nBuilding processed dataset (engineered features + preprocessing)...")
        train_proc, test_proc = build_processed_dataset(train_merged, test_merged)
        print(f"Train (processed) shape: {train_proc.shape}")

    # 4. Train requested models
    results = {}

    if run_raw:
        results["raw"] = run_pipeline("raw", train_raw, DEFAULT_PARAMS)

    if run_processed:
        results["processed"] = run_pipeline("processed", train_proc, DEFAULT_PARAMS)

    # 5. Evaluate and plot
    # Use whichever train split is available to recover y_true
    ref_train = train_raw if train_raw is not None else train_proc
    y_train = ref_train[TARGET_COL]

    print_summary_table(results, y_train)

    # ROC comparison and fold scores only make sense with multiple models
    if len(results) > 1:
        plot_roc_comparison(results, y_train, save_dir=args.results_dir)
        plot_fold_scores(results, save_dir=args.results_dir)
    else:
        print("(Skipping ROC comparison and fold-score plots — only one pipeline was run)")

    for idx, (label, result) in enumerate(results.items()):
        plot_feature_importance(result, label=label, save_dir=args.results_dir, idx=idx)

    print("\nDone. Results saved to:", args.results_dir)


if __name__ == "__main__":
    main()