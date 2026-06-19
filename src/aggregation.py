import pandas as pd

from src.config import * 

def aggregate_numeric(df, group_col, prefix):
    """
    Aggregate every numeric column in df using NUMERIC_AGGS.
    Non-numeric (categorical) columns are dropped.
    """
    id_like = {"SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"}
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in id_like
    ]

    if not numeric_cols:
        # Nothing to aggregate beyond a row count
        agg = df.groupby(group_col).size().to_frame(f"{prefix}_COUNT")
        agg.reset_index(inplace=True)
        return agg

    agg = df.groupby(group_col)[numeric_cols].agg(NUMERIC_AGGS)
    agg.columns = [f"{prefix}_{col}_{stat.upper()}" for col, stat in agg.columns]
    agg.reset_index(inplace=True)

    return agg

def build_bureau_features(bureau, bureau_bal):
    """
    bureau_balance (monthly history per SK_ID_BUREAU) -> aggregate to one
    row per SK_ID_BUREAU -> merge onto bureau -> aggregate to one row per
    SK_ID_CURR.
    """
    bureau_bal_agg = aggregate_numeric(bureau_bal, "SK_ID_BUREAU", "BUREAU_BAL")

    bureau = bureau.merge(bureau_bal_agg, on="SK_ID_BUREAU", how="left")
    bureau_features = aggregate_numeric(bureau, "SK_ID_CURR", "BUREAU")

    return bureau_features


def build_previous_application_features(
    prev_app,
    pos_cash,
    cc_bal,
    installments,
):
    """
    POS_CASH_balance, credit_card_balance, installments_payments (monthly
    history per SK_ID_PREV) -> aggregate each to one row per SK_ID_PREV ->
    merge onto previous_application -> aggregate to one row per SK_ID_CURR.
    """
    pos_cash_agg = aggregate_numeric(pos_cash, "SK_ID_PREV", "POS")
    cc_bal_agg = aggregate_numeric(cc_bal, "SK_ID_PREV", "CC")
    installments_agg = aggregate_numeric(installments, "SK_ID_PREV", "INSTAL")

    prev_app = prev_app.merge(pos_cash_agg, on="SK_ID_PREV", how="left")
    prev_app = prev_app.merge(cc_bal_agg, on="SK_ID_PREV", how="left")
    prev_app = prev_app.merge(installments_agg, on="SK_ID_PREV", how="left")

    prev_app_features = aggregate_numeric(prev_app, "SK_ID_CURR", "PREV")

    return prev_app_features


def merge_all_aux_tables(application, aux_tables):
    """
    Merge bureau-derived and previous_application-derived feature tables
    onto the application DataFrame via SK_ID_CURR.
    """
    df = application.copy()

    bureau_features = build_bureau_features(
        aux_tables["bureau"], aux_tables["bureau_bal"]
    )
    df = df.merge(bureau_features, on="SK_ID_CURR", how="left")

    prev_features = build_previous_application_features(
        aux_tables["prev_app"],
        aux_tables["pos_cash"],
        aux_tables["cc_bal"],
        aux_tables["installments"],
    )
    df = df.merge(prev_features, on="SK_ID_CURR", how="left")

    return df