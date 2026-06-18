from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from config import * 

if TYPE_CHECKING:
    from modelling import ModelResult


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _color(label, idx):
    return PALETTE.get(label.lower(), FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


# Scalar metrics
def evaluate(
    y_true: np.ndarray | pd.Series,
    oof_preds: np.ndarray,
    label: str = "model",
):
    """
    Compute and print key binary-classification metrics.
    """
    auc  = roc_auc_score(y_true, oof_preds)
    ap   = average_precision_score(y_true, oof_preds)

    print(f"\n[{label}]")
    print(f"ROC-AUC : {auc:.5f}")
    print(f"Avg Precision : {ap:.5f}")

    return {"roc_auc": auc, "avg_precision": ap}


# ROC curve comparison
def plot_roc_comparison(
    results_dict: dict[str, "ModelResult"],
    y_true: np.ndarray | pd.Series,
    save_dir: str = "results",
):
    """
    Plot overlaid ROC curves for every model in `results_dict`.
    """
    _ensure_dir(save_dir)

    fig, ax = plt.subplots(figsize=(7, 6))

    for idx, (label, result) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, result.oof_preds)
        auc = roc_auc_score(y_true, result.oof_preds)
        color = _color(label, idx)
        ax.plot(
            fpr, tpr,
            label=f"{label}  (AUC = {auc:.4f})",
            color=color,
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    out_path = os.path.join(save_dir, "roc_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# Feature importance
def plot_feature_importance(
    result: "ModelResult",
    label: str,
    top_n: int = 30,
    save_dir: str = "results",
):
    """
    Horizontal bar chart of the top-N features by mean gain.
    """
    _ensure_dir(save_dir)

    df = result.feature_importance.head(top_n).copy()
    df = df.sort_values("importance")           # ascending so top is at top of hbar

    color = _color(label)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.28)))
    ax.barh(df["feature"], df["importance"], color=color, alpha=0.85)
    ax.set_xlabel("Mean Split Importance", fontsize=10)
    ax.set_title(f"Top {top_n} Feature Importances — {label}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    safe_label = label.replace(" ", "_").lower()
    out_path = os.path.join(save_dir, f"feature_importance_{safe_label}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# Per-fold AUC comparison
def plot_fold_scores(
    results_dict: dict[str, "ModelResult"],
    save_dir: str = "results",
):
    """
    Box + strip plot comparing per-fold AUC distributions across models.
    """
    _ensure_dir(save_dir)

    records = []
    for label, result in results_dict.items():
        for fold_idx, score in enumerate(result.fold_scores, start=1):
            records.append({"model": label, "fold": fold_idx, "auc": score})

    df = pd.DataFrame(records)
    labels = list(results_dict.keys())
    colors = [_color(lbl, i) for i, lbl in enumerate(labels)]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 2.2), 5))

    for i, (lbl, color) in enumerate(zip(labels, colors)):
        scores = df.loc[df["model"] == lbl, "auc"].values
        bp = ax.boxplot(
            scores,
            positions=[i],
            widths=0.4,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay individual fold dots
        jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(scores))
        ax.scatter(np.full(len(scores), i) + jitter, scores,
                   color=color, s=50, zorder=3, alpha=0.9)

        mean_score = np.mean(scores)
        ax.text(i, mean_score + 0.0005, f"{mean_score:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("Per-fold AUC Distribution by Model", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    plt.tight_layout()
    out_path = os.path.join(save_dir, "fold_scores_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# Summary table
def print_summary_table(
    results_dict: dict[str, "ModelResult"],
    y_true: np.ndarray | pd.Series,
):
    """
    Print and return a comparison DataFrame with one row per model.
    """
    rows = []
    for label, result in results_dict.items():
        auc  = roc_auc_score(y_true, result.oof_preds)
        ap   = average_precision_score(y_true, result.oof_preds)
        rows.append({
            "model":        label,
            "cv_auc_mean":  result.mean_cv_score,
            "cv_auc_std":   result.std_cv_score,
            "oof_auc":      auc,
            "oof_avg_prec": ap,
            "n_features":   len(result.feature_importance),
        })

    summary = pd.DataFrame(rows).set_index("model")

    print("MODEL COMPARISON SUMMARY")
    print()
    print(summary.to_string(float_format="{:.5f}".format))

    return summary