from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }


def compare_reference(results, reference_csv, tolerance=0.01):
    reference = pd.read_csv(reference_csv)
    metrics = ["accuracy", "precision", "recall", "f1"]
    merged = results.merge(reference[["model_family", "feature_set", *metrics]], on=["model_family", "feature_set"], suffixes=("_new", "_reference"), validate="one_to_one")
    maximum_difference = 0.0
    for metric in metrics:
        difference = (merged[f"{metric}_new"] - merged[f"{metric}_reference"]).abs()
        maximum_difference = max(maximum_difference, float(difference.max()))
    if maximum_difference > tolerance:
        raise AssertionError(f"Results differ from the stored reference by {maximum_difference:.6f}")
    return maximum_difference


def save_outputs(results, family_winners, paired_configs, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "final_test_all_models.csv", index=False)

    selected = [{"model_family": "persistence", "feature_set": "benchmark"}]
    selected += [{"model_family": model, "feature_set": feature_set} for model, feature_set in family_winners.items()]
    winners = (pd.DataFrame(selected) .merge(results, on=["model_family", "feature_set"], how="left", validate="one_to_one",) .sort_values("f1", ascending=False))
    winners.to_csv(results_dir / "final_test_family_winners.csv", index=False)

    delta_rows = []
    for model in paired_configs:
        baseline = results[(results["model_family"] == model) & (results["feature_set"] == "baseline")].iloc[0]
        augmented = results[(results["model_family"] == model) & (results["feature_set"] == "augmented")].iloc[0]
        delta_rows.append({"model_family": model, "baseline_f1": baseline["f1"], "augmented_f1": augmented["f1"], "delta_f1": augmented["f1"] - baseline["f1"]})
    pd.DataFrame(delta_rows).to_csv(results_dir / "test_commodity_increment_summary.csv", index=False)

    labels = [f"{row.model_family.replace('_', ' ').title()} - {row.feature_set}" for _, row in winners.iterrows()]
    values = winners["f1"].to_numpy()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values)
    axis.set_ylabel("Test F1")
    axis.set_title("Final model comparison")
    axis.tick_params(axis="x", rotation=28)
    axis.set_ylim(0, max(values) * 1.15)
    for i, value in enumerate(values):
        axis.text(i, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(results_dir / "final_test_model_comparison.png", dpi=180)
    plt.close(figure)
    return winners
