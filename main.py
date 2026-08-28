"""Reproducible entry point for the capstone project.

Default behavior reproduces the final held-out test comparison from the frozen
modeling sample and the model configurations selected before the test was opened.

Optional commands:
  --rebuild-data   recreate the canonical modeling table from included snapshots
  --from-raw       regenerate source snapshots from original files in data/raw
  --rerun-selection rerun validation-only hyperparameter grids (slower)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FAMILY_WINNERS, PAIRED_CONFIGS, TARGET
from src.data_pipeline import (
    assert_matches_canonical,
    build_modeling_table,
    extract_ucdp_snapshot,
    extract_world_bank_snapshot,
)
from src.evaluation import binary_metrics
from src.models import fit_predict, split_modeling_table

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
GENERATED = RESULTS / "generated"
REFERENCE = RESULTS / "reference"


def _evaluate_frozen_models(modeling_table: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every frozen model after refitting on train plus validation."""
    train, validation, test = split_modeling_table(modeling_table)
    final_train = pd.concat([train, validation], ignore_index=True)
    y_test = test[TARGET].astype(int).to_numpy()

    rows = []

    # Learn the majority class from final training data only so test prevalence
    # cannot influence even the trivial benchmark.
    majority_class = int(
        pd.Series(final_train[TARGET].astype(int)).value_counts().idxmax()
    )
    majority_predictions = np.full(len(test), majority_class, dtype=int)
    rows.append(
        {
            "model_family": "majority",
            "feature_set": "benchmark",
            **binary_metrics(y_test, majority_predictions),
        }
    )

    # Persistence is deliberately simple: it tests how difficult it is for ML
    # models to improve on the strong month-to-month continuity of conflict.
    persistence_predictions = (
        test["events_t"].astype(int)
        >= test["high_conflict_cutoff_events"].astype(int)
    ).astype(int)
    rows.append(
        {
            "model_family": "persistence",
            "feature_set": "benchmark",
            **binary_metrics(y_test, persistence_predictions),
        }
    )

    for model_family, paired_configs in PAIRED_CONFIGS.items():
        for feature_set in ("baseline", "augmented"):
            predictions = fit_predict(
                final_train,
                test,
                model_family,
                feature_set,
                paired_configs[feature_set],
            )
            rows.append(
                {
                    "model_family": model_family,
                    "feature_set": feature_set,
                    **binary_metrics(y_test, predictions),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )


def _compare_reference(
    generated: pd.DataFrame,
    tolerance: float = 0.01,
) -> float:
    """Check that regenerated metrics remain within the documented tolerance."""
    reference = pd.read_csv(REFERENCE / "final_test_all_models.csv")
    metric_columns = ["accuracy", "precision", "recall", "f1"]
    merged = generated.merge(
        reference[["model_family", "feature_set", *metric_columns]],
        on=["model_family", "feature_set"],
        suffixes=("_generated", "_reference"),
        validate="one_to_one",
    )

    maximum_difference = 0.0
    for metric in metric_columns:
        difference = (
            merged[f"{metric}_generated"] - merged[f"{metric}_reference"]
        ).abs()
        maximum_difference = max(maximum_difference, float(difference.max()))

    if maximum_difference > tolerance:
        raise AssertionError(
            "Generated final metrics differ from stored reference by up to "
            f"{maximum_difference:.6f}, exceeding tolerance {tolerance}."
        )

    return maximum_difference


def _write_final_outputs(results: pd.DataFrame) -> None:
    """Write the final tables and comparison figure to results/generated/."""
    GENERATED.mkdir(parents=True, exist_ok=True)
    results.to_csv(GENERATED / "final_test_all_models.csv", index=False)

    winner_keys = [{"model_family": "persistence", "feature_set": "benchmark"}]
    winner_keys.extend(
        {"model_family": family, "feature_set": feature_set}
        for family, feature_set in FAMILY_WINNERS.items()
    )
    family_winners = pd.DataFrame(winner_keys).merge(
        results,
        on=["model_family", "feature_set"],
        how="left",
        validate="one_to_one",
    )
    family_winners = family_winners.sort_values("f1", ascending=False)
    family_winners.to_csv(
        GENERATED / "final_test_family_winners.csv",
        index=False,
    )

    delta_rows = []
    for model_family in PAIRED_CONFIGS:
        baseline = results[
            (results.model_family == model_family)
            & (results.feature_set == "baseline")
        ].iloc[0]
        augmented = results[
            (results.model_family == model_family)
            & (results.feature_set == "augmented")
        ].iloc[0]
        delta_rows.append(
            {
                "model_family": model_family,
                "baseline_f1": baseline.f1,
                "augmented_f1": augmented.f1,
                "delta_f1_augmented_minus_baseline": augmented.f1 - baseline.f1,
            }
        )
    pd.DataFrame(delta_rows).to_csv(
        GENERATED / "test_commodity_increment_summary.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    labels = [
        f"{row.model_family.replace('_', ' ').title()} — {row.feature_set}"
        for _, row in family_winners.iterrows()
    ]
    f1_values = family_winners["f1"].to_numpy()
    axis.bar(labels, f1_values)
    axis.set_ylabel("Test F1")
    axis.set_title("Frozen model-family winners on the held-out test period")
    axis.tick_params(axis="x", rotation=28)
    axis.set_ylim(0, max(f1_values) * 1.15)
    for index, value in enumerate(f1_values):
        axis.text(index, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(GENERATED / "final_test_model_comparison.png", dpi=180)
    plt.close(figure)


def rebuild_data(from_raw: bool = False) -> None:
    """Reconstruct and verify the canonical modeling table."""
    interim = DATA / "interim"
    metadata = DATA / "metadata" / "ssa_countries.csv"
    events = interim / "ucdp_ssa_2000_2025.csv"
    indices = interim / "world_bank_commodity_indices_1999_10_2025_12.csv"
    canonical = DATA / "processed" / "modeling_table.csv"

    if from_raw:
        raw_ged = DATA / "raw" / "GEDEvent_v26_1.csv"
        raw_pink_sheet = DATA / "raw" / "CMO-Historical-Data-Monthly.xlsx"
        missing_files = [
            str(path)
            for path in (raw_ged, raw_pink_sheet)
            if not path.exists()
        ]
        if missing_files:
            raise FileNotFoundError(
                "Missing raw source files required by --from-raw:\n  "
                + "\n  ".join(missing_files)
            )

        print("Extracting reproducibility snapshots from original raw sources...")
        extract_ucdp_snapshot(raw_ged, metadata, events)
        extract_world_bank_snapshot(raw_pink_sheet, indices)

    print("Rebuilding the country-month modeling table...")
    rebuilt = build_modeling_table(events, indices, metadata)
    assert_matches_canonical(rebuilt, canonical)
    GENERATED.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(GENERATED / "modeling_table_rebuilt.csv", index=False)
    print(
        "Data reconstruction passed: "
        f"{len(rebuilt):,} rows exactly match the canonical table."
    )


def main() -> int:
    """Parse CLI options and run the requested reproducibility workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="rebuild phases 4-9 from included source snapshots",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="regenerate snapshots from original raw data and rebuild",
    )
    parser.add_argument(
        "--rerun-selection",
        action="store_true",
        help="rerun all validation-only hyperparameter grids (slower)",
    )
    args = parser.parse_args()

    GENERATED.mkdir(parents=True, exist_ok=True)
    if args.from_raw:
        rebuild_data(from_raw=True)
    elif args.rebuild_data:
        rebuild_data(from_raw=False)

    if args.rerun_selection:
        from src.model_selection import rerun_all

        selection_table = pd.read_csv(DATA / "processed" / "modeling_table.csv")
        train_selection, validation_selection, _ = split_modeling_table(
            selection_table
        )
        print("Rerunning validation-only model selection grids...")
        rerun_all(
            train_selection,
            validation_selection,
            GENERATED / "model_selection",
        )
        print(
            "Validation grids written to results/generated/model_selection/. "
            "Test rows were not used.\n"
        )

    print("Running the frozen final out-of-sample evaluation...")
    modeling_table = pd.read_csv(DATA / "processed" / "modeling_table.csv")
    results = _evaluate_frozen_models(modeling_table)
    _write_final_outputs(results)
    maximum_difference = _compare_reference(results)

    print("\nFinal held-out test F1 scores:")
    print(
        results[
            [
                "model_family",
                "feature_set",
                "f1",
                "precision",
                "recall",
                "accuracy",
            ]
        ].to_string(index=False)
    )
    print(
        "\nReference reproduction check passed "
        f"(maximum metric difference = {maximum_difference:.6g})."
    )
    print(f"Generated outputs: {GENERATED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
