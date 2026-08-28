"""Reference test for the deterministic augmented Logistic Regression model."""

from pathlib import Path

import pandas as pd

from src.config import PAIRED_CONFIGS
from src.evaluation import binary_metrics
from src.models import fit_predict, split_modeling_table

ROOT = Path(__file__).resolve().parents[1]


def test_augmented_logistic_matches_reference():
    modeling_table = pd.read_csv(ROOT / "data/processed/modeling_table.csv")
    train, validation, test = split_modeling_table(modeling_table)
    final_train = pd.concat([train, validation], ignore_index=True)

    predictions = fit_predict(
        final_train,
        test,
        "logistic_regression",
        "augmented",
        PAIRED_CONFIGS["logistic_regression"]["augmented"],
    )
    generated_metrics = binary_metrics(test["target_high_conflict"], predictions)

    reference = pd.read_csv(ROOT / "results/reference/final_test_all_models.csv")
    reference = reference[
        (reference.model_family == "logistic_regression")
        & (reference.feature_set == "augmented")
    ].iloc[0]

    for metric in ["accuracy", "precision", "recall", "f1"]:
        assert abs(generated_metrics[metric] - float(reference[metric])) < 1e-12
