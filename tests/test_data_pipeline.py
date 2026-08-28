"""Regression tests for the frozen data construction and split design."""

from pathlib import Path

import pandas as pd

from src.data_pipeline import assert_matches_canonical, build_modeling_table

ROOT = Path(__file__).resolve().parents[1]


def test_rebuild_matches_canonical():
    rebuilt = build_modeling_table(
        ROOT / "data/interim/ucdp_ssa_2000_2025.csv",
        ROOT / "data/interim/world_bank_commodity_indices_1999_10_2025_12.csv",
        ROOT / "data/metadata/ssa_countries.csv",
    )
    assert_matches_canonical(
        rebuilt,
        ROOT / "data/processed/modeling_table.csv",
    )


def test_frozen_split_counts_and_boundaries():
    modeling_table = pd.read_csv(ROOT / "data/processed/modeling_table.csv")
    counts = modeling_table["target_split"].value_counts().to_dict()
    assert counts == {"train": 10800, "test": 2304, "validation": 1728}

    train = modeling_table[modeling_table.target_split == "train"]
    validation = modeling_table[modeling_table.target_split == "validation"]
    test = modeling_table[modeling_table.target_split == "test"]

    assert train.target_month.max() == "2018-12-01"
    assert validation.target_month.min() == "2019-01-01"
    assert validation.target_month.max() == "2021-12-01"
    assert test.target_month.min() == "2022-01-01"
    assert test.target_month.max() == "2025-12-01"


def test_no_target_columns_in_frozen_predictors():
    from src.config import AUGMENTED_FEATURES, BASELINE_FEATURES

    forbidden = {
        "target_high_conflict",
        "target_events_t_plus_1",
        "target_fatalities_t_plus_1",
        "train_event_q75",
        "high_conflict_cutoff_events",
    }
    assert forbidden.isdisjoint(BASELINE_FEATURES)
    assert forbidden.isdisjoint(AUGMENTED_FEATURES)
