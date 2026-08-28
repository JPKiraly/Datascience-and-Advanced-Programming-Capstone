"""Frozen project configuration selected before the held-out test was opened."""

RANDOM_STATE = 42
TARGET = "target_high_conflict"
SPLIT_COLUMN = "target_split"

CATEGORICAL_FEATURES = ["country_code", "month_of_year"]
CONFLICT_FEATURES = [
    "events_t",
    "fatalities_t",
    "events_lag1",
    "fatalities_lag1",
    "events_3m_sum",
    "fatalities_3m_sum",
]
COMMODITY_FEATURES = [
    "energy_change_pct",
    "energy_volatility_3m_pct",
    "food_change_pct",
    "food_volatility_3m_pct",
    "fertilizers_change_pct",
    "fertilizers_volatility_3m_pct",
    "metals_minerals_change_pct",
    "metals_minerals_volatility_3m_pct",
    "precious_metals_change_pct",
    "precious_metals_volatility_3m_pct",
]
BASELINE_FEATURES = CATEGORICAL_FEATURES + CONFLICT_FEATURES
AUGMENTED_FEATURES = BASELINE_FEATURES + COMMODITY_FEATURES

# Keep the paired baseline/augmented settings exactly as chosen on validation
# so the held-out test remains a one-shot evaluation rather than a tuning source.
PAIRED_CONFIGS = {
    "logistic_regression": {
        "baseline": {
            "feature_set": "baseline",
            "log_conflict": True,
            "C": 0.1,
            "class_weight": None,
        },
        "augmented": {
            "feature_set": "augmented",
            "log_conflict": True,
            "C": 0.1,
            "class_weight": None,
        },
    },
    "knn": {
        "baseline": {
            "feature_set": "baseline",
            "log_conflict": True,
            "n_neighbors": 21,
            "weights": "distance",
        },
        "augmented": {
            "feature_set": "augmented",
            "log_conflict": True,
            "n_neighbors": 21,
            "weights": "distance",
        },
    },
    "decision_tree": {
        "baseline": {
            "feature_set": "baseline",
            "max_depth": 3,
            "min_samples_leaf": 20,
        },
        "augmented": {
            "feature_set": "augmented",
            "max_depth": 3,
            "min_samples_leaf": 20,
        },
    },
    "adaboost": {
        "baseline": {
            "feature_set": "baseline",
            "n_estimators": 200,
            "learning_rate": 1.0,
        },
        "augmented": {
            "feature_set": "augmented",
            "n_estimators": 200,
            "learning_rate": 0.5,
        },
    },
    "mlp": {
        "baseline": {
            "feature_set": "baseline",
            "log_conflict": True,
            "architecture": (64, 32),
            "learning_rate": 0.001,
            "batch_size": 1024,
            "final_epochs": 30,
        },
        "augmented": {
            "feature_set": "augmented",
            "log_conflict": False,
            "architecture": (32,),
            "learning_rate": 0.001,
            "batch_size": 1024,
            "final_epochs": 100,
        },
    },
}

# These feature-set winners were frozen before the final test was opened.
FAMILY_WINNERS = {
    "logistic_regression": "augmented",
    "knn": "baseline",
    "decision_tree": "baseline",
    "adaboost": "augmented",
    "mlp": "baseline",
}

TRAIN_TARGET_END = "2018-12-01"
VALIDATION_TARGET_START = "2019-01-01"
VALIDATION_TARGET_END = "2021-12-01"
TEST_TARGET_START = "2022-01-01"
TEST_TARGET_END = "2025-12-01"
