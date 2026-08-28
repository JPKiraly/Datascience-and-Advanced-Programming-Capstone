"""Leakage-safe preprocessing for the frozen country-month classification task."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import (
    AUGMENTED_FEATURES,
    BASELINE_FEATURES,
    CATEGORICAL_FEATURES,
    COMMODITY_FEATURES,
    CONFLICT_FEATURES,
    TARGET,
)


def select_xy(df, feature_set: str, target: str = TARGET):
    """Select the frozen predictor set and binary target without audit columns."""
    if feature_set == "baseline":
        columns = BASELINE_FEATURES
    elif feature_set == "augmented":
        columns = AUGMENTED_FEATURES
    else:
        raise ValueError("feature_set must be 'baseline' or 'augmented'")

    return df[columns].copy(), df[target].astype(int).copy()


def _log1p_array(values):
    values = np.asarray(values, dtype=float)
    if np.any(values < 0):
        raise ValueError("Conflict count variables must be non-negative before log1p")
    return np.log1p(values)


def build_preprocessor(
    feature_set: str,
    model_family: str,
    log_conflict: bool = False,
):
    """Build a preprocessor that callers must fit on training observations only."""
    if feature_set not in {"baseline", "augmented"}:
        raise ValueError("Unknown feature set")
    if model_family not in {"scaled", "tree"}:
        raise ValueError("model_family must be 'scaled' or 'tree'")
    if model_family == "tree" and log_conflict:
        raise ValueError("Tree-based primary specifications use raw conflict counts")

    # Drop one category to avoid redundant dummy columns; unseen categories are
    # ignored so future-period rows cannot break the fitted encoder.
    categorical = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    if model_family == "scaled":
        conflict_steps = []
        if log_conflict:
            conflict_steps.append(
                (
                    "log1p",
                    FunctionTransformer(
                        _log1p_array,
                        validate=False,
                        feature_names_out="one-to-one",
                    ),
                )
            )
        conflict_steps.append(("scale", StandardScaler()))

        transformers = [
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("conflict", Pipeline(conflict_steps), CONFLICT_FEATURES),
        ]
        if feature_set == "augmented":
            transformers.append(("commodity", StandardScaler(), COMMODITY_FEATURES))
    else:
        # Scaling does not affect threshold-based tree splits, so retain raw
        # numeric values for the tree and AdaBoost specifications.
        transformers = [
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("conflict", "passthrough", CONFLICT_FEATURES),
        ]
        if feature_set == "augmented":
            transformers.append(("commodity", "passthrough", COMMODITY_FEATURES))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
