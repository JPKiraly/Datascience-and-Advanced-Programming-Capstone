"""Validation-only hyperparameter searches used before the test set was opened.

These functions deliberately accept only the train and validation partitions.
They are optional for final reproduction because the submitted final model
configurations are already frozen in :mod:`src.config`.
"""

from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE, TARGET
from .evaluation import binary_metrics
from .models import TorchMLP
from .preprocessing import build_preprocessor, select_xy


def _sort(results: pd.DataFrame) -> pd.DataFrame:
    """Apply the common validation ranking used during model selection."""
    return results.sort_values(
        ["f1", "recall", "precision", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def logistic_grid(train, validation) -> pd.DataFrame:
    """Run the frozen Logistic Regression validation grid."""
    rows = []
    for feature_set in ["baseline", "augmented"]:
        train_features_raw, train_target = select_xy(train, feature_set)
        validation_features_raw, validation_target = select_xy(
            validation, feature_set
        )

        for log_conflict in [False, True]:
            preprocessor = build_preprocessor(feature_set, "scaled", log_conflict)
            train_features = preprocessor.fit_transform(train_features_raw)
            validation_features = preprocessor.transform(validation_features_raw)

            for c_value in [0.01, 0.1, 1.0, 10.0]:
                for class_weight in [None, "balanced"]:
                    model = LogisticRegression(
                        C=c_value,
                        solver="liblinear",
                        class_weight=class_weight,
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    )
                    predictions = model.fit(
                        train_features, train_target
                    ).predict(validation_features)
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "log_conflict": log_conflict,
                            "C": c_value,
                            "class_weight": class_weight,
                            **binary_metrics(validation_target, predictions),
                        }
                    )

    return _sort(pd.DataFrame(rows))


def knn_grid(train, validation) -> pd.DataFrame:
    """Run the frozen k-nearest-neighbors validation grid."""
    rows = []
    for feature_set in ["baseline", "augmented"]:
        train_features_raw, train_target = select_xy(train, feature_set)
        validation_features_raw, validation_target = select_xy(
            validation, feature_set
        )

        for log_conflict in [False, True]:
            preprocessor = build_preprocessor(feature_set, "scaled", log_conflict)
            train_features = preprocessor.fit_transform(train_features_raw)
            validation_features = preprocessor.transform(validation_features_raw)

            for n_neighbors in [3, 5, 7, 11, 21]:
                for weights in ["uniform", "distance"]:
                    model = KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        metric="minkowski",
                        p=2,
                        n_jobs=-1,
                    )
                    predictions = model.fit(
                        train_features, train_target
                    ).predict(validation_features)
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "log_conflict": log_conflict,
                            "n_neighbors": n_neighbors,
                            "weights": weights,
                            **binary_metrics(validation_target, predictions),
                        }
                    )

    return _sort(pd.DataFrame(rows))


def decision_tree_grid(train, validation) -> pd.DataFrame:
    """Run the frozen Decision Tree validation grid."""
    rows = []
    for feature_set in ["baseline", "augmented"]:
        train_features_raw, train_target = select_xy(train, feature_set)
        validation_features_raw, validation_target = select_xy(
            validation, feature_set
        )
        preprocessor = build_preprocessor(feature_set, "tree", False)
        train_features = preprocessor.fit_transform(train_features_raw)
        validation_features = preprocessor.transform(validation_features_raw)

        for max_depth in [3, 5, 8, None]:
            for min_samples_leaf in [1, 5, 10, 20]:
                model = DecisionTreeClassifier(
                    criterion="entropy",
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_STATE,
                )
                predictions = model.fit(train_features, train_target).predict(
                    validation_features
                )
                rows.append(
                    {
                        "feature_set": feature_set,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        **binary_metrics(validation_target, predictions),
                    }
                )

    return _sort(pd.DataFrame(rows))


def adaboost_grid(train, validation) -> pd.DataFrame:
    """Run the frozen AdaBoost validation grid with decision stumps."""
    rows = []
    for feature_set in ["baseline", "augmented"]:
        train_features_raw, train_target = select_xy(train, feature_set)
        validation_features_raw, validation_target = select_xy(
            validation, feature_set
        )
        preprocessor = build_preprocessor(feature_set, "tree", False)
        train_features = preprocessor.fit_transform(train_features_raw)
        validation_features = preprocessor.transform(validation_features_raw)

        for n_estimators in [50, 100, 200]:
            for learning_rate in [0.01, 0.1, 0.5, 1.0]:
                stump = DecisionTreeClassifier(
                    max_depth=1,
                    random_state=RANDOM_STATE,
                )
                model = AdaBoostClassifier(
                    estimator=stump,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=RANDOM_STATE,
                )
                predictions = model.fit(train_features, train_target).predict(
                    validation_features
                )
                rows.append(
                    {
                        "feature_set": feature_set,
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        **binary_metrics(validation_target, predictions),
                    }
                )

    return _sort(pd.DataFrame(rows))


def mlp_grid(train, validation) -> pd.DataFrame:
    """Run the frozen feed-forward MLP validation grid with early stopping."""
    # Single-thread execution improves repeatability across ordinary CPU setups.
    torch.set_num_threads(1)
    rows = []
    train_target_array = train[TARGET].to_numpy(dtype="float32")
    validation_target_array = validation[TARGET].to_numpy(dtype="int64")

    for feature_set in ["baseline", "augmented"]:
        train_features_raw, _ = select_xy(train, feature_set)
        validation_features_raw, _ = select_xy(validation, feature_set)

        for log_conflict in [False, True]:
            preprocessor = build_preprocessor(feature_set, "scaled", log_conflict)
            train_features = torch.from_numpy(
                preprocessor.fit_transform(train_features_raw).astype("float32")
            )
            validation_features = torch.from_numpy(
                preprocessor.transform(validation_features_raw).astype("float32")
            )
            train_target = torch.from_numpy(train_target_array).view(-1, 1)
            validation_target = torch.from_numpy(
                validation_target_array.astype("float32")
            ).view(-1, 1)

            for architecture in [(32,), (64, 32), (128, 64)]:
                for learning_rate in [0.001, 0.0001]:
                    random.seed(RANDOM_STATE)
                    np.random.seed(RANDOM_STATE)
                    torch.manual_seed(RANDOM_STATE)

                    model = TorchMLP(train_features.shape[1], architecture)
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=learning_rate,
                    )
                    loss_function = torch.nn.BCEWithLogitsLoss()
                    best_loss = float("inf")
                    best_state = None
                    best_epoch = 0
                    bad_epochs = 0
                    generator = torch.Generator().manual_seed(RANDOM_STATE)

                    for epoch in range(1, 101):
                        model.train()
                        permutation = torch.randperm(
                            len(train_features),
                            generator=generator,
                        )
                        for start in range(0, len(train_features), 1024):
                            batch_indices = permutation[start : start + 1024]
                            optimizer.zero_grad()
                            loss = loss_function(
                                model(train_features[batch_indices]),
                                train_target[batch_indices],
                            )
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            validation_loss = float(
                                loss_function(
                                    model(validation_features),
                                    validation_target,
                                ).item()
                            )

                        if validation_loss < best_loss - 1e-4:
                            best_loss = validation_loss
                            best_state = copy.deepcopy(model.state_dict())
                            best_epoch = epoch
                            bad_epochs = 0
                        else:
                            bad_epochs += 1

                        if bad_epochs >= 8:
                            break

                    model.load_state_dict(best_state)
                    with torch.no_grad():
                        probabilities = (
                            torch.sigmoid(model(validation_features))
                            .numpy()
                            .reshape(-1)
                        )
                    predictions = (probabilities >= 0.5).astype(int)
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "log_conflict": log_conflict,
                            "architecture": "-".join(map(str, architecture)),
                            "learning_rate": learning_rate,
                            "best_epoch": best_epoch,
                            **binary_metrics(
                                validation_target_array,
                                predictions,
                            ),
                        }
                    )

    return _sort(pd.DataFrame(rows))


def rerun_all(train, validation, output_dir):
    """Rerun every validation grid without accessing the held-out test set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grids = {
        "logistic_regression": logistic_grid(train, validation),
        "knn": knn_grid(train, validation),
        "decision_tree": decision_tree_grid(train, validation),
        "adaboost": adaboost_grid(train, validation),
        "mlp": mlp_grid(train, validation),
    }

    winners = []
    for model_family, grid in grids.items():
        grid.to_csv(
            output_dir / f"{model_family}_validation_grid.csv",
            index=False,
        )
        for feature_set in ["baseline", "augmented"]:
            selected = grid[grid.feature_set == feature_set].iloc[0].to_dict()
            winners.append({"model_family": model_family, **selected})

    pd.DataFrame(winners).to_csv(
        output_dir / "validation_selected_models.csv",
        index=False,
    )
    return grids
