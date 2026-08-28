"""Fit frozen model configurations and generate predictions."""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor, select_xy


def split_modeling_table(df):
    """Return the chronological train, validation, and held-out test partitions."""
    return (
        df.loc[df["target_split"] == "train"].copy(),
        df.loc[df["target_split"] == "validation"].copy(),
        df.loc[df["target_split"] == "test"].copy(),
    )


def _transform(
    train_df,
    test_df,
    feature_set: str,
    model_family: str,
    log_conflict: bool = False,
):
    """Fit preprocessing on training data only and transform train/test inputs."""
    train_features_raw, train_target = select_xy(train_df, feature_set)
    test_features_raw, _ = select_xy(test_df, feature_set)
    preprocessor = build_preprocessor(feature_set, model_family, log_conflict)
    train_features = preprocessor.fit_transform(train_features_raw)
    test_features = preprocessor.transform(test_features_raw)

    return train_features, train_target.to_numpy(dtype=int), test_features


class TorchMLP(torch.nn.Module):
    """Simple feed-forward network used in the frozen MLP specification."""

    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []
        previous_dim = input_dim
        for units in hidden_layers:
            layers.extend(
                [
                    torch.nn.Linear(previous_dim, units),
                    torch.nn.ReLU(),
                ]
            )
            previous_dim = units
        layers.append(torch.nn.Linear(previous_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, values):
        return self.net(values)


def _mlp_predict(train_df, test_df, feature_set: str, config: dict):
    """Refit the frozen MLP and predict without using test data for stopping."""
    # A single Torch thread reduces run-to-run variation across common CPU
    # environments without changing the frozen architecture or optimization.
    torch.set_num_threads(1)
    train_features, train_target, test_features = _transform(
        train_df,
        test_df,
        feature_set,
        "scaled",
        bool(config["log_conflict"]),
    )
    train_features = torch.from_numpy(
        np.asarray(train_features, dtype="float32")
    )
    test_features = torch.from_numpy(np.asarray(test_features, dtype="float32"))
    train_target = torch.from_numpy(
        np.asarray(train_target, dtype="float32")
    ).view(-1, 1)

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    model = TorchMLP(train_features.shape[1], tuple(config["architecture"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    batch_size = int(config["batch_size"])
    generator = torch.Generator().manual_seed(RANDOM_STATE)

    # The epoch count was selected before the final test and is therefore fixed
    # here; the held-out test never controls early stopping or model selection.
    for _ in range(int(config["final_epochs"])):
        model.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        for start in range(0, len(train_features), batch_size):
            batch_indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_function(
                model(train_features[batch_indices]),
                train_target[batch_indices],
            )
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probabilities = (
            torch.sigmoid(model(test_features)).cpu().numpy().reshape(-1)
        )
    return (probabilities >= 0.5).astype(int)


def fit_predict(
    train_df,
    test_df,
    model_family: str,
    feature_set: str,
    config: dict,
):
    """Fit one frozen model configuration and return binary predictions."""
    if model_family == "logistic_regression":
        train_features, train_target, test_features = _transform(
            train_df,
            test_df,
            feature_set,
            "scaled",
            config["log_conflict"],
        )
        model = LogisticRegression(
            C=float(config["C"]),
            solver="liblinear",
            class_weight=config.get("class_weight"),
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        return model.fit(train_features, train_target).predict(test_features)

    if model_family == "knn":
        train_features, train_target, test_features = _transform(
            train_df,
            test_df,
            feature_set,
            "scaled",
            config["log_conflict"],
        )
        model = KNeighborsClassifier(
            n_neighbors=int(config["n_neighbors"]),
            weights=config["weights"],
            metric="minkowski",
            p=2,
            n_jobs=-1,
        )
        return model.fit(train_features, train_target).predict(test_features)

    if model_family == "decision_tree":
        train_features, train_target, test_features = _transform(
            train_df,
            test_df,
            feature_set,
            "tree",
            False,
        )
        model = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=config["max_depth"],
            min_samples_leaf=int(config["min_samples_leaf"]),
            random_state=RANDOM_STATE,
        )
        return model.fit(train_features, train_target).predict(test_features)

    if model_family == "adaboost":
        train_features, train_target, test_features = _transform(
            train_df,
            test_df,
            feature_set,
            "tree",
            False,
        )
        stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
        model = AdaBoostClassifier(
            estimator=stump,
            n_estimators=int(config["n_estimators"]),
            learning_rate=float(config["learning_rate"]),
            random_state=RANDOM_STATE,
        )
        return model.fit(train_features, train_target).predict(test_features)

    if model_family == "mlp":
        return _mlp_predict(train_df, test_df, feature_set, config)

    raise ValueError(f"Unknown model_family: {model_family}")
