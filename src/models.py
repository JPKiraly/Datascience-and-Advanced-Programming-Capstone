import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .data_loader import TARGET, build_preprocessor, select_xy
from .evaluation import binary_metrics

RANDOM_STATE = 42

PAIRED_CONFIGS = {
    "logistic_regression": {"baseline": {"log_conflict": True, "C": 0.1, "class_weight": None}, "augmented": {"log_conflict": True, "C": 0.1, "class_weight": None}},
    "knn": {"baseline": {"log_conflict": True, "n_neighbors": 21, "weights": "distance"}, "augmented": {"log_conflict": True, "n_neighbors": 21, "weights": "distance"}},
    "decision_tree": {"baseline": {"max_depth": 3, "min_samples_leaf": 20}, "augmented": {"max_depth": 3, "min_samples_leaf": 20}},
    "adaboost": {"baseline": {"n_estimators": 200, "learning_rate": 1.0}, "augmented": {"n_estimators": 200, "learning_rate": 0.5}},
    "mlp": {
        "baseline": {"log_conflict": True, "architecture": (64, 32), "learning_rate": 0.001, "batch_size": 1024, "final_epochs": 30},
        "augmented": {"log_conflict": False, "architecture": (32,), "learning_rate": 0.001, "batch_size": 1024, "final_epochs": 100},
    },
}

FAMILY_WINNERS = {"logistic_regression": "augmented", "knn": "baseline", "decision_tree": "baseline", "adaboost": "augmented", "mlp": "baseline"}


def _transform(train, test, feature_set, model_type, log_conflict=False):
    x_train_raw, y_train = select_xy(train, feature_set)
    x_test_raw, _ = select_xy(test, feature_set)
    preprocessor = build_preprocessor(feature_set, model_type, log_conflict)
    x_train = preprocessor.fit_transform(x_train_raw)
    x_test = preprocessor.transform(x_test_raw)
    return x_train, y_train.to_numpy(dtype=int), x_test


def _make_mlp(input_size, hidden_layers):
    layers = []
    previous = input_size
    for hidden in hidden_layers:
        layers.append(torch.nn.Linear(previous, hidden))
        layers.append(torch.nn.ReLU())
        previous = hidden
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


def _fit_mlp(train, test, feature_set, config):
    torch.set_num_threads(1)
    x_train, y_train, x_test = _transform(train, test, feature_set, "scaled", config["log_conflict"])
    x_train = torch.tensor(np.asarray(x_train, dtype="float32"))
    x_test = torch.tensor(np.asarray(x_test, dtype="float32"))
    y_train = torch.tensor(np.asarray(y_train, dtype="float32")).reshape(-1, 1)

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    model = _make_mlp(x_train.shape[1], config["architecture"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_function = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(RANDOM_STATE)

    for _ in range(config["final_epochs"]):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(x_train), config["batch_size"]):
            batch = order[start:start + config["batch_size"]]
            optimizer.zero_grad()
            loss = loss_function(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_test)).numpy().reshape(-1)
    return (probabilities >= 0.5).astype(int)


def fit_predict(train, test, model_family, feature_set, config):
    if model_family == "logistic_regression":
        x_train, y_train, x_test = _transform(train, test, feature_set, "scaled", config["log_conflict"])
        model = LogisticRegression(C=config["C"], solver="liblinear", class_weight=config["class_weight"], max_iter=2000, random_state=RANDOM_STATE)
    elif model_family == "knn":
        x_train, y_train, x_test = _transform(train, test, feature_set, "scaled", config["log_conflict"])
        model = KNeighborsClassifier(n_neighbors=config["n_neighbors"], weights=config["weights"], metric="minkowski", p=2, n_jobs=-1)
    elif model_family == "decision_tree":
        x_train, y_train, x_test = _transform(train, test, feature_set, "tree")
        model = DecisionTreeClassifier(criterion="entropy", max_depth=config["max_depth"], min_samples_leaf=config["min_samples_leaf"], random_state=RANDOM_STATE)
    elif model_family == "adaboost":
        x_train, y_train, x_test = _transform(train, test, feature_set, "tree")
        stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
        model = AdaBoostClassifier(estimator=stump, n_estimators=config["n_estimators"], learning_rate=config["learning_rate"], random_state=RANDOM_STATE)
    elif model_family == "mlp":
        return _fit_mlp(train, test, feature_set, config)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    model.fit(x_train, y_train)
    return model.predict(x_test)


def _sort_results(rows):
    return (pd.DataFrame(rows) .sort_values(["f1", "recall", "precision", "accuracy"], ascending=False) .reset_index(drop=True))


def logistic_validation(train, validation):
    rows = []
    for feature_set in ["baseline", "augmented"]:
        x_train_raw, y_train = select_xy(train, feature_set)
        x_val_raw, y_val = select_xy(validation, feature_set)
        for log_conflict in [False, True]:
            preprocessor = build_preprocessor(feature_set, "scaled", log_conflict)
            x_train = preprocessor.fit_transform(x_train_raw)
            x_val = preprocessor.transform(x_val_raw)
            for c_value in [0.01, 0.1, 1.0, 10.0]:
                for class_weight in [None, "balanced"]:
                    model = LogisticRegression(C=c_value, solver="liblinear", class_weight=class_weight, max_iter=2000, random_state=RANDOM_STATE)
                    predictions = model.fit(x_train, y_train).predict(x_val)
                    rows.append({"feature_set": feature_set, "log_conflict": log_conflict, "C": c_value, "class_weight": class_weight, **binary_metrics(y_val, predictions)})
    return _sort_results(rows)


def knn_validation(train, validation):
    rows = []
    for feature_set in ["baseline", "augmented"]:
        x_train_raw, y_train = select_xy(train, feature_set)
        x_val_raw, y_val = select_xy(validation, feature_set)
        for log_conflict in [False, True]:
            preprocessor = build_preprocessor(feature_set, "scaled", log_conflict)
            x_train = preprocessor.fit_transform(x_train_raw)
            x_val = preprocessor.transform(x_val_raw)
            for k in [3, 5, 7, 11, 21]:
                for weights in ["uniform", "distance"]:
                    model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric="minkowski", p=2, n_jobs=-1)
                    predictions = model.fit(x_train, y_train).predict(x_val)
                    rows.append({"feature_set": feature_set, "log_conflict": log_conflict, "n_neighbors": k, "weights": weights, **binary_metrics(y_val, predictions)})
    return _sort_results(rows)


def tree_validation(train, validation):
    rows = []
    for feature_set in ["baseline", "augmented"]:
        x_train_raw, y_train = select_xy(train, feature_set)
        x_val_raw, y_val = select_xy(validation, feature_set)
        preprocessor = build_preprocessor(feature_set, "tree")
        x_train = preprocessor.fit_transform(x_train_raw)
        x_val = preprocessor.transform(x_val_raw)
        for depth in [3, 5, 8, None]:
            for leaf in [1, 5, 10, 20]:
                model = DecisionTreeClassifier(criterion="entropy", max_depth=depth, min_samples_leaf=leaf, random_state=RANDOM_STATE)
                predictions = model.fit(x_train, y_train).predict(x_val)
                rows.append({"feature_set": feature_set, "max_depth": depth, "min_samples_leaf": leaf, **binary_metrics(y_val, predictions)})
    df = _sort_results(rows)
    return df.sort_values(["f1", "recall", "precision", "accuracy", "min_samples_leaf"], ascending=[False, False, False, False, False],).reset_index(drop=True)


def adaboost_validation(train, validation):
    rows = []
    for feature_set in ["baseline", "augmented"]:
        x_train_raw, y_train = select_xy(train, feature_set)
        x_val_raw, y_val = select_xy(validation, feature_set)
        preprocessor = build_preprocessor(feature_set, "tree")
        x_train = preprocessor.fit_transform(x_train_raw)
        x_val = preprocessor.transform(x_val_raw)
        for n_estimators in [50, 100, 200]:
            for learning_rate in [0.01, 0.1, 0.5, 1.0]:
                stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
                model = AdaBoostClassifier(estimator=stump, n_estimators=n_estimators, learning_rate=learning_rate, random_state=RANDOM_STATE)
                predictions = model.fit(x_train, y_train).predict(x_val)
                rows.append({"feature_set": feature_set, "n_estimators": n_estimators, "learning_rate": learning_rate, **binary_metrics(y_val, predictions)})
    return _sort_results(rows)


def mlp_validation(train, validation):
    torch.set_num_threads(1)
    rows = []
    y_train_array = train[TARGET].to_numpy(dtype="float32")
    y_val_array = validation[TARGET].to_numpy(dtype="int64")

    for feature_set in ["baseline", "augmented"]:
        x_train_raw, _ = select_xy(train, feature_set)
        x_val_raw, _ = select_xy(validation, feature_set)
        for log_conflict in [False, True]:
            preprocessor = build_preprocessor(feature_set, "scaled", log_conflict)
            x_train = torch.from_numpy(preprocessor.fit_transform(x_train_raw).astype("float32"))
            x_val = torch.from_numpy(preprocessor.transform(x_val_raw).astype("float32"))
            y_train = torch.from_numpy(y_train_array).view(-1, 1)
            y_val = torch.from_numpy(y_val_array.astype("float32")).view(-1, 1)

            for architecture in [(32,), (64, 32), (128, 64)]:
                for learning_rate in [0.001, 0.0001]:
                    random.seed(RANDOM_STATE)
                    np.random.seed(RANDOM_STATE)
                    torch.manual_seed(RANDOM_STATE)
                    model = _make_mlp(x_train.shape[1], architecture)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    loss_function = torch.nn.BCEWithLogitsLoss()
                    best_loss = float("inf")
                    best_state = None
                    best_epoch = 0
                    bad_epochs = 0
                    generator = torch.Generator().manual_seed(RANDOM_STATE)

                    for epoch in range(1, 101):
                        model.train()
                        order = torch.randperm(len(x_train), generator=generator)
                        for start in range(0, len(x_train), 1024):
                            batch = order[start:start + 1024]
                            optimizer.zero_grad()
                            loss = loss_function(model(x_train[batch]), y_train[batch])
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            val_loss = float(loss_function(model(x_val), y_val).item())

                        if val_loss < best_loss - 1e-4:
                            best_loss = val_loss
                            best_state = copy.deepcopy(model.state_dict())
                            best_epoch = epoch
                            bad_epochs = 0
                        else:
                            bad_epochs += 1

                        if bad_epochs >= 8:
                            break

                    model.load_state_dict(best_state)
                    with torch.no_grad():
                        predictions = (torch.sigmoid(model(x_val)).numpy().reshape(-1) >= 0.5).astype(int)
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "log_conflict": log_conflict,
                            "architecture": "-".join(map(str, architecture)),
                            "learning_rate": learning_rate,
                            "best_epoch": best_epoch,
                            **binary_metrics(y_val_array, predictions),
                        }
                    )

    return _sort_results(rows)
