import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import TARGET, load_data, rebuild_data, split_data
from src.evaluation import binary_metrics, compare_reference, save_outputs
from src.models import FAMILY_WINNERS, PAIRED_CONFIGS, fit_predict

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def evaluate_models(df):
    train, validation, test = split_data(df)
    final_train = pd.concat([train, validation], ignore_index=True)
    y_test = test[TARGET].astype(int).to_numpy()
    rows = []

    majority_class = int(train[TARGET].value_counts().idxmax())
    majority_predictions = np.full(len(test), majority_class, dtype=int)
    rows.append({"model_family": "majority", "feature_set": "benchmark", **binary_metrics(y_test, majority_predictions)})

    persistence_predictions = (test["events_t"].astype(int) >= test["high_conflict_cutoff_events"].astype(int)).astype(int)
    rows.append({"model_family": "persistence", "feature_set": "benchmark", **binary_metrics(y_test, persistence_predictions)})

    for model_family, configs in PAIRED_CONFIGS.items():
        for feature_set in ["baseline", "augmented"]:
            predictions = fit_predict(final_train, test, model_family, feature_set, configs[feature_set])
            rows.append({"model_family": model_family, "feature_set": feature_set, **binary_metrics(y_test, predictions)})

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--from-raw", action="store_true")
    args = parser.parse_args()

    if args.from_raw:
        rebuilt = rebuild_data(DATA, from_raw=True)
        print(f"Rebuilt data: {len(rebuilt):,} rows")
    elif args.rebuild_data:
        rebuilt = rebuild_data(DATA)
        print(f"Rebuilt data: {len(rebuilt):,} rows")

    df = load_data(DATA / "modeling_table.csv")
    results = evaluate_models(df)
    save_outputs(results, FAMILY_WINNERS, PAIRED_CONFIGS, RESULTS)
    difference = compare_reference(results, RESULTS / "reference_metrics.csv")

    print("\nFinal held-out test results:\n")
    print(results[["model_family", "feature_set", "f1", "precision", "recall", "accuracy"]].to_string(index=False))
    print(f"\nReference check passed. Maximum difference: {difference:.6f}")
    print(f"Results saved in: {RESULTS}")


if __name__ == "__main__":
    main()
