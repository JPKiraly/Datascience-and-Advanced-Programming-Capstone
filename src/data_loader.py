from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "target_high_conflict"
CATEGORICAL_FEATURES = ["country_code", "month_of_year"]
CONFLICT_FEATURES = ["events_t", "fatalities_t", "events_lag1", "fatalities_lag1", "events_3m_sum", "fatalities_3m_sum"]
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
INDEX_COLUMNS = ["energy_index", "food_index", "fertilizers_index", "metals_minerals_index", "precious_metals_index"]


def load_data(path):
    return pd.read_csv(path)


def split_data(df):
    train = df[df["target_split"] == "train"].copy()
    validation = df[df["target_split"] == "validation"].copy()
    test = df[df["target_split"] == "test"].copy()
    return train, validation, test


def select_xy(df, feature_set):
    columns = BASELINE_FEATURES if feature_set == "baseline" else AUGMENTED_FEATURES
    return df[columns].copy(), df[TARGET].astype(int).copy()


def _log1p(x):
    return np.log1p(np.asarray(x, dtype=float))


def build_preprocessor(feature_set, model_type, log_conflict=False):
    categorical = Pipeline([("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))])

    if model_type == "scaled":
        conflict_steps = []
        if log_conflict:
            conflict_steps.append(("log1p", FunctionTransformer(_log1p, validate=False)))
        conflict_steps.append(("scale", StandardScaler()))
        transformers = [("categorical", categorical, CATEGORICAL_FEATURES), ("conflict", Pipeline(conflict_steps), CONFLICT_FEATURES)]
        if feature_set == "augmented":
            transformers.append(("commodity", StandardScaler(), COMMODITY_FEATURES))
    else:
        transformers = [("categorical", categorical, CATEGORICAL_FEATURES), ("conflict", "passthrough", CONFLICT_FEATURES)]
        if feature_set == "augmented":
            transformers.append(("commodity", "passthrough", COMMODITY_FEATURES))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def extract_ucdp_snapshot(raw_csv, countries_csv, output_csv):
    countries = pd.read_csv(countries_csv)
    mapping = countries.set_index("ucdp_country_name")[["country_name", "country_code"]].to_dict("index")
    wanted = set(mapping)
    columns = ["id", "year", "country", "type_of_violence", "event_clarity", "date_prec", "date_start", "date_end", "deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown", "best", "low", "high"]
    pieces = []

    for chunk in pd.read_csv(raw_csv, usecols=columns, chunksize=100000):
        chunk = chunk[chunk["year"].between(2000, 2025) & chunk["country"].isin(wanted)].copy()
        if chunk.empty:
            continue
        chunk["country_name"] = chunk["country"].map(lambda x: mapping[x]["country_name"])
        chunk["country_code"] = chunk["country"].map(lambda x: mapping[x]["country_code"])
        chunk["ucdp_country_name"] = chunk["country"]
        chunk["violence_type_label"] = chunk["type_of_violence"].map({1: "state-based violence", 2: "non-state violence", 3: "one-sided violence"})
        chunk["monthly_eligible"] = chunk["date_prec"] <= 4
        chunk = chunk.rename(columns={"id": "event_id", "best": "fatalities_best", "low": "fatalities_low", "high": "fatalities_high"})
        pieces.append(
            chunk[
                [
                    "event_id",
                    "year",
                    "country_name",
                    "country_code",
                    "ucdp_country_name",
                    "type_of_violence",
                    "violence_type_label",
                    "event_clarity",
                    "date_prec",
                    "date_start",
                    "date_end",
                    "deaths_a",
                    "deaths_b",
                    "deaths_civilians",
                    "deaths_unknown",
                    "fatalities_best",
                    "fatalities_low",
                    "fatalities_high",
                    "monthly_eligible",
                ]
            ]
        )

    snapshot = pd.concat(pieces, ignore_index=True)
    snapshot.to_csv(output_csv, index=False)
    return snapshot


def extract_world_bank_snapshot(raw_xlsx, output_csv):
    raw = pd.read_excel(raw_xlsx, sheet_name="Monthly Indices", header=None, skiprows=9)
    selected = raw.iloc[:, [0, 2, 6, 13, 14, 16]].copy()
    selected.columns = ["period", *INDEX_COLUMNS]
    selected = selected[selected["period"].astype(str).str.match(r"^\d{4}M\d{2}$", na=False)].copy()
    selected["date"] = pd.to_datetime(selected["period"].str.replace("M", "-", regex=False) + "-01")
    selected = selected[(selected["date"] >= "1999-10-01") & (selected["date"] <= "2025-12-01")].copy()
    selected["year"] = selected["date"].dt.year
    selected["month"] = selected["date"].dt.month
    selected["analysis_period"] = selected["date"] >= "2000-01-01"
    selected = selected[["date", "year", "month", *INDEX_COLUMNS, "analysis_period"]]
    selected.to_csv(output_csv, index=False, date_format="%Y-%m-%d")
    return selected


def build_conflict_panel(events_csv, countries_csv):
    events = pd.read_csv(events_csv, parse_dates=["date_start", "date_end"])
    countries = pd.read_csv(countries_csv)
    eligible = events[events["monthly_eligible"].astype(str).str.lower().isin({"true", "1"})].copy()
    midpoint = eligible["date_start"] + (eligible["date_end"] - eligible["date_start"]) / 2
    eligible["date"] = midpoint.dt.to_period("M").dt.to_timestamp()
    aggregated = eligible.groupby(["country_code", "date"], as_index=False).agg(event_count_total=("event_id", "count"), fatalities_best_total=("fatalities_best", "sum"))
    dates = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
    panel = pd.MultiIndex.from_product([countries["country_code"], dates], names=["country_code", "date"],).to_frame(index=False)
    panel = panel.merge(countries[["country_code", "country_name", "ucdp_country_name"]], on="country_code", how="left", validate="many_to_one")
    panel = panel.merge(aggregated, on=["country_code", "date"], how="left", validate="one_to_one")
    panel[["event_count_total", "fatalities_best_total"]] = panel[["event_count_total", "fatalities_best_total"]].fillna(0)
    panel["event_count_total"] = panel["event_count_total"].astype(int)
    return panel


def build_commodity_features(indices_csv):
    data = pd.read_csv(indices_csv, parse_dates=["date"])[["date", *INDEX_COLUMNS]].copy()

    for column in INDEX_COLUMNS:
        stem = column.removesuffix("_index")
        change = f"{stem}_change_pct"
        volatility = f"{stem}_volatility_3m_pct"
        data[change] = 100 * (data[column] / data[column].shift(1) - 1)
        data[volatility] = data[change].rolling(3).std(ddof=1)

    return data[(data["date"] >= "2000-01-01") & (data["date"] <= "2025-12-01")].copy()


def build_modeling_table(events_csv, indices_csv, countries_csv):
    conflict = build_conflict_panel(events_csv, countries_csv)
    commodities = build_commodity_features(indices_csv)
    df = conflict.merge(commodities, on="date", how="left", validate="many_to_one")
    df = df.sort_values(["country_code", "date"]).reset_index(drop=True)
    grouped = df.groupby("country_code", sort=False)

    df["events_t"] = df["event_count_total"]
    df["fatalities_t"] = df["fatalities_best_total"]
    df["events_lag1"] = grouped["event_count_total"].shift(1)
    df["fatalities_lag1"] = grouped["fatalities_best_total"].shift(1)
    df["events_3m_sum"] = (grouped["event_count_total"].rolling(3).sum().reset_index(level=0, drop=True))
    df["fatalities_3m_sum"] = (grouped["fatalities_best_total"].rolling(3).sum().reset_index(level=0, drop=True))
    df["month_of_year"] = df["date"].dt.month

    threshold_sample = df[(df["date"] >= "2000-01-01") & (df["date"] <= "2018-12-01")]
    q75 = threshold_sample.groupby("country_code")["event_count_total"].quantile(0.75)
    cutoffs = np.maximum(2, np.floor(q75).astype(int) + 1)
    df["train_event_q75"] = df["country_code"].map(q75)
    df["high_conflict_cutoff_events"] = df["country_code"].map(cutoffs).astype(int)
    df["target_events_t_plus_1"] = grouped["event_count_total"].shift(-1)
    df["target_fatalities_t_plus_1"] = grouped["fatalities_best_total"].shift(-1)
    df["target_month"] = df["date"] + pd.offsets.MonthBegin(1)
    df[TARGET] = (df["target_events_t_plus_1"] >= df["high_conflict_cutoff_events"]).astype(int)
    df["target_split"] = np.select([df["target_month"] <= pd.Timestamp("2018-12-01"), df["target_month"] <= pd.Timestamp("2021-12-01")], ["train", "validation"], default="test")
    df = df[(df["date"] >= "2000-03-01") & (df["date"] <= "2025-11-01")].copy()
    df["observation_id"] = df["country_code"] + "_" + df["date"].dt.strftime("%Y-%m")
    df["feature_month"] = df["date"].dt.strftime("%Y-%m-%d")
    df["target_month"] = df["target_month"].dt.strftime("%Y-%m-%d")

    columns = [
        "observation_id",
        "country_code",
        "country_name",
        "feature_month",
        "target_month",
        "target_split",
        "month_of_year",
        "events_t",
        "fatalities_t",
        "events_lag1",
        "fatalities_lag1",
        "events_3m_sum",
        "fatalities_3m_sum",
        *COMMODITY_FEATURES,
        TARGET,
        "target_events_t_plus_1",
        "target_fatalities_t_plus_1",
        "train_event_q75",
        "high_conflict_cutoff_events",
    ]
    return df[columns].reset_index(drop=True)


def check_rebuild(rebuilt, canonical_csv):
    canonical = pd.read_csv(canonical_csv)
    if list(rebuilt.columns) != list(canonical.columns) or rebuilt.shape != canonical.shape:
        raise AssertionError("Rebuilt data do not match the stored modeling table")

    for column in canonical.columns:
        if pd.api.types.is_numeric_dtype(canonical[column]):
            a = pd.to_numeric(canonical[column], errors="coerce").to_numpy(float)
            b = pd.to_numeric(rebuilt[column], errors="coerce").to_numpy(float)
            if not np.allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True):
                raise AssertionError(f"Mismatch in {column}")
        elif not np.array_equal(canonical[column].astype(str).to_numpy(), rebuilt[column].astype(str).to_numpy(),):
            raise AssertionError(f"Mismatch in {column}")

    return True


def rebuild_data(data_dir, from_raw=False):
    data_dir = Path(data_dir)
    raw = data_dir / "raw"
    countries = raw / "ssa_countries.csv"
    events = raw / "ucdp_ssa_2000_2025.csv"
    indices = raw / "world_bank_commodity_indices_1999_10_2025_12.csv"
    canonical = data_dir / "modeling_table.csv"

    if from_raw:
        full_ucdp = raw / "GEDEvent_v26_1.csv"
        pink_sheet = raw / "CMO-Historical-Data-Monthly.xlsx"
        if not full_ucdp.exists() or not pink_sheet.exists():
            raise FileNotFoundError("Place GEDEvent_v26_1.csv and CMO-Historical-Data-Monthly.xlsx in data/raw/")
        extract_ucdp_snapshot(full_ucdp, countries, events)
        extract_world_bank_snapshot(pink_sheet, indices)

    rebuilt = build_modeling_table(events, indices, countries)
    check_rebuild(rebuilt, canonical)
    rebuilt.to_csv(data_dir / "modeling_table_rebuilt.csv", index=False)
    return rebuilt
