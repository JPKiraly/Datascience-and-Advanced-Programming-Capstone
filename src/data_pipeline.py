"""Reconstruct the frozen modeling table from included source snapshots.

The repository stores compact, versioned source snapshots rather than the full
262 MB UCDP GED file. The extraction helpers can regenerate those snapshots
when the original source files are provided under ``data/raw/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INDEX_COLUMNS = [
    "energy_index",
    "food_index",
    "fertilizers_index",
    "metals_minerals_index",
    "precious_metals_index",
]


def extract_ucdp_snapshot(
    raw_csv: str | Path,
    countries_csv: str | Path,
    output_csv: str | Path,
):
    """Extract the project country/year subset from the full UCDP GED file."""
    countries = pd.read_csv(countries_csv)
    ucdp_to_metadata = countries.set_index("ucdp_country_name")[
        ["country_name", "country_code"]
    ].to_dict("index")
    wanted_countries = set(ucdp_to_metadata)
    columns_to_read = [
        "id",
        "year",
        "country",
        "type_of_violence",
        "event_clarity",
        "date_prec",
        "date_start",
        "date_end",
        "deaths_a",
        "deaths_b",
        "deaths_civilians",
        "deaths_unknown",
        "best",
        "low",
        "high",
    ]

    pieces = []
    for chunk in pd.read_csv(raw_csv, usecols=columns_to_read, chunksize=100_000):
        chunk = chunk[
            chunk["year"].between(2000, 2025)
            & chunk["country"].isin(wanted_countries)
        ].copy()
        if chunk.empty:
            continue

        chunk["country_name"] = chunk["country"].map(
            lambda country: ucdp_to_metadata[country]["country_name"]
        )
        chunk["country_code"] = chunk["country"].map(
            lambda country: ucdp_to_metadata[country]["country_code"]
        )
        chunk["ucdp_country_name"] = chunk["country"]
        chunk["violence_type_label"] = chunk["type_of_violence"].map(
            {
                1: "state-based violence",
                2: "non-state violence",
                3: "one-sided violence",
            }
        )

        # Monthly assignment is restricted to events whose UCDP date precision
        # supports a defensible month; less precise events stay outside the
        # primary monthly panel rather than being assigned artificial precision.
        chunk["monthly_eligible"] = chunk["date_prec"] <= 4
        chunk = chunk.rename(
            columns={
                "id": "event_id",
                "best": "fatalities_best",
                "low": "fatalities_low",
                "high": "fatalities_high",
            }
        )
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


def extract_world_bank_snapshot(
    raw_xlsx: str | Path,
    output_csv: str | Path,
):
    """Extract the five World Bank monthly aggregate commodity indices."""
    raw = pd.read_excel(
        raw_xlsx,
        sheet_name="Monthly Indices",
        header=None,
        skiprows=9,
    )
    selected = raw.iloc[:, [0, 2, 6, 13, 14, 16]].copy()
    selected.columns = ["period", *INDEX_COLUMNS]
    selected = selected[
        selected["period"].astype(str).str.match(r"^\d{4}M\d{2}$", na=False)
    ].copy()
    selected["date"] = pd.to_datetime(
        selected["period"].str.replace("M", "-", regex=False) + "-01"
    )
    selected = selected[
        (selected["date"] >= "1999-10-01")
        & (selected["date"] <= "2025-12-01")
    ].copy()
    selected["year"] = selected["date"].dt.year
    selected["month"] = selected["date"].dt.month
    selected["analysis_period"] = selected["date"] >= "2000-01-01"
    selected = selected[
        ["date", "year", "month", *INDEX_COLUMNS, "analysis_period"]
    ]
    selected.to_csv(output_csv, index=False, date_format="%Y-%m-%d")
    return selected


def build_conflict_panel(
    events_csv: str | Path,
    countries_csv: str | Path,
) -> pd.DataFrame:
    """Aggregate eligible UCDP events to a balanced country-month panel."""
    events = pd.read_csv(events_csv, parse_dates=["date_start", "date_end"])
    countries = pd.read_csv(countries_csv)
    eligible = events[
        events["monthly_eligible"].astype(str).str.lower().isin({"true", "1"})
    ].copy()

    # Midpoint assignment avoids mechanically assigning a multi-day event to
    # its start or end month when both dates are known with adequate precision.
    midpoint = eligible["date_start"] + (
        eligible["date_end"] - eligible["date_start"]
    ) / 2
    eligible["date"] = midpoint.dt.to_period("M").dt.to_timestamp()

    aggregated = eligible.groupby(
        ["country_code", "date"],
        as_index=False,
    ).agg(
        event_count_total=("event_id", "count"),
        fatalities_best_total=("fatalities_best", "sum"),
    )

    dates = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
    panel = pd.MultiIndex.from_product(
        [countries["country_code"], dates],
        names=["country_code", "date"],
    ).to_frame(index=False)
    panel = panel.merge(
        countries[["country_code", "country_name", "ucdp_country_name"]],
        on="country_code",
        how="left",
        validate="many_to_one",
    )
    panel = panel.merge(
        aggregated,
        on=["country_code", "date"],
        how="left",
        validate="one_to_one",
    )

    # Explicit zero rows preserve quiet country-months instead of dropping them
    # from the classification universe.
    panel[["event_count_total", "fatalities_best_total"]] = panel[
        ["event_count_total", "fatalities_best_total"]
    ].fillna(0)
    panel["event_count_total"] = panel["event_count_total"].astype(int)
    return panel


def build_commodity_features(indices_csv: str | Path) -> pd.DataFrame:
    """Create monthly percentage changes and three-month realized volatility."""
    commodity_data = pd.read_csv(indices_csv, parse_dates=["date"])[
        ["date", *INDEX_COLUMNS]
    ].copy()

    for index_column in INDEX_COLUMNS:
        stem = index_column.removesuffix("_index")
        change_column = f"{stem}_change_pct"
        volatility_column = f"{stem}_volatility_3m_pct"
        commodity_data[change_column] = 100.0 * (
            commodity_data[index_column] / commodity_data[index_column].shift(1)
            - 1.0
        )
        commodity_data[volatility_column] = commodity_data[change_column].rolling(
            3
        ).std(ddof=1)

    return commodity_data[
        (commodity_data["date"] >= "2000-01-01")
        & (commodity_data["date"] <= "2025-12-01")
    ].copy()


def build_modeling_table(
    events_csv: str | Path,
    indices_csv: str | Path,
    countries_csv: str | Path,
) -> pd.DataFrame:
    """Build the frozen country-month supervised-learning table."""
    conflict = build_conflict_panel(events_csv, countries_csv)
    commodities = build_commodity_features(indices_csv)
    modeling_table = conflict.merge(
        commodities,
        on="date",
        how="left",
        validate="many_to_one",
    )
    modeling_table = modeling_table.sort_values(
        ["country_code", "date"]
    ).reset_index(drop=True)
    grouped = modeling_table.groupby("country_code", sort=False)

    modeling_table["events_t"] = modeling_table["event_count_total"]
    modeling_table["fatalities_t"] = modeling_table["fatalities_best_total"]
    modeling_table["events_lag1"] = grouped["event_count_total"].shift(1)
    modeling_table["fatalities_lag1"] = grouped["fatalities_best_total"].shift(1)
    modeling_table["events_3m_sum"] = (
        grouped["event_count_total"]
        .rolling(3)
        .sum()
        .reset_index(level=0, drop=True)
    )
    modeling_table["fatalities_3m_sum"] = (
        grouped["fatalities_best_total"]
        .rolling(3)
        .sum()
        .reset_index(level=0, drop=True)
    )
    modeling_table["month_of_year"] = modeling_table["date"].dt.month

    # Estimate country-specific high-conflict cutoffs only from the original
    # training period so later outcomes cannot influence the target definition.
    threshold_sample = modeling_table[
        (modeling_table["date"] >= "2000-01-01")
        & (modeling_table["date"] <= "2018-12-01")
    ]
    q75 = threshold_sample.groupby("country_code")["event_count_total"].quantile(
        0.75
    )
    cutoffs = np.maximum(2, np.floor(q75).astype(int) + 1)
    modeling_table["train_event_q75"] = modeling_table["country_code"].map(q75)
    modeling_table["high_conflict_cutoff_events"] = modeling_table[
        "country_code"
    ].map(cutoffs).astype(int)

    modeling_table["target_events_t_plus_1"] = grouped["event_count_total"].shift(-1)
    modeling_table["target_fatalities_t_plus_1"] = grouped[
        "fatalities_best_total"
    ].shift(-1)
    modeling_table["target_month"] = modeling_table["date"] + pd.offsets.MonthBegin(
        1
    )
    modeling_table["target_high_conflict"] = (
        modeling_table["target_events_t_plus_1"]
        >= modeling_table["high_conflict_cutoff_events"]
    ).astype(int)

    # Chronological target-month splits preserve the forecasting direction and
    # prevent future observations from entering model selection.
    modeling_table["target_split"] = np.select(
        [
            modeling_table["target_month"] <= pd.Timestamp("2018-12-01"),
            modeling_table["target_month"] <= pd.Timestamp("2021-12-01"),
        ],
        ["train", "validation"],
        default="test",
    )

    modeling_table = modeling_table[
        (modeling_table["date"] >= "2000-03-01")
        & (modeling_table["date"] <= "2025-11-01")
    ].copy()
    modeling_table["observation_id"] = (
        modeling_table["country_code"]
        + "_"
        + modeling_table["date"].dt.strftime("%Y-%m")
    )
    modeling_table["feature_month"] = modeling_table["date"].dt.strftime(
        "%Y-%m-%d"
    )
    modeling_table["target_month"] = modeling_table["target_month"].dt.strftime(
        "%Y-%m-%d"
    )

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
        "target_high_conflict",
        "target_events_t_plus_1",
        "target_fatalities_t_plus_1",
        "train_event_q75",
        "high_conflict_cutoff_events",
    ]
    return modeling_table[columns].reset_index(drop=True)


def assert_matches_canonical(
    rebuilt: pd.DataFrame,
    canonical_csv: str | Path,
    atol: float = 1e-10,
) -> bool:
    """Verify that a rebuilt table reproduces the frozen canonical dataset."""
    canonical = pd.read_csv(canonical_csv)
    if list(rebuilt.columns) != list(canonical.columns):
        raise AssertionError("Column order does not match canonical table")
    if rebuilt.shape != canonical.shape:
        raise AssertionError(f"Shape mismatch: {rebuilt.shape} != {canonical.shape}")

    for column in canonical.columns:
        if pd.api.types.is_numeric_dtype(canonical[column]):
            canonical_values = pd.to_numeric(
                canonical[column], errors="coerce"
            ).to_numpy(float)
            rebuilt_values = pd.to_numeric(
                rebuilt[column], errors="coerce"
            ).to_numpy(float)
            if not np.allclose(
                canonical_values,
                rebuilt_values,
                rtol=1e-10,
                atol=atol,
                equal_nan=True,
            ):
                raise AssertionError(f"Numeric mismatch in {column}")
        else:
            canonical_values = canonical[column].astype(str).to_numpy()
            rebuilt_values = rebuilt[column].astype(str).to_numpy()
            if not np.array_equal(canonical_values, rebuilt_values):
                raise AssertionError(f"Text mismatch in {column}")

    return True
