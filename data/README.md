# Data directories

- `metadata/ssa_countries.csv`: fixed 48-country Sub-Saharan African universe and UCDP name aliases.
- `interim/ucdp_ssa_2000_2025.csv`: UCDP events already filtered to the fixed country universe and years 2000-2025. Events with `date_prec=5` remain in the snapshot for auditability; the primary monthly panel excludes them.
- `interim/world_bank_commodity_indices_1999_10_2025_12.csv`: the five World Bank monthly index series used, including Oct-Dec 1999 solely as lookback for early 2000 features.
- `processed/modeling_table.csv`: frozen country-month modeling table used for the main experiments.
- `raw/`: optional location for the original full UCDP and World Bank files.

The analysis is predictive, not causal. Commodity values are global monthly signals repeated across countries within a month, so the effective time variation in those variables is much smaller than the row count of the country-month panel.
