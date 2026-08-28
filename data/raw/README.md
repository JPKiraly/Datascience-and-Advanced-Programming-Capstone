# Original raw sources

The submitted repository does **not** commit the full global UCDP GED file because it is about 262 MB and exceeds GitHub's normal 100 MB per-file limit.

To run `python main.py --from-raw`, place these exact files here:

- `GEDEvent_v26_1.csv` — UCDP Georeferenced Event Dataset (GED), Global v26.1.
- `CMO-Historical-Data-Monthly.xlsx` — World Bank Commodity Price Data (Pink Sheet), monthly historical workbook used for the project (workbook header: updated August 04, 2026).

The compact versioned extracts used for the submitted analysis are committed under `data/interim/`, so the default analysis and `--rebuild-data` mode do not require downloading the full raw files.
