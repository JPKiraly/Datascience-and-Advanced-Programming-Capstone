"""Data loading and preprocessing interface for the capstone project.

The substantive pipeline lives in :mod:`src.data_pipeline`.  This module keeps
an explicit ``data_loader.py`` entry in the repository structure requested for
the course while re-exporting the tested data-construction functions without
duplicating analytical logic.
"""

from src.data_pipeline import (
    assert_matches_canonical,
    build_commodity_features,
    build_conflict_panel,
    build_modeling_table,
    extract_ucdp_snapshot,
    extract_world_bank_snapshot,
)

__all__ = [
    "assert_matches_canonical",
    "build_commodity_features",
    "build_conflict_panel",
    "build_modeling_table",
    "extract_ucdp_snapshot",
    "extract_world_bank_snapshot",
]
