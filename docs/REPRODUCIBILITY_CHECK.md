# Reproducibility check performed after the final code-quality pass

The submission-ready repository was tested from the repository root after formatting, naming, comment, and documentation cleanup. No analytical logic, frozen hyperparameters, data, or reference results were changed during this pass.

Checks completed successfully:

- `python -m compileall -q .` completed without errors.
- `pytest -q` → **4 passed**.
- `python main.py --rebuild-data` reconstructed **14,832 rows** and matched the frozen canonical modeling table column-by-column.
- The same command completed the frozen final held-out test evaluation and passed the built-in reference check. The maximum absolute difference across accuracy, precision, recall, and F1 was approximately **0.00206**, below the documented tolerance of **0.01**. The small difference is confined to the PyTorch MLP and reflects platform-level numerical variation; deterministic scikit-learn results reproduce exactly or to ordinary floating-point precision.
- All Python source lines are at most 88 characters and the intended Black-compatible formatting target is recorded in `pyproject.toml`.

The optional full `--rerun-selection` mode is intentionally much slower and is not required for the standard submission check; the exact historical validation grids are preserved under `results/reference/model_selection/`.
