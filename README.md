# Commodity-Price Dynamics and One-Month-Ahead Conflict Prediction in Sub-Saharan Africa

Reproducible capstone project for **Introduction to Data Science and Advanced Programming**.

The repository root also contains the required project proposal (`PROPOSAL.md`) and the report (`REPORT.pdf`). The editable Word version of the report is kept under `docs/REPORT_EDITABLE.docx` for final personalization before submission.

## Research question

> **Can global commodity-price changes and short-run volatility improve the one-month-ahead prediction of high-conflict months in Sub-Saharan Africa beyond information contained in recent conflict history?**

The project is explicitly **predictive, not causal**. The unit of observation is the country-month: predictors observed in month `t` classify whether month `t+1` is a country-specific high-conflict month.

## Main result

Recent conflict history carries most of the predictive information. Commodity dynamics provide only a weak and specification-sensitive incremental signal. On the held-out 2022-2025 test period, the frozen commodity-augmented Logistic Regression improves F1 by about **+0.003** relative to the same Logistic model without commodity variables. The best frozen family-level test model is the conflict-history MLP (**F1 = 0.857**), closely followed by commodity-augmented Logistic Regression (**F1 = 0.855**).

Robustness checks weaken the commodity result: the Logistic commodity increment becomes slightly negative under a stricter q80 target and a six-month commodity-volatility window. The final interpretation is therefore that commodity-price dynamics add, at most, modest predictive information beyond the much stronger signal in recent conflict history.

## Quick reproduction

Create the environment, then run the project from the repository root.

### Conda

```bash
conda env create -f environment.yml
conda activate commodity-conflict-capstone
python main.py
```

### pip / virtual environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

`python main.py`:

1. loads the frozen country-month modeling table;
2. combines training and validation after model selection has already been frozen;
3. refits the frozen baseline and commodity-augmented versions of Logistic Regression, kNN, Decision Tree, AdaBoost and the MLP;
4. evaluates them on the held-out 2022-2025 test sample;
5. reproduces the stored final metrics; and
6. writes regenerated tables and a comparison figure to `results/generated/`.

The script checks the reproduced metrics against `results/reference/final_test_all_models.csv` and exits with an error if they differ materially.

## Rebuild the modeling table

The repository contains compact, versioned source snapshots sufficient to recreate the full modeling table:

```bash
python main.py --rebuild-data
```

This reconstructs the balanced country-month conflict panel, commodity features, conflict-history predictors, target, and chronological split. The resulting 14,832-row table is checked column-by-column against the frozen canonical table.

## Rebuild from the original raw files

The full global UCDP file is about 262 MB and therefore cannot be stored as a normal GitHub file. To reproduce the source snapshots themselves, place these files under `data/raw/`:

```text
data/raw/GEDEvent_v26_1.csv
data/raw/CMO-Historical-Data-Monthly.xlsx
```

Then run:

```bash
python main.py --from-raw
```

The expected World Bank workbook is the monthly historical Pink Sheet workbook used in the project (its header states **Updated on August 04, 2026**). The UCDP source is **GED Global v26.1**.

Official source pages:

- UCDP dataset downloads: https://ucdp.uu.se/downloads/
- World Bank Commodity Markets / Pink Sheet: https://www.worldbank.org/en/research/commodity-markets

## Optional: rerun validation model selection

The final test results do **not** depend on rerunning hyperparameter selection, because the model choices are frozen and stored. For a full validation-grid rerun:

```bash
python main.py --rerun-selection
```

This is substantially slower than the default command, especially for the neural-network grid. The selection code receives only the training and validation partitions; test rows are never passed to it.

## Data construction

### Country universe

The analysis uses a fixed balanced panel of **48 sovereign Sub-Saharan African countries**. Countries without UCDP events remain in the panel with explicit zero-event months.

### UCDP events

UCDP events with `date_prec <= 4` are eligible for the primary monthly allocation. When an event spans multiple dates, the assigned month is the month containing the midpoint between `date_start` and `date_end`. Events with `date_prec = 5` remain in the versioned source snapshot for auditability but are excluded from primary monthly allocation because their month is not observed precisely.

### Commodity variables

Five World Bank monthly commodity indices are used:

- Energy
- Food
- Fertilizers
- Metals & Minerals
- Precious Metals

For each index:

```text
monthly_change_t = 100 * (Index_t / Index_(t-1) - 1)
3m_volatility_t = sample SD(change_t, change_(t-1), change_(t-2))
```

No clipping, winsorization, smoothing or interpolation is applied to the primary commodity signals.

### Conflict-history predictors

The baseline uses:

- country;
- month-of-year seasonality;
- current-month events and fatalities;
- one-month-lagged events and fatalities;
- three-month event and fatality sums.

The augmented feature set adds the ten commodity change/volatility variables.

## Target construction

For each country `c`, the primary threshold is estimated using **January 2000-December 2018 outcomes only**:

```text
cutoff_c = max(2, floor(Q75_c) + 1)
```

The target is 1 when next month's UCDP event count is at least the fixed country cutoff.

The chronological target split is:

| Partition | Target months | Rows | Positive rate |
|---|---|---:|---:|
| Train | Apr 2000-Dec 2018 | 10,800 | 8.58% |
| Validation | Jan 2019-Dec 2021 | 1,728 | 21.24% |
| Test | Jan 2022-Dec 2025 | 2,304 | 22.01% |

The test sample remained untouched until the model and feature-set choices were frozen.

## Models and preprocessing

Five classifiers are implemented:

1. Logistic Regression
2. k-Nearest Neighbors
3. Decision Tree
4. AdaBoost with decision stumps
5. Feed-forward MLP

A majority classifier and a one-month conflict-persistence rule provide benchmarks.

Country and month are one-hot encoded. Logistic Regression, kNN and the MLP use numeric scaling fit on the relevant training sample only. Validation selected `log1p` conflict counts for the final Logistic and baseline MLP specifications. Tree-based models use raw numeric predictors. The MLP is implemented in **PyTorch**, because that was the reproducible neural-network runtime available during the final experiment.

Primary model selection uses **validation F1**, with precision, recall, accuracy and confusion matrices reported alongside it.

## Final held-out test results

Frozen family winners:

| Model | Frozen feature set | Test F1 | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|
| MLP | Baseline | **0.857** | 0.917 | 0.805 | 0.941 |
| Logistic Regression | Augmented | **0.855** | 0.924 | 0.795 | 0.941 |
| Persistence | Benchmark | 0.846 | 0.843 | 0.848 | 0.932 |
| kNN | Baseline | 0.844 | 0.935 | 0.769 | 0.938 |
| AdaBoost | Augmented | 0.832 | 0.879 | 0.789 | 0.930 |
| Decision Tree | Baseline | 0.802 | 0.876 | 0.740 | 0.920 |

Commodity test-F1 increments (augmented minus baseline):

| Model | ΔF1 |
|---|---:|
| Logistic Regression | **+0.0030** |
| kNN | -0.0446 |
| Decision Tree | 0.0000 |
| AdaBoost | -0.0123 |
| MLP | -0.0043 |

## Robustness

For frozen Logistic hyperparameters:

| Specification | Baseline F1 | Augmented F1 | Commodity ΔF1 |
|---|---:|---:|---:|
| Primary q75 target + 3m volatility | 0.852 | 0.855 | **+0.0030** |
| Alternative q80 target + 3m volatility | 0.858 | 0.851 | **-0.0064** |
| Primary q75 target + 6m volatility | 0.851 | 0.849 | **-0.0021** |

These results support a weak and specification-sensitive commodity signal rather than a robust improvement.

## Repository structure

```text
.
├── README.md
├── PROPOSAL.md
├── REPORT.pdf
├── SUBMISSION_CHECKLIST.md
├── main.py
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── pytest.ini
├── CITATION.cff
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_pipeline.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── model_selection.py
├── data/
│   ├── metadata/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── results/
│   ├── reference/
│   └── generated/
├── tests/
└── docs/
    └── REPORT_EDITABLE.docx
```

## Code quality

The Python source is organized into small modules under `src/` and uses descriptive function and variable names. Source files follow a consistent Black-compatible style (88-character line limit), with the formatting target documented in `pyproject.toml`. Comments and docstrings focus on methodological reasons—especially leakage prevention, temporal precision, frozen model selection, and deterministic evaluation—rather than narrating routine syntax.

Reproducibility is centralized through `RANDOM_STATE = 42`. Estimators that expose a `random_state` receive it explicitly; deterministic components such as kNN, `StandardScaler`, and one-hot encoding do not expose such a parameter. The PyTorch MLP seeds Python, NumPy, PyTorch, and its minibatch generator.

## Reproducibility checks

Run:

```bash
pytest
python main.py --rebuild-data
python main.py
```

In the packaged repository these checks pass. The rebuilt modeling table matches the frozen canonical table exactly, while the default final evaluation reproduces the stored test metrics within the documented numerical tolerance (`0.01`). In the final submission check, the maximum absolute metric difference was approximately `0.00206`, arising from the PyTorch MLP; deterministic scikit-learn results reproduce exactly or to ordinary floating-point precision.

## GitHub

The folder is prepared to be committed as a GitHub repository. The global UCDP raw file is explicitly excluded through `.gitignore`; the compact reproducibility snapshot is included. See `docs/GITHUB_SETUP.md` for the exact initialization and push commands.

## Data acknowledgements

UCDP GED should be cited using UCDP's version-specific citation and the foundational GED paper: Sundberg, Ralph, and Erik Melander (2013), *Introducing the UCDP Georeferenced Event Dataset*, Journal of Peace Research 50(4), 523-532.

Commodity data are from the World Bank **Commodity Price Data (The Pink Sheet)**.

## AI use

Generative AI was used as a programming, debugging, organization and writing assistant. It was not used to create the underlying observations or outcomes. See `docs/AI_USE.md`.
