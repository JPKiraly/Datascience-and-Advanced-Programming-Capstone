# Commodity-Price Dynamics and Conflict Prediction in Sub-Saharan Africa

The project examines the following question:

**Can global commodity-price changes and short-run volatility improve the one-month-ahead prediction of high-conflict months in Sub-Saharan Africa beyond information already contained in recent conflict history?**

The analysis covers 48 Sub-Saharan African countries from 2000 to 2025. It combines UCDP conflict data with monthly World Bank Pink Sheet commodity indices. The task at hand is a classification problem: information available in month t is used to predict whether month t+1 is a high-conflict month.

## Setup

### Conda

```bash
conda env create -f environment.yml
conda activate commodity-conflict-capstone
python main.py
```

### pip

```bash
python -m venv .venv
pip install -r requirements.txt
python main.py
```

On Windows, the command can also be:

```bash
py main.py
```

## Running the project

The main analysis is run with:

```bash
python main.py
```

The script trains the final baseline and augmented versions of the five models, evaluates them on the 2022-2025 test period, and saves the results in the `results/` folder.

## Rebuilding the data

The repository includes the smaller data files needed to reconstruct the modeling table. To rebuild it, run:

```bash
python main.py --rebuild-data
```

The full UCDP dataset is too large to include in the repository. To rebuild the smaller source files from the original UCDP and World Bank files, place these two files in `data/raw/`:

```text
GEDEvent_v26_1.csv
CMO-Historical-Data-Monthly.xlsx
```

and run:

```bash
python main.py --from-raw
```

Download pages are listed in `data/raw/README.md`.

## Data construction

UCDP events are aggregated to country-month observations. Events with `date_prec <= 4` are kept for the monthly analysis as they represent a precise assignment of the event to a specific month. When an event covers dates in two calendar months, it is assigned to the month containing the midpoint of its start and end dates.

Five World Bank commodity indices are used: Energy, Food, Fertilizers, Metals & Minerals, and Precious Metals. For each index, I calculated the monthly percentage change and the standard deviation of the last three monthly changes.

The baseline predictors are country, month of the year, current and lagged conflict events and fatalities, and three-month conflict totals. The augmented specification adds the ten commodity variables.

For each country, a high-conflict threshold is calculated from the 2000-2018 training period:

```text
cutoff = max(2, floor(75th percentile) + 1)
```

A target value of 1 means that the following month reaches or exceeds this country-specific cutoff.

| Sample | Target period | Observations | High-conflict share |
|---|---|---:|---:|
| Training | Apr 2000-Dec 2018 | 10,800 | 8.58% |
| Validation | Jan 2019-Dec 2021 | 1,728 | 21.24% |
| Test | Jan 2022-Dec 2025 | 2,304 | 22.01% |

## Models and preprocessing

Five classifiers are compared:

- Logistic Regression
- k-Nearest Neighbors (kNN)
- Decision Tree
- AdaBoost
- Multilayer Perceptron (MLP)

The project also includes a majority benchmark and a persistence benchmark.

Country and month are converted with one-hot encoding. Logistic Regression, kNN and the MLP use standardized numerical variables. For these three models, raw and `log1p` conflict variables were compared during validation. Decision Tree and AdaBoost use the original numerical values.

The final model settings are stored in `src/models.py`. They were selected using the validation period before the test results were evaluated.

## Main results

| Model | Feature set | F1 | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|
| MLP | Baseline | 0.857 | 0.917 | 0.805 | 0.941 |
| Logistic Regression | Augmented | 0.855 | 0.924 | 0.795 | 0.941 |
| Persistence | Benchmark | 0.846 | 0.843 | 0.848 | 0.932 |
| kNN | Baseline | 0.844 | 0.935 | 0.769 | 0.938 |
| AdaBoost | Augmented | 0.832 | 0.879 | 0.789 | 0.930 |
| Decision Tree | Baseline | 0.802 | 0.876 | 0.740 | 0.920 |

Adding commodity variables changes test F1 as follows:

| Model | Change in F1 |
|---|---:|
| Logistic Regression | +0.003 |
| kNN | -0.045 |
| Decision Tree | 0.000 |
| AdaBoost | -0.012 |
| MLP | -0.004 |

The main result is therefore that recent conflict history (persistence) is a strong predictor, while the global commodity variables provide little additional improvement.

The small positive result for Logistic Regression also disappears in the two robustness checks reported in the paper as per a stricter 80th-percentile conflict target and a six-month volatility measure.

## Repository structure

```text
commodity_conflict_capstone/
├── README.md
├── PROPOSAL.md
├── environment.yml
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   └── evaluation.py
├── data/
│   ├── modeling_table.csv
│   └── raw/
│       ├── README.md
│       ├── ssa_countries.csv
│       ├── ucdp_ssa_2000_2025.csv
│       └── world_bank_commodity_indices_1999_10_2025_12.csv
├── results/
└── notebooks/
```

## Reproducibility checks

```bash
python main.py
python main.py --rebuild-data
```

## AI mention

ChatGPT was used to support code drafting and debugging, clarify theoretical questions alongside the course material, and help structure the data pipeline, model comparison, reproducibility steps, and advised on the planning for the overall project. It was also used, together with DeepL, to support translation from French to English and to review the clarity of some parts of the report.
