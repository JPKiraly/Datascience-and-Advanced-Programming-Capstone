# Methods and frozen analysis protocol

## Unit and horizon

The unit is country-month. Predictors at month `t` classify whether month `t+1` is a high-conflict month.

## Country universe

The balanced panel uses 48 sovereign Sub-Saharan African countries. Countries without recorded UCDP events remain in the panel with explicit zero-event months.

## UCDP monthly allocation

Events with `date_prec <= 4` are eligible for monthly allocation. If an event spans multiple dates, its month is assigned using the midpoint between `date_start` and `date_end`. Events with `date_prec = 5` are retained in the source snapshot but excluded from the primary monthly allocation because their month is not observed precisely.

## Commodity features

Five World Bank Pink Sheet indices are used: Energy, Food, Fertilizers, Metals & Minerals, and Precious Metals. For each index:

`monthly_change = 100 * (Index_t / Index_(t-1) - 1)`

`3m_volatility = sample standard deviation of monthly changes at t, t-1, t-2`

No winsorization, clipping, or smoothing is applied to the primary commodity features.

## Conflict-history predictors

The baseline contains current-month event/fatality counts, one-month lags, and three-month sums, plus country and month-of-year indicators.

## Primary target

For country `c`, compute the 75th percentile of monthly UCDP event counts using Jan 2000-Dec 2018 outcomes only. The discrete high-conflict cutoff is:

`cutoff_c = max(2, floor(Q75_c) + 1)`

The next-month target is 1 if the next month's event count is at least this cutoff.

## Chronological split

- Train outcomes: Apr 2000-Dec 2018.
- Validation outcomes: Jan 2019-Dec 2021.
- Test outcomes: Jan 2022-Dec 2025.

No random train/test split is used.

## Preprocessing

Country and month-of-year are one-hot encoded. Logistic Regression, kNN, and the MLP use training-fitted standardization. Raw versus `log1p` conflict counts for scale-sensitive models was selected using validation F1 only. Decision Tree and AdaBoost use raw numerical predictors.

## Test protocol

All model/feature-set/hyperparameter choices were frozen before the held-out test period was opened. For the final test, training and validation were combined, preprocessing was fit on this final training sample, each frozen model was refit once, and the test set was evaluated once. No post-test retuning is permitted.

## Main conclusion

Recent conflict history contains most of the predictive signal. Commodity variables yield a small positive F1 increment for Logistic Regression in the primary held-out test specification, but this increment does not generalize across model families and becomes negative under the q80-target and six-month-volatility robustness specifications. The commodity contribution is therefore weak and specification-sensitive.
