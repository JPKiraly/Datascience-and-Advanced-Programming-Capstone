# Project proposal

## Predicting Conflict Escalation from Commodity-Price Dynamics

This project asks if global commodity-price changes and short-run volatility improve the one-month-ahead prediction of high-conflict months in Sub-Saharan Africa beyond information contained in recent conflict history. Therefore, the objective of the project is predictive rather than causal and is formulated as a binary-classification problem at the country-month level.

The analysis combines monthly World Bank Pink Sheet commodity indices with the Uppsala Conflict Data Program Georeferenced Event Dataset (UCDP GED), covering 2000-2025. Conflict events and fatalities are aggregated by country and month. A country-specific high-conflict outcome is constructed from the historical distribution of monthly event counts using training-period data only. Predictors include recent conflict history, country and seasonal information, plus monthly changes and three-month volatility for Energy, Food, Fertilizers, Metals & Minerals, and Precious Metals.

Five course-aligned classifiers are compared: Logistic Regression, k-nearest neighbors, Decision Tree, AdaBoost, and a feed-forward multilayer perceptron (MLP). The dataset is divided chronologically into training, validation, and held-out test periods to avoid bias. Model selection is based on validation F1, with precision, recall, accuracy also reported.

The central empirical comparison is therefore between a conflict-history baseline and the same model augmented with commodity variables. Success is therefore not defined by finding a positive commodity effect, but by assessing whether commodity dynamics provide stable incremental predictive information out of sample.
