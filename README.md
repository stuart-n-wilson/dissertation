# Imputation Error Propagation in Time Series Forecasts

MSc Data Science dissertation, University of Sheffield (Distinction).

A benchmarking framework that measures how errors from imputing missing values propagate into downstream time series forecasts, using a bias-variance decomposition.

## Research questions

1. How accurately do different imputation methods reconstruct missing values under synthetic missingness?
2. How do those imputation errors affect the bias and variance of downstream forecasts?

## Approach

- **Data:** ETTh1, a multivariate hourly sensor dataset (7 features).
- **Missingness:** synthetic missingness injected into the series so imputed values can be compared against the known truth.
- **Imputers:** mean, last observation carried forward (LOCF), Kalman smoothing, ARIMA, BRITS.
- **Forecasters:** Holt-Winters exponential smoothing (univariate) and XGBoost (multivariate, recursive).
- **Evaluation:** reconstruction error (MAE) and a bias-variance decomposition of forecast error for every imputer-forecaster pair.

## Key findings

- The best reconstruction method is not the best forecasting input. BRITS reconstructed best at 10% missingness but was the least stable of the competent forecasting inputs.
- XGBoost carried more bias and variance than Holt-Winters, because it is multivariate and recursive and so is exposed to imputed lags across all features.

## Repository structure

```
TODO: replace with your actual folders, e.g.
src/        imputation, forecasting and evaluation code
notebooks/  exploratory analysis
figures/    plots used in the write-up
```

## Full write-up

[MSc_Disseration.pdf](https://github.com/stuart-n-wilson/dissertation/blob/main/MSc_Dissertation.pdf)


## Author

Stuart Wilson · [LinkedIn](https://linkedin.com/in/stuart-n-wilson) · [Portfolio](https://stuart-n-wilson.github.io/)
