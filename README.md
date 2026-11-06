# Imputation Error Propagation in Time Series Forecasts

MSc Data Science dissertation, University of Sheffield (Distinction).

A benchmarking framework that measures how errors from imputing missing values propagate into downstream time series forecasts, using a bias-variance decomposition.

## Abstract

**Background**. Missing data is a key issue across many domains, and is widely categorised
into three mechanisms of missingness: Missing At Random (MAR), Missing Completely At
Random (MCAR) and Missing Not At Random (MNAR). The reconstruction, or imputation,
of this missing data risks propagating errors into downstream predictive tasks, such as
forecasting for time series data. However, current imputation benchmarking studies focus
on reconstruction accuracy, or basic downstream predictive performance, neglecting deep
error insight into the predictions.

**Aims**. This research aims to analyse how each of the missingness mechanisms, along with
missingness proportion affects reconstruction performance. This analysis is then extended
to examining how systematic error (bias) and instability (variance) are propagated through
a range of imputation methods into time series forecasts.

**Methods**. Using the Electricity Transformer (ETT) dataset, synthetic missingness was
repeatedly generated under each of the missingness mechanisms, before being imputed
with a range of imputation models, and reconstruction performance aggregated. These
reconstructed datasets were then passed to forecasting models, with forecasting errors
decomposed into bias and variance.

**Results**. Reconstruction accuracy declined with increased missingness proportions for all
competent imputers, with performance broadly reflective of model complexity; the neural
network based method reconstructed most accurately at low missingness but quickly
degraded as missingness proportions rose. MCAR was least affected by missingness
proportions, while MAR and MNAR suffered from skewed distributions of remaining values.
Downstream, the forecaster proved to dictate how errors are propagated into forecasts,
with the multivariate forecaster demonstrating much greater bias and variance compared
with the univariate, as it was exposed to reconstruction errors across all features. Critically,
the strongest reconstructor produced the least stable prediction.

**Conclusions**. Reconstruction accuracy is not necessarily reflective of forecasting quality.
Instead, imputers and forecasters should be regarded as a pair with consideration to
the type of errors induced, rather than reconstruction accuracy alone. Future work can
extend into synthetic temporal missingness patterns, additional forecasters, and generalise
findings to more datasets.

## Research questions

1. How do different missingness mechanisms (MAR, MCAR, MNAR) and missingness proportions affect reconstruction performance of time series imputation methods?
2. To what extent do different time series imputation techniques demonstrate differentbias-variance trade-offs within downstream forecasting tasks?

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
