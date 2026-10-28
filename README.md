# Defense Acquisition Valuation

Should an Italian company acquire a competitor in the defense sector at the current market price of **€49 per share**?

This project answers that question with time series analysis. It forecasts the target's quarterly free cash flow per share with a SARIMA model and turns the forecast into a 5-year Discounted Cash Flow (DCF) valuation. It then measures how the target's cash flows respond to macroeconomic and sector drivers with a SARIMAX model.

The project was developed as a university case study (Data Science for Economics and Business, Università Politecnica delle Marche).

## Business questions

1. **Valuation:** is the current market price consistent with the target's intrinsic value, so that the acquisition makes economic sense?
2. **Drivers:** which macroeconomic and sector variables explain the dynamics of the target's free cash flow?

Overpaying for an acquisition creates goodwill that future performance may not support. That can lead to impairments, weaker profitability and a high opportunity cost. Buying below intrinsic value, on the other hand, secures assets and market share on favorable terms.

## Data

| Source | Content | Frequency |
|---|---|---|
| `Defense Company Free Cash Flow.xlsx` | Free cash flow and free cash flow per share of the target | Quarterly, from 2006 (47 observations) |
| `variabili_macroeconomiche.xlsx` | US GDP growth (q/q %), geopolitical risk index, PPI for steel and titanium | Quarterly |
| `Brent Oil Futures Historical Data.csv` | Brent crude oil futures price | Daily, aggregated to quarterly mean |

> **Note:** the data files are not included in this repository.

## Methodology

### Question 1: SARIMA forecast and DCF valuation

1. **Exploratory analysis:** rolling mean and standard deviation (4 quarters) to inspect trend and changing variance. Seasonality checked with boxplots by quarter and a one-way ANOVA.
2. **Stationarity:** Augmented Dickey-Fuller test (lags selected by AIC) on the original series and after seasonal differencing (lag 4). Additive decomposition shows that seasonality is the main source of non-stationarity, so no regular differencing is needed (`d = 0`, `D = 1`).
3. **Model selection:** grid search over 36 SARIMA specifications compared by AIC and BIC.
4. **Final model:** **SARIMA(1,0,1)(1,1,0)[4]**. It was chosen as a trade-off between goodness of fit, interpretability and residual diagnostics (ACF, Ljung-Box), and it is estimated with robust (QML) standard errors.
5. **Forecast:** 20 quarters ahead with a 90% confidence interval, summed into 5 annual cash flows.
6. **DCF:** the annual flows are discounted at a **WACC of 11.91%**, and a terminal value is added using a **perpetual growth rate of 2%**. The DCF is computed for three scenarios (lower bound, mean, upper bound of the forecast) and floored at zero.

### Question 2: macroeconomic drivers (SARIMAX)

1. **Variables:** US GDP growth, geopolitical risk index, PPI for steel and titanium, and Brent oil (quarterly average).
2. **Stationarity:** ADF test on every series. Series that are not stationary are differenced, and oil is log-differenced, which approximates its quarterly percentage change. The target is differenced FCF per share.
3. **Exploratory relationships:** Pearson and Spearman correlations, lead/lag cross-correlations (0–6 quarters) and Granger causality tests (up to 4 lags), all on the transformed series to avoid spurious relationships.
4. **Final model:** **SARIMAX(0,0,2)(0,1,1)[4]** with standardized exogenous regressors and robust standard errors. PPI enters with a 3-quarter lag, as suggested by the exploratory tests.
5. **Diagnostics:** Ljung-Box, Jarque-Bera, Durbin-Watson, standardized residuals, Q-Q plot and residual ACF.

## Results

### Valuation

| Scenario | Value per share | vs. market price (€49) |
|---|---|---|
| Pessimistic | €0 | Overvalued |
| Central | between the two bounds | Overvalued |
| Optimistic | €45.27 | Overvalued |

The target appears **overvalued in every scenario**. This does not mean the company has no economic value. It means its ability to generate sustainable cash flows is weak.

- Annual FCF per share fell from **€2.80 in 2006** to a low of **−€0.87 in 2013**, with only a marginal recovery afterwards. The average was €0.56 and the median €0.35.
- Cash flows are **strongly seasonal**: on average negative or close to zero in Q1–Q3 (Q1 −€1.85, Q2 +€0.05, Q3 −€0.62) and sharply positive in Q4 (+€2.97). This is typical of the defense sector, with its long production cycles, few large contracts and dependence on public spending decisions.
- The decline follows the 2008 financial crisis and the defense spending cuts that came after it.

**Recommendation:** do not proceed at current conditions. The deal could become justifiable only if there are significant, measurable synergies, or if there is forward-looking information not yet reflected in historical data, such as new contracts.

### Macroeconomic drivers

| Variable | Effect on FCF per share | Economic reading |
|---|---|---|
| US GDP growth | ▲ Positive | Economic growth supports public and defense spending |
| Geopolitical risk | ▼ Negative | In the short run, uncertainty delays contract decisions and raises costs |
| Steel & titanium PPI (lag 3) | ▼ Negative | Fixed-price contracts pass higher input costs on to margins with a delay |
| Brent oil | ▼ Negative | Higher energy costs |

## Limitations

- **Small sample:** 47 quarterly observations limit the stability and significance of the estimates.
- **Sensitivity to assumptions:** small changes in WACC or `g` can move the DCF valuation considerably.
- **Residual issues:** the SARIMAX residuals are broadly acceptable, but they show some departures from normality, and the MA parameters are unstable.

## Repository structure

```
├── DEFINITIVO.py            # full pipeline for both questions
├── report.pdf               # final report (in Italian)
└── README.md
```

## Authors

- Daniel Galizia
- Chiara Paoloni
- Martina Leporati
- Matheus Soares Campos

Università Politecnica delle Marche, April 2026
