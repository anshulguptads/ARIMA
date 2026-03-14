# 📈 ARIMA Forecasting Masterclass — Interactive Dashboard

An interactive Streamlit dashboard for understanding, implementing, and evaluating ARIMA time series forecasting models. Built for the **Global MBA (GMBA)** program at **SP Jain School of Global Management**.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## Overview

This dashboard is a single, self-contained application that covers ARIMA from theory to results — no switching between slides, notebooks, or documentation. Select a section from the sidebar and work through it at your own pace.

**Dataset:** International Airline Passengers (1949–1960) — loads automatically, no setup required.

---

## Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | **Home & Overview** | The 7-step ARIMA workflow, dataset preview, and key statistics |
| 2 | **ARIMA Theory** | AR, I, and MA components explained with formulas, simulated process charts, and the ACF/PACF decision guide |
| 3 | **Dataset Exploration** | Time series visualization, descriptive statistics, seasonal decomposition (additive/multiplicative), monthly distributions, year-over-year comparison |
| 4 | **Stationarity Testing** | Log transformation, ADF test results, differencing, interactive rolling statistics |
| 5 | **ACF & PACF Analysis** | Interactive autocorrelation plots with significance highlighting and parameter selection guidance |
| 6 | **Model Building** | Interactive p/d/q selection, model fitting with AIC/BIC metrics, automated 10-model comparison grid |
| 7 | **Diagnostics** | 4-panel residual analysis, Ljung-Box test, Shapiro-Wilk normality test |
| 8 | **Forecasting & Results** | Test-set forecast with 95% confidence intervals, MAE/RMSE/MAPE evaluation, adjustable future forecast horizon |
| 9 | **Summary & Cheat Sheet** | One-page reference — formulas, ACF/PACF pattern recognition table, workflow summary, limitations |

---

## Getting Started

### Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/arima-masterclass.git
cd arima-masterclass

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run arima_dashboard.py
```

Opens at `http://localhost:8501`.

### Access Online

If deployed on Streamlit Cloud, simply open the link shared with you — no installation needed.

---

## Repository Structure

```
arima-masterclass/
├── arima_dashboard.py    # Streamlit application
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Dashboard framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts |
| `statsmodels` | ARIMA modeling, statistical tests, ACF/PACF |
| `scikit-learn` | Forecast accuracy metrics |
| `scipy` | Normality testing, Q-Q plot |

---

## Dataset

The dashboard uses the **Box-Jenkins Airline Passengers** dataset — a widely referenced benchmark in time series analysis.

- **Period:** January 1949 – December 1960
- **Frequency:** Monthly (144 observations)
- **Source:** Auto-downloaded from [GitHub](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv)
- **Offline fallback:** Synthetic data with matching characteristics is generated if the download fails

The series contains an upward trend, 12-month seasonality, and increasing variance — covering stationarity testing, differencing, log transforms, and seasonal pattern analysis.

---

## Deployment on Streamlit Cloud

1. Push this repo to your GitHub account (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. Click **New app** → select repo → set main file to `arima_dashboard.py` → **Deploy**
4. Share the generated URL

> **Note:** Community Cloud apps sleep after a few days of inactivity. The first visit after that triggers a ~30-second wake-up.

---

## Author

**Dr. Anshul Gupta**  
Associate Professor & Area Head — Technology Management  
SP Jain School of Global Management

---

<p align="center">
  <em>Built with ❤️ for the Global MBA Program</em>
</p>
