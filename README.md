# Arqyon MarketCast

**A growing collection of local machine-learning models for market forecasting.**

Arqyon MarketCast (`arqyon-marketcast`) is the home for market forecasting experiments. **TrendLens** is the first model in the collection. It estimates the next daily closing price and direction for cryptocurrencies, stocks, forex, and gold futures using Yahoo Finance data.

> **Educational only, not financial advice**

## Model catalog

| Model | Forecast | Method |
| --- | --- | --- |
| [TrendLens](#trendlens) | Next-session closing price, UP/DOWN direction, and probabilities | Random Forest and XGBoost, evaluated against a simple baseline |

Future models can be added to this catalog with their forecast horizon, data source, and evaluation results.

## TrendLens

TrendLens compares Random Forest and XGBoost on daily market data, runs locally on a CPU, and saves trained models for reuse. Change one `TICKER` setting to switch assets.

| Setup | Python | Data source | Compute |
| --- | --- | --- | --- |
| Windows 11 | 3.12 (64-bit) | Yahoo Finance via `yfinance` | CPU only, two training threads |

## Quick start on Windows 11

1. Install [64-bit Python 3.12](https://www.python.org/downloads/).
2. Open the folder containing `predictor.py` and `requirements.txt` in VS Code.
3. Open **Terminal → New Terminal**, select **PowerShell**, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predictor.py
```

The first run downloads five years of daily history for Bitcoin, compares and trains both model families, and opens the historical chart. Later runs for the same ticker load the saved models and fetch fresh data. You’ll need an internet connection for installation and data downloads. Yahoo Finance access through this project requires no API key.

## Switch assets

Change the `TICKER` line in [predictor.py](predictor.py), save the file, and run the same command again.

```python
TICKER = "AAPL"
```

| Asset | Yahoo Finance symbol |
| --- | --- |
| Bitcoin | `BTC-USD` |
| Ether | `ETH-USD` |
| Apple | `AAPL` |
| NVIDIA | `NVDA` |
| EUR/USD | `EURUSD=X` |
| Gold futures | `GC=F` |

Other Yahoo Finance symbols work when they have enough usable history and both directions appear in training. Each ticker gets its own saved models.

## Read the forecast

The terminal prints the latest usable daily close, the estimated next closing price, the predicted direction, and both probabilities. **UP** means a higher close. **DOWN** includes an unchanged close. Confidence is the larger of the two uncalibrated model probabilities. A 70% reading alone does not establish a 70% success rate.

The forecast covers the next session in the daily data, which can be after a weekend or holiday. The program uses Yahoo’s session metadata to avoid unfinished daily bars. The input close can lag a live quote.

| Output | What it contains |
| --- | --- |
| `output/<ticker-folder>/models.joblib` | Both fitted model families and their evaluation results |
| `output/<ticker-folder>/predicted_vs_actual.png` | Actual closes, both models’ historical estimates, and the last-close baseline |

The chart plots forecasts on the dates they target and shows up to 100 historical holdout sessions. It updates when you retrain. The terminal prints the exact output paths.

## How the backtest works

1. **Build features.** Moving averages, RSI, MACD, Bollinger Bands, relative volume, volatility, and lagged returns use data available by the input session’s close.
2. **Compare models in time order.** Three expanding-window folds score both families. A one-session gap keeps the last training target before the first validation input.
3. **Score a separate holdout.** The latest 20% of labeled rows are held out from model selection. The report shows direction accuracy, Brier score for probability error, and mean absolute closing-price error.
4. **Save the fitted models.** The family with the lowest average validation Brier score supplies the forecast. Both families are refit on all labeled rows and saved. Each family pairs a direction classifier with a regressor that estimates the size of the move.

The baseline uses the historical UP rate for probability and the latest known close for its price estimate. In the initial tests, that baseline had lower holdout probability error and price error than both model families across all six example tickers.

## Commands and troubleshooting

| Task | PowerShell command |
| --- | --- |
| Run normally | ` .\.venv\Scripts\python.exe predictor.py ` |
| Save the chart without opening a window | ` .\.venv\Scripts\python.exe predictor.py --no-show ` |
| Retrain on the latest available history | ` .\.venv\Scripts\python.exe predictor.py --retrain ` |

- **Python command unavailable.** Reopen the terminal after installing Python and check `py -3.12 --version`.
- **`requirements.txt` missing.** Open the extracted folder that directly contains the project files.
- **Yahoo download fails.** Check the exact symbol on Yahoo Finance and retry when the connection is available. For short histories, choose a symbol with at least 250 usable training rows after indicator warm-up.
- **Chart window fails to open.** Run with `--no-show` and open the saved PNG from File Explorer.

## TrendLens source files

| File | Role |
| --- | --- |
| [requirements.txt](requirements.txt) | Exact direct dependency versions |
| [predictor.py](predictor.py) | Ticker setting, data download, and main entry point |
| [features.py](features.py) | Technical indicators and next-session training targets |
| [train.py](train.py) | Model fitting, chronological evaluation, and baselines |
| [predict.py](predict.py) | Model persistence, forecast output, and chart generation |
