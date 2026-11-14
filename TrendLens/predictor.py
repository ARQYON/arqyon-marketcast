"""Run this file to download data, train or load models, and show a forecast."""

import argparse
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from features import make_features, make_training_rows
from predict import load_bundle, make_forecast, plot_backtest, print_results, save_bundle
from train import train_models


# Change only this line to try a different Yahoo Finance symbol.
# Examples: "BTC-USD", "ETH-USD", "AAPL", "NVDA", "EURUSD=X", "GC=F".
TICKER = "BTC-USD"

HISTORY_PERIOD = "5y"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DISCLAIMER = "Educational only, not financial advice"


def download_history(ticker):
    """Fetch daily bars and remove the current bar if it could still change."""
    print(f"Downloading daily history for {ticker} ...")
    asset = yf.Ticker(ticker)
    try:
        # Close is Yahoo's quoted close series. Disable dividend adjustments so
        # prices stay on the quote scale. Yahoo may normalize past stock splits.
        yf.config.debug.hide_exceptions = False
        history = asset.history(
            period=HISTORY_PERIOD, interval="1d", auto_adjust=False,
            actions=False, timeout=15,
        )
    except Exception as error:
        # Yahoo can reject an invalid ticker, rate-limit requests, or be offline.
        raise RuntimeError(f"Yahoo Finance download failed for {ticker}: {error}") from error

    if history.empty or "Close" not in history:
        raise ValueError(f"No daily close history for {ticker}. Check the Yahoo Finance symbol.")
    if history.index.tz is None:
        raise ValueError("Yahoo Finance did not supply a market time zone for the daily bars.")

    try:
        metadata = asset.get_history_metadata()
        currency = metadata.get("currency") or "quote currency"
        session_end = metadata.get("currentTradingPeriod", {}).get("regular", {}).get("end")
    except Exception:
        # Without reliable session metadata, keep only earlier dates.
        currency = "quote currency"
        session_end = None

    # Yahoo timestamps daily rows in the market's own time zone. If Yahoo
    # supplies a regular session end for today, allow five minutes for the
    # daily close to settle. For 24-hour bars, wait for the next market date.
    market_now = pd.Timestamp(datetime.now(history.index.tz))
    market_today = market_now.date()
    today_is_complete = False
    if isinstance(session_end, pd.Timestamp) and session_end.tzinfo is not None:
        session_end = session_end.tz_convert(history.index.tz)
        today_is_complete = (
            session_end.date() == market_today
            and market_now >= session_end + pd.Timedelta(minutes=5)
        )
    cutoff = pd.Timestamp(market_today) + pd.Timedelta(days=int(today_is_complete))
    history = history.copy()
    history.index = history.index.tz_localize(None).normalize()
    history = history.loc[history.index < cutoff].sort_index()
    history = history.loc[~history.index.duplicated(keep="last")].copy()

    # Missing closes cannot be used for returns. Never fill them from the future.
    close = pd.to_numeric(history["Close"], errors="coerce")
    valid = close.gt(0) & close.lt(float("inf"))
    history = history.loc[valid].copy()
    history["Close"] = close.loc[valid].astype(float)
    if history.empty:
        raise ValueError(f"No usable completed daily close is available for {ticker}.")

    return history, currency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true", help="Refit both models with current data.")
    parser.add_argument("--no-show", action="store_true", help="Save the PNG without opening a window.")
    args = parser.parse_args()

    print(DISCLAIMER)
    ticker = TICKER.strip().upper()
    if not ticker or any(char.isspace() for char in ticker) or "," in ticker:
        raise ValueError("Set TICKER to one Yahoo Finance symbol, such as BTC-USD or AAPL.")

    # Keep output paths Windows-safe and avoid collisions between similar symbols.
    folder = re.sub(r"[^A-Z0-9_-]", "_", ticker)
    folder += "_" + hashlib.sha256(ticker.encode()).hexdigest()[:8]
    output = OUTPUT_DIR / folder
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "models.joblib"
    chart_path = output / "predicted_vs_actual.png"
    history, currency = download_history(ticker)
    features = make_features(history)
    latest = features.iloc[[-1]]
    if latest.isna().any().any():
        raise ValueError("Insufficient usable history for the latest daily features.")
    bundle = None if args.retrain else load_bundle(model_path, ticker, list(features.columns))
    if bundle is None:
        rows = make_training_rows(history["Close"], features)
        bundle = train_models(rows, ticker)
        save_bundle(model_path, bundle)
    else:
        print("Loaded saved models without retraining.")
    current_close = float(history["Close"].iloc[-1])
    forecast = make_forecast(bundle, latest, current_close)
    print_results(bundle, forecast, current_close, history.index[-1], currency)
    plot_backtest(bundle, chart_path, currency, show=not args.no_show)
    print(f"Models: {model_path}\nChart: {chart_path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
