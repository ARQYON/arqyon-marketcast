"""Create causal daily features and the separate next-session training targets."""

import numpy as np
import pandas as pd


def make_features(history):
    """At row t, use only data that is available by the close of row t."""
    close = history["Close"]
    features = pd.DataFrame(index=close.index)

    # A rolling window ends at the current row. We express price-based
    # indicators relative to price so a long-term trend in the quote level
    # does not dominate the features.
    for days in (5, 10, 20, 50):
        average = close.rolling(days).mean()
        features[f"close_vs_sma_{days}"] = close / average - 1.0

    # RSI compares smoothed gains and losses over a 14-session span. Wilder's
    # smoothing uses alpha=1/14. Guard against division by zero on flat data.
    change = close.diff()
    avg_gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    total = avg_gain + avg_loss
    features["rsi_14"] = (100 * avg_gain / total).mask(total.eq(0), 50.0)

    # MACD is the difference between the fast and slow exponential averages.
    # A second exponential average of that difference gives the signal line.
    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    features["macd"] = macd / close
    features["macd_signal"] = signal / close
    features["macd_histogram"] = (macd - signal) / close

    # Bollinger Bands sit two standard deviations above and below a 20-session
    # average. Position and width summarize both bands in price-relative units.
    center = close.rolling(20).mean()
    deviation = close.rolling(20).std(ddof=0)
    lower = center - 2 * deviation
    upper = center + 2 * deviation
    width = upper - lower
    features["bollinger_position"] = ((close - lower) / width).mask(width.eq(0), 0.5)
    features["bollinger_width"] = width / center

    # Some feeds, especially spot forex, report zero volume or omit the column.
    # Use a neutral value and a flag for missing volume. The comparison average
    # is calculated from earlier reported volume only.
    volume = history.get("Volume", pd.Series(0.0, index=close.index))
    volume = pd.to_numeric(volume, errors="coerce")
    reported = volume.gt(0) & volume.lt(float("inf"))
    positive_volume = volume.where(reported)
    prior_volume_avg = positive_volume.shift(1).rolling(20, min_periods=5).mean()
    features["volume_reported"] = reported.astype(int)
    features["log_relative_volume"] = np.log(positive_volume / prior_volume_avg).fillna(0.0)

    # Today's return is already known after today's close. Positive shifts
    # below refer to earlier returns and can never read a future close.
    returns = np.log(close).diff()
    features["return_1"] = returns
    features["return_5"] = np.log(close).diff(5)
    features["volatility_20"] = returns.rolling(20).std()
    for lag in (1, 2, 3, 5):
        features[f"return_lag_{lag}"] = returns.shift(lag)

    # Keep warm-up rows here. Training drops them, while prediction validates
    # the very latest row so it cannot silently fall back to an older close.
    return features.replace([np.inf, -np.inf], np.nan)


def make_training_rows(close, features):
    """Attach targets for training. The latest row has no target yet."""
    rows = features.copy()
    next_close = close.shift(-1)
    next_log_return = np.log(next_close) - np.log(close)
    rows["current_close"] = close
    rows["target_close"] = next_close
    rows["target_date"] = close.index.to_series().shift(-1)

    # UP means strictly higher. DOWN groups lower and unchanged closes so the
    # app can keep the requested two-class output. Keep unknown targets null.
    rows["target_up"] = next_close.gt(close).where(next_close.notna())
    rows["target_magnitude"] = next_log_return.abs()
    rows = rows.dropna().copy()
    rows["target_up"] = rows["target_up"].astype(int)
    return rows
