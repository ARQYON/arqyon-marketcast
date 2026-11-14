"""Persist the model bundle, display a forecast, and plot historical predictions."""

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib

from train import BUNDLE_VERSION, forecast_arrays


def load_bundle(path, ticker, feature_columns):
    """Load a previously trained bundle if it matches this ticker and schema."""
    if not path.exists():
        return None
    try:
        # joblib uses pickle internally. Only load a file you created locally.
        bundle = joblib.load(path)
    except Exception as error:
        print(f"Saved models could not be read ({error}). Rebuilding them.", file=sys.stderr)
        return None

    if (
        not isinstance(bundle, dict)
        or bundle.get("version") != BUNDLE_VERSION
        or bundle.get("ticker") != ticker
        or bundle.get("feature_columns") != feature_columns
    ):
        print("Saved models do not match the current ticker or feature schema. Rebuilding them.")
        return None
    return bundle


def save_bundle(path, bundle):
    """Write to a temporary file first so a failed write preserves the old copy."""
    # Close the temporary file before joblib opens it. Windows may lock an
    # open file, which would otherwise prevent the write or final rename.
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary_path = Path(file.name)
    try:
        joblib.dump(bundle, temporary_path, compress=3)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def make_forecast(bundle, latest_features, current_close):
    """Return a single forecast from the model family selected during validation."""
    pair = bundle["models"][bundle["winner"]]
    probabilities, prices = forecast_arrays(pair, latest_features, [current_close])
    probability_up = float(probabilities[0])
    probability_down = 1.0 - probability_up
    return {
        "price": float(prices[0]),
        "direction": "UP" if probability_up > 0.5 else "DOWN",
        "confidence": max(probability_up, probability_down),
        "probability_up": probability_up,
        "probability_down": probability_down,
    }


def print_results(bundle, forecast, current_close, latest_date, currency):
    """Show comparable test metrics and explain the live probabilities."""
    backtest = bundle["backtest"]
    holdout_start = backtest.index[0].date()
    holdout_end = backtest.index[-1].date()
    print(f"\nModel comparison | {bundle['ticker']}")
    print("                           Time-series CV          Historical holdout")
    print(f"{'Model':<18}{'Brier':>9}{'Accuracy':>11}{'Brier':>10}{'Price MAE':>12}")
    for name, scores in bundle["holdout_scores"].items():
        cv_brier = bundle["cv_scores"][name]["brier"]
        print(
            f"{name:<18}{cv_brier:>9.4f}{scores['accuracy']:>11.1%}"
            f"{scores['brier']:>10.4f}{scores['price_mae']:>12,.4f}"
        )
    print(f"Holdout target dates: {holdout_start} to {holdout_end} ({len(backtest)} sessions)")
    print(f"Forecast model: {bundle['winner']} (lowest CV Brier score)")
    print(f"Saved models trained through: {bundle['trained_through'].date()}")
    print("\nNext-session forecast")
    # Keep standard currency precision for larger quotes and enough digits
    # to distinguish small changes in forex rates and low-priced assets.
    decimals = 2 if current_close >= 10 else 6
    while round(current_close, decimals) == round(forecast["price"], decimals) and decimals < 12:
        decimals += 1
    print(f"Current price (daily close on {latest_date.date()}): {current_close:,.{decimals}f} {currency}")
    print(f"Estimated next closing price: {forecast['price']:,.{decimals}f} {currency}")
    print(f"Direction: {forecast['direction']}")
    print(f"Model confidence: {forecast['confidence']:.1%}")
    print(f"UP probability: {forecast['probability_up']:.1%}")
    print(f"DOWN probability: {forecast['probability_down']:.1%}")
    print("Confidence is the model's uncalibrated probability for its chosen direction.")
    print("UP means a higher close. DOWN includes an unchanged close.")


def plot_backtest(bundle, chart_path, currency, show):
    """Save a PNG and optionally display it in a desktop window."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recent = bundle["backtest"].tail(100)
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in ("Actual", "Random Forest", "XGBoost", "Baseline"):
        ax.plot(recent.index, recent[name], label=name, linewidth=1.4)
    ax.set(title=f"{bundle['ticker']} — historical holdout forecasts",
           xlabel="Target session", ylabel=f"Closing price ({currency})")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)
