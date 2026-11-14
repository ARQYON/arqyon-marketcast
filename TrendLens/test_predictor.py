"""Deterministic offline regression checks: python -m unittest -v."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import predictor
import train
from features import make_features, make_training_rows
from predict import make_forecast


def synthetic_history(count=450):
    rng = np.random.default_rng(42)
    return pd.DataFrame({"Close": 100 * np.exp(np.cumsum(rng.normal(0, .015, count))),
                         "Volume": rng.integers(100, 10000, count)},
                        index=pd.bdate_range("2020-01-01", periods=count))


class PipelineTests(unittest.TestCase):
    def test_probability_and_price_direction(self):
        for probability in (0.0, 0.4999, 0.5, 0.5001, 1.0):
            classifier, regressor = Mock(), Mock()
            classifier.classes_ = np.array([0, 1])
            classifier.predict_proba.return_value = np.array([[1 - probability, probability]])
            regressor.predict.return_value = np.array([0.0])
            bundle = {"models": {"test": (classifier, regressor)}, "winner": "test"}
            forecast = make_forecast(bundle, pd.DataFrame({"x": [1]}), 100)
            self.assertEqual(forecast["direction"] == "UP", forecast["price"] > 100)
            self.assertAlmostEqual(round(100 * forecast["probability_up"], 1)
                                   + round(100 * forecast["probability_down"], 1), 100, places=1)

    def test_causal_features_and_next_session_targets(self):
        history = synthetic_history()
        full = make_features(history)
        for end in (80, 200, 449):
            pd.testing.assert_frame_equal(full.iloc[:end], make_features(history.iloc[:end]))
        rows = make_training_rows(history.Close, full)
        self.assertNotIn(history.index[-1], rows.index)
        self.assertFalse(full.iloc[-1].isna().any())
        for date, row in rows.iterrows():
            next_date = history.index[history.index.get_loc(date) + 1]
            self.assertEqual(row.target_date, next_date)
            self.assertEqual(row.target_close, history.loc[next_date, "Close"])

    def test_volume_and_short_history(self):
        for history in (synthetic_history().drop(columns="Volume"),
                        synthetic_history().assign(Volume=0),
                        synthetic_history().assign(Volume=np.nan)):
            self.assertFalse(make_features(history).iloc[-1].isna().any())
        history = synthetic_history(100)
        with self.assertRaisesRegex(ValueError, "250"):
            train.train_models(make_training_rows(history.Close, make_features(history)), "TEST")
        flat = synthetic_history().assign(Close=100.0)
        features = make_features(flat)
        self.assertFalse(features.iloc[-1].isna().any())
        with self.assertRaisesRegex(ValueError, "only one direction"):
            train.train_models(make_training_rows(flat.Close, features), "FLAT")

    def test_unfinished_bar_and_invalid_ticker(self):
        history = synthetic_history(100)
        today = pd.Timestamp.now(tz="UTC").normalize()
        history.index = pd.date_range(end=today, periods=100)
        asset = Mock()
        asset.history.return_value = history
        asset.get_history_metadata.return_value = {}
        with patch.object(predictor.yf, "Ticker", return_value=asset):
            cleaned, _ = predictor.download_history("TEST")
            self.assertLess(cleaned.index[-1].date(), today.date())
            asset.history.return_value = pd.DataFrame()
            with self.assertRaisesRegex(ValueError, "No daily close"):
                predictor.download_history("INVALID")

    def test_session_close_settlement(self):
        history = synthetic_history(100)
        history.index = pd.date_range(end="2026-11-13", periods=100, tz="America/New_York")
        session_end = pd.Timestamp("2026-11-13 16:00", tz="America/New_York")
        asset = Mock()
        asset.history.return_value = history
        asset.get_history_metadata.return_value = {
            "currency": "USD", "currentTradingPeriod": {"regular": {"end": session_end}}}
        for clock_time, included in (("15:00", False), ("16:04", False), ("16:06", True)):
            now = pd.Timestamp(f"2026-11-13 {clock_time}", tz="America/New_York")
            with patch.object(predictor.yf, "Ticker", return_value=asset), \
                 patch.object(predictor, "datetime") as clock:
                clock.now.return_value = now.to_pydatetime()
                cleaned, currency = predictor.download_history("TEST")
                self.assertEqual(cleaned.index[-1].day == 13, included)
                self.assertEqual(currency, "USD")

    def test_workflow_persistence_retrain_and_split_boundaries(self):
        history = synthetic_history()
        original_fit = train.fit_pair
        original_evaluate = train.evaluate_pair
        fitted = []
        evaluated = []

        def fit(name, rows, columns):
            fitted.append(rows.copy())
            return original_fit(name, rows, columns)

        def evaluate(pair, rows, columns):
            self.assertLess(fitted[-1].target_date.max(), rows.index.min())
            evaluated.append(rows.copy())
            return original_evaluate(pair, rows, columns)

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(predictor, "OUTPUT_DIR", Path(directory)), \
             patch.object(predictor, "download_history", return_value=(history, "USD")), \
             patch("sys.argv", ["predictor.py", "--no-show"]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            with patch.object(train, "fit_pair", side_effect=fit), \
                 patch.object(train, "evaluate_pair", side_effect=evaluate):
                predictor.main()
            path = next(Path(directory).rglob("models.joblib"))
            bundle = joblib.load(path)
            self.assertEqual(len(evaluated), 8)
            holdout = evaluated[-1]
            for validation in evaluated[:6]:
                self.assertLess(validation.target_date.max(), holdout.index.min())
            self.assertEqual(list(bundle["backtest"].index), list(holdout.target_date))
            forecast = make_forecast(bundle, make_features(history).iloc[[-1]], history.Close.iloc[-1])
            self.assertAlmostEqual(forecast["probability_up"] + forecast["probability_down"], 1)
            self.assertEqual(forecast["direction"] == "UP", forecast["price"] > history.Close.iloc[-1])
            with Image.open(next(Path(directory).rglob("*.png"))) as image:
                image.verify()
            saved_time = path.stat().st_mtime_ns
            with patch.object(predictor, "train_models", side_effect=AssertionError("Unexpected retraining")):
                predictor.main()
            self.assertEqual(saved_time, path.stat().st_mtime_ns)
            with patch("sys.argv", ["predictor.py", "--no-show", "--retrain"]), \
                 patch.object(predictor, "train_models", wraps=train.train_models) as retrain:
                predictor.main()
                retrain.assert_called_once()
            for label in ("Current price", "Estimated next closing price", "Direction:",
                          "Model confidence", "UP probability", "DOWN probability", "Loaded saved models"):
                self.assertIn(label, output.getvalue())


if __name__ == "__main__":
    unittest.main()
