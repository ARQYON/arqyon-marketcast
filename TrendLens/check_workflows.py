"""Run all example symbols; use --synthetic only when Yahoo is unavailable."""
import argparse
import contextlib
import io
import json
import re
from pathlib import Path
from unittest.mock import patch

import joblib
from PIL import Image

import predictor
from test_predictor import synthetic_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    mode = "synthetic" if args.synthetic else "live"
    results = {}
    for ticker in ("BTC-USD", "ETH-USD", "AAPL", "NVDA", "EURUSD=X", "GC=F"):
        capture = io.StringIO()
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(predictor, "TICKER", ticker))
                stack.enter_context(patch("sys.argv", ["predictor.py", "--no-show", "--retrain"]))
                if args.synthetic:
                    stack.enter_context(patch.object(predictor, "OUTPUT_DIR", Path("output/synthetic")))
                    history = synthetic_history()
                    if ticker == "EURUSD=X":
                        history["Close"] /= 100
                        history["Volume"] = 0
                    stack.enter_context(patch.object(predictor, "download_history", return_value=(history, "test units")))
                stack.enter_context(contextlib.redirect_stdout(capture))
                predictor.main()
            log = capture.getvalue()
            model_path = Path(log.split("Models: ")[-1].splitlines()[0])
            bundle = joblib.load(model_path)
            up = float(re.search(r"UP probability: ([\d.]+)%", log)[1])
            down = float(re.search(r"DOWN probability: ([\d.]+)%", log)[1])
            assert abs(up + down - 100) <= 0.11
            current = float(re.search(r"Current price .*?: ([\d,.]+)", log)[1].replace(",", ""))
            estimate = float(re.search(r"Estimated next closing price: ([\d,.]+)", log)[1].replace(",", ""))
            assert (estimate > current) == ("Direction: UP" in log)
            with Image.open(model_path.with_name("predicted_vs_actual.png")) as image:
                image.verify()
            results[ticker] = {"status": "passed", "mode": mode, "scores": bundle["holdout_scores"]}
        except Exception as error:
            results[ticker] = {"status": "failed", "mode": mode, "error": str(error)}
        print(capture.getvalue())
        print(ticker, results[ticker])
    Path(f"checks_{mode}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
