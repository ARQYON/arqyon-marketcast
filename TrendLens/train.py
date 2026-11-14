"""Compare two CPU model families on historical data and fit the saved models."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor


RANDOM_SEED = 42
CPU_THREADS = 2
CV_SPLITS = 3
MIN_TRAINING_ROWS = 250
# A row's label comes from the next session. Leave one row out so the last
# training label is strictly earlier than the first validation feature date.
GAP = 1
BUNDLE_VERSION = 1
TARGET_COLUMNS = ("current_close", "target_close", "target_date", "target_up", "target_magnitude")
MODEL_NAMES = ("Random Forest", "XGBoost")


def fit_pair(name, training, feature_columns):
    """Fit a classifier for direction and a regressor for size of the move."""
    settings = dict(n_estimators=120, random_state=RANDOM_SEED, n_jobs=CPU_THREADS)
    if name == "Random Forest":
        classifier = RandomForestClassifier(**settings, max_depth=5, min_samples_leaf=12)
        regressor = RandomForestRegressor(**settings, max_depth=5, min_samples_leaf=12)
    elif name == "XGBoost":
        # Histogram trees, shallow depths, and two threads keep CPU use modest.
        settings.update(
            tree_method="hist", device="cpu", max_depth=3, learning_rate=0.05,
            min_child_weight=8, subsample=0.85, colsample_bytree=0.85, verbosity=0,
        )
        classifier = XGBClassifier(**settings, objective="binary:logistic", eval_metric="logloss")
        regressor = XGBRegressor(**settings, objective="reg:squarederror")
    else:
        raise ValueError(f"Unknown model family: {name}")

    y_up = training["target_up"]
    if y_up.nunique() != 2:
        raise ValueError("A training period has only one direction. Try a ticker with more varied history.")
    x = training.loc[:, list(feature_columns)]
    classifier.fit(x, y_up)
    regressor.fit(x, training["target_magnitude"])
    return classifier, regressor


def forecast_arrays(pair, features, current_close):
    """Convert direction and move-size estimates back into prices."""
    classifier, regressor = pair
    # The classes are 0=DOWN and 1=UP. Find the UP column by its label.
    up_column = list(classifier.classes_).index(1)
    up_probability = classifier.predict_proba(features)[:, up_column]
    direction_up = up_probability > 0.5

    # The regressor estimates the absolute log-return. A tiny positive floor
    # keeps a zero-size model estimate consistent with the chosen direction.
    magnitude = np.maximum(regressor.predict(features).astype(float), 1e-8)
    signed_move = np.where(direction_up, magnitude, -magnitude)
    estimated_close = np.asarray(current_close) * np.exp(signed_move)
    return up_probability, estimated_close


def measure(actual_up, actual_close, up_probability, predicted_close):
    """Accuracy: higher is better. Brier score and price MAE: lower is better."""
    return {
        "accuracy": float(accuracy_score(actual_up, up_probability > 0.5)),
        "brier": float(brier_score_loss(actual_up, up_probability)),
        "price_mae": float(mean_absolute_error(actual_close, predicted_close)),
    }


def evaluate_pair(pair, validation, feature_columns):
    """Score predictions on future rows and keep prices for the chart."""
    up_probability, prices = forecast_arrays(
        pair, validation.loc[:, list(feature_columns)], validation["current_close"],
    )
    scores = measure(validation["target_up"], validation["target_close"], up_probability, prices)
    return scores, prices


def evaluate_baseline(training, validation):
    """Use past UP frequency for direction and the current close for price."""
    up_probability = np.full(len(validation), training["target_up"].mean())
    return measure(
        validation["target_up"], validation["target_close"],
        up_probability, validation["current_close"],
    )


def train_models(rows, ticker):
    """Select a model family on earlier folds, then evaluate on a later holdout."""
    if len(rows) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Need {MIN_TRAINING_ROWS} usable daily training rows after indicator warm-up. "
            f"Yahoo returned enough data for {len(rows)}. Try a ticker with more history."
        )
    feature_columns = [column for column in rows if column not in TARGET_COLUMNS]
    holdout_start = int(len(rows) * 0.8)
    development = rows.iloc[:holdout_start - GAP]
    holdout = rows.iloc[holdout_start:]
    cv_scores = {}
    for name in (*MODEL_NAMES, "Baseline"):
        scores = []
        for train_index, valid_index in TimeSeriesSplit(n_splits=CV_SPLITS, gap=GAP).split(development):
            training = development.iloc[train_index]
            validation = development.iloc[valid_index]
            assert training["target_date"].max() < validation.index.min()
            if name == "Baseline":
                score = evaluate_baseline(training, validation)
            else:
                pair = fit_pair(name, training, feature_columns)
                score, _ = evaluate_pair(pair, validation, feature_columns)
            scores.append(score)
        cv_scores[name] = {key: float(np.mean([score[key] for score in scores])) for key in scores[0]}

    # Select before looking at the holdout. Never tune against its scores.
    winner = min(MODEL_NAMES, key=lambda name: cv_scores[name]["brier"])
    assert development["target_date"].max() < holdout.index.min()
    backtest = holdout[["target_close", "current_close"]].copy()
    backtest.index = holdout["target_date"]
    backtest = backtest.rename(columns={"target_close": "Actual", "current_close": "Baseline"})
    holdout_scores = {}
    models = {}
    for name in MODEL_NAMES:
        print(f"Training and evaluating {name} ...")
        pair = fit_pair(name, development, feature_columns)
        holdout_scores[name], prices = evaluate_pair(pair, holdout, feature_columns)
        backtest[name] = prices
        models[name] = fit_pair(name, rows, feature_columns)
    holdout_scores["Baseline"] = evaluate_baseline(development, holdout)
    return dict(version=BUNDLE_VERSION, ticker=ticker, feature_columns=feature_columns,
                models=models, winner=winner, cv_scores=cv_scores,
                holdout_scores=holdout_scores, backtest=backtest,
                trained_through=rows["target_date"].max())
