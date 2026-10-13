"""Reproducible, leakage-free regression baseline for drone payload research."""

from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURES = ("speed", "altitude")
TARGET = "payload"


@dataclass
class Experiment:
    model: Pipeline
    y_test: pd.Series
    predictions: object
    r2: float
    mse: float
    threshold: float


def prepare_data(df: pd.DataFrame):
    """Coerce source columns to numeric; remove rows without a valid target."""
    required = set(FEATURES) | {TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    numeric = df.loc[:, [*FEATURES, TARGET]].copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.dropna(subset=[TARGET])
    if len(numeric) < 6:
        raise ValueError("At least six labeled rows are required")
    if numeric.loc[:, list(FEATURES)].isna().all().any():
        raise ValueError("Each feature must have at least one numeric value")
    return numeric[list(FEATURES)], numeric[TARGET]


def train_and_evaluate(df: pd.DataFrame, random_state: int = 5, test_size: float = 1 / 3, quantile: float = 0.90):
    """Split before imputing to avoid train/test leakage.

    Threshold is estimated from *training-set* predictions, not test labels.
    It is a triage heuristic, NOT a certified drone payload capacity.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1")
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("regressor", LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)
    import numpy as np
    threshold = float(np.quantile(pipeline.predict(X_train), quantile))
    predictions = pipeline.predict(X_test)
    return Experiment(
        model=pipeline,
        y_test=y_test,
        predictions=predictions,
        r2=float(r2_score(y_test, predictions)),
        mse=float(mean_squared_error(y_test, predictions)),
        threshold=threshold,
    )


def flag_heavy(predicted_payload: float, threshold: float) -> bool:
    """Flag a prediction above the empirical alert threshold."""
    return bool(predicted_payload > threshold)
