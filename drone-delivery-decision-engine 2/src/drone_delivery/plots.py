"""Diagnostic plots for a fitted baseline experiment."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_plots(experiment, directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(experiment.y_test)
    predicted = np.asarray(experiment.predictions)
    fig, ax = plt.subplots()
    ax.scatter(actual, predicted, alpha=0.7)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], linestyle="--", color="gray")
    ax.set(xlabel="Actual payload (g)", ylabel="Predicted payload (g)", title="Actual vs predicted")
    fig.tight_layout(); fig.savefig(path / "actual_vs_predicted.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots()
    ax.scatter(predicted, actual - predicted, alpha=0.7)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set(xlabel="Predicted payload (g)", ylabel="Residual (g)", title="Residuals vs predicted")
    fig.tight_layout(); fig.savefig(path / "residuals.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots()
    ax.hist(predicted, bins=min(15, max(3, len(predicted) // 3)))
    ax.axvline(experiment.threshold, linestyle="--", color="gray", label="Training 90th percentile")
    ax.legend()
    ax.set(xlabel="Predicted payload (g)", ylabel="Count", title="Predicted payload distribution")
    fig.tight_layout(); fig.savefig(path / "predicted_distribution.png", dpi=160); plt.close(fig)
