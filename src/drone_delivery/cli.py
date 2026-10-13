"""Command-line interface for the drone delivery prototype."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .model import train_and_evaluate, flag_heavy
from .plots import save_plots


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate a drone payload regression baseline")
    parser.add_argument("data", type=Path, help="Path to parameters.csv (not included)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    exp = train_and_evaluate(pd.read_csv(args.data, encoding="latin1"))
    args.out.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "actual_payload_g": exp.y_test.to_numpy(),
        "predicted_payload_g": exp.predictions,
        "heavy_flag": [flag_heavy(x, exp.threshold) for x in exp.predictions],
    })
    out.to_csv(args.out / "predictions.csv", index=False)
    metrics = {"r2": exp.r2, "mse": exp.mse, "training_prediction_p90_g": exp.threshold,
               "n_test": len(exp.y_test)}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_plots(exp, args.out)
    print(json.dumps(metrics, indent=2))
    print(f"Outputs written to {args.out}")

if __name__ == "__main__":
    main()
