"""Run reproducible random-forest proxy sweeps and collect metrics.

Each configuration is trained on all cases in the manifest's ``train`` list,
tuned on its ``val`` proxy cases, and evaluated on both proxy (``val``) and
confirmation (``test``) cases. The manifest's ``test`` list is not the
project's held-out test split; the real test split is never used here.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

PHASES = ("trees", "depth-leaf", "max-features", "class-weight", "random-state")
BASE = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "stage1_class_weight": "none",
    "stage2_class_weight": "balanced_subsample",
    "stage1_negative_ratio": 5,
    "dilation_iterations": 3,
    "min_component_size": 100,
    "seed": 42,
}


def configurations(phase: str) -> list[dict]:
    if phase == "trees":
        return [{**BASE, "n_estimators": value} for value in (50, 100, 150, 200, 300)]
    if phase == "depth-leaf":
        return [
            {**BASE, "max_depth": depth, "min_samples_leaf": leaf}
            for depth, leaf in itertools.product((10, 15, 20, 25), (1, 2, 4, 8))
        ]
    if phase == "max-features":
        return [{**BASE, "max_features": value} for value in ("sqrt", "log2", "all")]
    if phase == "class-weight":
        return [
            {**BASE, "stage1_class_weight": stage1, "stage2_class_weight": stage2}
            for stage1, stage2 in (
                ("none", "none"),
                ("none", "balanced"),
                ("none", "balanced_subsample"),
                ("balanced", "balanced_subsample"),
                ("balanced_subsample", "balanced_subsample"),
            )
        ]
    if phase == "random-state":
        return [{**BASE, "seed": seed} for seed in (0, 1, 21, 42, 123, 999)]
    raise ValueError(f"Unknown sweep phase: {phase}")


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def train_arguments(config: dict) -> list[str]:
    return [
        "--n-estimators", str(config["n_estimators"]),
        "--max-depth", str(config["max_depth"]),
        "--min-samples-leaf", str(config["min_samples_leaf"]),
        "--max-features", str(config["max_features"]),
        "--stage1-class-weight", str(config["stage1_class_weight"]),
        "--stage2-class-weight", str(config["stage2_class_weight"]),
        "--stage1-negative-ratio", str(config["stage1_negative_ratio"]),
        "--dilation-iterations", str(config["dilation_iterations"]),
        "--min-component-size", str(config["min_component_size"]),
        "--seed", str(config["seed"]),
    ]


def collect_results(output_dir: Path) -> None:
    rows: list[dict] = []
    for metrics_path in sorted(output_dir.glob("*/metrics.csv")):
        run_dir = metrics_path.parent
        manifest_path = run_dir / "manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["run_id"] = manifest["run_id"]
                row["phase"] = manifest["phase"]
                row["parameters"] = json.dumps(manifest["parameters"], sort_keys=True)
                rows.append(row)
    if not rows:
        return
    result_path = output_dir / "case_metrics.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} case metrics to {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("experiments/random_forest_proxy_split.json"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/random_forest"))
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    if not args.splits.is_file():
        raise FileNotFoundError(
            f"{args.splits} does not exist; run create_rf_proxy_split.py first"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    script_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = configurations(args.phase)
    if args.max_runs is not None:
        configs = configs[: args.max_runs]

    for config in configs:
        parameter_text = "_".join(
            f"{key}-{str(config[key]).replace('.', 'p')}"
            for key in sorted(config)
        )
        run_id = f"{args.phase}_{parameter_text}"
        run_dir = output_dir / run_id
        metrics_path = run_dir / "metrics.csv"
        if metrics_path.is_file():
            print(f"Skipping completed run {run_id}")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "random_forest.joblib"
        predictions_root = run_dir / "predictions"
        predictions_dir = predictions_root / "random_forest"
        manifest = {
            "run_id": run_id,
            "phase": args.phase,
            "parameters": config,
            "splits": str(args.splits),
            "data": str(args.data),
            "proxy_split": "val",
            "confirmation_split": "test",
            "python": sys.executable,
        }
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")

        common = [
            sys.executable,
            str(script_root / "models/random_forest.py"),
        ]
        start = time.monotonic()
        run(
            common
            + [
                "train",
                "--data", str(args.data),
                "--splits", str(args.splits),
                "--model", str(model_path),
            ]
            + train_arguments(config)
        )
        fit_seconds = time.monotonic() - start
        run(
            common
            + [
                "tune",
                "--data", str(args.data),
                "--splits", str(args.splits),
                "--model", str(model_path),
                "--batch-size", str(args.batch_size),
            ]
        )
        run(
            common
            + [
                "predict",
                "--data", str(args.data),
                "--splits", str(args.splits),
                "--model", str(model_path),
                "--split", "val",
                "--out", str(predictions_dir),
                "--batch-size", str(args.batch_size),
            ]
        )
        run(
            common
            + [
                "predict",
                "--data", str(args.data),
                "--splits", str(args.splits),
                "--model", str(model_path),
                "--split", "test",
                "--out", str(predictions_dir),
                "--batch-size", str(args.batch_size),
            ]
        )
        run(
            [
                sys.executable,
                str(script_root / "evaluate.py"),
                "--data", str(args.data),
                "--splits", str(args.splits),
                "--predictions", str(predictions_root),
                "--models", "random_forest",
                "--out", str(metrics_path),
            ]
        )
        manifest["fit_seconds"] = fit_seconds
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    collect_results(output_dir)


if __name__ == "__main__":
    main()
