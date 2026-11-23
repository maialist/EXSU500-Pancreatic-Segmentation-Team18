"""Plot random-forest sweep metrics without changing the final-results plots."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict]:
    """Read sweep metrics and convert serialized parameters and Dice to native types."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["parameters"] = json.loads(row["parameters"])
        row["dice"] = float(row["dice"])
    return rows


def grouped(rows: list[dict], phase: str, split: str, structure: str) -> dict[str, list[float]]:
    """Group pancreas-total Dice values by run for the requested sweep subset."""
    result: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["phase"] == phase and row["split"] == split and row["structure"] == structure:
            result[row["run_id"]].append(row["dice"])
    return result


def plot_phase(
    rows: list[dict], phase: str, parameter: str, out_dir: Path
) -> None:
    """Save a phase-specific validation plot, using a heatmap for depth and leaf."""
    if phase == "depth-leaf":
        combinations: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in rows:
            if (
                row["phase"] == phase
                and row["split"] == "val"
                and row["structure"] == "pancreas_total"
            ):
                parameters = row["parameters"]
                combinations[
                    (int(parameters["max_depth"]), int(parameters["min_samples_leaf"]))
                ].append(row["dice"])
        if not combinations:
            return
        depths = sorted({depth for depth, _ in combinations})
        leaves = sorted({leaf for _, leaf in combinations})
        matrix = np.full((len(depths), len(leaves)), np.nan)
        for (depth, leaf), values in combinations.items():
            matrix[depths.index(depth), leaves.index(leaf)] = np.mean(values)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(leaves)), leaves)
        ax.set_yticks(range(len(depths)), depths)
        ax.set(xlabel="min_samples_leaf", ylabel="max_depth", title="Depth × leaf: proxy pancreas-total Dice")
        fig.colorbar(image, ax=ax, label="Dice")
        fig.tight_layout()
        fig.savefig(out_dir / "depth-leaf_pancreas_total_dice.png", dpi=200)
        plt.close(fig)
        return
    # Other phases compare mean proxy Dice across runs, with case variation shown.
    runs = []
    for run_id, values in grouped(rows, phase, "val", "pancreas_total").items():
        row = next(item for item in rows if item["run_id"] == run_id)
        if phase == "class-weight":
            value = (
                f"{row['parameters']['stage1_class_weight']}/"
                f"{row['parameters']['stage2_class_weight']}"
            )
        else:
            value = row["parameters"][parameter]
        runs.append((value, np.mean(values), np.std(values)))
    if not runs:
        return
    runs.sort(key=lambda item: str(item[0]))
    labels = [str(item[0]) for item in runs]
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.2), 4.5))
    ax.errorbar(
        x,
        [item[1] for item in runs],
        yerr=[item[2] for item in runs],
        fmt="o-",
        capsize=3,
    )
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel(parameter)
    ax.set_ylabel("Pancreas-total Dice")
    ax.set_title(f"{phase}: proxy validation")
    fig.tight_layout()
    fig.savefig(out_dir / f"{phase}_pancreas_total_dice.png", dpi=200)
    plt.close(fig)


def plot_proxy_confirmation(rows: list[dict], out_dir: Path) -> None:
    """Plot each run's mean proxy score against its confirmation score."""
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["structure"] != "pancreas_total":
            continue
        values[row["run_id"]][row["split"]] = (
            values[row["run_id"]].get(row["split"], 0.0) + row["dice"]
        )
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        if row["structure"] == "pancreas_total":
            counts[(row["run_id"], row["split"])] += 1
    points = []
    for run_id, split_values in values.items():
        if "val" in split_values and "test" in split_values:
            proxy = split_values["val"] / counts[(run_id, "val")]
            confirm = split_values["test"] / counts[(run_id, "test")]
            points.append((proxy, confirm))
    if not points:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([point[0] for point in points], [point[1] for point in points])
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Proxy pancreas-total Dice", ylabel="Confirmation pancreas-total Dice")
    ax.set_title("Proxy versus confirmation")
    fig.tight_layout()
    fig.savefig(out_dir / "proxy_vs_confirmation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Load aggregated metrics and generate all sweep and confirmation figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/experiments/random_forest/case_metrics.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/experiments/random_forest/figures"),
    )
    args = parser.parse_args()
    rows = load_rows(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for phase, parameter in (
        ("trees", "n_estimators"),
        ("depth-leaf", "max_depth"),
        ("max-features", "max_features"),
        ("class-weight", "stage2_class_weight"),
        ("random-state", "seed"),
    ):
        plot_phase(rows, phase, parameter, args.out_dir)
    plot_proxy_confirmation(rows, args.out_dir)
    print(f"Saved sweep figures to {args.out_dir}")


if __name__ == "__main__":
    main()
