"""Train and run voxel-wise logistic regression for pancreas CT segmentation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASS_IDS = (0, 1, 2)
CLASS_NAMES = {0: "background", 1: "pancreas", 2: "tumour"}
NEIGHBOUR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
)
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MODEL_PATH = Path("data/models/logistic_regression.joblib")


def target_labels(label: np.ndarray, target: str) -> np.ndarray:
    """Return labels for multiclass segmentation or tumour-vs-rest."""
    if target == "multiclass":
        return label
    if target == "tumour":
        return (label == 2).astype(np.uint8)
    raise ValueError(f"Unsupported target: {target}")


ATLAS_SIZE = 24
SMOOTHING_SIGMAS = (1, 2, 4, 8)


def build_atlas(data_dir: Path, case_names: list[str]) -> np.ndarray:
    """Mean pancreas+tumour location in normalised volume coordinates (training cases only)."""
    total = np.zeros((ATLAS_SIZE,) * 3, dtype=np.float32)
    for case_name in case_names:
        with np.load(data_dir / f"{case_name}.npz") as case:
            mask = (case["label"] > 0).astype(np.float32)
        total += ndimage.zoom(mask, [ATLAS_SIZE / size for size in mask.shape], order=1)
    total = ndimage.gaussian_filter(total / len(case_names), 1)
    return total / total.max()


def context_maps(image: np.ndarray, atlas: np.ndarray) -> np.ndarray:
    """Per-voxel context channels: log spatial prior, multi-scale smoothed
    intensity, and local intensity spread. Shape image.shape + (6,)."""
    prior = ndimage.zoom(atlas, [size / ATLAS_SIZE for size in image.shape], order=1)
    prior = prior[tuple(slice(0, size) for size in image.shape)]
    prior = np.pad(prior, [(0, size - got) for size, got in zip(image.shape, prior.shape)], mode="edge")
    channels = [np.log(prior + 1e-3)]
    channels += [ndimage.gaussian_filter(image, sigma) for sigma in SMOOTHING_SIGMAS]
    mean = ndimage.gaussian_filter(image, 2)
    channels.append(np.sqrt(np.maximum(ndimage.gaussian_filter(image**2, 2) - mean**2, 0)))
    return np.stack(channels, axis=-1).astype(np.float32)


def extract_features(
    image: np.ndarray, coordinates: np.ndarray, maps: np.ndarray | None = None
) -> np.ndarray:
    """Edge-padded 3x3x3 intensity patch, plus context channels when given."""
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n_voxels, 3)")

    coordinates = coordinates.astype(np.intp, copy=False)
    if coordinates.size and (
        np.any(coordinates < 0)
        or any(np.any(coordinates[:, axis] >= image.shape[axis]) for axis in range(3))
    ):
        raise ValueError("Voxel coordinates are outside the image bounds")

    extra = 0 if maps is None else maps.shape[-1]
    features = np.empty((len(coordinates), len(NEIGHBOUR_OFFSETS) + extra), dtype=np.float32)
    for column, offset in enumerate(NEIGHBOUR_OFFSETS):
        indices = tuple(
            np.clip(coordinates[:, axis] + offset[axis], 0, image.shape[axis] - 1)
            for axis in range(3)
        )
        features[:, column] = image[indices]
    if maps is not None:
        features[:, len(NEIGHBOUR_OFFSETS) :] = maps[
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        ]
    return features


def _background_coordinates(
    raw_label: np.ndarray,
    count: int,
    hard_fraction: float,
    dilation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample background voxels, mostly from a shell around the pancreas/tumour.

    Uniform background is dominated by air and fat that are trivially separable;
    the shell supplies the confusable soft tissue seen at prediction time.
    """
    foreground = raw_label > 0
    parts: list[np.ndarray] = []
    n_hard = round(count * hard_fraction) if dilation > 0 else 0
    if n_hard and foreground.any():
        found = np.argwhere(foreground)
        low = np.maximum(found.min(axis=0) - dilation, 0)
        high = np.minimum(found.max(axis=0) + dilation + 1, raw_label.shape)
        window = tuple(slice(a, b) for a, b in zip(low, high))
        shell = ndimage.binary_dilation(foreground[window], iterations=dilation)
        shell &= ~foreground[window]
        shell_voxels = np.argwhere(shell) + low
        if len(shell_voxels):
            keep = rng.choice(len(shell_voxels), min(n_hard, len(shell_voxels)), replace=False)
            parts.append(shell_voxels[keep])
    n_random = count - sum(len(part) for part in parts)
    background = np.flatnonzero(~foreground.ravel())
    chosen = rng.choice(background, min(n_random, len(background)), replace=False)
    parts.append(np.column_stack(np.unravel_index(chosen, raw_label.shape)))
    return np.concatenate(parts)


def _sample_training_data(
    data_dir: Path,
    case_names: list[str],
    target: str,
    samples_per_class: int,
    samples_per_class_per_case: int,
    seed: int,
    background_samples: int,
    background_per_case: int,
    hard_fraction: float,
    dilation: int,
    atlas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    class_ids = (0, 1) if target == "tumour" else CLASS_IDS
    class_features: dict[int, list[np.ndarray]] = {class_id: [] for class_id in class_ids}

    for case_name in case_names:
        case_path = data_dir / f"{case_name}.npz"
        if not case_path.is_file():
            raise FileNotFoundError(f"Processed case is missing: {case_path}")
        with np.load(case_path) as case:
            image = case["image"].astype(np.float32, copy=False)
            raw_label = case["label"]
            labels = target_labels(raw_label, target)
        if image.ndim != 3 or labels.ndim != 3:
            raise ValueError(f"Expected a 3D image and label for {case_name}")
        if image.shape != labels.shape:
            raise ValueError(
                f"Image and label shapes differ for {case_name}: "
                f"{image.shape} != {labels.shape}"
            )

        maps = context_maps(image, atlas)
        coordinates = _background_coordinates(
            raw_label, background_per_case, hard_fraction, dilation, rng
        )
        class_features[0].append(extract_features(image, coordinates, maps))

        flat_labels = labels.ravel()
        for class_id in class_ids:
            if class_id == 0:
                continue
            voxels = np.flatnonzero(flat_labels == class_id)
            count = min(len(voxels), samples_per_class_per_case)
            if count == 0:
                continue
            selected = rng.choice(voxels, size=count, replace=False)
            coordinates = np.column_stack(np.unravel_index(selected, labels.shape))
            class_features[class_id].append(extract_features(image, coordinates, maps))

    features_by_class: list[np.ndarray] = []
    labels_by_class: list[np.ndarray] = []
    for class_id in class_ids:
        samples = class_features[class_id]
        if not samples:
            raise ValueError(
                f"No training voxels for class {class_id}; check the labels and split"
            )
        class_data = np.concatenate(samples, axis=0)
        count = min(len(class_data), background_samples if class_id == 0 else samples_per_class)
        if count < len(class_data):
            selected = rng.choice(len(class_data), size=count, replace=False)
            class_data = class_data[selected]
        features_by_class.append(class_data)
        labels_by_class.append(np.full(count, class_id, dtype=np.uint8))

    return np.concatenate(features_by_class), np.concatenate(labels_by_class)


def _load_splits(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as split_file:
        splits = json.load(split_file)
    required = ("train", "val", "test")
    if not all(isinstance(splits.get(name), list) for name in required):
        raise ValueError(f"{path} must contain train, val, and test case lists")
    if any(not names for names in (splits[name] for name in required)):
        raise ValueError("Train, validation, and test splits must all be non-empty")
    if any(not isinstance(case, str) for name in required for case in splits[name]):
        raise ValueError("Split entries must be case-name strings")
    if len(set(splits["train"] + splits["val"] + splits["test"])) != sum(
        len(splits[name]) for name in required
    ):
        raise ValueError("Patient cases overlap between train, validation, and test")
    return splits


THRESHOLD_GRID = tuple(
    np.round([0.05, 0.075, *np.arange(0.10, 0.96, 0.05)], 3)
)


def _predict_proba_volume(
    classifier: LogisticRegression, image: np.ndarray, batch_size: int, atlas: np.ndarray
) -> np.ndarray:
    """Return per-voxel class probabilities with shape image.shape + (n_classes,)."""
    maps = context_maps(image, atlas)
    probabilities = np.empty(image.shape + (len(classifier.classes_),), dtype=np.float32)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    for start in range(0, image.size, batch_size):
        flat_indices = np.arange(start, min(start + batch_size, image.size))
        coordinates = np.column_stack(np.unravel_index(flat_indices, image.shape))
        flat[start : start + len(flat_indices)] = classifier.predict_proba(
            extract_features(image, coordinates, maps)
        )
    return probabilities


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labelled, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == sizes.argmax()


def _decode(
    probabilities: np.ndarray, classes: np.ndarray, decision: dict | None
) -> np.ndarray:
    """Turn probabilities into labels using tuned per-class thresholds.

    Without tuned settings this falls back to the plain argmax.
    """
    if not decision:
        return classes[probabilities.argmax(axis=-1)].astype(np.uint8)
    prediction = np.zeros(probabilities.shape[:-1], dtype=np.uint8)
    for class_id in sorted(decision):
        settings = decision[class_id]
        column = int(np.flatnonzero(classes == class_id)[0])
        mask = probabilities[..., column] >= settings["threshold"]
        if settings["largest_component"]:
            mask = _keep_largest_component(mask)
        prediction[mask] = class_id
    return prediction


def _dice(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = int(truth.sum() + prediction.sum())
    if denominator == 0:
        return float("nan")
    return 2.0 * int(np.logical_and(truth, prediction).sum()) / denominator


def _tune_decision(
    classifier: LogisticRegression,
    cases: list[tuple[np.ndarray, np.ndarray]],
    class_ids: tuple[int, ...],
    batch_size: int,
    atlas: np.ndarray,
) -> dict:
    """Pick, per foreground class, the threshold and largest-component option
    that maximise mean per-case Dice on the supplied (validation) cases."""
    floor = float(THRESHOLD_GRID[0])
    cache: dict[int, list] = {class_id: [] for class_id in class_ids if class_id}
    for image, truth in cases:
        probabilities = _predict_proba_volume(classifier, image, batch_size, atlas)
        for class_id in cache:
            column = int(np.flatnonzero(classifier.classes_ == class_id)[0])
            scores = probabilities[..., column].ravel()
            candidates = np.flatnonzero(scores >= floor)
            cache[class_id].append(
                (truth == class_id, candidates, scores[candidates])
            )

    decision: dict[int, dict] = {}
    for class_id, entries in cache.items():
        best = (-1.0, None)
        for largest in (False, True):
            for threshold in THRESHOLD_GRID:
                dices = []
                for truth_mask, candidates, scores in entries:
                    mask = np.zeros(truth_mask.size, dtype=bool)
                    mask[candidates[scores >= threshold]] = True
                    mask = mask.reshape(truth_mask.shape)
                    if largest:
                        mask = _keep_largest_component(mask)
                    dices.append(_dice(truth_mask, mask))
                score = float(np.nanmean(dices)) if not np.all(np.isnan(dices)) else 0.0
                if score > best[0]:
                    best = (score, {"threshold": float(threshold), "largest_component": largest})
        print(
            f"  class {class_id}: threshold={best[1]['threshold']:.2f} "
            f"largest_component={best[1]['largest_component']} "
            f"validation mean Dice={best[0]:.4f}"
        )
        decision[class_id] = best[1]
    return decision


def _parse_class_weight(value: str, class_ids: tuple[int, ...]):
    if value in ("balanced", "none"):
        return "balanced" if value == "balanced" else None
    weights = [float(part) for part in value.split(",")]
    if len(weights) != len(class_ids) or any(weight <= 0 for weight in weights):
        raise ValueError(
            f"--class-weight needs 'balanced', 'none' or {len(class_ids)} positive "
            "comma-separated values"
        )
    return dict(zip(class_ids, weights))


def train(args: argparse.Namespace) -> None:
    splits = _load_splits(args.splits)
    atlas = build_atlas(args.data, splits["train"])
    features, labels = _sample_training_data(
        args.data,
        splits["train"],
        args.target,
        args.samples_per_class,
        args.samples_per_class_per_case,
        args.seed,
        args.background_samples,
        args.background_per_case,
        args.hard_negative_fraction,
        args.hard_negative_dilation,
        atlas,
    )
    train_ids = (0, 1) if args.target == "tumour" else CLASS_IDS
    class_weight = _parse_class_weight(args.class_weight, train_ids)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.c,
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=args.max_iter,
            class_weight=class_weight,
            random_state=args.seed,
        ),
    )
    classifier.fit(features, labels)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "atlas": atlas,
            "target": args.target,
            "class_ids": train_ids,
            "config": {
                "C": args.c,
                "l1_ratio": 0.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": args.max_iter,
                "class_weight": args.class_weight,
                "samples_per_class": args.samples_per_class,
                "samples_per_class_per_case": args.samples_per_class_per_case,
                "background_samples": args.background_samples,
                "background_per_case": args.background_per_case,
                "hard_negative_fraction": args.hard_negative_fraction,
                "hard_negative_dilation": args.hard_negative_dilation,
                "seed": args.seed,
                "feature": "3x3x3 intensity patch + log spatial prior + multi-scale smoothed intensity + local spread (standardised)",
            },
            "train_cases": tuple(splits["train"]),
            "decision": None,
        },
        args.model,
    )
    print(f"Trained on {len(labels):,} sampled voxels; saved model to {args.model}")
    print("Run `tune` on the validation split before `predict` to set decision thresholds.")
    for class_id in np.unique(labels):
        class_name = (
            ("non-tumour" if class_id == 0 else "tumour")
            if args.target == "tumour"
            else CLASS_NAMES[int(class_id)]
        )
        print(f"  {class_name}: {(labels == class_id).sum():,}")


def tune(args: argparse.Namespace) -> None:
    splits = _load_splits(args.splits)
    artifact = joblib.load(args.model)
    target = artifact["target"]
    cases = []
    for case_name in splits[args.split]:
        with np.load(args.data / f"{case_name}.npz") as case:
            cases.append(
                (case["image"].astype(np.float32, copy=False), target_labels(case["label"], target))
            )
    print(f"Tuning decision rule on {len(cases)} {args.split} cases")
    artifact["decision"] = _tune_decision(
        artifact["classifier"], cases, tuple(artifact["class_ids"]), args.batch_size, artifact["atlas"]
    )
    joblib.dump(artifact, args.model)
    print(f"Saved decision settings to {args.model}")


def predict(args: argparse.Namespace) -> None:
    splits = _load_splits(args.splits)
    artifact = joblib.load(args.model)
    classifier: LogisticRegression = artifact["classifier"]
    target = artifact["target"]
    class_ids = tuple(artifact["class_ids"])
    decision = artifact.get("decision")
    if not decision:
        print("Warning: no tuned decision settings; using argmax (run `tune` first).")
    args.out.mkdir(parents=True, exist_ok=True)
    totals = {
        class_id: {"dice_sum": 0.0, "case_count": 0}
        for class_id in class_ids
        if class_id != 0
    }

    for case_name in splits[args.split]:
        case_path = args.data / f"{case_name}.npz"
        if not case_path.is_file():
            raise FileNotFoundError(f"Processed case is missing: {case_path}")
        with np.load(case_path) as case:
            image = case["image"].astype(np.float32, copy=False)
            truth = target_labels(case["label"], target)
        if image.ndim != 3 or truth.ndim != 3:
            raise ValueError(f"Expected a 3D image and label for {case_name}")
        if image.shape != truth.shape:
            raise ValueError(
                f"Image and label shapes differ for {case_name}: "
                f"{image.shape} != {truth.shape}"
            )
        prediction = _decode(
            _predict_proba_volume(classifier, image, args.batch_size, artifact["atlas"]),
            classifier.classes_,
            decision,
        )

        for class_id, values in totals.items():
            truth_mask = truth == class_id
            prediction_mask = prediction == class_id
            denominator = int(truth_mask.sum() + prediction_mask.sum())
            if denominator:
                values["dice_sum"] += (
                    2.0 * int(np.logical_and(truth_mask, prediction_mask).sum()) / denominator
                )
                values["case_count"] += 1

        if target == "tumour":
            prediction *= 2
        prediction_path = args.out / f"{case_name}.npz"
        np.savez_compressed(prediction_path, prediction=prediction)
        print(f"Predicted {case_name} -> {prediction_path}")

    print(f"Mean per-case Dice on the {args.split} split:")
    for class_id, values in totals.items():
        class_name = "tumour" if target == "tumour" else CLASS_NAMES[class_id]
        score = (
            0.0
            if values["case_count"] == 0
            else values["dice_sum"] / values["case_count"]
        )
        print(f"  {class_name}: {score:.4f} ({values['case_count']} cases)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="fit and save the logistic-regression model"
    )
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--splits", type=Path, default=DEFAULT_DATA_DIR / "splits.json")
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument(
        "--target",
        choices=("multiclass", "tumour"),
        default="multiclass",
        help="segment background/pancreas/tumour or tumour versus everything else",
    )
    train_parser.add_argument("--c", type=float, default=1.0)
    train_parser.add_argument("--max-iter", type=int, default=500)
    train_parser.add_argument("--samples-per-class", type=int, default=10_000)
    train_parser.add_argument("--samples-per-class-per-case", type=int, default=250)
    train_parser.add_argument(
        "--background-samples", type=int, default=40_000,
        help="total background voxels sampled (foreground classes use --samples-per-class)",
    )
    train_parser.add_argument("--background-per-case", type=int, default=400)
    train_parser.add_argument(
        "--hard-negative-fraction", type=float, default=0.75,
        help="fraction of background sampled from the shell around pancreas/tumour",
    )
    train_parser.add_argument(
        "--hard-negative-dilation", type=int, default=10,
        help="shell width in voxels (about 15 mm in-plane)",
    )
    train_parser.add_argument(
        "--class-weight", default="1,1,1",
        help="'balanced', 'none', or one weight per class (background,pancreas,tumour)",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)

    tune_parser = commands.add_parser(
        "tune",
        help="choose per-class probability thresholds and largest-component "
        "post-processing on a labelled split (use val) and store them in the model",
    )
    tune_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    tune_parser.add_argument("--splits", type=Path, default=DEFAULT_DATA_DIR / "splits.json")
    tune_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    tune_parser.add_argument("--split", choices=("val",), default="val")
    tune_parser.add_argument("--batch-size", type=int, default=100_000)
    tune_parser.set_defaults(func=tune)

    predict_parser = commands.add_parser(
        "predict", help="predict a labelled split and report mean per-case Dice"
    )
    predict_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    predict_parser.add_argument("--splits", type=Path, default=DEFAULT_DATA_DIR / "splits.json")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    predict_parser.add_argument(
        "--out", type=Path, default=Path("data/predictions/logistic_regression")
    )
    predict_parser.add_argument("--batch-size", type=int, default=100_000)
    predict_parser.set_defaults(func=predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        if not np.isfinite(args.c) or args.c <= 0 or args.max_iter <= 0:
            raise ValueError("--c and --max-iter must be positive")
        if args.samples_per_class <= 0 or args.samples_per_class_per_case <= 0:
            raise ValueError("Sample limits must be positive")
        if args.background_samples <= 0 or args.background_per_case <= 0:
            raise ValueError("Background sample limits must be positive")
        if not 0 <= args.hard_negative_fraction <= 1 or args.hard_negative_dilation < 0:
            raise ValueError("Invalid hard-negative settings")
    elif args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
