"""Train and run a two-stage random forest for pancreas CT segmentation.

Protocol adjustments:
    * Stage 1 classifies background versus pancreas, where pancreas includes
      both the pancreas and tumour labels.
    * Stage 2 classifies tumour versus non-tumour only inside a dilated
      pancreas region. It is trained with dilated ground-truth pancreas masks
      and uses dilated Stage 1 predictions during validation and testing.
    * The original 27 raw 3x3x3 intensity features are replaced by 14 channels
      computed once per volume: the normalized intensity, Gaussian-smoothed
      intensities, local means and standard deviations, gradient magnitude,
      Laplacian-of-Gaussian responses, and normalized spatial coordinates.

This staged hierarchy is based on a similar pancreas-segmentation protocol
described by Farag et al. (2014), which uses random forest models in a
hierarchical segmentation approach. The present implementation adapts that
idea to this project's preprocessed Medical Segmentation Decathlon Task07
Pancreas data, shared patient-level split, labels, feature channels, and
validation/test protocol. The model is a research baseline and is not
clinically validated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    generate_binary_structure,
    gaussian_filter,
    gaussian_laplace,
    label as connected_components,
    uniform_filter,
)
from sklearn.ensemble import RandomForestClassifier

CLASS_NAMES = {0: "background", 1: "pancreas", 2: "tumour"}
# These fixed feature settings define the input protocol for both forests.
GAUSSIAN_SIGMAS = (1.0, 2.0, 4.0)
LOCAL_WINDOW_SIZES = (5, 9)
DEFAULT_DILATION_ITERATIONS = 3
DEFAULT_MIN_COMPONENT_SIZE = 100
THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MODEL_PATH = Path("data/models/random_forest.joblib")


def compute_feature_channels(image: np.ndarray) -> tuple[np.ndarray, ...]:
    """Compute reusable filter and position channels for one volume."""
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    image = image.astype(np.float32, copy=False)
    # Keep one array per channel so the same filters can serve sampling and
    # batched prediction without recalculating them for individual voxels.
    channels: list[np.ndarray] = [image]
    # Smoothed intensities provide context at multiple spatial scales.
    for sigma in GAUSSIAN_SIGMAS:
        channels.append(gaussian_filter(image, sigma=sigma).astype(np.float32))
    # Local moments describe regional brightness and texture.
    for size in LOCAL_WINDOW_SIZES:
        mean = uniform_filter(image, size=size).astype(np.float32)
        mean_squared = uniform_filter(image * image, size=size).astype(np.float32)
        variance = np.maximum(mean_squared - mean * mean, 0.0)
        channels.extend((mean, np.sqrt(variance).astype(np.float32)))
    # Derivative responses capture boundaries and local structural changes.
    gradients = np.gradient(image)
    channels.append(
        np.sqrt(sum(gradient * gradient for gradient in gradients)).astype(np.float32)
    )
    for sigma in (1.0, 2.0):
        channels.append(gaussian_laplace(image, sigma=sigma).astype(np.float32))
    # Consistent preprocessing makes normalized coordinates useful as an
    # anatomical position prior.
    coordinates = np.meshgrid(
        *(np.linspace(0.0, 1.0, size, dtype=np.float32) for size in image.shape),
        indexing="ij",
    )
    channels.extend(coordinates)
    return tuple(channels)


def _features_at_coordinates(
    channels: tuple[np.ndarray, ...], coordinates: np.ndarray
) -> np.ndarray:
    # Convert volume channels into one feature row per requested voxel.
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n_voxels, 3)")
    if coordinates.size and (
        np.any(coordinates < 0)
        or any(np.any(coordinates[:, axis] >= channels[0].shape[axis]) for axis in range(3))
    ):
        raise ValueError("Voxel coordinates are outside the image bounds")
    indices = tuple(coordinates[:, axis] for axis in range(3))
    return np.column_stack([channel[indices] for channel in channels]).astype(
        np.float32, copy=False
    )


def _stage1_labels(label: np.ndarray) -> np.ndarray:
    # Stage 1 treats tumour as part of the pancreas foreground.
    return (label > 0).astype(np.uint8)


def _stage2_labels(label: np.ndarray) -> np.ndarray:
    # Stage 2 separates tumour from all other voxels in its pancreas ROI.
    return (label == 2).astype(np.uint8)


def _sample_training_data(
    data_dir: Path,
    case_names: list[str],
    stage: int,
    samples_per_class: int,
    samples_per_class_per_case: int,
    dilation_iterations: int,
    negative_ratio: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample balanced rows for one stage, using ground-truth Stage 2 ROIs."""
    # Balanced sampling prevents the background or non-tumour class from
    # dominating the forest fit.
    rng = np.random.default_rng(seed)
    class_features: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for case_name in case_names:
        case_path = data_dir / f"{case_name}.npz"
        if not case_path.is_file():
            raise FileNotFoundError(f"Processed case is missing: {case_path}")
        with np.load(case_path) as case:
            image = case["image"].astype(np.float32, copy=False)
            label = case["label"]
        if image.shape != label.shape:
            raise ValueError(
                f"Image and label shapes differ for {case_name}: "
                f"{image.shape} != {label.shape}"
            )
        channels = compute_feature_channels(image)
        if stage == 1:
            # Stage 1 samples across the complete volume.
            targets = _stage1_labels(label)
            region = np.ones(label.shape, dtype=bool)
        else:
            # Stage 2 uses ground-truth ROIs during training. Background in the
            # dilation shell remains a valid non-tumour training example.
            targets = _stage2_labels(label)
            region = binary_dilation(
                label > 0, iterations=dilation_iterations
            )
        flat_targets = targets.ravel()
        flat_region = region.ravel()
        for class_id in (0, 1):
            voxels = np.flatnonzero(flat_region & (flat_targets == class_id))
            per_case_limit = (
                samples_per_class_per_case * negative_ratio
                if class_id == 0
                else samples_per_class_per_case
            )
            count = min(len(voxels), per_case_limit)
            if count == 0:
                continue
            selected = rng.choice(voxels, size=count, replace=False)
            coordinates = np.column_stack(np.unravel_index(selected, label.shape))
            class_features[class_id].append(
                _features_at_coordinates(channels, coordinates)
            )

    features_by_class: list[np.ndarray] = []
    labels_by_class: list[np.ndarray] = []
    for class_id in (0, 1):
        # Apply both the per-case cap and the overall cap after collecting
        # samples, while keeping equal class counts where possible.
        if not class_features[class_id]:
            raise ValueError(
                f"No training voxels for class {class_id} in Stage {stage}"
            )
        class_data = np.concatenate(class_features[class_id], axis=0)
        overall_limit = (
            samples_per_class * negative_ratio if class_id == 0 else samples_per_class
        )
        count = min(len(class_data), overall_limit)
        if count < len(class_data):
            selected = rng.choice(len(class_data), size=count, replace=False)
            class_data = class_data[selected]
        features_by_class.append(class_data)
        labels_by_class.append(np.full(count, class_id, dtype=np.uint8))
    return np.concatenate(features_by_class), np.concatenate(labels_by_class)


def _load_splits(path: Path) -> dict[str, list[str]]:
    # All models must use the same non-overlapping patient-level split.
    with path.open(encoding="utf-8") as split_file:
        splits = json.load(split_file)
    required = ("train", "val", "test")
    if not all(isinstance(splits.get(name), list) for name in required):
        raise ValueError(f"{path} must contain train, val, and test case lists")
    if any(not splits[name] for name in required):
        raise ValueError("Train, validation, and test splits must all be non-empty")
    if any(not isinstance(case, str) for name in required for case in splits[name]):
        raise ValueError("Split entries must be case-name strings")
    if len(set(splits["train"] + splits["val"] + splits["test"])) != sum(
        len(splits[name]) for name in required
    ):
        raise ValueError("Patient cases overlap between train, validation, and test")
    return splits


def _fit_forest(
    args: argparse.Namespace,
    features: np.ndarray,
    labels: np.ndarray,
    class_weight: str | None,
):
    # Both stages share the forest hyperparameters but receive different
    # targets, regions of interest, and class-weighting strategies.
    max_features = 1.0 if args.max_features == "all" else args.max_features
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=args.seed,
        n_jobs=-1,
    )
    classifier.fit(features, labels)
    return classifier, max_features


def _predict_probabilities(
    classifier: RandomForestClassifier,
    channels: tuple[np.ndarray, ...],
    coordinates: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    # Batch probability calls to bound the temporary feature matrix size.
    probabilities = np.empty(len(coordinates), dtype=np.float32)
    for start in range(0, len(coordinates), batch_size):
        batch = coordinates[start : start + batch_size]
        probabilities[start : start + len(batch)] = classifier.predict_proba(
            _features_at_coordinates(channels, batch)
        )[:, 1]
    return probabilities


def _largest_component(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Keep the largest connected foreground component above a size threshold."""
    components, count = connected_components(mask, structure=generate_binary_structure(3, 1))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    if sizes[largest] < min_size:
        return np.zeros(mask.shape, dtype=bool)
    return components == largest


def _predict_stage1(
    classifier: RandomForestClassifier,
    channels: tuple[np.ndarray, ...],
    batch_size: int,
    threshold: float,
    min_component_size: int,
) -> np.ndarray:
    # Stage 1 classifies every voxel because it defines the Stage 2 ROI.
    coordinates = np.column_stack(np.unravel_index(np.arange(channels[0].size), channels[0].shape))
    probabilities = _predict_probabilities(classifier, channels, coordinates, batch_size)
    mask = probabilities.reshape(channels[0].shape) >= threshold
    return _largest_component(mask, min_component_size)


def _predict_case(
    stage1_classifier: RandomForestClassifier,
    stage2_classifier: RandomForestClassifier,
    image: np.ndarray,
    batch_size: int,
    dilation_iterations: int,
    stage1_threshold: float,
    stage2_threshold: float,
    min_component_size: int,
) -> np.ndarray:
    # Compute features once, use Stage 1 to define a dilated ROI, then let
    # Stage 2 add tumour labels within that ROI.
    channels = compute_feature_channels(image)
    stage1_mask = _predict_stage1(
        stage1_classifier,
        channels,
        batch_size,
        stage1_threshold,
        min_component_size,
    )
    roi = binary_dilation(stage1_mask, iterations=dilation_iterations)
    coordinates = np.column_stack(np.nonzero(roi))
    stage2_probability = _predict_probabilities(
        stage2_classifier, channels, coordinates, batch_size
    )
    prediction = np.where(stage1_mask, 1, 0).astype(np.uint8)
    # The dilation supplies Stage 2 context, but tumour output stays within
    # the Stage 1 pancreas mask to avoid labelling the shell as tumour.
    stage2_mask = stage2_probability >= stage2_threshold
    stage2_mask &= stage1_mask[tuple(coordinates.T)]
    prediction[tuple(coordinates[stage2_mask].T)] = 2
    return prediction


def train(args: argparse.Namespace) -> None:
    # Fit Stage 1 and Stage 2 only on the shared training patients.
    splits = _load_splits(args.splits)
    stage1_features, stage1_labels = _sample_training_data(
        args.data, splits["train"], 1, args.samples_per_class,
        args.samples_per_class_per_case, args.dilation_iterations,
        args.stage1_negative_ratio, args.seed,
    )
    stage2_features, stage2_labels = _sample_training_data(
        args.data, splits["train"], 2, args.samples_per_class,
        args.samples_per_class_per_case, args.dilation_iterations, 1, args.seed + 1,
    )
    stage1_classifier, max_features = _fit_forest(
        args, stage1_features, stage1_labels, class_weight=args.stage1_class_weight
    )
    stage2_classifier, _ = _fit_forest(
        args, stage2_features, stage2_labels, class_weight=args.stage2_class_weight
    )
    # Store both forests and all protocol settings together so prediction
    # cannot silently use a different dilation or feature configuration.
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "stage1_classifier": stage1_classifier,
            "stage2_classifier": stage2_classifier,
            "config": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "max_features": max_features,
                "stage1_class_weight": args.stage1_class_weight,
                "stage2_class_weight": args.stage2_class_weight,
                "stage1_negative_ratio": args.stage1_negative_ratio,
                "samples_per_class": args.samples_per_class,
                "samples_per_class_per_case": args.samples_per_class_per_case,
                "dilation_iterations": args.dilation_iterations,
                "min_component_size": args.min_component_size,
                "stage1_threshold": 0.5,
                "stage2_threshold": 0.5,
                "feature_channels": (
                    "intensity",
                    "gaussian_sigma_1",
                    "gaussian_sigma_2",
                    "gaussian_sigma_4",
                    "local_mean_5",
                    "local_std_5",
                    "local_mean_9",
                    "local_std_9",
                    "gradient_magnitude",
                    "laplacian_gaussian_sigma_1",
                    "laplacian_gaussian_sigma_2",
                    "coordinate_0",
                    "coordinate_1",
                    "coordinate_2",
                ),
            },
            "train_cases": tuple(splits["train"]),
        },
        args.model,
    )
    print(f"Trained Stage 1 on {len(stage1_labels):,} sampled voxels")
    print(f"Trained Stage 2 on {len(stage2_labels):,} sampled voxels")
    print(f"Saved model to {args.model}")


def _dice(mask: np.ndarray, truth: np.ndarray) -> float:
    denominator = int(mask.sum() + truth.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int(np.logical_and(mask, truth).sum()) / denominator


def _validation_probabilities(
    artifact: dict,
    data_dir: Path,
    case_names: list[str],
    batch_size: int,
) -> list[dict[str, np.ndarray]]:
    """Compute reusable Stage 1 and Stage 2 validation probabilities."""
    results = []
    for case_name in case_names:
        case_path = data_dir / f"{case_name}.npz"
        if not case_path.is_file():
            raise FileNotFoundError(f"Processed case is missing: {case_path}")
        with np.load(case_path) as case:
            image = case["image"].astype(np.float32, copy=False)
            truth = case["label"]
        channels = compute_feature_channels(image)
        coordinates = np.column_stack(
            np.unravel_index(np.arange(image.size), image.shape)
        )
        stage1_probability = _predict_probabilities(
            artifact["stage1_classifier"], channels, coordinates, batch_size
        ).reshape(image.shape)
        roi = binary_dilation(
            stage1_probability >= 0.5,
            iterations=artifact["config"]["dilation_iterations"],
        )
        roi_coordinates = np.column_stack(np.nonzero(roi))
        stage2_probability = _predict_probabilities(
            artifact["stage2_classifier"],
            channels,
            roi_coordinates,
            batch_size,
        )
        results.append(
            {
                "truth": truth,
                "stage1_probability": stage1_probability,
                "roi_coordinates": roi_coordinates,
                "stage2_probability": stage2_probability,
            }
        )
    return results


def tune(args: argparse.Namespace) -> None:
    """Select thresholds on validation data and store them in the artifact."""
    splits = _load_splits(args.splits)
    artifact = joblib.load(args.model)
    validation = _validation_probabilities(
        artifact, args.data, splits["val"], args.batch_size
    )
    min_component_size = artifact["config"]["min_component_size"]

    best_stage1 = (0.0, 0.5)
    for threshold in THRESHOLD_GRID:
        scores = []
        for result in validation:
            mask = _largest_component(
                result["stage1_probability"] >= threshold,
                min_component_size,
            )
            scores.append(_dice(mask, result["truth"] > 0))
        score = float(np.mean(scores))
        if score > best_stage1[0]:
            best_stage1 = (score, threshold)

    best_stage2 = (0.0, 0.5)
    for threshold in THRESHOLD_GRID:
        scores = []
        for result in validation:
            stage1_mask = _largest_component(
                result["stage1_probability"] >= best_stage1[1],
                min_component_size,
            )
            stage2_mask = np.zeros(stage1_mask.shape, dtype=bool)
            selected = result["stage2_probability"] >= threshold
            stage2_mask[tuple(result["roi_coordinates"][selected].T)] = True
            scores.append(_dice(stage2_mask & stage1_mask, result["truth"] == 2))
        score = float(np.mean(scores))
        if score > best_stage2[0]:
            best_stage2 = (score, threshold)

    artifact["config"]["stage1_threshold"] = best_stage1[1]
    artifact["config"]["stage2_threshold"] = best_stage2[1]
    joblib.dump(artifact, args.model)
    print(
        f"Selected Stage 1 threshold {best_stage1[1]:.2f} "
        f"(mean pancreas Dice {best_stage1[0]:.3f})"
    )
    print(
        f"Selected Stage 2 threshold {best_stage2[1]:.2f} "
        f"(mean tumour Dice {best_stage2[0]:.3f})"
    )
    print(f"Updated model artifact: {args.model}")


def predict(args: argparse.Namespace) -> None:
    # Validation and test inference intentionally use predicted, not
    # ground-truth, Stage 1 masks.
    splits = _load_splits(args.splits)
    artifact = joblib.load(args.model)
    stage1_classifier = artifact["stage1_classifier"]
    stage2_classifier = artifact["stage2_classifier"]
    config = artifact["config"]
    dilation_iterations = config["dilation_iterations"]
    stage1_threshold = (
        config["stage1_threshold"]
        if args.stage1_threshold is None
        else args.stage1_threshold
    )
    stage2_threshold = (
        config["stage2_threshold"]
        if args.stage2_threshold is None
        else args.stage2_threshold
    )
    args.out.mkdir(parents=True, exist_ok=True)
    totals = {1: {"dice_sum": 0.0, "case_count": 0}, 2: {"dice_sum": 0.0, "case_count": 0}}
    for case_name in splits[args.split]:
        # The same saved feature protocol and ROI composition are used for
        # every case in the selected shared split.
        case_path = args.data / f"{case_name}.npz"
        if not case_path.is_file():
            raise FileNotFoundError(f"Processed case is missing: {case_path}")
        with np.load(case_path) as case:
            image = case["image"].astype(np.float32, copy=False)
            truth = case["label"]
        prediction = _predict_case(
            stage1_classifier,
            stage2_classifier,
            image,
            args.batch_size,
            dilation_iterations,
            stage1_threshold,
            stage2_threshold,
            config["min_component_size"],
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
        prediction_path = args.out / f"{case_name}.npz"
        np.savez_compressed(prediction_path, prediction=prediction)
        print(f"Predicted {case_name} -> {prediction_path}")
    print(f"Mean per-case Dice on the {args.split} split:")
    for class_id, values in totals.items():
        score = values["dice_sum"] / values["case_count"] if values["case_count"] else 0.0
        print(f"  {CLASS_NAMES[class_id]}: {score:.4f} ({values['case_count']} cases)")


def build_parser() -> argparse.ArgumentParser:
    # Keep the original train/predict workflow while exposing the new
    # two-stage dilation setting.
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="fit and save both random forests")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--splits", type=Path, default=DEFAULT_DATA_DIR / "splits.json")
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--n-estimators", type=int, default=200)
    train_parser.add_argument("--max-depth", type=int, default=20)
    train_parser.add_argument("--min-samples-leaf", type=int, default=2)
    train_parser.add_argument("--max-features", choices=("sqrt", "log2", "all"), default="sqrt")
    train_parser.add_argument(
        "--stage1-class-weight",
        choices=("none", "balanced", "balanced_subsample"),
        default="none",
        help="class weighting for the Stage 1 forest",
    )
    train_parser.add_argument(
        "--stage2-class-weight",
        choices=("none", "balanced", "balanced_subsample"),
        default="balanced_subsample",
        help="class weighting for the Stage 2 forest",
    )
    train_parser.add_argument("--samples-per-class", type=int, default=10_000)
    train_parser.add_argument("--samples-per-class-per-case", type=int, default=250)
    train_parser.add_argument(
        "--dilation-iterations", type=int, default=DEFAULT_DILATION_ITERATIONS
    )
    train_parser.add_argument(
        "--stage1-negative-ratio",
        type=int,
        default=5,
        help="background samples per pancreas sample during Stage 1 training",
    )
    train_parser.add_argument(
        "--min-component-size",
        type=int,
        default=DEFAULT_MIN_COMPONENT_SIZE,
        help="minimum Stage 1 connected-component size in voxels",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)
    tune_parser = commands.add_parser(
        "tune", help="select Stage 1 and Stage 2 thresholds on validation data"
    )
    tune_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    tune_parser.add_argument("--splits", type=Path, default=DEFAULT_DATA_DIR / "splits.json")
    tune_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    tune_parser.add_argument("--batch-size", type=int, default=100_000)
    tune_parser.set_defaults(func=tune)
    predict_parser = commands.add_parser(
        "predict", help="predict a labelled split and report mean per-case Dice"
    )
    predict_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    predict_parser.add_argument("--splits", type=Path, default=DEFAULT_DATA_DIR / "splits.json")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    predict_parser.add_argument("--out", type=Path, default=Path("data/predictions/random_forest"))
    predict_parser.add_argument("--batch-size", type=int, default=100_000)
    predict_parser.add_argument("--stage1-threshold", type=float)
    predict_parser.add_argument("--stage2-threshold", type=float)
    predict_parser.set_defaults(func=predict)
    return parser


def main() -> None:
    # Reject invalid resource and morphology settings before training or
    # prediction begins.
    args = build_parser().parse_args()
    if args.command == "train":
        args.stage1_class_weight = (
            None if args.stage1_class_weight == "none" else args.stage1_class_weight
        )
        args.stage2_class_weight = (
            None if args.stage2_class_weight == "none" else args.stage2_class_weight
        )
        if (
            args.n_estimators <= 0
            or args.max_depth <= 0
            or args.min_samples_leaf <= 0
            or args.samples_per_class <= 0
            or args.samples_per_class_per_case <= 0
            or args.dilation_iterations < 0
            or args.stage1_negative_ratio <= 0
            or args.min_component_size < 0
        ):
            raise ValueError("Forest, sample, and dilation settings are invalid")
    elif args.command in ("tune", "predict") and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.command == "predict":
        for threshold in (args.stage1_threshold, args.stage2_threshold):
            if threshold is not None and not 0.0 <= threshold <= 1.0:
                raise ValueError("Prediction thresholds must be between 0 and 1")
    args.func(args)


if __name__ == "__main__":
    main()
