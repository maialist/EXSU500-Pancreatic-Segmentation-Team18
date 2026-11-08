
"""
Calculate dataset statistics for Task07_Pancreas.

Inputs:
    imagesTr/  - original CT images (.nii.gz)
    labelsTr/  - corresponding segmentation masks (.nii.gz)

Outputs:
    dataset_statistics/
        per_case_statistics.csv
        dataset_summary.csv

Label convention:
    0 = background
    1 = pancreas
    2 = tumour
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


CLASS_NAMES = {
    0: "background",
    1: "pancreas",
    2: "tumour",
}


def get_case_statistics(image_path, label_path):
    """Calculate image and segmentation statistics for one patient."""

    # get image and label data
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))

    # get shape
    image_shape = image.shape[:3]
    label_shape = label.shape[:3]

    if image_shape != label_shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: "
            f"image={image_shape}, label={label_shape}"
        )

    # get spacing (voxel size in mm)
    spacing = image.header.get_zooms()[:3]

    if len(spacing) != 3:
        raise ValueError(f"Invalid spacing: {image_path.name}")

    voxel_volume = float(np.prod(spacing))

    segmentation = np.asarray(label.dataobj)

    if not np.all(np.isin(segmentation, list(CLASS_NAMES))):
        raise ValueError(
            f"Unexpected label values in {label_path.name}: "
            f"{np.unique(segmentation)}"
        )

    total_voxels = segmentation.size

    # print stats
    stats = {
        "case": image_path.name,
        "voxels_x": image_shape[0],
        "voxels_y": image_shape[1],
        "voxels_z": image_shape[2],
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "voxel_volume_mm3": voxel_volume,
    }

    # Per-class voxel fractions
    for class_id, class_name in CLASS_NAMES.items():
        count = np.count_nonzero(segmentation == class_id)

        stats[f"{class_name}_voxels"] = count
        stats[f"{class_name}_fraction"] = count / total_voxels

    # Tumour volume in mm3
    tumour_voxels = stats["tumour_voxels"]
    stats["tumour_volume_mm3"] = tumour_voxels * voxel_volume

    return stats


def summarize_column(data, column):
    """Return descriptive statistics for a column."""

    values = data[column].dropna()

    return {
        "metric": column,
        "min": values.min(),
        "Q1": values.quantile(0.25),
        "median": values.median(),
        "Q3": values.quantile(0.75),
        "max": values.max(),
        "IQR": values.quantile(0.75) - values.quantile(0.25),
        "mean": values.mean(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Task07_Pancreas dataset statistics."
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing CT images."
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Directory containing segmentation labels."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_statistics"),
        help="Output directory."
    )

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    image_files = sorted(args.images.glob("*.nii.gz"))

    if not image_files:
        raise FileNotFoundError(
            f"No .nii.gz images found in {args.images}"
        )

    results = []

    for image_path in image_files:

        label_path = args.labels / image_path.name

        if not label_path.is_file():
            print(f"WARNING: Missing label for {image_path.name}")
            continue

        print(f"Processing: {image_path.name}")

        results.append(
            get_case_statistics(image_path, label_path)
        )

    if not results:
        raise RuntimeError("No matching image-label pairs found.")

    df = pd.DataFrame(results)

    # Save per-case statistics
    per_case_path = args.output / "per_case_statistics.csv"
    df.to_csv(per_case_path, index=False)

    # Summarize image dimensions, spacing, and tumour volume
    columns = [
        "voxels_x", "voxels_y", "voxels_z",
        "spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
        "tumour_volume_mm3",
    ]

    summary = [
        summarize_column(df, column)
        for column in columns
    ]

    # Per-class fractions: summarize across individual patients
    for class_name in CLASS_NAMES.values():
        column = f"{class_name}_fraction"
        summary.append(summarize_column(df, column))

    summary_df = pd.DataFrame(summary)

    # Also calculate pooled voxel fractions across all patients
    pooled = []
    total_voxels = df[
        [f"{name}_voxels" for name in CLASS_NAMES.values()]
    ].sum().sum()

    for class_name in CLASS_NAMES.values():
        count = df[f"{class_name}_voxels"].sum()

        pooled.append({
            "metric": f"{class_name}_pooled_fraction",
            "min": np.nan,
            "Q1": np.nan,
            "median": np.nan,
            "Q3": np.nan,
            "max": np.nan,
            "IQR": np.nan,
            "mean": count / total_voxels,
        })

    summary_df = pd.concat(
        [summary_df, pd.DataFrame(pooled)],
        ignore_index=True
    )

    summary_path = args.output / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Report number of cases with tumour labels
    tumour_cases = (df["tumour_voxels"] > 0).sum()

    print("\n--- Dataset summary ---")
    print(f"Cases analyzed: {len(df)}")
    print(f"Cases with tumour labels: {tumour_cases}")
    print(f"Cases without tumour labels: {len(df) - tumour_cases}")

    print("\nTumour volume (mm3):")
    print(f"Median: {df['tumour_volume_mm3'].median():.2f}")
    print(
        f"IQR: "
        f"{df['tumour_volume_mm3'].quantile(0.25):.2f} - "
        f"{df['tumour_volume_mm3'].quantile(0.75):.2f}"
    )

    print(f"\nSaved results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
