
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

# Calculate statistics for a single case.
def get_case_statistics(image_path, label_path):
    """Calculate image and segmentation statistics for one patient."""

    # get image and label
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))

    # get shape 
    image_shape = image.shape[:3]
    label_shape = label.shape[:3]

    # ensure image and label shapes match
    if image_shape != label_shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: "
            f"image={image_shape}, label={label_shape}"
        )

    # get spacing (voxel size in mm)
    spacing = image.header.get_zooms()[:3]

    # ensure spacing is in 3 dimensions
    if len(spacing) != 3:
        raise ValueError(f"Invalid spacing: {image_path.name}")

    # calculate voxel volume in mm^3 from spacing
    voxel_volume = float(np.prod(spacing))

    # get segmentation array
    segmentation = np.asarray(label.dataobj)

    # ensure segmentation contains only expected class labels
    if not np.all(np.isin(segmentation, list(CLASS_NAMES))):
        raise ValueError(
            f"Unexpected label values in {label_path.name}: "
            f"{np.unique(segmentation)}"
        )

    # calculate total number of voxels in the segmentation
    total_voxels = segmentation.size

    # print stats
    stats = {
        "case": image_path.name, # case name
        "voxels_x": image_shape[0], # number of voxels along x-axis
        "voxels_y": image_shape[1], # number of voxels along y-axis
        "voxels_z": image_shape[2], # number of voxels along z-axis
        "spacing_x_mm": spacing[0], # spacing along x-axis in mm
        "spacing_y_mm": spacing[1], # spacing along y-axis in mm
        "spacing_z_mm": spacing[2], # spacing along z-axis in mm
        "voxel_volume_mm3": voxel_volume, # volume of a single voxel in mm^3
    }

    # Calculate per-class voxel fractions for each class
    for class_id, class_name in CLASS_NAMES.items():
        # count number of voxels for this class
        count = np.count_nonzero(segmentation == class_id)
        # calculate fraction of total voxels for this class
        stats[f"{class_name}_voxels"] = count
        stats[f"{class_name}_fraction"] = count / total_voxels

    # Tumour volume in mm3
    # get number of tumour voxels
    tumour_voxels = stats["tumour_voxels"]
    # calculate tumour volume in mm^3
    stats["tumour_volume_mm3"] = tumour_voxels * voxel_volume

    return stats

# Summarize statistics for a column.
def summarize_column(data, column):
    """Return descriptive statistics for a column."""

    # drop NaN values for the column
    values = data[column].dropna()

    # return summary statistics as a dictionary
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

    # make output directory if it doesn't exist
    args.output.mkdir(parents=True, exist_ok=True)

    # get list of image files
    image_files = sorted(args.images.glob("*.nii.gz"))
    # check if any image files were found
    if not image_files:
        raise FileNotFoundError(
            f"No .nii.gz images found in {args.images}"
        )

    results = [] # list to store per-case statistics

    # iterate over image files and calculate statistics
    for image_path in image_files:
        # get corresponding label path
        label_path = args.labels / image_path.name
        # ensure label file exists
        if not label_path.is_file():
            print(f"WARNING: Missing label for {image_path.name}")
            continue

        print(f"Processing: {image_path.name}")
        # calculate statistics for this case and append to results
        results.append(
            get_case_statistics(image_path, label_path)
        )

    # check if any results were collected
    if not results:
        raise RuntimeError("No matching image-label pairs found.")

    # convert results to a DataFrame 
    df = pd.DataFrame(results)

    # save per-case statistics file
    per_case_path = args.output / "per_case_statistics.csv"
    df.to_csv(per_case_path, index=False)

    # create columns for image dimensions, spacing, and tumour volume
    columns = [
        "voxels_x", "voxels_y", "voxels_z",
        "spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
        "tumour_volume_mm3",
    ]

    # summarize statistics for each column and store in a list of dictionaries
    summary = [
        summarize_column(df, column)
        for column in columns
    ]

    # summarize columns across patients
    for class_name in CLASS_NAMES.values():
        column = f"{class_name}_fraction"
        summary.append(summarize_column(df, column))
    # convert summary to a DataFrame
    summary_df = pd.DataFrame(summary)

    # calculate pooled voxel fractions across all patients
    pooled = []
    total_voxels = df[
        # sum the total number of voxels across all patients
        [f"{name}_voxels" for name in CLASS_NAMES.values()]
    ].sum().sum()

    # calculate pooled fractions for each class
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

    # add pooled fractions to the summary DataFrame
    summary_df = pd.concat(
        [summary_df, pd.DataFrame(pooled)],
        ignore_index=True
    )

    # save dataset summary file
    summary_path = args.output / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # find total number of cases with tumour labels
    tumour_cases = (df["tumour_voxels"] > 0).sum()

    # print dataset summary to console
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
