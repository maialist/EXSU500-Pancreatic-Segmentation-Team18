"""Preprocess the Medical Segmentation Decathlon pancreas CT data (.nii.gz).

Requires: pip install nibabel numpy scipy

Pipeline per case:
  1. Load image + label, reorient to RAS+
  2. Resample to a common voxel spacing (linear for CT, nearest for labels)
  3. Clip HU to a soft-tissue window and scale to [0, 1]
  4. Save as compressed .npz (float16 image, uint8 label) in the output folder

Also writes a patient-level train/val/test split (splits.json) so that all three
models (logistic regression, random forest, 3D UNet) are evaluated on identical
cases.

Inputs:
    Task07_raw_data/  - Task07 raw data directory from the Medical Segmentation Decathlon (MSD)

Outputs:
    data/
        processed/
            case_00000.npz - all processed cases (image + label)
            splits.json - train/val/test split of cases for reproducibility

Usage:
  python preprocess.py --raw "/path/to/EXSU500_PancreasSeg_raw_data" --out data/processed
"""
import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

DEFAULT_RAW = Path.home() / "Downloads/McGill/Master's/EXSU 500/EXSU500_PancreasSeg_raw_data"
HU_MIN, HU_MAX = -100.0, 240.0  # soft-tissue window
TARGET_SPACING = (1.5, 1.5, 2.5)  # mm, RAS axes

# Reorients NIfTI image to nearest RAS+ orientation to obtain NumPy array and spacing. Returns (array, spacing).
def load_ras(path):
    """Load a NIfTI file reoriented to RAS+; returns (array, spacing)."""
    # load image and reorient to RAS+ 
    img = nib.as_closest_canonical(nib.load(str(path)))
    # get data as array 
    data = np.asarray(img.dataobj, dtype=np.float32)
    # return data and spacing (x, y, z) (zoom = voxel size in mm)   
    return data, img.header.get_zooms()[:3]

# Resample volume to target spacing using specified interpolation order. Returns resampled volume.
def resample(vol, spacing, target, order):
    # compute factors for each axis (current spacing / target spacing)
    factors = [s / t for s, t in zip(spacing, target)]
    return zoom(vol, factors, order=order)

# Preprocess a single case: load, reorient, resample, clip, scale, and return (image, label).
def preprocess_case(image_path, label_path):
    # load image and reorient to RAS+
    img, spacing = load_ras(image_path)
    # load label and reorient to RAS+
    lbl, _ = load_ras(label_path)
    # resample to target spacing (linear for image, nearest for label)
    img = resample(img, spacing, TARGET_SPACING, order=1)
    lbl = resample(lbl, spacing, TARGET_SPACING, order=0).astype(np.uint8)
    # clip intensity to soft-tissue window and scale to [0, 1]
    img = (np.clip(img, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    # return image as float16 and label as uint8
    return img.astype(np.float16), lbl

# Create train/val/test splits for a list of case names.
def make_splits(names, val_frac, test_frac, seed):
    # sort list for reproducibility
    names = sorted(names)
    # shuffle with fixed seed
    random.Random(seed).shuffle(names)
    # compute number of cases for each split
    n_test = round(len(names) * test_frac)
    n_val = round(len(names) * val_frac)
    # return dictionary of splits
    return {
        "test": names[:n_test],
        "val": names[n_test:n_test + n_val],
        "train": names[n_test + n_val:],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="process only N cases (debug)")
    args = p.parse_args()

    # get images in the raw folder (ignore hidden files)
    images = sorted(p for p in (args.raw / "imagesTr").glob("*.nii.gz") if not p.name.startswith("._"))
    # limit number of cases if specified
    if args.limit:
        images = images[: args.limit]
    # make out directory if it doesn't exist
    args.out.mkdir(parents=True, exist_ok=True)

    done = [] # list of processed case names
    # iterate over images and preprocess each case
    for i, img_path in enumerate(images, 1):
        name = img_path.name[: -len(".nii.gz")] # case name without extension
        label_path = args.raw / "labelsTr" / img_path.name # label path
        out_path = args.out / f"{name}.npz" # output path
        # skip if label doesn't exist
        if not label_path.exists():
            print(f"[skip] no label for {name}")
            continue
        # process and save (if output doesn't exist)
        if not out_path.exists():
            img, lbl = preprocess_case(img_path, label_path)
            np.savez_compressed(out_path, image=img, label=lbl)
        # add case to done list
        done.append(name)
        print(f"[{i}/{len(images)}] {name}")

    # make train/val/test splits and save to JSON
    splits = make_splits(done, args.val_frac, args.test_frac, args.seed)
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2))
    print({k: len(v) for k, v in splits.items()})


def load_case(path):
    """Helper for training scripts: returns (image float32, label uint8)."""
    d = np.load(path)
    return d["image"].astype(np.float32), d["label"]


if __name__ == "__main__":
    main()
