# EXSU500-Pancreatic-Segmentation

## Project Overview

This project investigates the use of artificial intelligence to accurately identify and delineate pancreatic tumours on CT imaging. Automated tumour segmentation could provide a foundation for computer-assisted visualization of tumor anatomy during preoperative planning and future image-guided surgical systems. 

The project will involve training and comparing multiple machine learning models to determine how accurately they can perform the segmentation task.

The project is being completed as part of **EXSU 500: Fundamentals of Artificial Intelligence in Medicine** at McGill University.

The intended use of the model is **research**. It is not intended for clinical decision-making.

---

## Task Definition

**Task type:** Multi-class image segmentation

**Specific task:** Voxel-wise segmentation of background, pancreas, and pancreatic tumour

### Task Description

The models take **CT** medical images as input and predict background (label 0), pancreas (label 1), and pancreatic tumour (label 2) at the voxel level. The classical models and 3D U-Net also support a tumour-versus-rest binary target.

---

## Dataset

Dataset from Task07 of the Medical Segmentation Decathlon - 421 3D volumes of portal venous phase CT scans of patients with pancreatic tumours. 

All scans are sourced from the Memorial Sloan Kettering Cancer Center (New York City, NY, USA). 

### Source

**Dataset provider:** Medical Segmentation Decathlon

**URL:** http://medicaldecathlon.com/

**Citation:** https://arxiv.org/abs/1902.09063 

---

## Python environment

The project dependencies are pinned in `requirements.txt` and were exported from Python 3.11.7. To recreate the environment from the repository root:

```bash
python3.11 -m venv .pancsegenv
source .pancsegenv/bin/activate
python -m pip install -r requirements.txt
```

The virtual environment itself is excluded by `.gitignore`; share the requirements file rather than committing the machine-specific environment directory.

---

## Logistic regression baseline

The voxel-wise logistic-regression baseline uses the shared patient-level `train`/`val`/`test` split and preprocessed arrays. Each voxel is represented by its edge-padded 3x3x3 neighbourhood of normalized CT intensities (27 features), and each volume is predicted in batches. It supports three-class segmentation and tumour-versus-rest prediction.

Run training and validation prediction from the repository root:

```bash
python models/logistic_regression.py train
python models/logistic_regression.py predict --split val
```

Training saves `data/models/logistic_regression.joblib`. Predictions are written to `data/predictions/logistic_regression/`; mean per-case Dice is reported for the selected labelled split, excluding cases where both the reference and prediction are empty for a class. Use validation predictions for model selection; after choices are fixed, evaluate the test split once:

```bash
python models/logistic_regression.py predict --split test
```

Train a binary tumour-versus-rest model with `python models/logistic_regression.py train --target tumour`. Its saved predictions use the original label values (0 for background/non-tumour and 2 for tumour).

### Fixed comparison protocol and logistic-regression settings

The source data, label definitions, patient-level split (`data/processed/splits.json`), preprocessing (RAS orientation, 1.5 x 1.5 x 2.5 mm spacing, -100 to 240 HU window, and [0, 1] intensity scaling), and validation/test protocol are shared across models. Do not change preprocessing or the split for this baseline alone.

The starting model uses L2 regularization (`l1_ratio=0.0`), `solver="lbfgs"`, `C=1.0`, `max_iter=500`, balanced class weights, and seed 42. Training samples at most 250 voxels per class per training patient and 10,000 per class overall, with equal class sample counts when each class has enough voxels. Intensities already share the same [0, 1] scale, so no additional feature scaler is applied. These are reproducible starting settings, not tuned or clinically validated recommendations.

You can change `--target`, `--c`, `--max-iter`, `--samples-per-class`, `--samples-per-class-per-case`, and `--seed`. `--data`, `--splits`, and `--model` select input, split, and model artifact paths; prediction additionally requires `--split train|val|test` and accepts `--batch-size` and `--out`. Training settings and training case names are saved in the model artifact. Tune on validation data and reserve test data for the final evaluation.

## Random forest

The voxel-wise random forest uses the same preprocessed volumes and shared patient-level split as logistic regression. It extracts the same edge-padded 3x3x3 normalized CT intensity neighbourhood (27 features) and samples a balanced number of voxels from the training patients. It supports both three-class pancreas/tumour segmentation and tumour-versus-rest prediction.

Install scikit-learn and joblib in the same environment as the preprocessing dependencies, then run:

```bash
python models/random_forest.py train
python models/random_forest.py predict --split val
```

Training saves `data/models/random_forest.joblib`; predictions and mean per-case Dice are written for labelled validation or test cases, using the same Dice convention as logistic regression. The default model is three-class. Use validation predictions to select settings, then evaluate the test split once after choices are fixed:

```bash
python models/random_forest.py predict --split test
```

The project baseline sets `n_estimators=200` (number of trees), `max_depth=20` (limits tree complexity), `min_samples_leaf=2` (discourages leaves based on a single voxel), and `max_features="sqrt"` (each split considers the square root of the 27 input features). It uses `class_weight="balanced_subsample"` to account for class imbalance within each bootstrap sample, `n_jobs=-1` to train trees in parallel, and seed 42 for reproducibility. Training samples at most 250 voxels per class per patient and 10,000 per class overall, making the three classes equally represented in the fit data. These are starting settings for this dataset and runtime, not clinically validated or guaranteed optimal settings.

You can change the target, tree count, depth, minimum leaf size, feature subset, sample caps, and seed with `--target`, `--n-estimators`, `--max-depth`, `--min-samples-leaf`, `--max-features`, `--samples-per-class`, `--samples-per-class-per-case`, and `--seed`. `--max-features` accepts `sqrt`, `log2`, or `all` (all 27 features). For example:

```bash
python models/random_forest.py train --n-estimators 300 --max-depth 24 --min-samples-leaf 3
python models/random_forest.py train --target tumour
```

The exact model and sampling settings are stored in the model artifact. Keep preprocessing and the split unchanged for comparison; tune on validation data only and record selected settings before the single test evaluation.

## 3D U-Net

The 3D U-Net trains on random patches from the preprocessed volumes and the same patient-level split as the classical baselines. It uses a four-level encoder/decoder with skip connections, GroupNorm, and a combined multiclass cross-entropy plus foreground Dice loss. Training evaluates full validation volumes with sliding-window inference and saves the checkpoint with the best mean foreground per-case Dice. Predictions are also produced with sliding windows and saved as compressed `.npz` files.

Train and predict on the validation split:

```bash
python models/unet3d.py train
python models/unet3d.py predict --split val
```

The default model artifact is `data/models/unet3d.pt`. Once settings are chosen on validation data, run final test evaluation with:

```bash
python models/unet3d.py predict --split test
```

Training supports multiclass background/pancreas/tumour segmentation by default, or tumour-versus-rest with `--target tumour`. For example:

```bash
python models/unet3d.py train --patch-size 64 64 32 --base-channels 16 --epochs 50
```

### Editable U-Net parameters and starting point

The starting configuration uses multiclass labels, a `64 x 64 x 32` voxel patch, 16 base feature channels, 50 epochs, one sampled patch per training case per epoch, batch size 2, and a 75% chance that a patch is centered on a labelled foreground voxel. Foreground patch centers are chosen among the foreground classes present in that case, giving pancreas and tumour equal chances when both are present. Optimization uses AdamW with learning rate `0.001` and weight decay `0.00001`; cross-entropy and soft Dice losses each have weight `1`. Seed 42 is used for reproducibility, validation is run every 5 epochs, and the checkpoint with the best validation mean foreground Dice is retained.

The following training options can be changed from the command line:

| Option | Starting value | Purpose |
|---|---:|---|
| `--target` | `multiclass` | Predict all three labels, or use `tumour` for tumour-versus-rest. |
| `--patch-size` | `64 64 32` | 3D patch dimensions in voxels. |
| `--base-channels` | `16` | Feature channels in the first U-Net level; later levels double each time. |
| `--epochs` | `50` | Maximum training passes through the training cases. |
| `--patches-per-case` | `1` | Random patches sampled from each training case per epoch. |
| `--batch-size` | `2` | Number of patches per optimizer update. |
| `--foreground-probability` | `0.75` | Probability of centering a patch on a labelled foreground voxel rather than a random location. |
| `--learning-rate` | `0.001` | AdamW learning rate. |
| `--weight-decay` | `0.00001` | AdamW weight decay. |
| `--ce-weight` | `1` | Cross-entropy contribution to the loss. |
| `--dice-weight` | `1` | Foreground soft-Dice contribution to the loss. |
| `--seed` | `42` | Random seed for initialization, case order, and patch sampling. |
| `--validation-every` | `5` | Run full-volume validation after this many epochs (and always on the last epoch). |
| `--overlap` | `0.5` | Fractional overlap of sliding windows during validation or prediction. |
| `--inference-batch-size` | `1` | Number of windows evaluated together during training validation. |
| `--device` | `auto` | `auto` selects CUDA, Apple MPS, or CPU; a device can also be selected explicitly. |

`--data`, `--splits`, and `--model` change the data, shared split, and checkpoint paths. Prediction additionally requires `--split val` or `--split test`, and accepts `--batch-size` for inference windows plus `--out` for the output directory. Training and prediction use the same saved patch size and architecture. The initial values are reproducible starting settings, not tuned or clinically validated recommendations. Keep preprocessing and `splits.json` unchanged for comparisons, select settings using validation data, and use test data only for final evaluation. This project remains research-only and is not for clinical decision-making.

The CLI exposes the main training and inference hyperparameters. The current implementation keeps the four-level network topology, two 3D convolutions per block, GroupNorm/LeakyReLU, AdamW, and the cross-entropy/soft-Dice loss formulation fixed; changing those requires editing `models/unet3d.py`. The dataset, preprocessing, and shared split are protocol choices rather than U-Net-specific tuning parameters.
