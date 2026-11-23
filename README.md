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

The voxel-wise logistic-regression baseline uses the shared patient-level `train`/`val`/`test` split and preprocessed arrays. Each voxel is represented by its edge-padded 3x3x3 neighbourhood of normalized CT intensities (27 features) plus 6 context features (a log spatial atlas prior built from training labels in normalised coordinates, Gaussian-smoothed intensities at four scales, and local standard deviation). Training samples hard negatives near the organs; `tune` (validation only) then picks per-class probability thresholds and largest-component post-processing, which `predict` applies. These features differ from the random forest and U-Net inputs. Run `train`, `tune`, then `predict`. It supports three-class segmentation and tumour-versus-rest prediction.

Run training and validation prediction from the repository root:

```bash
python models/logistic_regression.py train
python models/logistic_regression.py tune
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

The random forest uses the same preprocessed volumes and shared patient-level split as logistic regression, but is trained as a two-stage model. Stage 1 classifies background versus pancreas (pancreas and tumour combined). Stage 2 classifies tumour versus non-tumour inside a dilation of the pancreas region. Stage 2 uses ground-truth pancreas masks while training and predicted Stage 1 masks during validation and test prediction, so validation and test results include Stage 1 errors. This protocol was developed from Farag et al. to better represent the features of the dataset using a hierarchichal approach.

Install scikit-learn and joblib in the same environment as the preprocessing dependencies, then run:

```bash
python models/random_forest.py train
python models/random_forest.py tune
python models/random_forest.py predict --split val
```

Training saves both forests and their configuration in `data/models/random_forest.joblib`; predictions and mean per-case Dice are written for labelled validation or test cases, using the same Dice convention as logistic regression. Stage 2 is evaluated only in a dilation of the predicted Stage 1 mask, and the final output retains labels 0 (background), 1 (pancreas), and 2 (tumour). Use validation predictions to select settings, then evaluate the test split once after choices are fixed:

```bash
python models/random_forest.py predict --split test
```

Instead of raw 3x3x3 intensity patches, each volume is transformed once into reusable channels: normalized intensity, Gaussian-smoothed intensity at sigma 1, 2, and 4, local mean and standard deviation with window sizes 5 and 9, gradient magnitude, Laplacian of Gaussian at sigma 1 and 2, and three normalized spatial coordinates. This gives both forests local intensity context, texture, edges, and a coarse anatomical position prior without recomputing filters for every voxel batch.

The project baseline sets `n_estimators=200` (per stage), `max_depth=20`, `min_samples_leaf=2`, and `max_features="sqrt"`. Stage 1 samples five background voxels per pancreas voxel and uses no additional class weighting; Stage 2 samples balanced tumour/non-tumour voxels inside dilated ground-truth pancreas regions and uses `class_weight="balanced_subsample"`. Both forests use `n_jobs=-1` and seed 42. The Stage 1 foreground ratio is intentionally less extreme than the full-scan ratio, while exposing the forest to substantially more background than the previous balanced sampler.

Training initializes conservative probability thresholds at 0.5. Run `tune` on the validation split after training to select separate Stage 1 and Stage 2 thresholds. Tuning also uses the largest connected Stage 1 component and stores the selected thresholds in the model artifact. Prediction uses those saved thresholds. The Stage 2 dilation supplies context, but tumour labels are emitted only inside the cleaned Stage 1 mask.

You can change the tree count, depth, minimum leaf size, feature subset, class weights, sample caps, Stage 1 negative ratio, dilation size, minimum component size, and seed with `--n-estimators`, `--max-depth`, `--min-samples-leaf`, `--max-features`, `--stage1-class-weight`, `--stage2-class-weight`, `--samples-per-class`, `--samples-per-class-per-case`, `--stage1-negative-ratio`, `--dilation-iterations`, `--min-component-size`, and `--seed`. `--max-features` accepts `sqrt`, `log2`, or `all` (all 14 filter and position channels); class weights accept `none`, `balanced`, or `balanced_subsample`. For example:

```bash
python models/random_forest.py train --n-estimators 300 --max-depth 24 --min-samples-leaf 3
python models/random_forest.py train --dilation-iterations 4
python models/random_forest.py tune
```

The exact model, feature, sampling, class-weight, dilation, component, and threshold settings are stored in the model artifact. Keep preprocessing and the split unchanged for comparison; tune on validation data only and record selected settings before the single test evaluation. The model remains a research baseline and is not clinically validated.

### Random-forest experiments

The repository includes a reproducible proxy-sweep workflow under `experiments/`. It is currently implemented for the random forest, but its run manifests and long-format metrics are intentionally model-neutral so equivalent logistic-regression and U-Net sweeps can be added later.

First create a fixed proxy/confirmation split from the original validation patients:

```bash
python experiments/create_rf_proxy_split.py --seed 42 --proxy-cases 8
```

The generated `experiments/random_forest_proxy_split.json` keeps the original training patients, places eight complete validation volumes in `val` for proxy screening, and places the remaining validation volumes in `test` for confirmation. The `test` key in this experiment manifest is **not** the project's held-out test split. The real `data/processed/splits.json` must remain unchanged.

Run one proxy sweep phase at a time:

```bash
python experiments/run_random_forest_sweep.py --phase trees
python experiments/run_random_forest_sweep.py --phase depth-leaf
python experiments/run_random_forest_sweep.py --phase max-features
python experiments/run_random_forest_sweep.py --phase class-weight
python experiments/run_random_forest_sweep.py --phase random-state
```

Each configuration trains on all manifest training patients, tunes thresholds on the proxy `val` patients, and evaluates both proxy `val` and confirmation `test` patients. Completed runs are skipped, so an interrupted phase can be resumed. Results and per-run manifests are written under `data/experiments/random_forest/`; the aggregate case-level file is `data/experiments/random_forest/case_metrics.csv`. The sweep currently tests:

| Phase | Values |
|---|---|
| `trees` | 50, 100, 150, 200, 300 estimators |
| `depth-leaf` | depths 10/15/20/25 × leaf sizes 1/2/4/8 |
| `max-features` | `sqrt`, `log2`, `all` |
| `class-weight` | Stage 1/Stage 2 combinations of `none`, `balanced`, and `balanced_subsample` |
| `random-state` | 0, 1, 21, 42, 123, 999 |

Generate tuning figures separately from the final model-comparison figures:

```bash
python experiments/plot_random_forest_sweeps.py
```

Figures are saved to `data/experiments/random_forest/figures/`. The existing `plot_results.py` is intentionally unchanged and remains reserved for the final logistic-regression/random-forest/U-Net comparison.

The sweep calls `evaluate.py` with the experiment manifest and records Dice, precision, recall, truth voxels, predicted voxels, and intersection voxels. `evaluate.py` accepts the same custom `--splits`, `--predictions`, and `--out` options for these runs while retaining its normal `data/results/dice_per_case.csv` workflow. Select a configuration using proxy results, confirm it on the remaining validation patients, then retrain/tune the selected configuration using the full validation split before evaluating the real test split once.

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
