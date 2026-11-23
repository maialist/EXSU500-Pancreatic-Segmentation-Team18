# Random-forest experiments

These scripts run and visualize reproducible random-forest parameter sweeps.
All generated models, predictions, metrics, and figures belong under
`data/experiments/` and are excluded from version control.

## Validation proxy manifest

`random_forest_proxy_split.json` is a generated experiment manifest based on
`data/processed/splits.json`. It keeps the original training and actual test
patients unchanged, and divides only the original validation patients:

- `val` contains the proxy patients used to tune thresholds and screen
  parameter configurations.
- `test` contains confirmation patients used to check selected runs. This is
  not the project's held-out test split.
- `actual_test` records the original held-out test patients and must not be
  used for model or parameter selection.

Regenerate the manifest with `python experiments/create_rf_proxy_split.py`.
The case lists are JSON data, so explanatory comments are kept here rather
than inserted into the machine-readable manifest.

## Workflow scripts

- `create_rf_proxy_split.py` makes the seeded proxy/confirmation split while
  preserving training and actual test membership.
- `run_random_forest_sweep.py` creates configurations for one parameter
  sweep, trains and tunes each model, predicts both experiment subsets, and
  combines their case metrics. Use `--phase` to select `trees`,
  `depth-leaf`, `max-features`, `class-weight`, or `random-state`.
- `plot_random_forest_sweeps.py` reads the combined case-metrics CSV and saves
  per-phase plots plus a proxy-versus-confirmation plot.

The sweep expects preprocessed cases and the manifest to be available.
`run_random_forest_sweep.py --help` and `plot_random_forest_sweeps.py --help`
show their input and output path options.
