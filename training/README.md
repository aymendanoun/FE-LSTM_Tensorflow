# Stage 2: Training

Trains the LSTM surrogates on the database from `../database_generation`: a **stress network**
(6 strains -> 6 stresses) and, optionally, a **tangent network** (6 strains -> 21 terms of the
symmetric tangent). Their weights are exported as `.npz` files that the fe_lstm law
(`../fe_lstm`) reads.

Since 2026-11-28 only the stress network is trained by default (`networks.tangent.enabled:
false`): fe_lstm uses the finite-difference tangent of the stress network, which always agrees
with the stress and gives the same converged FE solution. Train the tangent network with
`--networks stress tangent` (or `enabled: true`) to run fe_lstm with `--tangent network`.

Port of `../legacy/training/LSTM_model_Training_*.py` and `LSTM_model_Get_weights_scaler_*.py`
to Keras 3 / pandas 2, with every setting in a config file.

## Layout

| Path | Content |
|------|---------|
| `src/lstm_training/data.py` | Reads `path_*.csv`, `Cij_m` columns, MinMax scaling, train / test split |
| `src/lstm_training/model.py` | Keras model, losses, optimizers, keep-best callback |
| `src/lstm_training/export.py` | `.npz` export for fe_lstm + NumPy check of the exported files |
| `src/lstm_training/train.py` | Trains one network from the config and exports it |
| `scripts/train.py` | Entry point |
| `configs/training.yaml` | Default config (database of `database_generation/configs/database.yaml`) |
| `configs/training_gyroid.yaml` | Same settings on the gyroid database |
| `configs/training_gyroid_3d.yaml` | Same settings on the gyroid database loaded on all six components |
| `configs/training_gyroid_3d_weighted.yaml` | Same, with the loss weighted per increment (`time_weights`) |
| `tests/` | Tests on a synthetic database; regression against `reference_results/` |
| `models/` | Trained networks (not in git) |

## Usage

From this folder, in the `felstm` environment:

```bash
python scripts/train.py                                       # stress network, configs/training.yaml
python scripts/train.py --networks stress tangent             # also the thesis tangent network
python scripts/train.py --config configs/training_gyroid.yaml
python scripts/train.py --networks stress --epochs 50         # quick run of one network
python scripts/train.py --n-paths 200 --output models/test    # fewer paths, other output folder
```

Long runs keep going when the conversation or terminal is closed with
`nohup caffeinate -i python -u scripts/train.py ... > train.log 2>&1 &` (macOS).
On the 10-core Mac, the default network on 1000 paths x 100 increments takes about 1.1 s per
epoch, so about 20 min per network for 1000 epochs.

## Settings (`configs/training.yaml`)

| Key | Meaning | Legacy value |
|-----|---------|--------------|
| `data.database_dir`, `data.n_paths` | database and number of paths (`null` = all) | 10000 paths |
| `data.test_fraction` | last paths held out for the final test | 0.1 (tangent), 0.01 (stress) |
| `data.validation_split` | last fraction of the training paths used for validation | 0.2 |
| `model.lstm_units` | one entry per LSTM layer: `[50, 50]` = 2 layers of 50 units | `[50, 50]` |
| `model.dense_units`, `model.dense_activations` | hidden Dense layers and their activations | `[50]`, `[relu]` |
| `model.output_activation` | activation of the output layer | `linear` |
| `training.optimizer` | `adam`, or `{name: adam, learning_rate: 1.0e-3, ...}` (any Keras optimizer) | `adam` |
| `training.loss` | any Keras loss name (`mse`, `mae`, `huber`, `log_cosh`...) or `nmse` | `mse` |
| `training.epochs`, `training.batch_size`, `training.shuffle` | | 1000, 32, true |
| `training.keep_best`, `training.monitor` | export the epoch with the lowest `monitor` | true, `val_nmse` |
| `training.time_weights` | loss weight per increment, `{power: 2.0, floor: 0.05}` (see below; `null` = plain loss) | – |
| `training.early_stopping_patience` | stop after N epochs without improvement (`null` = never) | – |
| `training.reduce_lr` | Keras `ReduceLROnPlateau` options, e.g. `{factor: 0.5, patience: 50}` | – |
| `networks.<name>.inputs` / `outputs` | column names; `Cij_m` = `(Cij + Cji) / 2` | see config |
| `networks.<name>.model` / `training` | overrides of the global `model` / `training` for this network | – |

Dense activations are limited to those the NumPy inference of fe_lstm implements: `linear`,
`relu`, `tanh`, `sigmoid`, `elu`, `selu`, `softplus`, `softsign`, `silu` (= `swish`),
`leaky_relu`. The LSTM layers keep the Keras defaults (tanh, sigmoid gates).
`nmse` is the legacy metric, `100 sum((y - y_pred)^2) / sum(y^2)` on the scaled outputs; it is
always reported as a metric (unweighted).

`time_weights` weights the loss of increment `t` by `(r_max / max(r_t, floor r_max))^power`
(normalized to mean 1), with `r_t` the RMS output at `t` over the training paths. With plain MSE
on MinMax-scaled outputs the first increments of a path, where the stresses are small, barely
count: the gyroid 3D stress network came out about 7 times too soft there. `power: 2` gives
every increment the same weight relative to its output level; `floor` caps the weight of
near-zero increments. Use `monitor: val_loss` with it (`val_nmse` is unweighted).
`metrics.json` reports the test RMSE at each increment (`rmse_by_increment`).

## Output (input of stage 3)

`<output.dir>/weights_scaler_stress/` (and `<output.dir>/weights_scaler_Lt/` when the tangent network
is trained), each with:

| File | Content |
|------|---------|
| `LSTM_layer_1.npz` ... `LSTM_layer_n.npz` | `W_*`, `U_*`, `b_*` per gate (thesis format) |
| `Dense_layer_<n+1>.npz` ... | `W`, `b` |
| `scaling_stress.npz` / `scaling_Lt.npz` | `scalingX_min/max`, `scalingY_min/max` |
| `architecture.json` | layer sizes, activations, input / output names (read by fe_lstm) |
| `history.csv`, `history.png` | loss and NMSE per epoch, training and validation |
| `metrics.json` | best epoch, test errors per output (RMSE, NMSE, R²), settings |
| `test_predictions.npz` | strain, true and predicted outputs on the test paths |
| `model.keras` | the Keras model (to continue training or inspect) |

To use them in fe_lstm, set `material.weights_dir: ../training/<output.dir>` in its config, or run
`python scripts/run_fe.py --weights ../training/<output.dir>` from `../fe_lstm`. After the
export, `train.py` re-runs the test paths through the `.npz` files with NumPy and stops with an
error if they differ from Keras.

## Differences from the legacy scripts

- Same architecture, scaling, split, loss, optimizer, epochs and batch size by default.
- Paths are read from the stage 1 database (`path_*.csv`); `Cij_m` is computed from the full
  tangent (the stage 1 files no longer hold the `_m` columns).
- The best epoch is kept in memory instead of saving 1000 checkpoints and reloading one.
- A constant input or output column (e.g. `Ezz = 0` when only `Exx, Eyy, Exy` are loaded) is
  exported with `max = min + 1`, so the fe_lstm formula `(x - min) / (max - min)` gives the same
  scaling as sklearn (unit scale) instead of dividing by zero. Note that such a network never
  saw that strain component: in a 3D FE simulation it will extrapolate.
- `architecture.json` is new; fe_lstm still reads the thesis weights without it.
- The tangent network is not trained by default (the thesis trained it); see above.
- `time_weights` (off by default) is new.
