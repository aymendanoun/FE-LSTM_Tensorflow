# FE-LSTM: context for Claude Code

## What this project is
Reproduction and modernisation of FE-LSTM, from Aymen Danoun's Ph.D. (I2M Bordeaux, 2019-2022).
LSTM networks learn the history-dependent (nonlinear, path-dependent) constitutive response of
heterogeneous / architected materials from micro-scale simulations. The trained network is then used
as the constitutive law inside the Fedoo finite element code, which made multiscale simulations
up to 99% faster.

## Pipeline
1. `src/fe_lstm/data/`: generate strain-stress histories with Simcoon / Fedoo micro-simulations
2. `src/fe_lstm/model/`: define, train and export the LSTM (TensorFlow / Keras 3)
3. `src/fe_lstm/fem/`: call the LSTM at each integration point / increment inside Fedoo

## Rules
- **Never modify anything in `legacy/`.** It holds the original PhD scripts and is the reference.
- Port code into `src/fe_lstm/`. Keep the scientific behaviour identical (same architecture, scaling,
  loading paths) unless asked otherwise, and note every deliberate change.
- Each ported stage gets a regression test in `tests/` comparing against `reference_results/`
  with an explicit numerical tolerance.
- Do not commit large data or trained models (see `.gitignore`); small reference files only.
- Hyperparameters and paths belong in `configs/*.yaml`, not hard-coded.

## Environments
- `felstm` (`environment.yml`): latest fedoo (1.x), simcoon (2.x), TensorFlow 2.21 (Keras 3), Python 3.12
- `felstm-legacy` (`environment-legacy.yml`): versions pinned to the PhD period, used to run `legacy/`
- Fedoo and Simcoon come from the `set3mah` conda channel; TensorFlow from pip.
- Machine: macOS, Apple Silicon. TensorFlow runs on the CPU.

## Commands
```bash
conda activate felstm
pip install -e .
pytest
```

## Known porting hazards
- Keras 3 (TF >= 2.16): `tf.keras` changes, `.h5` saving, custom layers and serialization.
- Fedoo 1.x API differs a lot from 2021-2022 versions (ModelingSpace, constitutivelaw, weakform,
  problem, user-defined materials).
- Simcoon Python bindings: renamed functions and changed signatures across major versions.
