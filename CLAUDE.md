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

## Port status (end of session 2026-11-19, transcript in docs/claude_sessions/)
- Done: FE coupling. `src/fe_lstm/model/numpy_lstm.py` (NumPy LSTM inference, tested against
  Keras 3), `src/fe_lstm/fem/lstm_law.py` (fedoo 1.x law), `src/fe_lstm/fem/legacy_nr.py`
  (2022 Newton criterion), `src/fe_lstm/fem/specimen.py` + `scripts/run_fe.py` +
  `configs/specimen_with_holes_tensile.yaml`, `scripts/plot_results.py`. 24 tests pass.
- Law options: `state_update` = "legacy" (LSTM steps at every Newton iteration, thesis code) |
  "consistent" (one step per converged increment); `tangent` = "network" (tangent LSTM, 21
  terms, symmetric) | "finite_difference" (derivative of the stress LSTM, unsymmetric).
- fedoo 1.x assembles one triangle of K in small strain (`assume_sym=True`): unsymmetric
  tangents need `weakform.assembly_options["assume_sym"] = False` (done in specimen.py).
- Results so far (thesis stress weights, on Aymen's Mac and here):
  - `--tangent finite_difference --state-update consistent`: converges, 100 increments,
    ~2 Newton iterations each. Force-displacement saved as CSV; one .vtk per increment.
  - `--tangent finite_difference --state-update legacy`: fails at increment 88 (LSTM has run
    200+ steps vs 100-step training sequences).
  - tangent network (default): fails to converge on the Mac. Suspect: weights_scaler_Lt/ not
    matching the stress network. Not yet checked: need the 5 Lt .npz files.
- Weights: `legacy/fedoo_coupling/weights_scaler_3D/{weights_scaler_stress,weights_scaler_Lt}/`
  on Aymen's Mac only (*.npz ignored by git).
- Next steps:
  1. Check the tangent network against the finite-difference tangent of the stress network.
  2. Compare the force-displacement curve / stress fields with the thesis figures.
  3. Analytic (exact) Jacobian of the LSTM step to replace finite differences.
  4. Port training to Keras 3 / pandas 2 (`DataFrame.append` removed).
  5. Port database generation: fedoo 1.x has no `Homogen.GetResultsUnitCell`; `output.dat`
     missing from the repo. Note: legacy loadings only vary e11, e12, e22.
- Trained weights are not in git; tests use synthetic weights (`tests/conftest.py`).

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
