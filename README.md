# FE-LSTM (TensorFlow)

Reproduction and modernisation of **FE-LSTM**, developed during my Ph.D. in Mechanical
Engineering at I2M Bordeaux (2019-2022). The project uses LSTM networks as data-driven constitutive
laws for nonlinear multiscale simulations of heterogeneous and architected materials, and embeds them
in a finite element solver (Fedoo).

The original work reduced computational time by up to 99%, so HPC-scale multiscale simulations
could run on a standard desktop. The thesis received the **CSMA PhD Thesis Award 2022**.

> Thesis: TODO add title and link (e.g. theses.fr / HAL)

## Goals

1. **Reproduce** the original results with the versions used at the time (`environment-legacy.yml`).
2. **Modernise** the code for the latest Fedoo, Simcoon and TensorFlow/Keras 3 (`environment.yml`)
   and check the results against the reference.

## Pipeline

1. **Database generation** (`src/fe_lstm/data/`): micro-scale simulations with Simcoon/Fedoo produce
   strain-stress histories.
2. **Training** (`src/fe_lstm/model/`): an LSTM learns the history-dependent constitutive response.
3. **FE coupling** (`src/fe_lstm/fem/`): the trained LSTM replaces the constitutive law in a Fedoo
   finite element simulation.

## Installation

```bash
conda env create -f environment.yml
conda activate felstm
pip install -e .
pytest
```

## Usage

```bash
python scripts/generate.py --config configs/default.yaml
python scripts/train.py    --config configs/default.yaml
python scripts/run_fe.py   --config configs/default.yaml
```

## Repository structure

| Path | Content |
|------|---------|
| `legacy/` | Original PhD scripts, unmodified |
| `src/fe_lstm/` | Ported, installable package |
| `scripts/` | Command-line entry points |
| `configs/` | YAML configuration files |
| `tests/` | Regression tests against `reference_results/` |
| `reference_results/` | Small reference outputs from the legacy run |
| `notebooks/` | Legacy vs modern comparison plots |

## Status

- [x] Fedoo coupling ported to fedoo 1.x (`src/fe_lstm/fem/`, `scripts/run_fe.py`)
- [ ] FE-LSTM run with the trained thesis weights and compared to the thesis results
- [ ] Training ported to Keras 3
- [ ] Database generation ported (fedoo 1.x has no path-driven unit-cell solver any more)
- [ ] Legacy environment pinned and original results reproduced

## Running the FE-LSTM simulation

The trained weights are not in git (`*.npz` is ignored). Copy the folder exported by
`legacy/training/LSTM_model_Get_weights_scaler_*.py` (the one holding `weights_scaler_stress/`
and `weights_scaler_Lt/`) to `models/weights_scaler_3D/`, or pass its path with `--weights`:

```bash
python scripts/run_fe.py                                  # FE-LSTM, settings from configs/
python scripts/run_fe.py --weights /path/to/weights_scaler_3D
python scripts/run_fe.py --material epicp                 # simcoon EPICP reference
python scripts/run_fe.py --state-update consistent        # see "Porting notes"
```

Results are written to `results/specimen_with_holes_tensile/` (`.fdh5`, plus `.vtk` for ParaView).

## Porting notes (fedoo 2022 -> fedoo 1.1)

- The LSTM inference was already pure NumPy (weights exported from Keras), so TensorFlow is only
  needed for training. `src/fe_lstm/model/numpy_lstm.py` is tested against a Keras 3 model.
- The law (`src/fe_lstm/fem/lstm_law.py`) now derives from fedoo's `Mechanical3D`. The Voigt
  convention is unchanged: `[e_xx, e_yy, e_zz, g_xy, g_xz, g_yz]`, engineering shear.
- **Recurrent state and Newton-Raphson.** In the 2022 code the LSTM state advanced at every
  Newton-Raphson iteration, and only the cell state `c` was restored when an increment was
  restarted. `state_update: legacy` (default) keeps this behaviour; `state_update: consistent`
  advances the LSTM exactly once per converged increment, as in the training sequences.
- The weights are read once into memory instead of from disk at every call (same numbers).

## License

MIT, see [LICENSE](LICENSE).
