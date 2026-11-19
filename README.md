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

- [ ] Legacy environment pinned and original results reproduced
- [ ] Database generation ported
- [ ] Training ported to Keras 3
- [ ] Fedoo coupling ported to fedoo 1.x
- [ ] Regression tests passing

## License

MIT, see [LICENSE](LICENSE).
