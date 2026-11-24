# FE-LSTM (TensorFlow)

Reproduction and modernisation of **FE-LSTM**, developed during my Ph.D. in Mechanical
Engineering at I2M Bordeaux (2019-2022). The project uses LSTM networks as data-driven constitutive
laws for nonlinear multiscale simulations of heterogeneous and architected materials, and embeds them
in a finite element solver (Fedoo).

The original work reduced computational time by up to 99%, so HPC-scale multiscale simulations
could run on a standard desktop. The thesis received the **CSMA PhD Thesis Award 2022**.

> Thesis: TODO add title and link (e.g. theses.fr / HAL)

## Workflow

The project has three self-contained stages, one folder each. Each stage has its own package,
scripts, configs, tests and README, and hands its output to the next stage through files.

| Stage | Folder | Does | Produces |
|-------|--------|------|----------|
| 1 | [`database_generation/`](database_generation/) | RVE with microgen, micro-simulations with fedoo + simcoon along random strain paths | strain-stress-tangent histories (`data/`) |
| 2 | [`training/`](training/) | Trains the stress and tangent LSTMs (TensorFlow / Keras 3) | exported `.npz` weights (`models/`) |
| 3 | [`fe_lstm/`](fe_lstm/) | Uses the LSTMs as the constitutive law of a fedoo FE simulation | FE results (`results/`) |

`legacy/` holds the original PhD scripts, unmodified, as the reference.

## Installation

```bash
conda env create -f environment.yml
conda activate felstm
pip install -e database_generation -e training -e fe_lstm
pytest                      # runs the tests of the three stages
```

Stage 1 comes with ready-made RVE meshes ([`database_generation/rve_examples/`](database_generation/rve_examples/README.md)).
Building your own RVE with microgen needs a small `felstm-geometry` environment: see
[`database_generation/README.md`](database_generation/README.md).

## Usage

```bash
cd database_generation && python scripts/build_rve.py           # stage 1a, optional (felstm-geometry env)
cd database_generation && python scripts/generate_database.py   # stage 1b
cd training            && python scripts/train.py               # stage 2
cd fe_lstm             && python scripts/run_fe.py              # stage 3
```

## Status

- [x] Stage 3 ported to fedoo 1.x and run with the thesis weights
- [x] Stage 1 rebuilt on microgen + fedoo 1.x (thesis RVE reproduced, database not generated yet)
- [ ] Stage 2 ported to Keras 3 / pandas 2
- [ ] Full workflow run end to end and compared with the thesis results
- [ ] Legacy environment pinned and original results reproduced

## License

MIT, see [LICENSE](LICENSE).
