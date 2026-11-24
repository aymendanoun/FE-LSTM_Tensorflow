# Stage 2: Training

Trains the LSTM surrogates on the database from `../database_generation`: a stress network
(strain -> 6 stresses) and a tangent network (strain -> 21 tangent terms). Then exports their
weights as `.npz` files read by `../fe_lstm`.

**Status:** skeleton. Port of `../legacy/training/` to Keras 3 / pandas 2 to come
(`DataFrame.append` was removed in pandas 2).

## Layout

| Path | Content |
|------|---------|
| `src/lstm_training/` | Installable package (data loading, scaling, model, export) |
| `scripts/train.py` | Entry point |
| `configs/training.yaml` | Data, architecture and training settings |
| `tests/` | Tests (regression against `reference_results/`) |
| `models/` | Trained weights (not in git) |

## Usage

```bash
pip install -e .            # from this folder, in the felstm environment
python scripts/train.py --config configs/training.yaml
```

## Output (input of stage 3)

`models/weights_scaler_3D/weights_scaler_stress/` and `.../weights_scaler_Lt/`, each with
`LSTM_layer_1.npz`, `LSTM_layer_2.npz`, `Dense_layer_3.npz`, `Dense_layer_4.npz` and
`scaling_stress.npz` / `scaling_Lt.npz` (same keys as the thesis export, see
`../fe_lstm/src/fe_lstm/model/numpy_lstm.py`).
