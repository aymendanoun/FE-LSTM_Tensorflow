# Stage 3: FE-LSTM

Uses the trained LSTMs as the constitutive law of a fedoo 1.x finite element simulation: the
stress network gives the stress at each Gauss point, and a finite-difference derivative of the
stress network (default) or the thesis tangent network gives the stiffness for Newton-Raphson.

**Status:** ported to fedoo 1.1 and run with the thesis weights (see `../CLAUDE.md`).

## Layout

| Path | Content |
|------|---------|
| `src/fe_lstm/model/numpy_lstm.py` | NumPy LSTM inference (weights exported by stage 2) |
| `src/fe_lstm/fem/lstm_law.py` | fedoo 1.x constitutive law |
| `src/fe_lstm/fem/legacy_nr.py` | 2022 Newton-Raphson displacement criterion |
| `src/fe_lstm/fem/specimen.py` | 3D specimen-with-holes tensile test |
| `scripts/run_fe.py` | Run the FE simulation |
| `scripts/plot_results.py` | Force-displacement plots |
| `scripts/compare_tangents.py` | Tangent network vs finite-difference tangent |
| `configs/specimen_with_holes_tensile.yaml` | Mesh, material, loading, solver settings |
| `tests/` | Tests (synthetic weights, no trained model needed) |
| `results/` | Outputs (not in git) |

## Usage

```bash
pip install -e .            # from this folder, in the felstm environment
python scripts/run_fe.py                                  # FE-LSTM, settings from configs/
python scripts/run_fe.py --weights ../training/models/weights_scaler_3D
python scripts/run_fe.py --material epicp                 # simcoon EPICP reference
python scripts/run_fe.py --tangent network                # thesis tangent network (needs weights_scaler_Lt/)
python scripts/run_fe.py --state-update legacy            # thesis behaviour (reference only)
```

By default the config reads the thesis weights from
`../legacy/fedoo_coupling/weights_scaler_3D` (not in git). Point `weights_dir` to
`../training/models/weights_scaler_3D` once stage 2 has produced new networks.

## Porting notes (fedoo 2022 -> fedoo 1.1)

- The law derives from fedoo's `Mechanical3D`. Voigt convention unchanged:
  `[e_xx, e_yy, e_zz, g_xy, g_xz, g_yz]`, engineering shear.
- **Recurrent state** (deliberate change): `state_update: consistent` (default) advances the
  LSTM once per converged increment, as in training. `legacy` advances it at every Newton
  iteration like the 2022 code; on the specimen it fails at increment 77 with fedoo 1.x.
- **Tangent** (deliberate change, 2026-11-28): `finite_difference` (default) differentiates the
  stress network (6 extra stress evaluations per update; unsymmetric, so full assembly is switched
  on, since fedoo 1.x assembles one triangle in small strain). `network` is the thesis tangent
  network (symmetric, 21 terms). The converged solution depends only on the stress network: with
  the thesis weights both give the same curve (max diff 0.05%). The finite-difference tangent is
  always consistent with the stress, so it converges where a separately trained tangent network
  disagrees with the stress network (gyroid 3D networks: `network` fails at increment 1,
  `finite_difference` converges in 100 increments, 2 Newton iterations each).
  `weights_scaler_Lt/` is only needed for `network`.
- **Newton-Raphson criterion:** `LegacyDisplacement` (default) reproduces the 2022 definition,
  `max|correction| / max|total displacement|`. fedoo 1.x's `Displacement` normalises by the
  increment, which is much stricter: the thesis tolerance 0.05 then fails.
- `numpy_lstm.py` reads any number of LSTM and Dense layers (counted from the file names) and the
  Dense activations from `architecture.json`, written by the training stage. Without that file
  (thesis weights) the hidden Dense layers are relu and the output layer linear, as in the thesis.
  With it, the law also checks the input / output names and order.
- `numpy_lstm.py` is tested against a Keras 3 model, so TensorFlow is only needed for tests here.
- Weights are read into memory once instead of at every call (same numbers).
