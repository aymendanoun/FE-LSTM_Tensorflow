# Stage 3: FE-LSTM

Uses the trained LSTMs as the constitutive law of a fedoo 1.x finite element simulation: the
stress network gives the stress at each Gauss point, the tangent network (or a finite-difference
derivative of the stress network) gives the stiffness for Newton-Raphson.

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
python scripts/run_fe.py --tangent finite_difference      # tangent from the stress network
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
- **Tangent:** `network` (default, symmetric) or `finite_difference` (unsymmetric; fedoo 1.x
  assembles one triangle in small strain, so full assembly is switched on for it).
- **Newton-Raphson criterion:** `LegacyDisplacement` (default) reproduces the 2022 definition,
  `max|correction| / max|total displacement|`. fedoo 1.x's `Displacement` normalises by the
  increment, which is much stricter: the thesis tolerance 0.05 then fails.
- `numpy_lstm.py` is tested against a Keras 3 model, so TensorFlow is only needed for tests here.
- Weights are read into memory once instead of at every call (same numbers).
