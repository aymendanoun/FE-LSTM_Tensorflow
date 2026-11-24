# FE-LSTM: context for Claude Code

## What this project is
Reproduction and modernisation of FE-LSTM, from Aymen Danoun's Ph.D. (I2M Bordeaux, 2019-2022).
LSTM networks learn the history-dependent (nonlinear, path-dependent) constitutive response of
heterogeneous / architected materials from micro-scale simulations. The trained network is then used
as the constitutive law inside the Fedoo finite element code, which made multiscale simulations
up to 99% faster.

## Workflow: three self-contained stage folders (since 2026-11-24)
Each folder has its own `src/<package>/`, `scripts/`, `configs/`, `tests/`, `reference_results/`,
`pyproject.toml` and README. Stages exchange files only (no imports between stages).
1. `database_generation/` (package `database_generation`): octet-truss RVE with microgen
   (https://github.com/3MAH/microgen), fedoo 1.x + simcoon EPICP periodic strain-driven
   micro-simulations -> `data/database/path_*.csv` (strain, stress, full 6x6 tangent C11..C66
   per increment, legacy column names). Implemented 2026-11-24; full database not generated yet.
2. `training/` (package `lstm_training`): train stress and tangent LSTMs (Keras 3) -> `.npz`
   weights in `models/weights_scaler_3D/` (thesis export format). Skeleton only.
3. `fe_lstm/` (package `fe_lstm`): the LSTMs as fedoo 1.x constitutive law. Ported and working.
Paths in each stage's configs are relative to that stage folder (e.g. `../legacy/...`).
Root `pytest.ini` runs the three test folders; test file and helper module names must be unique
across stages (never `from conftest import` in a new stage: use a uniquely named helper module).

## Stage 1 notes (2026-11-24)
- microgen only in a separate env `felstm-geometry` (`database_generation/environment-geometry.yml`,
  Python 3.11, conda-forge microgen 1.3.1): conda microgen needs Python < 3.12, PyPI 1.3.2 would
  downgrade numpy/pyvista/VTK in felstm. `scripts/build_rve.py` runs there, writes
  `data/rve/my_rve.msh` from `geometry.build`; `scripts/generate_database.py` runs in felstm.
  microgen 1.3.1 has no lattice class and names the mesher `meshPeriodic` (not `mesh_periodic`);
  it leaves gmsh open after a meshing error (geometry.mesh_rve finalizes it).
- RVE examples (2026-11-24): `database_generation/rve_examples/*.msh` (committed, ~10 MB) built
  from `examples.yaml` by `scripts/build_rve_examples.py`, workflow of the microgen docs: CAD ->
  gmsh periodic mesh -> mmg remesh keeping periodicity (`remesh_keeping_periodicity_for_fem`,
  boundary-face triangles frozen). octet_truss_thesis (default `geometry.mesh_file`) stays
  gmsh-only to reproduce the thesis; octet_truss_thesis_remeshed, octet_truss_uniform,
  gyroid/neovius sheets, schwarz_p_skeletal, schwarz_d_sheet (voxel `method: grid`, CAD route not
  periodic-meshable), spherical_pore, honeycomb are remeshed (slivers 22% -> ~1% for gyroid).
  `scripts/check_rve.py` (felstm) validates any mesh (tet4, periodic, quality, C_hom, EPICP).
- mmg 5.8.0 has no Apple Silicon package: built from official source into felstm-geometry by
  `database_generation/scripts/install_mmg.sh` (cmake/make from conda-forge).
- Thesis RVE = legacy `octet_surf.msh`: unit cell, face struts r=0.10, inner struts r=0.05, mesh
  size 0.1. microgen version: C11/C12/C44 within 0.7% of the thesis mesh (8705/3694/3564, equal
  to the thesis Lt network's scaling maxima). Not mesh-converged (size 0.05: ~8% stiffer).
- Loading (Aymen's change 2026-11-24): 1000 paths of 2 random targets x 50 increments = 100 per
  path (legacy: 10000 paths, 4 x 25); Wm no longer written. Exx, Eyy in [0, 0.05], Exy in
  [0, 0.025], others 0 (default). Thesis 3D networks used all 6 components with +/- ranges:
  generation script missing; Aymen to choose before the full run (~70 min for 1000 paths on the
  Mac, 10 workers, one BLAS thread each).
Aymen's plan: rebuild the project from stage 1 with microgen, not port the legacy generation.

## Port status (end of session 2026-11-19, transcript in docs/claude_sessions/)
- Done: FE coupling, in `fe_lstm/`: `src/fe_lstm/model/numpy_lstm.py` (NumPy LSTM inference,
  tested against Keras 3), `src/fe_lstm/fem/lstm_law.py` (fedoo 1.x law),
  `src/fe_lstm/fem/legacy_nr.py` (2022 Newton criterion), `src/fe_lstm/fem/specimen.py` +
  `scripts/run_fe.py` + `configs/specimen_with_holes_tensile.yaml`, `scripts/plot_results.py`,
  `scripts/compare_tangents.py`.
- Law options: `state_update` = "consistent" (default: one step per converged increment) |
  "legacy" (LSTM steps at every Newton iteration, thesis code, reference only); `tangent` = "network" (tangent LSTM, 21
  terms, symmetric) | "finite_difference" (derivative of the stress LSTM, unsymmetric).
- fedoo 1.x assembles one triangle of K in small strain (`assume_sym=True`): unsymmetric
  tangents need `weakform.assembly_options["assume_sym"] = False` (done in specimen.py).
- Results so far (thesis stress weights, on Aymen's Mac and here):
  - `--tangent finite_difference --state-update consistent`: converges, 100 increments,
    ~2 Newton iterations each. Force-displacement saved as CSV; one .vtk per increment.
  - `--tangent finite_difference --state-update legacy`: fails at increment 88 (LSTM has run
    200+ steps vs 100-step training sequences).
  - `--tangent network --state-update consistent`: converges, same Newton counts (2/incr) and
    same force-displacement as finite_difference (max diff 0.05%).
  - Increment-size check (network, consistent): 200 increments converges; 50 fails on the last
    increment unless `--update-dt`. Curves differ by up to ~3% from 100 increments: the LSTM
    is mildly step-size dependent (trained on 100-step sequences only).
  - `--tangent network --state-update legacy`: fails at increment 77,
    Err 1.78180, the error seen on the Mac. So the Mac failure comes from the legacy state
    update (LSTM runs ~155 steps by then), not from the tangent weights.
- Tangent check (2026-11-20, `scripts/compare_tangents.py`): weights_scaler_Lt/ belongs with
  the stress network (same input scaling, same elastic stiffness, no factor-2 shear issue,
  same softening trends). It is smoother and ~5-20% lower than the exact step derivative
  (median rel. Frobenius diff 0.11-0.26, up to 0.87 late in biaxial), but predicts the
  stress net's increments along a path better than the step derivative does.
- Weights: `legacy/fedoo_coupling/weights_scaler_3D/{weights_scaler_stress,weights_scaler_Lt}/`
  on Aymen's Mac only (*.npz ignored by git).
- Next steps:
  1. (done 2026-11-20, see above.) Open question: why legacy mode converged in 2022 (old
     fedoo needing fewer Newton iterations per increment?).
  2. Compare the force-displacement curve / stress fields with the thesis figures.
  3. Analytic (exact) Jacobian of the LSTM step to replace finite differences.
  4. Port training to Keras 3 / pandas 2 (`DataFrame.append` removed).
  5. (done 2026-11-24) Database generation on microgen + fedoo 1.x. Next: choose the loading
     range, generate the database, then port training (step 4) on it.
- Trained weights are not in git; tests use synthetic weights (`fe_lstm/tests/conftest.py`).

## Rules
- **Never modify anything in `legacy/`.** It holds the original PhD scripts and is the reference.
- Put code in the stage folder it belongs to (`<stage>/src/<package>/`), keeping stages
  self-contained. Keep the scientific behaviour identical to the thesis (same architecture,
  scaling, loading paths) unless asked otherwise, and note every deliberate change.
- Each stage gets regression tests in `<stage>/tests/` comparing against
  `<stage>/reference_results/` with an explicit numerical tolerance.
- Do not commit large data or trained models (see `.gitignore`); small reference files only.
- Hyperparameters and paths belong in `<stage>/configs/*.yaml`, not hard-coded.
- **Always use `state_update: consistent`** (Aymen's decision, 2026-11-23) for FE runs, configs,
  scripts and new code. Use `legacy` only when explicitly asked to reproduce the thesis code.

## Environments
- `felstm` (`environment.yml`): latest fedoo (1.x), simcoon (2.x), TensorFlow 2.21 (Keras 3), Python 3.12
- `felstm-legacy` (`environment-legacy.yml`): versions pinned to the PhD period, used to run `legacy/`
- Fedoo and Simcoon come from the `set3mah` conda channel; TensorFlow from pip.
- Machine: macOS, Apple Silicon. TensorFlow runs on the CPU.

## Commands
```bash
conda activate felstm
pip install -e database_generation -e training -e fe_lstm
pytest                                   # from the repo root: all three stages
cd fe_lstm && python scripts/run_fe.py   # stage scripts run from their stage folder
```

## Known porting hazards
- Keras 3 (TF >= 2.16): `tf.keras` changes, `.h5` saving, custom layers and serialization.
- Fedoo 1.x API differs a lot from 2021-2022 versions (ModelingSpace, constitutivelaw, weakform,
  problem, user-defined materials).
- Simcoon Python bindings: renamed functions and changed signatures across major versions.
