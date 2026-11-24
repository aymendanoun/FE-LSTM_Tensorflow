# Stage 1: Database generation

Builds the octet-truss RVE with [microgen](https://github.com/3MAH/microgen), then runs
strain-driven periodic micro-simulations (fedoo 1.x + simcoon EPICP) along random strain paths.
The output is the database of strain, stress and tangent histories used by `../training`.

This replaces the legacy scripts (`../legacy/database_generation/`), which relied on the old fedoo
`Homogen.GetResultsUnitCell` (code in `../legacy/fedoo_coupling/fedoo/libHomogen/Homog_path.py`).
The procedure is the same: same RVE, material, loading paths, solver tolerance and recorded
quantities.

## RVE: an example mesh or your own

The micro-simulations read one periodic RVE mesh, `geometry.mesh_file` in
`configs/database.yaml`:

- **Example meshes** ([`rve_examples/`](rve_examples/README.md)): octet truss (the thesis RVE,
  default), gyroid, Neovius and Schwarz P TPMS, spherical pore, honeycomb. Ready to use, no
  microgen needed.
- **Your own RVE**: edit `geometry.build` in the config and run `scripts/build_rve.py`, or build
  any geometry with microgen. Check it with `scripts/check_rve.py`.

## Two environments

| Step | Environment | Why |
|------|-------------|-----|
| Build an RVE mesh (microgen, CadQuery, gmsh) | `felstm-geometry` ([`environment-geometry.yml`](environment-geometry.yml)) | microgen's conda build needs Python < 3.12; its PyPI build would downgrade numpy / pyvista / VTK in `felstm` |
| Check meshes, run the micro-simulations (fedoo, simcoon) | `felstm` (`../environment.yml`) | |

The two steps only share the mesh file. With an example mesh, only `felstm` is needed.

```bash
# once
conda activate felstm && pip install -e .
conda env create -f environment-geometry.yml                     # only to build RVEs
conda activate felstm-geometry && pip install --no-deps -e .

# optional: your own RVE (felstm-geometry), then check it (felstm)
conda activate felstm-geometry && python scripts/build_rve.py     # -> data/rve/my_rve.msh
conda activate felstm && python scripts/check_rve.py data/rve/my_rve.msh

# database (felstm; all CPU cores; restart the same command to resume)
conda activate felstm
python scripts/generate_database.py --paths 0:20      # try a few paths first
python scripts/generate_database.py                   # all n_paths of the config
```

Run time on a 10-core Apple Silicon Mac: about 4 s per path with 10 workers, so roughly
1 h 10 min for the 1000 paths of the config (the thesis used 10 000).

## What it does

**RVE** (`src/database_generation/geometry.py`, default `rve_examples/octet_truss_thesis.msh`):
unit cube with the 36 struts of the octet truss.
The 24 face struts (radius 0.10) lie on the cell faces and are cut in half; the 12 inner struts
have radius 0.05. microgen builds the cylinders, fuses them, cuts them to the cell and meshes the
result periodically (linear tetrahedra, gmsh). These radii and the mesh size 0.1 were measured on
the thesis mesh `octet_surf.msh`:

| | nodes | tet4 | solid fraction | C11 | C12 | C44 |
|---|---|---|---|---|---|---|
| thesis `octet_surf.msh` | 1694 | 4072 | 0.2272 | 8705.2 | 3694.3 | 3563.7 |
| microgen, mesh size 0.1 (default) | 1728 | 4164 | 0.2277 | 8643.1 | 3683.3 | 3579.6 |
| microgen, mesh size 0.05 | 5535 | 16304 | 0.2472 | 9406.4 | 4127.0 | 3846.0 |

(Homogenized elastic stiffness in MPa, struts E = 113800 MPa, nu = 0.342.) The thesis C11 and C44
are exactly the maxima of the thesis tangent network's training data. The thesis mesh is not
mesh-converged: refining it makes the RVE about 8% stiffer.

**Loading paths** (`loading.py`): each path has 2 random target strains, reached one after the
other along straight lines in 50 equal increments each, so 100 increments per path (the
training sequence length). Deliberate change: the legacy paths had 4 targets x 25 increments
(`N_Steps = 4`, `Dn_inc = 0.04`) and the thesis database 10 000 paths; the config uses 1000. By default, as in the legacy script, `Exx`, `Eyy` and `Exy` are drawn
uniformly in `[0, 0.05]`, `[0, 0.05]` and `[0, 0.025]`, the other components stay zero. Path `i`
uses the random stream `(seed, i)`: results do not depend on the number of workers.

**Micro-simulation** (`rve.py`): periodic boundary conditions with the macroscopic strain as
global unknowns (`fd.constraint.PeriodicBC("small_strain")`, engineering shear), the node nearest
the centre fixed, Newton-Raphson tolerance 0.05 with automatic time step. At the end of each
increment it records the mean strain, the mean stress (volume integral over the solid divided by
the cell volume, as in the legacy code) and the homogenized 6x6
tangent (`fd.homogen.get_tangent_stiffness`).

## Output (input of stage 2)

`data/database/path_00000.csv`, ... (not in git): 100 rows per path, columns

`Increment, Time, Exx, Eyy, Ezz, Exy, Exz, Eyz, Sxx, Syy, Szz, Sxy, Sxz, Syz, C11, C12, ..., C66`

Strains with engineering shear, stresses in MPa, `Cij` = d(stress i)/d(strain j). These are the
legacy column names (the legacy work columns `Wm, Wm_r, Wm_ir, Wm_d` are not written); the symmetric `Cij_m` used to train the tangent network are computed in the
training stage. `config.json` stores the configuration used; failed paths, if any, are listed in
`failed_paths.csv` and skipped.

## Checks (`tests/`)

- Linear elastic struts: mean stress = homogenized stiffness x strain, tangent = homogenized
  stiffness at every increment, and the thesis stiffness (C11 = 8705.2) on the thesis mesh.
- EPICP regression on the thesis mesh (path 0, 2 x 5 increments) against
  `reference_results/legacy_mesh_epicp_path0_2x5.csv` (generated with this code, fedoo 1.1.0 /
  simcoon 2.5.1: no output of the legacy runs exists). Tolerance 1e-6 of the maximum stress.
- Loading paths (bounds, reproducibility, piecewise-linear discretization), parallel driver and
  resume.
- Example meshes: each one is periodic tet4 with a symmetric positive-definite homogenized
  stiffness (`check_mesh`); the thesis example is within 1% of the thesis mesh.
- Geometry: strut layout always; in `felstm-geometry` (`pytest tests/test_geometry.py`): thesis
  octet mesh (volume fraction, size, periodicity) and the quick builders.
- Path 0 on the microgen RVE and on the thesis mesh: stresses within about 1-3% over the whole
  path, through yielding and unloading (checked once, not a test).

## Open points

- **Loading range.** The default reproduces the legacy script in `legacy/` (`Exx`, `Eyy`, `Exy`,
  positive only). The thesis 3D networks were trained on all six components with negative values
  too (`[-0.05, 0.05]` normal, `[-0.025, 0.025]` shear); that generation script is not in
  `legacy/`. `configs/database.yaml` shows the settings for that range.
- **Legacy shear bound.** The legacy path files set `e12` through simcoon, which may have read it
  as the tensor component (engineering shear up to 0.05). The default uses engineering shear
  up to 0.025, the range of the thesis networks.
