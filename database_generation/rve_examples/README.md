# Example RVE meshes

Ready-to-use periodic RVE meshes for the database generation, built with
[microgen](https://github.com/3MAH/microgen) 1.3.1 from its
[examples](https://github.com/3MAH/microgen/tree/v1.3.1/examples). Each one is a box-shaped
periodic cell of one material, meshed with linear tetrahedra (gmsh format 2.2), and passes
`scripts/check_rve.py`. To use one, set in `../configs/database.yaml`:

```yaml
geometry:
  mesh_file: rve_examples/gyroid_sheet.msh
```

| Mesh | Geometry | microgen example | Cell | Solid fraction | nodes / tet4 | C11 | C12 | C44 | EPICP, 10 increments |
|------|----------|------------------|------|---------------:|-------------:|----:|----:|----:|---------------------:|
| `octet_truss_thesis.msh` | Octet truss of the thesis, face struts r = 0.10, inner struts r = 0.05 (default) | `Lattices/octetTruss` | 1 x 1 x 1 | 0.228 | 1728 / 4164 | 8643 | 3683 | 3580 | 1.9 s |
| `octet_truss_uniform.msh` | Octet truss, all struts r = 0.07 | `Lattices/octetTruss` | 1 x 1 x 1 | 0.192 | 2757 / 7069 | 6619 | 3090 | 3036 | 3.2 s |
| `gyroid_sheet.msh` | Gyroid TPMS sheet | `TPMS/gyroid` | 1 x 1 x 1 | 0.250 | 6922 / 21995 | 16581 | 7489 | 5271 | 16.5 s |
| `neovius_sheet.msh` | Neovius TPMS sheet | `TPMS/tpms` | 1 x 1 x 1 | 0.250 | 6311 / 21116 | 17788 | 2764 | 1577 | 12.0 s |
| `schwarz_p_skeletal.msh` | Schwarz P TPMS, skeletal (lower) solid | `TPMS/tpms` | 1 x 1 x 1 | 0.300 | 4306 / 13963 | 14772 | 1619 | 2468 | 12.9 s |
| `spherical_pore.msh` | Cube with a centred spherical pore, r = 0.4 | `BasicShapes/shapes` | 1 x 1 x 1 | 0.736 | 2110 / 7853 | 95568 | 39059 | 24008 | 4.6 s |
| `honeycomb.msh` | Hexagonal honeycomb (side 0.5, wall 0.06), walls along x | `Lattices/honeycomb` | 0.5 x 0.866 x 1.5 | 0.134 | 2113 / 5990 | 17528 | 3389 | 3128 | 2.7 s |

Stiffness: homogenized elastic stiffness in MPa with the thesis struts (E = 113800 MPa,
nu = 0.342), Voigt order `[11, 22, 33, 12, 13, 23]` with engineering shear. Last column: one
10-increment EPICP path on a 10-core Mac (a 100-increment path takes about 10 times longer).
The thesis mesh `legacy/database_generation/octet_surf.msh` gives C11 = 8705, C12 = 3694,
C44 = 3564: `octet_truss_thesis.msh` is within 0.7%.

## Rebuilding or adding examples

The parameters of every example are in [`examples.yaml`](examples.yaml). In the
`felstm-geometry` environment, from `database_generation/`:

```bash
python scripts/build_rve_examples.py                 # all
python scripts/build_rve_examples.py gyroid_sheet    # one
```

To add an example, add an entry to `examples.yaml` (`kind` = `octet_truss`, `tpms`,
`spherical_pore` or `honeycomb`, see `src/database_generation/geometry.py`), build it, check it in
`felstm` with `python scripts/check_rve.py rve_examples/<name>.msh` and add its row here.

## Your own RVE

Either edit `geometry.build` in `configs/database.yaml` and run `scripts/build_rve.py`, or build
any geometry directly with microgen and mesh it with `microgen.meshPeriodic`
(`mshFileVersion=2`). The mesh must:

- contain linear tetrahedra and a single material;
- fill a box-shaped periodic cell that is its bounding box (the solid touches all six faces);
- have matching nodes on opposite faces (a periodic mesh).

`python scripts/check_rve.py my_rve.msh` checks all three and runs a short elastoplastic path.

## microgen 1.3.1 limitations met

- TPMS: the periodic meshing depends on the tessellation `resolution`. Schwarz D sheet could not
  be meshed periodically (resolutions 16-30; at 16 microgen even returns the full cube); gyroid
  skeletal fails at 20. If gmsh reports "Different number of points ... for periodic
  correspondence", try another resolution. The `tpms` builder checks the solid fraction.
- The microgen `honeycomb` example meshes a finite block; the `honeycomb` builder here uses the
  periodic rectangular cell instead.
- Examples that need external tools (Neper for Voronoi, mmg for remeshing) are not included.
