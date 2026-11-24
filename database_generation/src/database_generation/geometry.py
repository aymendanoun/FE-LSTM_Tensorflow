"""Periodic RVEs built with microgen and meshed with gmsh (linear tetrahedra).

Runs in the ``felstm-geometry`` environment (``environment-geometry.yml``), which has microgen.
The rest of the stage (micro-simulations) runs in ``felstm`` and only reads the mesh file.

Workflow of microgen's examples (https://3mah.github.io/microgen-docs/examples/index.html,
"Periodic gyroid mesh from a STEP file"), in :func:`build_rve`:

1. a builder returns the geometry of a box-shaped periodic cell ``[0, Lx] x [0, Ly] x [0, Lz]``
   (one material): a CadQuery solid, or directly a tetrahedral volume grid (TPMS ``method="grid"``);
2. a CAD solid is meshed periodically with gmsh (:func:`mesh_rve`, ``microgen.meshPeriodic``);
3. optionally, the mesh is remeshed with mmg keeping its periodicity (:func:`remesh`,
   ``microgen.remesh.remesh_keeping_periodicity_for_fem``, named
   ``remesh_keeping_boundaries_for_fem(..., periodic=True)`` in microgen 2): the triangles on the
   cell faces are frozen, the rest is remeshed, which removes the slivers gmsh leaves along
   tessellated surfaces;
4. the mesh is written in gmsh format 2.2.

fedoo's ``PeriodicBC`` pairs the nodes of opposite faces. The cell is the bounding box of the
mesh: the solid must touch the six faces (true for all builders here). mmg (``mmg3d_O3``) is not
packaged for Apple Silicon: ``scripts/install_mmg.sh`` builds it into ``felstm-geometry``.

Builders (adapted from https://github.com/3MAH/microgen/tree/v1.3.1/examples):

* :func:`octet_truss`: octet-truss strut lattice (``examples/Lattices/octetTruss``), with separate
  radii for the face and inner struts. The thesis RVE (``legacy/database_generation/octet_surf.msh``,
  unit cell, 4072 tet4, meshed solid fraction 0.227) has face struts 0.10 and inner struts 0.05.
* :func:`tpms`: triply periodic minimal surface solids, sheet or skeletal
  (``examples/TPMS/gyroid``, ``examples/TPMS/tpms``), at a target solid fraction.
* :func:`spherical_pore`: cubic cell with a centred spherical pore (``examples/BasicShapes``).
* :func:`honeycomb`: periodic rectangular cell of a hexagonal honeycomb extruded along x
  (``examples/Lattices/honeycomb``, which meshes a finite, non-periodic block).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- octet truss


def octet_struts(cell_size: float = 1.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """End points of the 36 struts of the octet-truss unit cell ``[0, L]^3``: 24 face struts
    (face centre to the 4 corners of its face, cut in half by the cell, the halves on opposite
    faces matching periodically), then 12 inner struts (between neighbouring face centres)."""
    L = cell_size
    corners = [np.array(c, dtype=float) * L for c in itertools.product((0, 1), repeat=3)]
    face_centres = []
    for axis in range(3):
        for side in (0.0, 1.0):
            c = np.full(3, 0.5 * L)
            c[axis] = side * L
            face_centres.append(c)
    struts = []
    for fc in face_centres:
        axis = int(np.flatnonzero((fc == 0) | (fc == L))[0])
        for c in corners:
            if np.isclose(c[axis], fc[axis]):
                struts.append((fc, c))
    for a, b in itertools.combinations(face_centres, 2):
        if np.isclose(np.linalg.norm(a - b), L / np.sqrt(2)):
            struts.append((a, b))
    assert len(struts) == 36
    return struts


def _rotated_x_axis(angles) -> np.ndarray:
    """Direction of microgen's cylinder axis (x before rotation) after its XZX Euler rotation."""
    import pyvista as pv
    from microgen.operations import rotate_pv_euler

    line = pv.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    pts = rotate_pv_euler(line, (0.0, 0.0, 0.0), angles).points
    d = pts[-1] - pts[0]
    return d / np.linalg.norm(d)


def euler_angles_for(direction: np.ndarray) -> tuple[float, float, float]:
    """Euler angles (degrees, microgen convention) turning the x axis onto ``direction``."""
    d = direction / np.linalg.norm(direction)
    grid = np.arange(0.0, 360.0, 45.0)
    for angles in itertools.product(grid, grid, grid):
        if abs(abs(_rotated_x_axis(angles) @ d) - 1.0) < 1e-6:  # pyvista points are float32
            return angles
    raise ValueError(f"No Euler angles found for direction {direction}")


def octet_truss(cell_size: float = 1.0, face_strut_radius: float = 0.1,
                inner_strut_radius: float = 0.05):
    """Octet-truss lattice cut to the cubic cell. Returns ``(solid, (L, L, L))``."""
    import cadquery as cq
    import microgen

    cylinders, cache = [], {}
    for k, (a, b) in enumerate(octet_struts(cell_size)):
        d = b - a
        key = tuple(np.round(d / np.linalg.norm(d), 6))
        if key not in cache:
            cache[key] = euler_angles_for(d)
        cylinders.append(microgen.Cylinder(
            center=tuple((a + b) / 2), orientation=cache[key], height=float(np.linalg.norm(d)),
            radius=face_strut_radius if k < 24 else inner_strut_radius,
        ).generate())
    solid = microgen.fuseShapes(cylinders, retain_edges=False)
    solid = solid.intersect(cq.Solid.makeBox(cell_size, cell_size, cell_size))
    return solid, (cell_size,) * 3


# ------------------------------------------------------------------------------- TPMS


def tpms(surface: str = "gyroid", part: str = "sheet", solid_fraction: float = 0.25,
         cell_size: float = 1.0, resolution: int = 20, method: str = "cad"):
    """TPMS solid in a cubic cell (one period), offset chosen to reach ``solid_fraction``.

    ``method="cad"`` (default): CadQuery solid, meshed with gmsh then (optionally) mmg.
    ``method="grid"``: microgen's voxel-based volume grid (``Tpms.grid_sheet`` /
    ``grid_lower_skeletal`` / ``grid_upper_skeletal``, as in the microgen "Periodic remeshing"
    example), no CAD nor gmsh: always periodic, but its cell faces carry clipped voxel triangles,
    so the remeshed quality is lower. Use it when the CAD route cannot be meshed periodically.

    ``surface``: a function name of ``microgen.shape.surface_functions`` (gyroid, schwarzP,
    schwarzD, neovius, ...). ``part``: "sheet", "lower skeletal" or "upper skeletal".

    The surface is tessellated at ``resolution`` points per period; periodic meshing needs the same
    boundary points on opposite faces, which depends on it (microgen 1.3.1: Schwarz D sheet fails
    at 20 and 30, gives the full cell at 16; gyroid skeletal fails at 20). If gmsh reports "Different number of
    points ... for periodic correspondence", try another resolution.
    """
    from microgen import Tpms
    from microgen.shape import surface_functions

    geometry = Tpms(
        surface_function=getattr(surface_functions, surface), density=solid_fraction,
        cell_size=cell_size, resolution=resolution, center=(cell_size / 2,) * 3,
    )
    if method == "grid":
        import pyvista as pv

        grid = {"sheet": geometry.grid_sheet, "lower skeletal": geometry.grid_lower_skeletal,
                "upper skeletal": geometry.grid_upper_skeletal}[part].triangulate()
        grid = grid.translate(-np.array(grid.bounds[::2]), inplace=False)  # cell = [0, L]^3
        grid = grid.extract_cells_by_type(pv.CellType.TETRA)
        fraction = abs(grid.volume) / cell_size**3
        if abs(fraction - solid_fraction) > 0.02:
            raise ValueError(f"TPMS grid solid fraction {fraction:.3f} instead of {solid_fraction}")
        return grid, (cell_size,) * 3
    if method != "cad":
        raise ValueError(f"method must be 'cad' or 'grid', got {method!r}")
    solid = geometry.generate(type_part=part)
    fraction = solid.Volume() / cell_size**3
    if abs(fraction - solid_fraction) > 0.02:  # microgen can return a wrong solid (e.g. the full cell)
        raise ValueError(f"TPMS solid fraction {fraction:.3f} instead of {solid_fraction}: "
                         f"try another resolution than {resolution}")
    return solid, (cell_size,) * 3


# --------------------------------------------------------------------- spherical pore


def spherical_pore(cell_size: float = 1.0, pore_radius: float = 0.4):
    """Cubic cell with a centred spherical pore (simple cubic array of pores)."""
    import microgen

    c = (cell_size / 2,) * 3
    box = microgen.Box(center=c, dim=(cell_size,) * 3).generate()
    pore = microgen.Sphere(center=c, radius=pore_radius).generate()
    return box.cut(pore), (cell_size,) * 3


# -------------------------------------------------------------------------- honeycomb


def honeycomb(side_length: float = 0.5, wall_thickness: float = 0.06, depth: float = 0.5):
    """Hexagonal honeycomb, walls along x, in its periodic rectangular cell.

    Hexagons of side ``side_length`` (measured at mid-wall) tile the (y, z) plane with vertices
    pointing along z. The periodic cell is ``depth x sqrt(3) a x 3 a`` and contains one full hole
    at its centre plus four quarter holes at its corners.
    """
    import microgen

    a = side_length
    ly, lz = np.sqrt(3) * a, 3.0 * a
    r_hole = a - wall_thickness / np.sqrt(3)  # circumradius of the holes
    if r_hole <= 0:
        raise ValueError("wall_thickness too large for side_length")
    corners = [(r_hole * np.cos(t), r_hole * np.sin(t))
               for t in np.deg2rad(90 + 60 * np.arange(6))]
    box = microgen.Box(center=(depth / 2, ly / 2, lz / 2), dim=(depth, ly, lz)).generate()
    solid = box
    for y, z in [(ly / 2, lz / 2), (0, 0), (ly, 0), (0, lz), (ly, lz)]:
        hole = microgen.ExtrudedPolygon(
            center=(depth / 2, y, z), list_corners=corners, height=2 * depth
        ).generate()
        solid = solid.cut(hole)
    return solid, (depth, ly, lz)


BUILDERS = {
    "octet_truss": octet_truss,
    "tpms": tpms,
    "spherical_pore": spherical_pore,
    "honeycomb": honeycomb,
}


# ------------------------------------------------------------------------------- mesh


def mesh_rve(solid, dims, output_file: str | Path, mesh_size: float, order: int = 1) -> Path:
    """Mesh ``solid`` periodically in the cell ``[0, dims]`` (gmsh, format 2.2, tetrahedra)."""
    import cadquery as cq
    import microgen
    from microgen import meshPeriodic  # microgen 1.3.1 name

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rve = microgen.Rve(dim_x=dims[0], dim_y=dims[1], dim_z=dims[2],
                       center=tuple(d / 2 for d in dims))
    phase = microgen.Phase(shape=solid)
    step_file = output_file.with_suffix(".step")
    cq.exporters.export(cq.Workplane().add(solid), str(step_file))
    try:
        meshPeriodic(str(step_file), rve, [phase], size=mesh_size, order=order,
                     output_file=str(output_file), mshFileVersion=2)
    finally:
        step_file.unlink(missing_ok=True)  # the CAD file is only an intermediate
        import gmsh

        if gmsh.isInitialized():  # microgen leaves gmsh open when meshing fails
            gmsh.finalize()
    return output_file


def remesh(grid, hmax: float, hausd: float | None = None, hmin: float | None = None,
           hgrad: float | None = None):
    """Remesh a periodic tetrahedral grid with mmg, keeping its periodicity (microgen)."""
    import shutil

    from microgen.remesh import remesh_keeping_periodicity_for_fem

    if shutil.which("mmg3d_O3") is None:
        raise RuntimeError("mmg3d_O3 not found: run scripts/install_mmg.sh in felstm-geometry")
    return remesh_keeping_periodicity_for_fem(grid, hmax=hmax, hausd=hausd, hmin=hmin, hgrad=hgrad)


def write_msh(grid, output_file: str | Path) -> Path:
    """Write the tetrahedra of a pyvista grid in gmsh format 2.2 (read by fedoo)."""
    import meshio
    import pyvista as pv

    tets = grid.extract_cells_by_type(pv.CellType.TETRA)
    cells = tets.cells_dict[pv.CellType.TETRA]
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(output_file), meshio.Mesh(np.asarray(tets.points, dtype=float), [("tetra", cells)]),
                 file_format="gmsh22", binary=False)
    return output_file


def build_rve(output_file: str | Path, kind: str, mesh_size: float, remesh_options: dict | None = None,
              **params) -> Path:
    """Build an RVE with ``BUILDERS[kind](**params)`` and write its periodic tetrahedral mesh.

    ``mesh_size``: gmsh size for CAD geometries. ``remesh_options``: if given, mmg options of
    :func:`remesh` (``hmax`` required, ``hausd``, ``hmin``, ``hgrad``).
    """
    import pyvista as pv

    if kind not in BUILDERS:
        raise ValueError(f"Unknown RVE kind {kind!r}; available: {sorted(BUILDERS)}")
    geometry, dims = BUILDERS[kind](**params)
    output_file = Path(output_file)
    if isinstance(geometry, pv.DataSet):
        if not remesh_options:
            raise ValueError("A grid geometry (TPMS method='grid') needs remesh_options")
        grid = geometry
    else:
        mesh_rve(geometry, dims, output_file, mesh_size)
        if not remesh_options:
            return output_file
        grid = pv.read(str(output_file)).extract_cells_by_type(pv.CellType.TETRA)
    return write_msh(remesh(grid, **remesh_options), output_file)


def mesh_octet_rve(output_file: str | Path, cell_size: float = 1.0, face_strut_radius: float = 0.1,
                   inner_strut_radius: float = 0.05, mesh_size: float = 0.1) -> Path:
    """Thesis octet-truss RVE (kept for compatibility: ``build_rve(..., "octet_truss", ...)``)."""
    return build_rve(output_file, "octet_truss", mesh_size, cell_size=cell_size,
                     face_strut_radius=face_strut_radius, inner_strut_radius=inner_strut_radius)
