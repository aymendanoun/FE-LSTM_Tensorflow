"""Octet-truss RVE. The strut layout is pure NumPy; building and meshing need microgen and only
run in the felstm-geometry environment:
    conda activate felstm-geometry && pytest tests/test_geometry.py
"""

import itertools

import numpy as np
import pytest

from database_generation.geometry import octet_struts


def test_octet_struts():
    struts = octet_struts(2.0)
    assert len(struts) == 36
    lengths = [np.linalg.norm(b - a) for a, b in struts]
    np.testing.assert_allclose(lengths, 2.0 / np.sqrt(2))
    keys = {tuple(sorted((tuple(a), tuple(b)))) for a, b in struts}
    assert len(keys) == 36  # no duplicate strut
    on_face = [np.any(np.isclose(a, 0) & np.isclose(b, 0)) or np.any(np.isclose(a, 2) & np.isclose(b, 2))
               for a, b in struts]
    assert sum(on_face) == 24  # face struts first, then 12 inner struts
    assert all(on_face[:24]) and not any(on_face[24:])


def test_euler_angles_align_cylinder_axis():
    pytest.importorskip("microgen")
    from database_generation.geometry import _rotated_x_axis, euler_angles_for

    for d in {tuple(np.sign(b - a)) for a, b in octet_struts()}:
        d = np.array(d, float)
        axis = _rotated_x_axis(euler_angles_for(d))
        assert abs(abs(axis @ d / np.linalg.norm(d)) - 1) < 1e-6


def test_mesh_matches_thesis_rve(tmp_path):
    pytest.importorskip("microgen")
    import meshio

    from database_generation.geometry import mesh_octet_rve

    f = mesh_octet_rve(tmp_path / "rve.msh", 1.0, 0.10, 0.05, mesh_size=0.1)
    m = meshio.read(f)
    x = m.points
    tets = np.vstack([c.data for c in m.cells if c.type == "tetra"])
    a, b, c, d = (x[tets[:, i]] for i in range(4))
    vf = (np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6).sum()
    assert np.allclose(x.min(0), 0, atol=1e-8) and np.allclose(x.max(0), 1, atol=1e-8)
    assert abs(vf - 0.2272) < 0.005  # thesis mesh octet_surf.msh: 0.2272
    assert 3500 < len(tets) < 5000  # thesis mesh: 4072 tet4
    # periodicity: every node on a face has a partner on the opposite face
    for ax, (lo, hi) in itertools.product(range(3), [(0.0, 1.0)]):
        p_lo = x[np.isclose(x[:, ax], lo, atol=1e-8)]
        p_hi = x[np.isclose(x[:, ax], hi, atol=1e-8)]
        assert len(p_lo) == len(p_hi) > 0
        key = lambda p: sorted(map(tuple, np.round(np.delete(p, ax, axis=1), 6)))
        assert key(p_lo) == key(p_hi)


@pytest.mark.parametrize("kind,params,size", [
    ("spherical_pore", {"cell_size": 1.0, "pore_radius": 0.4}, 0.1),
    ("honeycomb", {"side_length": 0.5, "wall_thickness": 0.06, "depth": 0.5}, 0.08),
])
def test_quick_builders_are_periodic(tmp_path, kind, params, size):
    pytest.importorskip("microgen")
    import meshio

    from database_generation.geometry import build_rve

    m = meshio.read(build_rve(tmp_path / f"{kind}.msh", kind, size, **params))
    x = m.points
    assert not (tmp_path / f"{kind}.step").exists()
    for ax in range(3):
        lo, hi = x[:, ax].min(), x[:, ax].max()
        key = lambda p: sorted(map(tuple, np.round(np.delete(p, ax, axis=1), 6)))
        assert key(x[np.isclose(x[:, ax], lo)]) == key(x[np.isclose(x[:, ax], hi)])


def test_unknown_kind(tmp_path):
    from database_generation.geometry import build_rve

    with pytest.raises(ValueError, match="Unknown RVE kind"):
        build_rve(tmp_path / "x.msh", "foam", 0.1)


def test_remesh_keeps_periodicity_and_improves_quality(tmp_path):
    pytest.importorskip("microgen")
    import shutil

    import pyvista as pv

    if shutil.which("mmg3d_O3") is None:
        pytest.skip("mmg3d_O3 not installed (scripts/install_mmg.sh)")
    from microgen import is_periodic

    from database_generation.check import element_quality
    from database_generation.geometry import build_rve

    raw = build_rve(tmp_path / "raw.msh", "spherical_pore", 0.1, cell_size=1.0, pore_radius=0.4)
    rem = build_rve(tmp_path / "rem.msh", "spherical_pore", 0.1, remesh_options={"hmax": 0.1, "hausd": 0.005},
                    cell_size=1.0, pore_radius=0.4)
    g_raw, g_rem = (pv.read(str(f)).extract_cells_by_type(pv.CellType.TETRA) for f in (raw, rem))
    assert is_periodic(g_rem.points)
    assert abs(g_rem.volume - g_raw.volume) < 0.01
    assert element_quality(g_rem)["sj_min"] > element_quality(g_raw)["sj_min"]


def test_grid_tpms_needs_remesh(tmp_path):
    pytest.importorskip("microgen")
    from database_generation.geometry import build_rve

    with pytest.raises(ValueError, match="remesh_options"):
        build_rve(tmp_path / "x.msh", "tpms", None, surface="gyroid", resolution=12, method="grid")
