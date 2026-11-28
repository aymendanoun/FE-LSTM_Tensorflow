"""Strain-driven periodic RVE simulation (port of SolverUnitCell, legacy Homog_path.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from database_generation import io
from database_generation.generate import generate, run_single
from database_generation.loading import discretize
from database_generation.rve import RVESimulation, load_tet_mesh

from dbgen_paths import LEGACY_MESH, STAGE

REFERENCE = STAGE / "reference_results/legacy_mesh_epicp_path0_2x5.csv"


def test_elastic_rve_matches_homogenized_stiffness():
    """Linear elastic struts: mean stress = C_hom : strain and tangent = C_hom at every increment."""
    import fedoo as fd

    E, nu = 113800.0, 0.342
    mesh = load_tet_mesh(LEGACY_MESH)
    fd.ModelingSpace("3D")
    asm = fd.Assembly.create(fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(E, nu)), mesh)
    C_hom = fd.homogen.get_homogenized_stiffness(asm)
    # thesis values (C11, C12, C44 of the thesis tangent network's training data)
    np.testing.assert_allclose([C_hom[0, 0], C_hom[0, 1], C_hom[3, 3]], [8705.2, 3694.3, 3563.7], atol=0.1)

    sim = RVESimulation(mesh, "ELISO", [E, nu, 0.0])
    targets = np.array([[0.002, -0.001, 0.0005, 0.001, -0.0005, 0.0003], [-0.001, 0.002, 0, 0, 0.001, 0]])
    res = sim.run_path(targets, 3)
    np.testing.assert_allclose(res.strain, discretize(targets, 3), atol=1e-10)
    scale = np.abs(res.stress).max()
    np.testing.assert_allclose(res.stress, res.strain @ C_hom.T, atol=2e-3 * scale)
    np.testing.assert_allclose(res.tangent, np.broadcast_to(C_hom, res.tangent.shape), atol=1e-6 * C_hom.max())


def test_epicp_regression(short_cfg):
    """EPICP on the thesis mesh, path 0 shortened to 2 x 5 increments, against the stored reference
    (generated with this code, fedoo 1.1.0 / simcoon 2.5.1; no legacy output exists)."""
    df = io.to_dataframe(run_single(short_cfg, STAGE, 0))
    ref = pd.read_csv(REFERENCE)
    assert list(df.columns) == list(ref.columns)
    for cols, scale in [(io.STRAIN_COLUMNS, 1e-12), (io.STRESS_COLUMNS, 1e-6), (io.TANGENT_COLUMNS, 1e-6)]:
        tol = scale * np.abs(ref[cols].values).max()
        np.testing.assert_allclose(df[cols].values, ref[cols].values, atol=tol, err_msg=str(cols[:2]))
    assert np.abs(ref[io.STRESS_COLUMNS].values).max() > 100  # the path reaches plasticity


def test_generate_parallel_and_resume(short_cfg):
    out = Path(short_cfg["output"]["dir"])
    s1 = generate(short_cfg, STAGE, range(3), n_workers=2, verbose=False)
    assert s1["done"] == 3 and not s1["failed"]
    assert sorted(p.name for p in out.glob("path_*.csv")) == [f"path_0000{i}.csv" for i in range(3)]
    df0 = io.read_path(out, 0)
    assert len(df0) == 10 and df0["Increment"].tolist() == list(range(1, 11))
    s2 = generate(short_cfg, STAGE, range(4), n_workers=2, verbose=False)
    assert s2["skipped"] == 3 and s2["done"] == 1
    pd.testing.assert_frame_equal(io.read_path(out, 0), df0)


def test_resume_refuses_other_config(short_cfg):
    generate(short_cfg, STAGE, range(1), n_workers=1, verbose=False)
    short_cfg["loading"]["increments_per_segment"] = 4
    with pytest.raises(ValueError, match="different configuration"):
        generate(short_cfg, STAGE, range(2), n_workers=1, verbose=False)


def test_missing_mesh_message(cfg, tmp_path):
    cfg["geometry"]["mesh_file"] = str(tmp_path / "missing.msh")
    with pytest.raises(FileNotFoundError, match="build_rve.py"):
        generate(cfg, STAGE, range(1), verbose=False)
