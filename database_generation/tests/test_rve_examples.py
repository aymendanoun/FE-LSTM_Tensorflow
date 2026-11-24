"""The example RVE meshes of rve_examples/ can be used by the database generation."""

import numpy as np
import pytest
import yaml

from database_generation.check import _face_pairs_match, check_mesh

from dbgen_paths import LEGACY_MESH, STAGE

EXAMPLES_DIR = STAGE / "rve_examples"
EXAMPLES = yaml.safe_load((EXAMPLES_DIR / "examples.yaml").read_text())


def test_every_listed_example_has_a_mesh():
    assert sorted(p.stem for p in EXAMPLES_DIR.glob("*.msh")) == sorted(EXAMPLES)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_is_usable(name):
    info = check_mesh(EXAMPLES_DIR / f"{name}.msh")  # raises if not tet4 / periodic / stiffness
    assert info["periodic"]
    assert 0.05 < info["solid_fraction"] < 0.95
    C = info["C_hom"]
    np.testing.assert_allclose(C, C.T, atol=1e-6 * np.abs(C).max())


def test_default_config_uses_thesis_example(cfg):
    assert cfg["geometry"]["mesh_file"] == "rve_examples/octet_truss_thesis.msh"


def test_thesis_example_matches_legacy_mesh():
    C_ex = check_mesh(EXAMPLES_DIR / "octet_truss_thesis.msh")["C_hom"]
    C_legacy = check_mesh(LEGACY_MESH)["C_hom"]
    for i, j in [(0, 0), (0, 1), (3, 3)]:
        assert abs(C_ex[i, j] / C_legacy[i, j] - 1) < 0.01


def test_face_pairs_match():
    g = np.array(np.meshgrid(*[np.linspace(0, 1, 3)] * 3)).reshape(3, -1).T
    assert _face_pairs_match(g, 1e-9)
    g[0] += [0, 0.1, 0.0] if g[0, 0] == 0 else 0  # move a node of the x=0 face within the face
    assert not _face_pairs_match(g, 1e-9)
