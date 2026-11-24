"""Shared fixtures of the database generation tests."""

import pytest
import yaml
from dbgen_paths import LEGACY_MESH, STAGE


@pytest.fixture
def cfg():
    return yaml.safe_load((STAGE / "configs/database.yaml").read_text())


@pytest.fixture
def short_cfg(cfg, tmp_path):
    """Config on the legacy mesh with a short path: 2 segments x 5 increments."""
    cfg["geometry"]["mesh_file"] = str(LEGACY_MESH)
    cfg["loading"]["n_segments"] = 2
    cfg["loading"]["increments_per_segment"] = 5
    cfg["output"]["dir"] = str(tmp_path / "database")
    return cfg
