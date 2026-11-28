"""Fixtures of the training stage: a small synthetic database and a fast config."""

import copy
from pathlib import Path

import pytest
import yaml

from training_synthetic import write_database

STAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load((STAGE / "configs" / "training.yaml").read_text())


@pytest.fixture
def synthetic_db(tmp_path):
    return write_database(tmp_path / "database")


@pytest.fixture
def tiny_cfg(cfg, synthetic_db, tmp_path):
    """Config of the stage with the synthetic database, small networks and few epochs."""
    c = copy.deepcopy(cfg)
    c["data"]["database_dir"] = str(synthetic_db)
    c["model"] = {"lstm_units": [8, 8], "dense_units": [8], "dense_activations": ["relu"],
                  "output_activation": "linear"}
    c["training"].update(epochs=5, batch_size=4, verbose=0)
    c["output"]["dir"] = str(tmp_path / "models")
    return c
