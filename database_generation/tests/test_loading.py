"""Random strain paths (port of generate_loadings / create_path of the legacy scripts)."""

import numpy as np
import pytest

from database_generation.generate import path_targets
from database_generation.loading import COMPONENTS, discretize, path_rng, random_targets


def test_legacy_defaults(cfg):
    lo = cfg["loading"]
    assert lo["n_segments"] * lo["increments_per_segment"] == 100  # training sequence length
    assert lo["active_components"] == ["Exx", "Eyy", "Exy"]


def test_targets_bounds_and_inactive_components(cfg):
    lo = cfg["loading"]
    t = np.array([path_targets(cfg, i) for i in range(500)])  # (500, 4, 6)
    assert t.shape == (500, lo["n_segments"], 6)
    for k, name in enumerate(COMPONENTS):
        if name in lo["active_components"]:
            assert t[..., k].min() >= lo["strain_min"][k] and t[..., k].max() <= lo["strain_max"][k]
            assert t[..., k].std() > 0
        else:
            assert np.all(t[..., k] == 0)


def test_paths_reproducible_and_distinct(cfg):
    np.testing.assert_array_equal(path_targets(cfg, 7), path_targets(cfg, 7))
    flat = {path_targets(cfg, i).tobytes() for i in range(1000)}
    assert len(flat) == 1000
    assert not np.array_equal(path_targets(cfg, 3), random_targets(
        path_rng(cfg["seed"] + 1, 3), 4, cfg["loading"]["active_components"],
        cfg["loading"]["strain_min"], cfg["loading"]["strain_max"]))


def test_discretize_is_piecewise_linear():
    targets = np.array([[0.02, 0, 0, 0.01, 0, 0], [-0.01, 0.03, 0, 0, 0, 0]])
    eps = discretize(targets, 4)
    assert eps.shape == (8, 6)
    np.testing.assert_allclose(eps[3], targets[0])
    np.testing.assert_allclose(eps[7], targets[1])
    np.testing.assert_allclose(eps[0], targets[0] / 4)
    np.testing.assert_allclose(eps[5], targets[0] + 0.5 * (targets[1] - targets[0]))


def test_bad_inputs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        random_targets(rng, 4, ["Exx", "Bad"], [0] * 6, [1] * 6)
    with pytest.raises(ValueError):
        random_targets(rng, 4, ["Exx"], [1] * 6, [0] * 6)
