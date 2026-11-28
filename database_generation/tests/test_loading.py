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


# ------------------------------------------------------------------------- "mixed" sampling
def _mixed_cfg(cfg, sampler="sobol"):
    import copy

    c = copy.deepcopy(cfg)
    c["loading"]["sampling"] = "mixed"
    c["loading"]["n_segments"] = 2
    c["loading"]["mixed"] = {"strain_scale": [0.05, 0.05, 0.05, 0.025, 0.025, 0.025], "max_norm": 1.25,
                             "fractions": {"proportional": 0.4, "reversal": 0.3, "random": 0.3},
                             "reversal_factor": [-1.0, 0.5], "sampler": sampler}
    return c


@pytest.mark.parametrize("sampler", ["sobol", "random"])
def test_mixed_sampling(cfg, sampler):
    from database_generation.loading import mixed_path_targets

    c = _mixed_cfg(cfg, sampler)
    lo = c["loading"]
    scale = np.array(lo["mixed"]["strain_scale"])
    out = [mixed_path_targets(c["seed"], i, lo["n_paths"], 2, lo["mixed"]) for i in range(lo["n_paths"])]
    t = np.array([o[0] for o in out])
    kinds = np.array([o[1] for o in out])
    np.testing.assert_array_equal(path_targets(c, 5), t[5])
    assert t.shape == (lo["n_paths"], 2, 6)
    assert np.all(np.linalg.norm(t / scale, axis=2) <= 1.25 + 1e-12)
    assert np.all(np.abs((t[:, 0] > 0).mean(0) - 0.5) < 0.06)  # both signs, every component
    frac = {k: (kinds == k).mean() for k in ("proportional", "reversal", "random")}
    tol = 0.01 if sampler == "sobol" else 0.05
    assert abs(frac["proportional"] - 0.4) < tol and abs(frac["reversal"] - 0.3) < tol
    p = t[kinds == "proportional"]
    np.testing.assert_allclose(p[:, 1], 2 * p[:, 0])  # monotonic straight path
    r = t[kinds == "reversal"]
    c_fac = np.einsum("ij,ij->i", r[:, 1], r[:, 0]) / np.einsum("ij,ij->i", r[:, 0], r[:, 0])
    assert c_fac.min() >= -1 - 1e-9 and c_fac.max() <= 0.5 + 1e-9
    np.testing.assert_allclose(r[:, 1], c_fac[:, None] * r[:, 0], atol=1e-12)


def test_mixed_sampling_errors(cfg):
    from database_generation.loading import mixed_path_targets

    c = _mixed_cfg(cfg)
    with pytest.raises(IndexError):
        mixed_path_targets(0, c["loading"]["n_paths"], c["loading"]["n_paths"], 2, c["loading"]["mixed"])
    c["loading"]["mixed"]["fractions"] = {"cyclic": 1.0}
    with pytest.raises(ValueError):
        path_targets(c, 0)
    c["loading"]["sampling"] = "grid"
    with pytest.raises(ValueError, match="box"):
        path_targets(c, 0)


def test_gyroid_3d_config_is_mixed():
    import yaml

    from dbgen_paths import STAGE

    c = yaml.safe_load((STAGE / "configs" / "database_gyroid_3d.yaml").read_text())
    assert c["loading"]["sampling"] == "mixed" and c["output"]["dir"] == "data/database_gyroid_3d"
    assert path_targets(c, 0).shape == (2, 6)
