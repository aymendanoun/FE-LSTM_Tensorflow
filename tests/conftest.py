"""Shared fixtures: synthetic network weights in the exact .npz format of the legacy export
(legacy/training/LSTM_model_Get_weights_scaler_*.py), so tests run without the trained models."""

import numpy as np
import pytest

UNITS = 50  # as in the thesis networks
E, NU = 113800.0, 0.342


def isotropic_stiffness(E=E, nu=NU):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    H = np.zeros((6, 6))
    H[:3, :3] = lam
    H[np.arange(3), np.arange(3)] = lam + 2 * mu
    H[np.arange(3, 6), np.arange(3, 6)] = mu  # engineering shear strains
    return H


def write_network(folder, scaling_name, n_in, n_out, rng, units=UNITS, out_bias=None, scale=0.3):
    """Write LSTM_layer_1/2, Dense_layer_3/4 and scaling_<name>.npz with random weights."""
    folder.mkdir(parents=True, exist_ok=True)
    n_prev = n_in
    for k in (1, 2):
        layer = {}
        for g in "ifco":
            layer[f"W_{g}"] = rng.normal(0, scale, (n_prev, units))
            layer[f"U_{g}"] = rng.normal(0, scale, (units, units))
            layer[f"b_{g}"] = rng.normal(0, scale, units)
        np.savez(folder / f"LSTM_layer_{k}.npz", **layer)
        n_prev = units
    np.savez(folder / "Dense_layer_3.npz", W=rng.normal(0, scale, (units, units)), b=rng.normal(0, scale, units))
    if out_bias is None:
        W4, b4 = rng.normal(0, scale, (units, n_out)), rng.normal(0, scale, n_out)
    else:  # constant output = out_bias (with y_min = 0, y_max = 1)
        W4, b4 = np.zeros((units, n_out)), np.asarray(out_bias, dtype=float)
    np.savez(folder / "Dense_layer_4.npz", W=W4, b=b4)
    if out_bias is None:
        y_min, y_max = -rng.uniform(1, 2, n_out), rng.uniform(1, 2, n_out)
    else:
        y_min, y_max = np.zeros(n_out), np.ones(n_out)
    np.savez(
        folder / f"scaling_{scaling_name}.npz",
        scalingX_min=-0.05 * np.ones(n_in),
        scalingX_max=0.05 * np.ones(n_in),
        scalingY_min=y_min,
        scalingY_max=y_max,
    )


@pytest.fixture
def weights_dir(tmp_path):
    """Random stress network + tangent network returning a constant isotropic stiffness."""
    from fe_lstm.fem.lstm_law import TANGENT_INDEX

    rng = np.random.default_rng(0)
    root = tmp_path / "weights_scaler_3D"
    write_network(root / "weights_scaler_stress", "stress", 6, 6, rng)
    H = isotropic_stiffness()
    write_network(root / "weights_scaler_Lt", "Lt", 6, 21, rng, out_bias=[H[i, j] for i, j in TANGENT_INDEX])
    return root
