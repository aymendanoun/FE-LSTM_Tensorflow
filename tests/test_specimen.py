"""The ported specimen script runs end to end in fedoo 1.x (short run, reference material)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from fe_lstm.fem.specimen import run

ROOT = Path(__file__).resolve().parents[1]


def short_config(tmp_path, material):
    cfg = yaml.safe_load((ROOT / "configs/specimen_with_holes_tensile.yaml").read_text())
    cfg["material"]["type"] = material
    cfg["solver"].update(dt=0.1, tmax=0.2, print_info=0)
    cfg["output"].update(dir=str(tmp_path), vtk=False)
    return cfg


@pytest.mark.parametrize("material", ["elastic", "epicp"])
def test_specimen_runs(tmp_path, material):
    out = run(short_config(tmp_path, material), ROOT)
    disp = out["problem"].get_disp()
    assert np.isfinite(disp).all()
    assert np.isclose(disp[0].max(), 0.04)  # imposed DispX, fully applied at tmax
    curve = out["curve"]  # time, ux, Fx: one row per increment + origin
    assert curve.shape == (3, 3) and np.isclose(curve[-1, 1], 0.04)
    assert (tmp_path / "RNN_specimen_with_holes_tensile_force_displacement.csv").exists()


def test_specimen_runs_with_lstm_law(tmp_path, weights_dir):
    cfg = short_config(tmp_path, "lstm")
    cfg["material"]["weights_dir"] = str(weights_dir)
    cfg["solver"].update(tol_nr=10.0)  # random weights: wiring smoke test only
    out = run(cfg, ROOT)
    assert np.isfinite(np.asarray(out["assembly"].sv["Stress"])).all()


def test_unsymmetric_tangent_is_assembled_in_full(tmp_path, weights_dir):
    """fedoo assembles one triangle when assume_sym is True: the FD tangent must disable it."""
    cfg = short_config(tmp_path, "lstm")
    cfg["material"].update(weights_dir=str(weights_dir), tangent="finite_difference")
    cfg["solver"].update(tol_nr=10.0)
    out = run(cfg, ROOT)
    assert out["assembly"].assume_sym is False
