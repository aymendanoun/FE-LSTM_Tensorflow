"""Regression: a short training on the synthetic database reproduces reference_results/.

Set FELSTM_UPDATE_REFERENCE=1 to rewrite the reference file (after a deliberate change).
Tolerance: 5% relative on the losses (TensorFlow CPU results vary slightly across versions and
thread counts even with a fixed seed).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("keras")

from lstm_training.export import predict_exported  # noqa: E402
from lstm_training.train import train_network  # noqa: E402

REFERENCE = Path(__file__).resolve().parents[1] / "reference_results" / "tiny_training_synthetic.json"
RTOL = 0.05


def test_train_and_export(tiny_cfg, tmp_path):
    s = train_network(tiny_cfg, "stress", tmp_path, verbose=False)
    out = Path(s["folder"])
    for f in ["LSTM_layer_1.npz", "LSTM_layer_2.npz", "Dense_layer_3.npz", "Dense_layer_4.npz",
              "scaling_stress.npz", "architecture.json", "history.csv", "history.png", "metrics.json",
              "model.keras", "test_predictions.npz"]:
        assert (out / f).exists(), f
    assert s["export_check_max_rel_diff"] < 1e-4
    assert s["epochs_run"] == 5 and 1 <= s["best_epoch"] <= 5
    assert len(s["test"]["paths"]) == 2  # 10% of 20 paths, the last ones
    p = np.load(out / "test_predictions.npz")
    np.testing.assert_allclose(predict_exported(out, p["x"]), p["y_pred"], rtol=1e-4, atol=1e-3)

    hist = np.genfromtxt(out / "history.csv", delimiter=",", names=True)
    result = {"loss": hist["loss"].tolist(), "val_loss": hist["val_loss"].tolist(),
              "test_nmse_percent": s["test"]["nmse_percent"]}
    if os.environ.get("FELSTM_UPDATE_REFERENCE"):
        REFERENCE.write_text(json.dumps(result, indent=2))
    ref = json.loads(REFERENCE.read_text())
    np.testing.assert_allclose(result["loss"], ref["loss"], rtol=RTOL)
    np.testing.assert_allclose(result["val_loss"], ref["val_loss"], rtol=RTOL)
    np.testing.assert_allclose(result["test_nmse_percent"], ref["test_nmse_percent"], rtol=RTOL)


def test_deeper_tangent_network(tiny_cfg, tmp_path):
    tiny_cfg["networks"]["tangent"]["model"] = {"lstm_units": [6, 6, 6], "dense_units": [8, 8],
                                                "dense_activations": ["tanh", "silu"]}
    tiny_cfg["networks"]["tangent"]["training"] = {"epochs": 2, "loss": "nmse",
                                                   "optimizer": {"name": "rmsprop", "learning_rate": 1e-3}}
    s = train_network(tiny_cfg, "tangent", tmp_path, verbose=False)
    arch = json.loads((Path(s["folder"]) / "architecture.json").read_text())
    assert arch["lstm_units"] == [6, 6, 6] and arch["dense_activations"] == ["tanh", "silu"]
    assert len(arch["outputs"]) == 21 and s["epochs_run"] == 2
    assert (Path(s["folder"]) / "Dense_layer_6.npz").exists()


def test_time_weights_change_training_and_report_increments(tiny_cfg, tmp_path):
    base = train_network(tiny_cfg, "stress", tmp_path / "a", verbose=False)
    tiny_cfg["training"].update(time_weights={"power": 2.0, "floor": 0.05}, monitor="val_loss")
    weighted = train_network(tiny_cfg, "stress", tmp_path / "b", verbose=False)
    assert weighted["best_value"] != base["best_value"]
    t = weighted["test"]
    assert len(t["rmse_by_increment"]) == len(t["rms_by_increment"]) > 1
