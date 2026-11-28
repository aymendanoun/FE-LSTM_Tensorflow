"""The exported .npz files reproduce the Keras model (any number of layers, any activation)."""

import json

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from lstm_training.data import MinMaxScaling  # noqa: E402
from lstm_training.export import _ACT, export_network, predict_exported  # noqa: E402
from lstm_training.model import SUPPORTED_ACTIVATIONS, build_model, get_optimizer  # noqa: E402


@pytest.mark.parametrize("act", SUPPORTED_ACTIVATIONS)
def test_numpy_activations_match_keras(act):
    z = np.linspace(-6, 6, 241)
    np.testing.assert_allclose(_ACT[act](z), keras.ops.convert_to_numpy(keras.activations.get(act)(z)),
                               rtol=1e-6, atol=1e-6)


def test_unsupported_activation():
    with pytest.raises(ValueError, match="gelu"):
        build_model(6, 6, [8], [8], ["gelu"])
    with pytest.raises(ValueError, match="one entry per hidden"):
        build_model(6, 6, [8], [8, 8], ["relu"])


def test_optimizer_spec():
    opt = get_optimizer({"name": "rmsprop", "learning_rate": 0.01})
    assert type(opt).__name__ == "RMSprop" and np.isclose(float(opt.learning_rate), 0.01)


def test_export_three_lstm_layers(tmp_path):
    keras.utils.set_random_seed(3)
    rng = np.random.default_rng(0)
    model = build_model(6, 5, [7, 9, 6], [11, 4], ["tanh", "elu"], "softplus")
    x = rng.uniform(0, 0.05, (4, 13, 6))
    x[..., 2] = 0.0  # a constant input, left unscaled
    y = rng.normal(0, 100, (4, 13, 5))
    xs, ys = MinMaxScaling().fit(x), MinMaxScaling().fit(y)
    arch = {"inputs": list("abcdef"), "outputs": list("vwxyz"), "lstm_units": [7, 9, 6],
            "dense_units": [11, 4], "dense_activations": ["tanh", "elu"], "output_activation": "softplus"}
    out = export_network(model, tmp_path / "net", "test", xs, ys, arch)
    names = sorted(p.name for p in out.glob("*.npz"))
    assert names == ["Dense_layer_4.npz", "Dense_layer_5.npz", "Dense_layer_6.npz", "LSTM_layer_1.npz",
                     "LSTM_layer_2.npz", "LSTM_layer_3.npz", "scaling_test.npz"]
    assert json.loads((out / "architecture.json").read_text())["scaling_file"] == "scaling_test.npz"
    y_keras = ys.inverse_transform(model.predict(xs.transform(x), verbose=0))
    np.testing.assert_allclose(predict_exported(out, x), y_keras, rtol=1e-4, atol=1e-4 * np.abs(y_keras).max())

    # re-export with fewer layers: no stale files left
    export_network(build_model(6, 5, [7], [], [], "linear"), out, "test", xs, ys,
                   {**arch, "lstm_units": [7], "dense_units": [], "dense_activations": [], "output_activation": "linear"})
    assert sorted(p.name for p in out.glob("*_layer_*.npz")) == ["Dense_layer_2.npz", "LSTM_layer_1.npz"]
