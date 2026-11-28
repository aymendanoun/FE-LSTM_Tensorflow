"""The NumPy LSTM must reproduce the Keras network it was exported from."""

import numpy as np
import pytest

from fe_lstm.model.numpy_lstm import LSTMSurrogate


def keras_style_reference(net: LSTMSurrogate, x_seq):
    """Independent implementation using Keras' fused kernels (W = [W_i W_f W_c W_o])."""
    z_seq = (x_seq - net.x_min) / (net.x_max - net.x_min)
    for layer in net.lstm_layers:
        W = np.concatenate([layer[f"W_{g}"] for g in "ifco"], axis=1)
        U = np.concatenate([layer[f"U_{g}"] for g in "ifco"], axis=1)
        b = np.concatenate([layer[f"b_{g}"] for g in "ifco"])
        n, T, _ = z_seq.shape
        u = U.shape[0]
        h, c = np.zeros((n, u)), np.zeros((n, u))
        out = np.empty((n, T, u))
        for t in range(T):
            gates = z_seq[:, t] @ W + h @ U + b
            i, f, g, o = np.split(gates, 4, axis=1)
            sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
            c = sig(f) * c + sig(i) * np.tanh(g)
            h = sig(o) * np.tanh(c)
            out[:, t] = h
        z_seq = out
    d1, d2 = net.dense_layers
    y = np.maximum(z_seq @ d1["W"] + d1["b"], 0) @ d2["W"] + d2["b"]
    return y * (net.y_max - net.y_min) + net.y_min


def test_shapes_and_units(weights_dir):
    net = LSTMSurrogate(weights_dir / "weights_scaler_stress", "scaling_stress.npz")
    assert net.units == [50, 50]
    assert (net.n_inputs, net.n_outputs) == (6, 6)


def test_matches_keras_formulation(weights_dir):
    net = LSTMSurrogate(weights_dir / "weights_scaler_stress", "scaling_stress.npz")
    x = np.random.default_rng(1).uniform(-0.05, 0.05, (7, 20, 6))
    np.testing.assert_allclose(net.predict_sequence(x), keras_style_reference(net, x), rtol=1e-12, atol=1e-12)


def test_step_does_not_mutate_state(weights_dir):
    net = LSTMSurrogate(weights_dir / "weights_scaler_stress", "scaling_stress.npz")
    state = net.zero_state(3)
    x = np.full((3, 6), 0.01)
    y1, _ = net.step(x, state)
    y2, _ = net.step(x, state)
    np.testing.assert_array_equal(y1, y2)
    assert all(not h.any() and not c.any() for h, c in state)


def test_matches_tensorflow_keras(weights_dir):
    """Rebuild the thesis architecture in Keras with the same weights (skipped without TF)."""
    keras = pytest.importorskip("tensorflow").keras
    net = LSTMSurrogate(weights_dir / "weights_scaler_stress", "scaling_stress.npz")
    model = keras.Sequential(
        [
            keras.Input((None, 6)),
            keras.layers.LSTM(50, return_sequences=True),
            keras.layers.LSTM(50, return_sequences=True),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dense(6, activation="linear"),
        ]
    )
    for keras_layer, layer in zip(model.layers[:2], net.lstm_layers):
        keras_layer.set_weights(
            [np.concatenate([layer[f"{k}_{g}"] for g in "ifco"], axis=-1) for k in "WUb"]
        )
    for keras_layer, layer in zip(model.layers[2:], net.dense_layers):
        keras_layer.set_weights([layer["W"], layer["b"]])
    x = np.random.default_rng(2).uniform(-0.05, 0.05, (4, 15, 6))
    y_keras = model.predict((x - net.x_min) / (net.x_max - net.x_min), verbose=0)
    y_keras = y_keras * (net.y_max - net.y_min) + net.y_min
    np.testing.assert_allclose(net.predict_sequence(x), y_keras, rtol=1e-4, atol=1e-4)


def _write_general_network(folder, rng, lstm_units, dense_units, acts, n_in=6, n_out=6, arch=True):
    """Weights of an n-LSTM + m-Dense network in the training-stage export format."""
    import json

    folder.mkdir(parents=True, exist_ok=True)
    n_prev = n_in
    for k, u in enumerate(lstm_units, 1):
        np.savez(folder / f"LSTM_layer_{k}.npz", **{
            f"{w}_{g}": rng.normal(0, 0.3, shape) for g in "ifco"
            for w, shape in (("W", (n_prev, u)), ("U", (u, u)), ("b", (u,)))
        })
        n_prev = u
    for k, u in enumerate(list(dense_units) + [n_out], len(lstm_units) + 1):
        np.savez(folder / f"Dense_layer_{k}.npz", W=rng.normal(0, 0.3, (n_prev, u)), b=rng.normal(0, 0.3, u))
        n_prev = u
    np.savez(folder / "scaling_stress.npz", scalingX_min=np.zeros(n_in), scalingX_max=0.05 * np.ones(n_in),
             scalingY_min=-np.ones(n_out), scalingY_max=np.ones(n_out))
    if arch:
        (folder / "architecture.json").write_text(json.dumps({
            "inputs": ["Exx", "Eyy", "Ezz", "Exy", "Exz", "Eyz"], "outputs": ["Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
            "lstm_units": list(lstm_units), "dense_units": list(dense_units),
            "dense_activations": list(acts[:-1]), "output_activation": acts[-1]}))


def test_general_architecture_matches_keras(tmp_path):
    """3 LSTM layers, 2 hidden Dense layers (tanh, elu), softplus output, read from architecture.json."""
    keras = pytest.importorskip("tensorflow").keras
    acts = ["tanh", "elu", "softplus"]
    _write_general_network(tmp_path, np.random.default_rng(5), [7, 9, 5], [8, 4], acts)
    net = LSTMSurrogate(tmp_path, "scaling_stress.npz")
    assert net.n_lstm_layers == 3 and net.units == [7, 9, 5] and net.activations == acts
    model = keras.Sequential(
        [keras.Input((None, 6))]
        + [keras.layers.LSTM(u, return_sequences=True) for u in (7, 9, 5)]
        + [keras.layers.Dense(u, activation=a) for u, a in zip((8, 4, 6), acts)]
    )
    for keras_layer, layer in zip(model.layers[:3], net.lstm_layers):
        keras_layer.set_weights([np.concatenate([layer[f"{k}_{g}"] for g in "ifco"], axis=-1) for k in "WUb"])
    for keras_layer, layer in zip(model.layers[3:], net.dense_layers):
        keras_layer.set_weights([layer["W"], layer["b"]])
    x = np.random.default_rng(6).uniform(0, 0.05, (3, 12, 6))
    y_keras = model.predict((x - net.x_min) / (net.x_max - net.x_min), verbose=0)
    y_keras = y_keras * (net.y_max - net.y_min) + net.y_min
    np.testing.assert_allclose(net.predict_sequence(x), y_keras, rtol=1e-4, atol=1e-4)


def test_layer_count_without_architecture_file(tmp_path):
    """Thesis-style export (no architecture.json): relu hidden layers, linear output."""
    _write_general_network(tmp_path, np.random.default_rng(7), [5], [6, 6], None, arch=False)
    net = LSTMSurrogate(tmp_path, "scaling_stress.npz")
    assert net.n_lstm_layers == 1 and net.activations == ["relu", "relu", "linear"]


def test_unsupported_activation_rejected(tmp_path):
    _write_general_network(tmp_path, np.random.default_rng(8), [5], [6], ["gelu", "linear"])
    with pytest.raises(ValueError, match="gelu"):
        LSTMSurrogate(tmp_path, "scaling_stress.npz")
