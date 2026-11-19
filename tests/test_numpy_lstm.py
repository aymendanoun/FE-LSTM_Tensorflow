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
