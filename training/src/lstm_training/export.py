"""Export of a trained network in the ``.npz`` format read by the fe_lstm stage.

Same files and keys as ``legacy/training/LSTM_model_Get_weights_scaler_*.py``:

* ``LSTM_layer_k.npz`` (k = 1 .. n_lstm): ``W_i W_f W_c W_o`` (input kernels), ``U_*`` (recurrent
  kernels), ``b_*`` (biases), Keras gate order i, f, c, o;
* ``Dense_layer_k.npz`` (k = n_lstm + 1 .. n_lstm + n_dense): ``W``, ``b``;
* ``scaling_<name>.npz``: ``scalingX_min scalingX_max scalingY_min scalingY_max``.

New: ``architecture.json`` (layer sizes, activations, input / output names). fe_lstm reads the
activations from it; without it (thesis weights) it assumes hidden Dense layers in relu and a
linear output layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lstm_training.data import MinMaxScaling

FORMAT_VERSION = 1

_ACT = {
    "linear": lambda z: z,
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "elu": lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0.0))),
    "selu": lambda z: 1.0507009873554805 * np.where(z > 0, z, 1.6732632423543772 * np.expm1(np.minimum(z, 0.0))),
    "softplus": lambda z: np.logaddexp(0.0, z),
    "softsign": lambda z: z / (1.0 + np.abs(z)),
    "silu": lambda z: z / (1.0 + np.exp(-z)),
    "swish": lambda z: z / (1.0 + np.exp(-z)),
    "leaky_relu": lambda z: np.where(z > 0, z, 0.2 * z),
}


def export_network(model, folder: str | Path, scaling_name: str, x_scaling: MinMaxScaling,
                   y_scaling: MinMaxScaling, architecture: dict) -> Path:
    """Write the weights, scaling and ``architecture.json`` of ``model`` into ``folder``."""
    import keras

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for old in list(folder.glob("LSTM_layer_*.npz")) + list(folder.glob("Dense_layer_*.npz")):
        old.unlink()  # a previous export with more layers must not leave stale files
    k = 0
    for layer in model.layers:
        k += 1
        if isinstance(layer, keras.layers.LSTM):
            W, U, b = layer.get_weights()
            u = layer.units
            gates = {}
            for n, g in enumerate("ifco"):
                gates[f"W_{g}"] = W[:, n * u:(n + 1) * u]
                gates[f"U_{g}"] = U[:, n * u:(n + 1) * u]
                gates[f"b_{g}"] = b[n * u:(n + 1) * u]
            np.savez(folder / f"LSTM_layer_{k}.npz", **gates)
        elif isinstance(layer, keras.layers.Dense):
            W, b = layer.get_weights()
            np.savez(folder / f"Dense_layer_{k}.npz", W=W, b=b)
        else:
            raise TypeError(f"Cannot export layer {layer.name} ({type(layer).__name__})")
    np.savez(
        folder / f"scaling_{scaling_name}.npz",
        scalingX_min=x_scaling.data_min, scalingX_max=x_scaling.export_max,
        scalingY_min=y_scaling.data_min, scalingY_max=y_scaling.export_max,
    )
    arch = {"format_version": FORMAT_VERSION, "scaling_file": f"scaling_{scaling_name}.npz", **architecture}
    (folder / "architecture.json").write_text(json.dumps(arch, indent=2))
    return folder


def predict_exported(folder: str | Path, x_seq: np.ndarray) -> np.ndarray:
    """Run the exported files with NumPy (unscaled x (n, T, n_in) -> unscaled y (n, T, n_out)).

    Independent of Keras: used to check an export before handing it to fe_lstm.
    """
    folder = Path(folder)
    arch = json.loads((folder / "architecture.json").read_text())
    n_lstm, n_dense = len(arch["lstm_units"]), len(arch["dense_units"]) + 1
    acts = list(arch["dense_activations"]) + [arch["output_activation"]]
    with np.load(folder / arch["scaling_file"]) as s:
        x_min, x_max, y_min, y_max = (s[k] for k in ("scalingX_min", "scalingX_max", "scalingY_min", "scalingY_max"))
    z = (np.asarray(x_seq, float) - x_min) / (x_max - x_min)
    sig = _ACT["sigmoid"]
    for k in range(1, n_lstm + 1):
        with np.load(folder / f"LSTM_layer_{k}.npz") as L:
            p = {key: L[key] for key in L.files}
        n, T, _ = z.shape
        u = p["U_i"].shape[0]
        h, c, out = np.zeros((n, u)), np.zeros((n, u)), np.empty((n, T, u))
        for t in range(T):
            pre = {g: z[:, t] @ p[f"W_{g}"] + h @ p[f"U_{g}"] + p[f"b_{g}"] for g in "ifco"}
            c = sig(pre["f"]) * c + sig(pre["i"]) * np.tanh(pre["c"])
            h = sig(pre["o"]) * np.tanh(c)
            out[:, t] = h
        z = out
    for k, act in zip(range(n_lstm + 1, n_lstm + n_dense + 1), acts):
        with np.load(folder / f"Dense_layer_{k}.npz") as D:
            z = _ACT[act](z @ D["W"] + D["b"])
    return z * (y_max - y_min) + y_min
