"""NumPy inference of the exported FE-LSTM networks.

Ported from ``legacy/fedoo_coupling/fedoo/libConstitutiveLaw/ConstitutiveLaw_LSTM.py``.

The network trained in ``legacy/training`` is::

    LSTM(50, return_sequences=True) -> LSTM(50, return_sequences=True)
    -> Dense(50, relu) -> Dense(n_outputs, linear)

with MinMax scaling of inputs and outputs. Its weights are exported to ``.npz`` files by
``legacy/training/LSTM_model_Get_weights_scaler_*.py``:

* ``LSTM_layer_1.npz``, ``LSTM_layer_2.npz``: keys ``W_i, W_f, W_c, W_o`` (input kernels),
  ``U_i, U_f, U_c, U_o`` (recurrent kernels), ``b_i, b_f, b_c, b_o`` (biases), Keras gate order.
* ``Dense_layer_3.npz``, ``Dense_layer_4.npz``: keys ``W, b``.
* ``scaling_<name>.npz``: keys ``scalingX_min, scalingX_max, scalingY_min, scalingY_max``
  (``MinMaxScaler.data_min_`` / ``data_max_``).

Inference runs one time step at a time for many material points at once, so the network can
serve as a constitutive law: the recurrent state (h, c) of each layer is the material memory.

Deliberate difference from the legacy code: the ``.npz`` arrays are read once into memory.
The legacy law indexed lazy ``NpzFile`` objects, which re-read the arrays from disk at every
call. The numerical result is identical.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_GATES = ("i", "f", "c", "o")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.array(data[key]) for key in data.files}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LSTMSurrogate:
    """Step-by-step NumPy inference of a 2-LSTM + 2-Dense network.

    Parameters
    ----------
    weights_dir : str or Path
        Folder containing ``LSTM_layer_1.npz``, ``LSTM_layer_2.npz``, ``Dense_layer_3.npz``,
        ``Dense_layer_4.npz`` and the scaling file.
    scaling_file : str
        Name of the scaling file in ``weights_dir`` (``scaling_stress.npz`` or
        ``scaling_Lt.npz`` in the original project).
    """

    n_lstm_layers = 2

    def __init__(self, weights_dir: str | Path, scaling_file: str):
        weights_dir = Path(weights_dir)
        self.lstm_layers = [
            _load_npz(weights_dir / f"LSTM_layer_{k}.npz")
            for k in range(1, self.n_lstm_layers + 1)
        ]
        self.dense_layers = [
            _load_npz(weights_dir / f"Dense_layer_{k}.npz")
            for k in range(self.n_lstm_layers + 1, self.n_lstm_layers + 3)
        ]
        scaling = _load_npz(weights_dir / scaling_file)
        self.x_min = scaling["scalingX_min"]
        self.x_max = scaling["scalingX_max"]
        self.y_min = scaling["scalingY_min"]
        self.y_max = scaling["scalingY_max"]

    @property
    def units(self) -> list[int]:
        """Number of units of each LSTM layer."""
        return [layer["W_i"].shape[1] for layer in self.lstm_layers]

    @property
    def n_inputs(self) -> int:
        return self.lstm_layers[0]["W_i"].shape[0]

    @property
    def n_outputs(self) -> int:
        return self.dense_layers[-1]["W"].shape[1]

    def zero_state(self, n_points: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Initial recurrent state: one ``(h, c)`` pair of zeros per LSTM layer."""
        return [(np.zeros((n_points, u)), np.zeros((n_points, u))) for u in self.units]

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_min) / (self.x_max - self.x_min)

    def _inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * (self.y_max - self.y_min) + self.y_min

    @staticmethod
    def _lstm_cell(x, h_prev, c_prev, layer):
        pre = {g: x @ layer[f"W_{g}"] + layer[f"b_{g}"] + h_prev @ layer[f"U_{g}"] for g in _GATES}
        i_t = _sigmoid(pre["i"])
        f_t = _sigmoid(pre["f"])
        o_t = _sigmoid(pre["o"])
        c_t = f_t * c_prev + i_t * np.tanh(pre["c"])
        h_t = o_t * np.tanh(c_t)
        return h_t, c_t

    def step(self, x: np.ndarray, state):
        """Advance the network by one time step.

        Parameters
        ----------
        x : array of shape (n_points, n_inputs)
            Unscaled inputs (strain in Voigt notation for FE-LSTM).
        state : list of (h, c)
            Recurrent state of each LSTM layer, arrays of shape (n_points, units).
            Not modified.

        Returns
        -------
        y : array of shape (n_points, n_outputs)
            Unscaled outputs.
        new_state : list of (h, c)
            Recurrent state after this step.
        """
        z = self._transform(np.asarray(x, dtype=float))
        new_state = []
        for layer, (h_prev, c_prev) in zip(self.lstm_layers, state):
            h, c = self._lstm_cell(z, h_prev, c_prev, layer)
            new_state.append((h, c))
            z = h
        dense_hidden, dense_out = self.dense_layers
        z = np.maximum(z @ dense_hidden["W"] + dense_hidden["b"], 0.0)  # relu
        y = z @ dense_out["W"] + dense_out["b"]  # linear
        return self._inverse_transform(y), new_state

    def predict_sequence(self, x_seq: np.ndarray) -> np.ndarray:
        """Run full sequences, shape (n_sequences, timesteps, n_inputs) -> (.., n_outputs).

        Equivalent to ``model.predict`` on the Keras model; used for validation.
        """
        x_seq = np.asarray(x_seq, dtype=float)
        state = self.zero_state(x_seq.shape[0])
        outputs = []
        for t in range(x_seq.shape[1]):
            y, state = self.step(x_seq[:, t, :], state)
            outputs.append(y)
        return np.stack(outputs, axis=1)
