"""Keras 3 model of the FE-LSTM surrogates, losses, optimizers and callbacks.

Architecture (legacy defaults in brackets)::

    LSTM(lstm_units[0], return_sequences=True) -> ... -> LSTM(lstm_units[-1])      [50, 50]
    -> Dense(dense_units[k], dense_activations[k]) for each hidden layer          [50, relu]
    -> Dense(n_outputs, output_activation)                                        [linear]

The LSTM layers keep the Keras defaults (tanh, sigmoid gates), which the NumPy inference of the
fe_lstm stage assumes. Dense activations are restricted to those it implements
(:data:`SUPPORTED_ACTIVATIONS`).
"""

from __future__ import annotations

import numpy as np

# Activations implemented by fe_lstm/src/fe_lstm/model/numpy_lstm.py (keep both lists in sync).
SUPPORTED_ACTIVATIONS = ("linear", "relu", "tanh", "sigmoid", "elu", "selu", "softplus",
                         "softsign", "silu", "swish", "leaky_relu")


def check_architecture(lstm_units, dense_units, dense_activations, output_activation):
    if not lstm_units:
        raise ValueError("lstm_units needs at least one LSTM layer")
    if len(dense_activations) != len(dense_units):
        raise ValueError(f"dense_activations {dense_activations} needs one entry per hidden Dense "
                         f"layer of dense_units {dense_units}")
    for act in list(dense_activations) + [output_activation]:
        if act not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Activation {act!r} is not supported by the fe_lstm law; "
                             f"use one of {SUPPORTED_ACTIVATIONS}")


def build_model(n_inputs: int, n_outputs: int, lstm_units, dense_units, dense_activations,
                output_activation: str = "linear"):
    """Uncompiled Keras Sequential model; sequences of any length."""
    import keras

    check_architecture(lstm_units, dense_units, dense_activations, output_activation)
    layers = [keras.Input((None, n_inputs))]
    layers += [keras.layers.LSTM(int(u), return_sequences=True, name=f"lstm_{k + 1}")
               for k, u in enumerate(lstm_units)]
    n_lstm = len(lstm_units)
    layers += [keras.layers.Dense(int(u), activation=a, name=f"dense_{n_lstm + k + 1}")
               for k, (u, a) in enumerate(zip(dense_units, dense_activations))]
    layers.append(keras.layers.Dense(n_outputs, activation=output_activation,
                                     name=f"dense_{n_lstm + len(dense_units) + 1}"))
    return keras.Sequential(layers)


def nmse(y_true, y_pred):
    """Percentage normalized MSE of the legacy code: 100 sum((y - y_pred)^2) / sum(y^2) (per batch)."""
    from keras import ops

    return 100.0 * ops.sum(ops.square(y_true - y_pred)) / ops.sum(ops.square(y_true))


def get_loss(name: str):
    import keras

    return nmse if name == "nmse" else keras.losses.get(name)


def get_optimizer(spec):
    """``"adam"`` or ``{"name": "adam", "learning_rate": 1e-3, ...}`` -> Keras optimizer."""
    import keras

    spec = {"name": spec} if isinstance(spec, str) else dict(spec)
    name = spec.pop("name")
    return keras.optimizers.get({"class_name": name, "config": spec})


def best_weights_callback(monitor: str):
    """Callback keeping the weights of the epoch with the lowest ``monitor`` and restoring them at
    the end of training (the legacy code saved a checkpoint per epoch and reloaded the best)."""
    import keras

    class KeepBest(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.best_epoch, self.weights = np.inf, None, None

        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get(monitor)
            if value is None:
                raise KeyError(f"monitor {monitor!r} not in the training logs {sorted(logs or {})}")
            if value < self.best:
                self.best, self.best_epoch = float(value), epoch + 1
                self.weights = self.model.get_weights()

        def on_train_end(self, logs=None):
            if self.weights is not None:
                self.model.set_weights(self.weights)

    return KeepBest()
