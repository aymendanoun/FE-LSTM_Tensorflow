"""Database reading, MinMax scaling and train / test split.

The database is the output of the database_generation stage: one CSV per strain path
(``path_00000.csv``, ...), all with the same number of increments, columns ``Exx ... Eyz``,
``Sxx ... Syz`` and the full tangent ``C11 ... C66``. A column named ``Cij_m`` is the symmetric
part ``(Cij + Cji) / 2``, the label of the legacy tangent network.

Legacy procedure (``legacy/training/LSTM_model_Training_*.py``), kept here:

* inputs and outputs scaled to [0, 1] by a MinMax scaler fitted on **all** paths;
* test set = the last ``test_fraction`` of the paths (``train_test_split(..., shuffle=False)``);
* validation set = the last ``validation_split`` of the training paths (Keras ``validation_split``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_SYM = re.compile(r"^C([1-6])([1-6])_m$")


def path_files(database_dir: str | Path, n_paths: int | None = None) -> list[Path]:
    """Sorted path files of the database (the first ``n_paths`` if given)."""
    files = sorted(Path(database_dir).glob("path_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No path_*.csv in {database_dir}: generate the database first "
            "(database_generation/scripts/generate_database.py)."
        )
    if n_paths is not None:
        if n_paths > len(files):
            raise ValueError(f"n_paths = {n_paths} but {database_dir} holds {len(files)} paths")
        files = files[:n_paths]
    return files


def columns(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Columns ``names`` of one path as an array (n_increments, len(names)); ``Cij_m`` is computed."""
    out = np.empty((len(df), len(names)))
    for k, name in enumerate(names):
        m = _SYM.match(name)
        if m:
            i, j = m.groups()
            out[:, k] = 0.5 * (df[f"C{i}{j}"].to_numpy() + df[f"C{j}{i}"].to_numpy())
        elif name in df.columns:
            out[:, k] = df[name].to_numpy()
        else:
            raise KeyError(f"Column {name!r} not in the database (columns: {list(df.columns)})")
    return out


def load_sequences(database_dir: str | Path, inputs: list[str], outputs: list[str],
                   n_paths: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read the database: X (n_paths, timesteps, n_inputs), Y (n_paths, timesteps, n_outputs)."""
    files = path_files(database_dir, n_paths)
    xs, ys = [], []
    for f in files:
        df = pd.read_csv(f)
        xs.append(columns(df, inputs))
        ys.append(columns(df, outputs))
    lengths = {x.shape[0] for x in xs}
    if len(lengths) != 1:
        raise ValueError(f"All paths must have the same number of increments, found {sorted(lengths)}")
    return np.stack(xs), np.stack(ys), [f.name for f in files]


class MinMaxScaling:
    """MinMax scaling to [0, 1] per feature, same result as ``sklearn.preprocessing.MinMaxScaler``.

    A constant feature (max = min) gets a unit scale, as in sklearn. :attr:`export_max` is
    ``data_max`` with ``min + 1`` for those features, so that the fe_lstm formula
    ``(x - min) / (max - min)`` reproduces this scaling (the plain data_max would divide by 0).
    """

    def __init__(self):
        self.data_min = self.data_max = None

    def fit(self, a: np.ndarray) -> "MinMaxScaling":
        flat = a.reshape(-1, a.shape[-1])
        self.data_min, self.data_max = flat.min(0), flat.max(0)
        return self

    @property
    def constant(self) -> np.ndarray:
        return self.data_max == self.data_min

    @property
    def export_max(self) -> np.ndarray:
        return np.where(self.constant, self.data_min + 1.0, self.data_max)

    def transform(self, a: np.ndarray) -> np.ndarray:
        return (a - self.data_min) / (self.export_max - self.data_min)

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        return a * (self.export_max - self.data_min) + self.data_min


def split_paths(n_paths: int, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the training and test paths: the test paths are the last ones (no shuffle)."""
    n_test = int(np.ceil(test_fraction * n_paths)) if test_fraction > 0 else 0
    if n_test >= n_paths:
        raise ValueError(f"test_fraction {test_fraction} leaves no training path out of {n_paths}")
    idx = np.arange(n_paths)
    return idx[: n_paths - n_test], idx[n_paths - n_test:]


def time_weights(y: np.ndarray, power: float = 2.0, floor: float = 0.05) -> np.ndarray:
    """Loss weight of each increment, shape (timesteps,), mean 1.

    ``y`` is the unscaled output (n_paths, timesteps, n_outputs) of the training paths. With
    ``r_t`` the RMS output at increment ``t`` (over paths and outputs) and ``r_max`` its maximum::

        w_t ∝ (r_max / max(r_t, floor * r_max)) ** power

    ``power = 2`` gives every increment the same weight in relative terms (the squared error is
    divided by the squared output level); ``power = 0`` is the plain unweighted loss. ``floor``
    caps the weight of increments whose outputs are close to zero.

    Deliberate change from the thesis (2026-11-28): with plain MSE on MinMax-scaled outputs, the
    first increments of a path (small stresses) barely count, and the stress network trained on
    the gyroid 3D database came out ~7x too soft there.
    """
    r = np.sqrt(np.mean(np.asarray(y, float) ** 2, axis=(0, 2)))
    r_max = float(r.max())
    if r_max == 0:
        return np.ones_like(r)
    w = (r_max / np.maximum(r, floor * r_max)) ** float(power)
    return w / w.mean()
