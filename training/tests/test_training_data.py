"""Database reading, scaling and split (legacy procedure)."""

import numpy as np
import pandas as pd
import pytest

from lstm_training.data import MinMaxScaling, columns, load_sequences, split_paths
from lstm_training.train import network_config


def test_load_shapes_and_symmetric_tangent(cfg, synthetic_db):
    tangent = network_config(cfg, "tangent")
    x, y, files = load_sequences(synthetic_db, tangent["inputs"], tangent["outputs"])
    assert x.shape == (20, 20, 6) and y.shape == (20, 20, 21)
    assert files[0] == "path_00000.csv"
    df = pd.read_csv(synthetic_db / "path_00003.csv")
    k = tangent["outputs"].index("C24_m")
    np.testing.assert_allclose(y[3, :, k], 0.5 * (df["C24"] + df["C42"]))
    assert y[3, 0, 0] == df["C11"][0]


def test_unknown_column(synthetic_db):
    with pytest.raises(KeyError, match="Wm"):
        columns(pd.read_csv(synthetic_db / "path_00000.csv"), ["Wm"])


def test_scaling_equals_sklearn_with_constant_columns(cfg, synthetic_db):
    sklearn = pytest.importorskip("sklearn.preprocessing")
    x, _, _ = load_sequences(synthetic_db, network_config(cfg, "stress")["inputs"], ["Sxx"])
    s = MinMaxScaling().fit(x)
    ref = sklearn.MinMaxScaler().fit(x.reshape(-1, 6))
    assert list(s.constant) == [False, False, True, False, True, True]  # Ezz, Exz, Eyz = 0
    np.testing.assert_allclose(s.transform(x).reshape(-1, 6), ref.transform(x.reshape(-1, 6)))
    np.testing.assert_allclose(s.inverse_transform(s.transform(x)), x)
    # exported (min, max) reproduce the scaling with the fe_lstm formula (x - min) / (max - min)
    np.testing.assert_allclose((x - s.data_min) / (s.export_max - s.data_min), s.transform(x))


def test_split_is_the_legacy_one():
    sel = pytest.importorskip("sklearn.model_selection")
    train, test = split_paths(1000, 0.1)
    assert (len(train), len(test)) == (900, 100) and test[0] == 900
    rows = np.arange(1000 * 3).reshape(1000, 3)  # 3 rows (increments) per path
    _, rows_test = sel.train_test_split(rows.reshape(-1, 1), test_size=0.1, shuffle=False)
    np.testing.assert_array_equal(rows_test.ravel(), rows[test].ravel())
    with pytest.raises(ValueError):
        split_paths(3, 1.0)


def test_network_overrides(cfg):
    import copy

    c = copy.deepcopy(cfg)
    c["networks"]["tangent"]["model"] = {"lstm_units": [64, 64, 64]}
    c["networks"]["tangent"]["training"] = {"epochs": 7}
    t = network_config(c, "tangent")
    assert t["model"]["lstm_units"] == [64, 64, 64] and t["model"]["dense_units"] == [50]
    assert t["training"]["epochs"] == 7 and t["training"]["optimizer"] == "adam"
    assert network_config(c, "stress")["training"]["epochs"] == 1000


def test_time_weights():
    import numpy as np
    from lstm_training.data import time_weights

    level = np.array([1.0, 2.0, 4.0, 10.0])  # RMS output per increment
    y = level[None, :, None] * np.ones((3, 4, 2))
    w = time_weights(y, power=2.0, floor=0.0)
    np.testing.assert_allclose(w * level ** 2 / np.mean(w * level ** 2), 1.0)  # same relative weight
    assert w.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(time_weights(y, power=0.0), 1.0)
    w_floor = time_weights(y, power=2.0, floor=0.4)  # increments below 4 capped at the 4 weight
    assert w_floor[0] == w_floor[1] == w_floor[2] > w_floor[3]
