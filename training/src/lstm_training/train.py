"""Train one network (stress or tangent) from the config, then export it for fe_lstm."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from lstm_training.data import MinMaxScaling, load_sequences, split_paths, time_weights
from lstm_training.export import export_network, predict_exported
from lstm_training.model import (best_weights_callback, build_model, check_architecture, get_loss,
                                 get_optimizer, nmse)


def network_config(cfg: dict, name: str) -> dict:
    """Settings of network ``name``: global ``model`` / ``training`` updated by the network's own."""
    if name not in cfg["networks"]:
        raise KeyError(f"No network {name!r} in the config (networks: {list(cfg['networks'])})")
    net = copy.deepcopy(cfg["networks"][name])
    net["model"] = {**cfg["model"], **net.get("model", {})}
    net["training"] = {**cfg["training"], **net.get("training", {})}
    return net


def _plot_history(history: dict, monitor: str, best_epoch, out_file: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, key in zip(axes, ("loss", "nmse")):
        ax.semilogy(epochs, history[key], "b--", label="training")
        if f"val_{key}" in history:
            ax.semilogy(epochs, history[f"val_{key}"], "r", label="validation")
        if best_epoch:
            ax.axvline(best_epoch, color="k", lw=0.8, ls=":", label=f"kept (epoch {best_epoch}, {monitor})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("normalized MSE [%]" if key == "nmse" else "loss")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def _test_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict:
    """Errors in physical units on the test paths, per output and overall."""
    err = (y_pred - y_true).reshape(-1, len(names))
    ref = y_true.reshape(-1, len(names))
    per = {}
    for k, n in enumerate(names):
        ss = float(np.sum(ref[:, k] ** 2))
        var = float(np.sum((ref[:, k] - ref[:, k].mean()) ** 2))
        per[n] = {
            "rmse": float(np.sqrt(np.mean(err[:, k] ** 2))),
            "nmse_percent": 100.0 * float(np.sum(err[:, k] ** 2)) / ss if ss > 0 else None,
            "r2": 1.0 - float(np.sum(err[:, k] ** 2)) / var if var > 0 else None,
        }
    by_step = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=(0, 2)))
    level = np.sqrt(np.mean(y_true ** 2, axis=(0, 2)))
    return {"nmse_percent": 100.0 * float(np.sum(err ** 2) / np.sum(ref ** 2)), "outputs": per,
            # RMSE and RMS output at each increment (all outputs and test paths)
            "rmse_by_increment": [float(v) for v in by_step],
            "rms_by_increment": [float(v) for v in level]}


def train_network(cfg: dict, name: str, stage_dir: str | Path, verbose: bool = True) -> dict:
    """Train network ``name`` of the config and export it to ``<output.dir>/<folder>``.

    Returns a summary dict (also written to ``metrics.json``).
    """
    import keras

    stage_dir = Path(stage_dir)
    net = network_config(cfg, name)
    m, tr, data = net["model"], net["training"], cfg["data"]
    check_architecture(m["lstm_units"], m["dense_units"], m["dense_activations"], m["output_activation"])
    keras.utils.set_random_seed(int(cfg.get("seed", 0)))

    database_dir = (stage_dir / data["database_dir"]).resolve()
    x, y, files = load_sequences(database_dir, net["inputs"], net["outputs"], data.get("n_paths"))
    x_scaling, y_scaling = MinMaxScaling().fit(x), MinMaxScaling().fit(y)  # all paths, as legacy
    train_idx, test_idx = split_paths(len(x), data["test_fraction"])
    xs, ys = x_scaling.transform(x), y_scaling.transform(y)
    n_val = int(np.ceil(data["validation_split"] * len(train_idx))) if data["validation_split"] else 0
    if verbose:
        print(f"[{name}] {len(x)} paths x {x.shape[1]} increments from {database_dir}: "
              f"{len(train_idx) - n_val} training, {n_val} validation, {len(test_idx)} test")
        const_in = [n for n, c in zip(net["inputs"], x_scaling.constant) if c]
        if const_in:
            print(f"[{name}] constant inputs in this database (left unscaled): {const_in}")

    model = build_model(len(net["inputs"]), len(net["outputs"]), m["lstm_units"], m["dense_units"],
                        m["dense_activations"], m["output_activation"])
    model.compile(optimizer=get_optimizer(tr["optimizer"]), loss=get_loss(tr["loss"]), metrics=[nmse])
    sample_weight = None
    if tr.get("time_weights"):
        if tr["loss"] == "nmse":
            raise ValueError("training.time_weights needs a Keras loss (mse, mae, ...), not nmse")
        tw = time_weights(y[train_idx], **tr["time_weights"])
        sample_weight = np.repeat(tw[None, :], len(train_idx), axis=0)
        if verbose:
            print(f"[{name}] increment weights {tr['time_weights']}: first {tw[:3].round(2)}, "
                  f"min {tw.min():.3g}, max {tw.max():.3g} (mean 1)")
    if verbose:
        model.summary()

    callbacks, keep = [], None
    monitor = tr.get("monitor", "val_nmse")
    if not n_val and monitor.startswith("val_"):
        monitor = monitor[4:]  # no validation set: monitor the training value
    if tr.get("keep_best", True):
        keep = best_weights_callback(monitor)
        callbacks.append(keep)
    if tr.get("early_stopping_patience"):
        callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=int(tr["early_stopping_patience"])))
    if tr.get("reduce_lr"):
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor=monitor, verbose=int(verbose), **tr["reduce_lr"]))

    t0 = time.time()
    hist = model.fit(
        xs[train_idx], ys[train_idx], sample_weight=sample_weight,
        epochs=int(tr["epochs"]), batch_size=int(tr["batch_size"]),
        validation_split=float(data["validation_split"]), shuffle=bool(tr["shuffle"]),
        callbacks=callbacks, verbose=tr.get("verbose", 2) if verbose else 0,
    ).history
    train_seconds = time.time() - t0

    out = stage_dir / cfg["output"]["dir"] / net["folder"]
    architecture = {
        "network": name, "inputs": list(net["inputs"]), "outputs": list(net["outputs"]),
        "lstm_units": [int(u) for u in m["lstm_units"]], "dense_units": [int(u) for u in m["dense_units"]],
        "dense_activations": list(m["dense_activations"]), "output_activation": m["output_activation"],
        "timesteps": int(x.shape[1]), "database_dir": str(database_dir), "n_paths": len(x),
    }
    export_network(model, out, net["scaling_name"], x_scaling, y_scaling, architecture)
    model.save(out / "model.keras")

    # Check the export with an independent NumPy forward pass, then evaluate on the test paths.
    y_keras = y_scaling.inverse_transform(model.predict(xs[test_idx if len(test_idx) else train_idx[:4]], verbose=0))
    y_numpy = predict_exported(out, x[test_idx if len(test_idx) else train_idx[:4]])
    export_err = float(np.max(np.abs(y_numpy - y_keras)) / max(np.max(np.abs(y_keras)), 1e-30))

    summary = {
        "network": name, "folder": str(out), "epochs_run": len(hist["loss"]),
        "best_epoch": keep.best_epoch if keep else None, "monitor": monitor,
        "best_value": keep.best if keep else None, "train_seconds": train_seconds,
        "export_check_max_rel_diff": export_err,
        "settings": {"model": m, "training": tr, "data": data, "seed": cfg.get("seed", 0)},
    }
    if len(test_idx):
        summary["test"] = _test_metrics(y[test_idx], y_keras, net["outputs"])
        summary["test"]["paths"] = [files[i] for i in test_idx]
        np.savez_compressed(out / "test_predictions.npz", x=x[test_idx], y_true=y[test_idx], y_pred=y_keras,
                            inputs=net["inputs"], outputs=net["outputs"])
    hist_arr = {k: np.asarray(v, float) for k, v in hist.items()}
    header = list(hist_arr)
    np.savetxt(out / "history.csv", np.column_stack([np.arange(1, len(hist["loss"]) + 1)] + list(hist_arr.values())),
               delimiter=",", header=",".join(["epoch"] + header), comments="", fmt="%.8g")
    _plot_history(hist_arr, monitor, summary["best_epoch"], out / "history.png")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    if verbose:
        print(f"[{name}] exported to {out} (best epoch {summary['best_epoch']}, {monitor} = "
              f"{summary['best_value']}; export check {export_err:.1e}; "
              f"test NMSE {summary.get('test', {}).get('nmse_percent', float('nan')):.4g} %)")
    if export_err > 1e-4:
        raise RuntimeError(f"Exported NumPy network differs from Keras by {export_err:.2e} (relative)")
    return summary
