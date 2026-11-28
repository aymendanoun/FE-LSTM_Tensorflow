"""Train the FE-LSTM surrogates (stress and tangent networks) and export their weights for fe_lstm.

Every setting is in the config (configs/training.yaml): database, number of LSTM layers and
units, Dense layers and activations, loss, optimizer, epochs, batch size, validation split...
The options below override the config for quick runs.

Usage (from the training/ folder, felstm environment):
    python scripts/train.py                                   # configs/training.yaml (stress network)
    python scripts/train.py --networks stress tangent         # also the thesis tangent network
    python scripts/train.py --config configs/training_gyroid.yaml
    python scripts/train.py --networks stress --epochs 50     # quick test of one network

Output: <output.dir>/weights_scaler_stress/ (and weights_scaler_Lt/ if trained), to use in fe_lstm
with material.weights_dir: ../training/<output.dir>.
"""

import argparse
from pathlib import Path

import yaml

from lstm_training.train import train_network

STAGE = Path(__file__).resolve().parents[1]  # the training/ folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=STAGE / "configs/training.yaml")
    parser.add_argument("--networks", nargs="+", help="networks to train (default: those enabled in the config)")
    parser.add_argument("--epochs", type=int, help="override training.epochs of every network")
    parser.add_argument("--n-paths", type=int, help="override data.n_paths")
    parser.add_argument("--output", help="override output.dir (relative to training/)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
        for net in cfg["networks"].values():
            net.get("training", {}).pop("epochs", None)
    if args.n_paths is not None:
        cfg["data"]["n_paths"] = args.n_paths
    if args.output:
        cfg["output"]["dir"] = args.output
    names = args.networks or [n for n, net in cfg["networks"].items() if net.get("enabled", True)]

    for name in names:
        train_network(cfg, name, STAGE)
    out = Path(cfg["output"]["dir"])
    print(f"Done. In fe_lstm, set material.weights_dir: {out if out.is_absolute() else '../training' / out}")


if __name__ == "__main__":
    main()
