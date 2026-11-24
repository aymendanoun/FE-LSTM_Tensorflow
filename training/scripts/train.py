"""Train an FE-LSTM surrogate (stress or tangent) and export its weights for the fe_lstm stage.

Usage (from the training/ folder):
    python scripts/train.py --config configs/training.yaml
"""

import argparse
from pathlib import Path

import yaml

STAGE = Path(__file__).resolve().parents[1]  # the training/ folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=STAGE / "configs/training.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())  # noqa: F841
    raise NotImplementedError("Training is not ported to Keras 3 yet.")


if __name__ == "__main__":
    main()
