"""Train the LSTM surrogate on the generated database.

Usage: python scripts/train.py --config configs/default.yaml
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    raise NotImplementedError("Port from legacy/training/ into src/fe_lstm/model/")


if __name__ == "__main__":
    main()
