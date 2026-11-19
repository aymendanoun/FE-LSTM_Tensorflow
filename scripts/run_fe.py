"""Run a finite element simulation in Fedoo with the trained LSTM as constitutive law.

Usage: python scripts/run_fe.py --config configs/default.yaml
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    raise NotImplementedError("Port from legacy/fedoo_coupling/ into src/fe_lstm/fem/")


if __name__ == "__main__":
    main()
