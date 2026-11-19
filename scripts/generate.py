"""Generate the training database (strain-stress histories).

Usage: python scripts/generate.py --config configs/default.yaml
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    raise NotImplementedError("Port from legacy/database_generation/ into src/fe_lstm/data/")


if __name__ == "__main__":
    main()
