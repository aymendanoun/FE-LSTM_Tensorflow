"""Run the FE-LSTM tensile test on the 3D specimen with holes (fedoo 1.x).

Port of legacy/fedoo_coupling/RNN_3D_specimen_with_holes_tensile.py.

Usage (from the repository root):
    python scripts/run_fe.py
    python scripts/run_fe.py --material epicp                # simcoon reference instead of LSTM
    python scripts/run_fe.py --weights path/to/weights_scaler_3D --state-update consistent
"""

import argparse
from pathlib import Path

import yaml

from fe_lstm.fem.specimen import run

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=ROOT / "configs/specimen_with_holes_tensile.yaml")
    parser.add_argument("--material", choices=["lstm", "elastic", "eliso", "epicp"])
    parser.add_argument("--weights", help="folder with weights_scaler_stress/ and weights_scaler_Lt/")
    parser.add_argument("--state-update", choices=["legacy", "consistent"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.material:
        cfg["material"]["type"] = args.material
    if args.weights:
        cfg["material"]["weights_dir"] = str(Path(args.weights).resolve())
    if args.state_update:
        cfg["material"]["state_update"] = args.state_update
    if cfg["material"]["type"] != "lstm":
        cfg["output"]["name"] = f"{cfg['material']['type']}_specimen_with_holes_tensile"

    out = run(cfg, ROOT)
    print(f"Done in {out['time']:.1f} s - results in {cfg['output']['dir']}/")


if __name__ == "__main__":
    main()
