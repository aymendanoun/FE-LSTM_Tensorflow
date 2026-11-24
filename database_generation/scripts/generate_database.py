"""Generate the FE-LSTM training database (strain, stress, tangent histories on the RVE).

Run in the felstm environment, from database_generation/, after scripts/build_rve.py:
    python scripts/generate_database.py                       # all paths of the config
    python scripts/generate_database.py --paths 0:20          # a subset (start:stop)
    python scripts/generate_database.py --workers 4
Completed paths are skipped, so an interrupted run can be restarted with the same command.
"""

import argparse
from pathlib import Path

import yaml

from database_generation.generate import generate

STAGE = Path(__file__).resolve().parents[1]  # the database_generation/ folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=STAGE / "configs/database.yaml")
    parser.add_argument("--paths", help="range of path indices start:stop (default: all)")
    parser.add_argument("--workers", type=int, help="number of parallel processes")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    indices = None
    if args.paths:
        start, stop = (int(v) for v in args.paths.split(":"))
        indices = range(start, stop)
    summary = generate(cfg, STAGE, indices, n_workers=args.workers)
    print(f"done {summary['done']}, failed {len(summary['failed'])}, skipped {summary['skipped']}, "
          f"{summary['elapsed_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
