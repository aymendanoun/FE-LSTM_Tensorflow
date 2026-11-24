"""Build your own RVE with microgen from the ``geometry.build`` block of the config, and write its
periodic tetrahedral mesh to ``geometry.build.output``.

Run in the felstm-geometry environment (see environment-geometry.yml), from database_generation/:
    conda activate felstm-geometry
    python scripts/build_rve.py --config configs/database.yaml
Then set ``geometry.mesh_file`` to that output, and check the mesh in felstm with
``python scripts/check_rve.py <mesh>``. Ready-made meshes: rve_examples/.
"""

import argparse
import time
from pathlib import Path

import yaml

from database_generation.geometry import build_rve

STAGE = Path(__file__).resolve().parents[1]  # the database_generation/ folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=STAGE / "configs/database.yaml")
    args = parser.parse_args()
    b = yaml.safe_load(Path(args.config).read_text())["geometry"]["build"]
    t0 = time.time()
    out = build_rve(STAGE / b["output"], b["kind"], b["mesh_size"], **b["params"])
    print(f"RVE mesh written to {out} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
