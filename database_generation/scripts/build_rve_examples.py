"""Build the example RVE meshes listed in rve_examples/examples.yaml with microgen.

Run in the felstm-geometry environment, from database_generation/:
    conda activate felstm-geometry
    python scripts/build_rve_examples.py                 # all examples
    python scripts/build_rve_examples.py gyroid_sheet honeycomb
"""

import argparse
import time
from pathlib import Path

import yaml

from database_generation.geometry import build_rve

STAGE = Path(__file__).resolve().parents[1]  # the database_generation/ folder
EXAMPLES_DIR = STAGE / "rve_examples"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="examples to build (default: all)")
    parser.add_argument("--list", default=EXAMPLES_DIR / "examples.yaml")
    args = parser.parse_args()
    examples = yaml.safe_load(Path(args.list).read_text())
    names = args.names or list(examples)
    unknown = set(names) - set(examples)
    if unknown:
        raise SystemExit(f"Unknown examples {sorted(unknown)}; available: {list(examples)}")
    failed = []
    for name in names:
        ex = examples[name]
        t0 = time.time()
        try:
            out = build_rve(EXAMPLES_DIR / f"{name}.msh", ex["kind"], ex["mesh_size"],
                            remesh_options=ex.get("remesh"), **ex["params"])
            print(f"{name:22s} -> {out.name}  ({time.time() - t0:.1f} s)", flush=True)
        except Exception as exc:  # report and go on with the other examples
            failed.append(name)
            print(f"{name:22s} FAILED: {exc}", flush=True)
    if failed:
        raise SystemExit(f"Failed: {failed}")


if __name__ == "__main__":
    main()
