"""Check RVE meshes before generating a database: tet4, periodic, homogenized elastic stiffness,
and one short elastoplastic (EPICP) path. Works for the examples and for your own microgen meshes.

Run in the felstm environment, from database_generation/:
    python scripts/check_rve.py rve_examples/*.msh
    python scripts/check_rve.py my_rve.msh --no-epicp
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from database_generation.check import check_mesh
from database_generation.generate import material_props

STAGE = Path(__file__).resolve().parents[1]  # the database_generation/ folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("meshes", nargs="+")
    parser.add_argument("--config", default=STAGE / "configs/database.yaml", help="material from this config")
    parser.add_argument("--no-epicp", action="store_true", help="skip the elastoplastic path")
    args = parser.parse_args()
    mat = yaml.safe_load(Path(args.config).read_text())["material"]
    props = material_props(mat)
    print(f"{'mesh':24s} {'nodes':>6s} {'tet4':>6s} {'cell':>22s} {'solid':>6s} "
          f"{'C11':>8s} {'C12':>8s} {'C44':>8s} {'EPICP 10 inc':>13s}")
    bad = []
    for f in args.meshes:
        try:
            i = check_mesh(f, mat["props"]["E"], mat["props"]["nu"], None if args.no_epicp else props)
        except Exception as exc:
            bad.append(f)
            print(f"{Path(f).stem:24s} NOT USABLE: {exc}")
            continue
        C = i["C_hom"]
        cell = "x".join(f"{v:g}" for v in i["cell"])
        ep = f"{i['epicp_seconds_10_increments']:.1f}s" if "epicp_seconds_10_increments" in i else "-"
        print(f"{Path(f).stem:24s} {i['nodes']:6d} {i['tet4']:6d} {cell:>22s} {i['solid_fraction']:6.3f} "
              f"{C[0, 0]:8.1f} {C[0, 1]:8.1f} {C[3, 3]:8.1f} {ep:>13s}")
    if bad:
        raise SystemExit(f"Not usable: {bad}")


if __name__ == "__main__":
    main()
