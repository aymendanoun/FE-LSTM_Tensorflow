"""Generate the database: run the RVE along ``n_paths`` random strain paths, in parallel.

* Path ``i`` uses the random stream ``(seed, i)``, so results do not depend on the number of
  workers or on the order in which paths finish.
* Completed paths are skipped when the generation is restarted (one CSV per path).
* A path whose simulation fails is recorded in ``failed_paths.csv`` and skipped (the legacy
  script printed ``Convergence Failed`` and moved on).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from database_generation import io
from database_generation.loading import path_rng, random_targets

_WORKER = {}
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def material_props(cfg_material: dict) -> np.ndarray:
    """Ordered umat properties from the config (``props`` keeps the umat's order)."""
    return np.array(list(cfg_material["props"].values()), dtype=float)


def path_targets(cfg: dict, path_index: int) -> np.ndarray:
    lo = cfg["loading"]
    return random_targets(
        path_rng(cfg["seed"], path_index), lo["n_segments"], lo["active_components"],
        lo["strain_min"], lo["strain_max"],
    )


def _init_worker(cfg: dict, mesh_file: str):
    from database_generation.rve import RVESimulation

    _WORKER["cfg"] = cfg
    _WORKER["sim"] = RVESimulation(
        mesh_file, cfg["material"]["umat"], material_props(cfg["material"]),
        tol_nr=cfg["solver"]["tol_nr"],
    )


def _run_one(args):
    path_index, out_dir = args
    cfg, sim = _WORKER["cfg"], _WORKER["sim"]
    t0 = time.time()
    try:
        result = sim.run_path(path_targets(cfg, path_index), cfg["loading"]["increments_per_segment"])
        io.write_path(out_dir, path_index, result)
        return path_index, True, time.time() - t0, ""
    except Exception as exc:  # keep going, as the legacy script did
        return path_index, False, time.time() - t0, f"{type(exc).__name__}: {exc}".replace("\n", " ")


def generate(cfg: dict, stage_dir: str | Path, path_indices=None, n_workers: int | None = None,
             verbose: bool = True) -> dict:
    """Run the paths not yet in the database. Returns counts and the failed indices."""
    stage_dir = Path(stage_dir)
    mesh_file = stage_dir / cfg["geometry"]["mesh_file"]
    if not mesh_file.exists():
        raise FileNotFoundError(
            f"{mesh_file} not found: use a mesh of rve_examples/, or build one with "
            "scripts/build_rve.py (felstm-geometry environment)."
        )
    out_dir = stage_dir / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = out_dir / "config.json"
    if cfg_file.exists() and any(out_dir.glob("path_*.csv")):
        if json.loads(cfg_file.read_text()) != json.loads(json.dumps(cfg)):
            raise ValueError(
                f"{out_dir} holds paths generated with a different configuration "
                f"(see {cfg_file}). Use another output dir, or delete it to start again."
            )
    cfg_file.write_text(json.dumps(cfg, indent=2))

    if path_indices is None:
        path_indices = range(cfg["loading"]["n_paths"])
    todo = [i for i in path_indices if not io.path_file(out_dir, i).exists()]
    n_workers = n_workers or cfg.get("parallel", {}).get("n_workers") or os.cpu_count()
    n_workers = max(1, min(n_workers, len(todo) or 1))
    if verbose:
        print(f"{len(todo)} paths to run ({len(path_indices) - len(todo)} already done), "
              f"{n_workers} worker(s), output {out_dir}")

    failed, done, t0 = [], 0, time.time()
    if todo:
        # One BLAS thread per worker: with one process per core, multithreaded BLAS only adds
        # contention (8 workers on a 10-core Mac: 48 s -> 33 s per path). Spawned workers inherit
        # the environment at start-up; a value already set by the user is kept.
        for var in _THREAD_VARS:
            os.environ.setdefault(var, "1")
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=(cfg, str(mesh_file))) as pool:
            for k, (i, ok, dt, msg) in enumerate(
                pool.imap_unordered(_run_one, [(i, str(out_dir)) for i in todo]), 1
            ):
                if ok:
                    done += 1
                else:
                    failed.append((i, msg))
                    with open(out_dir / "failed_paths.csv", "a") as f:
                        f.write(f"{i},{msg}\n")
                if verbose and (k % max(1, len(todo) // 100) == 0 or k == len(todo)):
                    rate = (time.time() - t0) / k
                    print(f"  {k}/{len(todo)}  last path {dt:.1f}s  failed {len(failed)}  "
                          f"ETA {rate * (len(todo) - k) / 60:.1f} min", flush=True)
    return {"done": done, "failed": failed, "skipped": len(path_indices) - len(todo),
            "elapsed_s": time.time() - t0}


def run_single(cfg: dict, stage_dir: str | Path, path_index: int = 0):
    """Run one path in this process (debugging, tests) and return the PathResult."""
    from database_generation.rve import RVESimulation

    sim = RVESimulation(
        Path(stage_dir) / cfg["geometry"]["mesh_file"], cfg["material"]["umat"],
        material_props(cfg["material"]), tol_nr=cfg["solver"]["tol_nr"],
    )
    return sim.run_path(path_targets(cfg, path_index), cfg["loading"]["increments_per_segment"])
