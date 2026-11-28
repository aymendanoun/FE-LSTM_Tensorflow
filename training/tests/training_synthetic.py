"""Synthetic database in the format of the database_generation stage (tests only).

Strain paths of 2 random segments; a history-dependent toy response (stress depends on the
running maximum of the strain), and a symmetric 6x6 "tangent" (Cij = Cji up to a tiny
asymmetry, to test the Cij_m columns). Only Exx, Eyy and Exy are loaded, as in the default
database: the other strain columns are constant (zero).
"""

from pathlib import Path

import numpy as np
import pandas as pd

STRAIN = ["Exx", "Eyy", "Ezz", "Exy", "Exz", "Eyz"]
STRESS = ["Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"]
TANGENT = [f"C{i + 1}{j + 1}" for i in range(6) for j in range(6)]


def synthetic_path(rng, n_segments=2, increments=10):
    targets = np.zeros((n_segments, 6))
    targets[:, [0, 1, 3]] = rng.uniform(0, [0.05, 0.05, 0.025], (n_segments, 3))
    start, strain = np.zeros(6), []
    for t in targets:
        strain += [start + (t - start) * (k + 1) / increments for k in range(increments)]
        start = t
    e = np.array(strain)
    emax = np.maximum.accumulate(np.abs(e), axis=0)
    stress = 3000.0 * np.tanh(40.0 * e) + 500.0 * emax
    stress[:, 2] = 0.3 * (stress[:, 0] + stress[:, 1])  # Szz from the in-plane stresses
    tangent = np.empty((len(e), 6, 6))
    for n in range(len(e)):
        a = 100.0 * np.outer(np.cos(e[n] * 50), np.cos(e[n] * 50)) + np.diag(1e4 / (1 + 50 * emax[n]))
        tangent[n] = a + 1e-3 * np.triu(np.ones((6, 6)), 1)  # tiny asymmetry
    return e, stress, tangent


def write_database(folder: Path, n_paths=20, increments=10, seed=0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_paths):
        e, s, c = synthetic_path(rng, increments=increments)
        n = len(e)
        df = pd.DataFrame(np.hstack([np.arange(1, n + 1)[:, None], np.linspace(0, 2, n)[:, None], e, s,
                                     c.reshape(n, 36)]), columns=["Increment", "Time"] + STRAIN + STRESS + TANGENT)
        df.to_csv(folder / f"path_{i:05d}.csv", index=False)
    return folder
