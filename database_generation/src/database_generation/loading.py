"""Random macroscopic strain paths.

Port of ``generate_loadings`` / ``create_path`` in ``legacy/database_generation/Create_DataBase_octet*.py``:
each path has ``n_segments`` random target strains, reached one after the other along straight
lines, each in ``increments_per_segment`` equal increments. Legacy: 4 targets x 25 increments
(``N_Steps = 4``, ``Dn_inc = 0.04``); current config: 2 targets x 50 increments. Each active
component of a target is drawn independently from a uniform distribution on
``[strain_min, strain_max]``; the other components stay at zero.

Voigt order ``[Exx, Eyy, Ezz, Exy, Exz, Eyz]`` with engineering shear strains (as fedoo's
``PeriodicBC`` and the fe_lstm stage).
"""

from __future__ import annotations

import numpy as np

COMPONENTS = ("Exx", "Eyy", "Ezz", "Exy", "Exz", "Eyz")


def path_rng(seed: int, path_index: int) -> np.random.Generator:
    """Independent, reproducible random stream for one path (same result in any process)."""
    return np.random.default_rng([seed, path_index])


def random_targets(
    rng: np.random.Generator,
    n_segments: int,
    active: list[str],
    strain_min: list[float],
    strain_max: list[float],
) -> np.ndarray:
    """Target strains of one path, shape (n_segments, 6)."""
    lo = np.asarray(strain_min, dtype=float)
    hi = np.asarray(strain_max, dtype=float)
    if lo.shape != (6,) or hi.shape != (6,) or np.any(hi < lo):
        raise ValueError("strain_min and strain_max need 6 values with strain_min <= strain_max")
    unknown = set(active) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown strain components {sorted(unknown)}; use {COMPONENTS}")
    targets = np.zeros((n_segments, 6))
    for k, name in enumerate(COMPONENTS):
        if name in active:
            targets[:, k] = rng.uniform(lo[k], hi[k], n_segments)
    return targets


def discretize(targets: np.ndarray, increments_per_segment: int, start=None) -> np.ndarray:
    """Strain at the end of every increment, shape (n_segments * increments_per_segment, 6)."""
    start = np.zeros(6) if start is None else np.asarray(start, dtype=float)
    points = np.vstack([start, targets])
    frac = np.arange(1, increments_per_segment + 1)[:, None] / increments_per_segment
    return np.vstack([a + frac * (b - a) for a, b in zip(points[:-1], points[1:])])
