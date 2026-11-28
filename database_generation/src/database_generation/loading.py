"""Random macroscopic strain paths.

Port of ``generate_loadings`` / ``create_path`` in ``legacy/database_generation/Create_DataBase_octet*.py``:
each path has ``n_segments`` random target strains, reached one after the other along straight
lines, each in ``increments_per_segment`` equal increments. Legacy: 4 targets x 25 increments
(``N_Steps = 4``, ``Dn_inc = 0.04``); current config: 2 targets x 50 increments. Each active
component of a target is drawn independently from a uniform distribution on
``[strain_min, strain_max]``; the other components stay at zero.

``sampling: mixed`` (new, 2026-11-25) draws targets on all six components in a normalized ball,
mixing proportional, reversal and random paths, with scrambled Sobol numbers for an even
coverage of the space with few paths: see :func:`mixed_targets`.

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


# ------------------------------------------------------------------ "mixed" sampling (2026-11-25)
PATH_TYPES = ("proportional", "reversal", "random")
DEFAULT_SCALE = (0.05, 0.05, 0.05, 0.025, 0.025, 0.025)  # thesis 3D ranges (normal, shear)
_SOBOL_CACHE: dict = {}


def _sobol_points(seed: int, n_points: int, dims: int) -> np.ndarray:
    """Scrambled Sobol points in [0, 1)^dims (first ``n_points`` of 2^m), cached per process."""
    key = (seed, n_points, dims)
    if key not in _SOBOL_CACHE:
        from scipy.stats import qmc

        m = max(1, int(np.ceil(np.log2(n_points))))
        _SOBOL_CACHE[key] = qmc.Sobol(dims, scramble=True, seed=seed).random_base2(m)[:n_points]
    return _SOBOL_CACHE[key]



def _direction(u6: np.ndarray) -> np.ndarray:
    """Uniform direction on the 6D unit sphere from 6 uniforms (Gaussian, normalized)."""
    from scipy.stats import norm

    g = norm.ppf(np.clip(u6, 1e-12, 1 - 1e-12))
    n = np.linalg.norm(g)
    return g / n if n > 0 else np.eye(6)[0]


def mixed_targets(uniforms: np.ndarray, n_segments: int, strain_scale, max_norm: float,
                  fractions: dict, reversal_factor) -> tuple[np.ndarray, str]:
    """Targets of one path from ``8 * n_segments + 1`` numbers in [0, 1).

    Strains are drawn in the normalized space ``u = strain / strain_scale``: a direction uniform
    on the 6D sphere and a norm uniform in ``[0, max_norm]`` (all directions and all amplitudes,
    from elastic to strongly plastic, equally represented). Path types:

    * ``proportional``: monotonic straight path to ``R d`` (targets at k/n of it): the loading of
      a Gauss point in a monotonic FE test;
    * ``reversal``: load to ``R d``, then to ``c R d`` with ``c`` in ``reversal_factor``
      (partial / full unloading, or reversed loading): elastic unloading, Bauschinger effect;
    * ``random``: independent targets (non-proportional loading, change of direction).
    """
    scale = np.asarray(strain_scale, dtype=float)
    names = [t for t in PATH_TYPES if fractions.get(t, 0) > 0]
    w = np.array([fractions[t] for t in names], dtype=float)
    if not names or np.any(w < 0):
        raise ValueError(f"fractions needs positive weights for some of {PATH_TYPES}, got {fractions}")
    unknown = set(fractions) - set(PATH_TYPES)
    if unknown:
        raise ValueError(f"Unknown path types {sorted(unknown)}; use {PATH_TYPES}")
    kind = names[int(np.searchsorted(np.cumsum(w) / w.sum(), uniforms[0], side="right"))]
    seg = uniforms[1:].reshape(n_segments, 8)  # per segment: 6 direction, 1 norm, 1 factor
    first = _direction(seg[0, :6]) * max_norm * seg[0, 6]
    if kind == "proportional":
        u = np.array([first * (k + 1) / n_segments for k in range(n_segments)])
    elif kind == "reversal":
        lo, hi = reversal_factor
        u = np.array([first] + [first * (lo + (hi - lo) * s[7]) for s in seg[1:]])
    else:
        u = np.array([_direction(s[:6]) * max_norm * s[6] for s in seg])
    return u * scale, kind


def mixed_path_targets(seed: int, path_index: int, n_paths: int, n_segments: int, opts: dict):
    """Targets and type of path ``path_index`` with the "mixed" sampling options ``opts``."""
    dims = 8 * n_segments + 1
    if opts.get("sampler", "sobol") == "sobol":
        if not 0 <= path_index < n_paths:
            raise IndexError(f"path {path_index} outside 0..{n_paths - 1}")
        uniforms = _sobol_points(seed, n_paths, dims)[path_index]
    else:
        uniforms = path_rng(seed, path_index).random(dims)
    return mixed_targets(uniforms, n_segments, opts.get("strain_scale", DEFAULT_SCALE),
                         float(opts.get("max_norm", 1.0)), opts["fractions"],
                         opts.get("reversal_factor", (-1.0, 0.5)))
