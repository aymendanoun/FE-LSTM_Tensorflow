"""Check that an RVE mesh can be used by the database generation (runs in ``felstm``).

Requirements of :class:`database_generation.rve.RVESimulation`:

* linear tetrahedra (``tet4``), one material;
* the periodic cell is the mesh bounding box: the solid touches the six faces;
* periodic node layout: every node on a face has a partner at the same position on the
  opposite face (needed by ``fd.constraint.PeriodicBC``).

:func:`check_mesh` also computes the solid fraction and the homogenized elastic stiffness, and can
run a short elastoplastic path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _face_pairs_match(nodes: np.ndarray, tol: float) -> bool:
    lo, hi = nodes.min(0), nodes.max(0)
    for ax in range(3):
        a = nodes[np.abs(nodes[:, ax] - lo[ax]) < tol]
        b = nodes[np.abs(nodes[:, ax] - hi[ax]) < tol]
        if len(a) == 0 or len(a) != len(b):
            return False
        key = lambda p: np.array(sorted(map(tuple, np.round(np.delete(p, ax, axis=1) / tol))))
        if not np.array_equal(key(a), key(b)):
            return False
    return True


def check_mesh(mesh_file: str | Path, E: float = 113800.0, nu: float = 0.342,
               epicp_props=None, tol: float = 1e-6) -> dict:
    """Return a dict of mesh properties; raises ``ValueError`` if the mesh cannot be used.

    ``epicp_props``: if given, also run one short EPICP path (2 segments x 5 increments) and
    report its time and maximum stress.
    """
    import time

    import fedoo as fd

    raw = fd.mesh.import_file(str(mesh_file))
    elm_types = list(raw.elm_types) if hasattr(raw, "elm_types") else [raw.elm_type]
    if "tet4" not in elm_types:
        raise ValueError(f"{mesh_file}: no linear tetrahedra (tet4), found {elm_types}")
    from database_generation.rve import RVESimulation, load_tet_mesh

    mesh = load_tet_mesh(mesh_file)
    x, e = mesh.nodes, mesh.elements
    a, b, c, d = (x[e[:, i]] for i in range(4))
    vol = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6
    dims = x.max(0) - x.min(0)
    info = {
        "nodes": mesh.n_nodes, "tet4": mesh.n_elements, "cell": tuple(np.round(dims, 6)),
        "solid_fraction": vol.sum() / np.prod(dims), "periodic": _face_pairs_match(x, tol * dims.max()),
    }
    if not info["periodic"]:
        raise ValueError(f"{mesh_file}: nodes of opposite faces do not match (not periodic)")

    fd.ModelingSpace("3D")
    asm = fd.Assembly.create(
        fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(E, nu)), mesh,
        name="check_assembly",
    )
    C = fd.homogen.get_homogenized_stiffness(asm)
    eig = np.linalg.eigvalsh(0.5 * (C + C.T))
    info["C_hom"] = C
    if eig.min() <= 0:
        raise ValueError(f"{mesh_file}: homogenized stiffness not positive definite ({eig.min():.3g})")

    if epicp_props is not None:
        sim = RVESimulation(mesh, "EPICP", epicp_props)
        targets = np.array([[0.03, 0.01, 0.0, 0.015, 0.0, 0.0], [0.0, 0.03, 0.0, 0.0, 0.0, 0.0]])
        t0 = time.time()
        res = sim.run_path(targets, 5)
        info["epicp_seconds_10_increments"] = time.time() - t0
        info["epicp_max_stress"] = float(np.abs(res.stress).max())
    return info
