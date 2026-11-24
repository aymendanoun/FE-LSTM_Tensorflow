"""Strain-driven periodic RVE simulation with fedoo 1.x and a simcoon umat.

Port of ``SolverUnitCell`` in ``legacy/fedoo_coupling/fedoo/libHomogen/Homog_path.py`` (old fedoo
API) to fedoo 1.x:

* periodic boundary conditions (``fd.constraint.PeriodicBC("small_strain")``): the macroscopic
  strain components are global unknowns ``E_xx ... E_yz`` (engineering shear), imposed as
  Dirichlet conditions; the node nearest to the RVE centre is fixed (rigid body motion);
* each segment of the path is solved with ``nlsolve`` (Newton-Raphson tolerance 0.05, automatic
  time step), and the results are recorded at the end of each of its equal increments;
* recorded per increment: mean strain, mean stress (volume integral of the stress over the solid
  divided by the RVE volume, as in the legacy code) and the homogenized 6x6 tangent
  (``fd.homogen.get_tangent_stiffness``). The legacy database also stored the mechanical work
  ``Wm``; it is not recorded here (not used for training).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STRAIN_DOFS = ("E_xx", "E_yy", "E_zz", "E_xy", "E_xz", "E_yz")


@dataclass
class PathResult:
    strain: np.ndarray  # (n_increments, 6)
    stress: np.ndarray  # (n_increments, 6)
    tangent: np.ndarray  # (n_increments, 6, 6)
    time: np.ndarray = field(default=None)  # (n_increments,) cumulated pseudo-time


def load_tet_mesh(mesh_file: str | Path):
    """Read the tetrahedral part of a gmsh mesh (as written by geometry.mesh_octet_rve)."""
    import fedoo as fd

    mesh = fd.mesh.import_file(str(mesh_file))
    if not hasattr(mesh, "elements"):  # MultiMesh: keep the volume elements
        mesh = mesh["tet4"]
    return mesh


class RVESimulation:
    """Run strain paths on one RVE mesh with one simcoon material.

    Parameters
    ----------
    mesh : fedoo Mesh, or path to a mesh file
    umat : simcoon umat name (``"EPICP"`` in the thesis)
    props : material properties passed to the umat
    tol_nr : Newton-Raphson tolerance (legacy ``ToleranceNR = 0.05``)
    """

    def __init__(self, mesh, umat: str, props, tol_nr: float = 0.05, dt_min: float = 1e-6):
        import fedoo as fd

        self.fd = fd
        fd.ModelingSpace("3D")
        self.mesh = load_tet_mesh(mesh) if isinstance(mesh, (str, Path)) else mesh
        self.volume = self.mesh.bounding_box.volume
        self.umat = umat
        self.props = np.asarray(props, dtype=float)
        self.tol_nr = tol_nr
        self.dt_min = dt_min
        self.center = [
            int(np.linalg.norm(self.mesh.nodes - self.mesh.bounding_box.center, axis=1).argmin())
        ]

    def _new_problem(self):
        fd = self.fd
        material = fd.constitutivelaw.Simcoon(self.umat, self.props, name="rve_material")
        wf = fd.weakform.StressEquilibrium(material, nlgeom=False, name="rve_weakform")
        assembly = fd.Assembly.create(wf, self.mesh, name="rve_assembly")
        pb = fd.problem.NonLinear(assembly, name="rve_problem")
        pb.bc.add(fd.constraint.PeriodicBC("small_strain"))
        pb.bc.add("Dirichlet", self.center, "Disp", 0, name="center")
        return assembly, pb

    def _mean(self, assembly, field_name: str, n: int) -> np.ndarray:
        values = assembly.sv[field_name]
        return np.array([assembly.integrate_field(values[i]) for i in range(n)]) / self.volume

    def run_path(self, targets: np.ndarray, increments_per_segment: int) -> PathResult:
        """Apply the segments ``targets`` (n_segments, 6) one after the other from zero strain."""
        assembly, pb = self._new_problem()
        rows = []

        def record(pb):
            rows.append((
                pb.time + offset[0],
                np.array([pb.get_dof_solution(c)[0] for c in STRAIN_DOFS]),
                self._mean(assembly, "Stress", 6),
                self.fd.homogen.get_tangent_stiffness(pb),
            ))

        dt = 1.0 / increments_per_segment
        previous = np.zeros(6)
        offset = [0.0]
        for target in np.asarray(targets, dtype=float):
            pb.bc.remove("Strain")
            for dof, value, start in zip(STRAIN_DOFS, target, previous):
                pb.bc.add("Dirichlet", dof, value, start_value=start, name="Strain")
            pb.nlsolve(dt=dt, tmax=1.0, update_dt=True, dt_min=self.dt_min * dt, tol_nr=self.tol_nr,
                       print_info=0, interval_output=dt, callback=record)
            previous = target
            offset[0] += 1.0

        n_expected = len(targets) * increments_per_segment
        if len(rows) != n_expected:
            raise RuntimeError(f"Recorded {len(rows)} increments instead of {n_expected}")
        t, strain, stress, tangent = (np.array(v) for v in zip(*rows))
        return PathResult(strain=strain, stress=stress, tangent=tangent, time=t)
