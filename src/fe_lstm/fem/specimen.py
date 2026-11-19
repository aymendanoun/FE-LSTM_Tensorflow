"""3D specimen with holes under tension, with the FE-LSTM law or a reference law (fedoo 1.x).

Port of ``legacy/fedoo_coupling/RNN_3D_specimen_with_holes_tensile.py``.

Legacy -> fedoo 1.x mapping:

=====================================================  ===================================================
legacy (2022)                                          fedoo 1.x
=====================================================  ===================================================
``Util.ProblemDimension("3D")``                        ``fd.ModelingSpace("3D")``
``Mesh.ImportFromFile(f); Mesh.GetAll()["Domain2"]``   ``fd.mesh.import_file(f)["tet4"]``
``mesh.AddSetOfNodes(np.where(crd[:,0]==xmin))``       ``mesh.find_nodes("X", xmin)``
``ConstitutiveLaw.LSTM(ID=...)``                       ``fe_lstm.fem.LSTMConstitutiveLaw(weights_dir)``
``ConstitutiveLaw.Simcoon("EPICP", props, 8)``         ``fd.constitutivelaw.Simcoon("EPICP", props)``
``WeakForm.InternalForce(law, nlgeom=False)``          ``fd.weakform.StressEquilibrium(law, nlgeom=False)``
``Assembly.Create(law, meshID, "tet4")``               ``fd.Assembly.create(weakform, mesh, "tet4")``
``Problem.NonLinearStatic("Assembling")``              ``fd.problem.NonLinear(assembly)``
``Problem.SetNewtonRaphsonErrorCriterion("Displ..")``  ``use_legacy_displacement_criterion(pb)``  
``Problem.BoundaryCondition("Dirichlet","DispX",..)``  ``pb.bc.add("Dirichlet", nodes, "DispX", value)``
``Problem.AddOutput(..., file_format="vtk")``          ``pb.add_output(...)`` + ``results.to_vtk``
``Problem.NLSolve(dt, tmax, update_dt, ToleranceNR)``  ``pb.nlsolve(dt, tmax, update_dt, tol_nr)``
=====================================================  ===================================================
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import fedoo as fd
import numpy as np

from fe_lstm.fem.legacy_nr import use_legacy_displacement_criterion
from fe_lstm.fem.lstm_law import LSTMConstitutiveLaw


def build_material(cfg: dict, root: Path):
    mat = cfg["material"]
    kind = mat["type"]
    if kind == "lstm":
        return LSTMConstitutiveLaw(
            root / mat["weights_dir"],
            state_update=mat["state_update"],
            tangent=mat.get("tangent", "network"),
        )
    if kind == "elastic":
        return fd.constitutivelaw.ElasticIsotrop(mat["E"], mat["nu"])
    if kind == "eliso":
        return fd.constitutivelaw.Simcoon("ELISO", np.array([mat["E"], mat["nu"], mat["alpha"]]))
    if kind == "epicp":
        props = np.array([mat["E"], mat["nu"], mat["alpha"], mat["Re"], mat["k"], mat["m"]])
        return fd.constitutivelaw.Simcoon("EPICP", props)
    raise ValueError(f"Unknown material type {kind!r}")


def load_mesh(cfg: dict, root: Path) -> fd.Mesh:
    mesh = fd.mesh.import_file(str(root / cfg["mesh"]["file"]))
    if hasattr(mesh, "elm_types"):  # gmsh file with several element types -> MultiMesh
        mesh = mesh[cfg["mesh"]["elm_type"]]
    return mesh


def run(cfg: dict, root: str | Path = ".", material=None) -> dict:
    """Run the simulation described by ``cfg`` (see configs/specimen_with_holes_tensile.yaml).

    Returns a dict with the problem, assembly, results object and wall-clock time.
    """
    root = Path(root)
    start = perf_counter()

    fd.ModelingSpace("3D")
    mesh = load_mesh(cfg, root)
    if material is None:
        material = build_material(cfg, root)

    weakform = fd.weakform.StressEquilibrium(material, nlgeom=False)
    if not getattr(material, "symmetric_tangent", True):
        # fedoo assumes a symmetric stiffness in small strain and assembles one triangle only:
        # an unsymmetric tangent (finite differences) must be assembled in full.
        weakform.assembly_options["assume_sym"] = False
    assembly = fd.Assembly.create(weakform, mesh, cfg["mesh"]["elm_type"])
    pb = fd.problem.NonLinear(assembly)

    sol = cfg["solver"]
    criterion = sol["nr_criterion"]
    if criterion == "LegacyDisplacement":  # 2022 fedoo definition, see legacy_nr.py
        pb.set_nr_criterion("Displacement", tol=sol["tol_nr"], max_subiter=sol["max_subiter"])
        use_legacy_displacement_criterion(pb)
    else:  # fedoo 1.x criteria: "Displacement", "Force" or "Work"
        pb.set_nr_criterion(criterion, tol=sol["tol_nr"], max_subiter=sol["max_subiter"])

    crd = mesh.nodes
    left = mesh.find_nodes("X", crd[:, 0].min())
    right = mesh.find_nodes("X", crd[:, 0].max())
    pb.bc.add("Dirichlet", left, "Disp", 0)  # DispX = DispY = DispZ = 0
    pb.bc.add("Dirichlet", right, "DispY", 0)
    pb.bc.add("Dirichlet", right, "DispX", cfg["loading"]["ux_right"])

    out = cfg["output"]
    out_dir = root / out["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pb.add_output(str(out_dir / out["name"]), assembly, out["fields"])

    pb.nlsolve(
        dt=sol["dt"],
        tmax=sol["tmax"],
        update_dt=sol["update_dt"],
        tol_nr=sol["tol_nr"],
        max_subiter=sol["max_subiter"],
        print_info=sol.get("print_info", 1),
    )
    elapsed = perf_counter() - start

    if out.get("vtk", False):
        results.to_vtk(str(out_dir / out["name"]), "Node")

    return {"problem": pb, "assembly": assembly, "results": results, "time": elapsed}
