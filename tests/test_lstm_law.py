"""LSTMConstitutiveLaw: tangent layout, recurrent-state handling and fedoo 1.x wiring."""

import fedoo as fd
import numpy as np
import pytest

from conftest import isotropic_stiffness
from fe_lstm.fem.lstm_law import TANGENT_INDEX, LSTMConstitutiveLaw, tangent_from_vector


# ----------------------------------------------------------------------------- tangent layout
def test_tangent_index_follows_training_labels():
    labels = ["C11", "C22", "C33", "C44", "C55", "C66", "C12_m", "C13_m", "C14_m", "C15_m", "C16_m",
              "C23_m", "C24_m", "C25_m", "C26_m", "C34_m", "C35_m", "C36_m", "C45_m", "C46_m", "C56_m"]
    expected = [(int(lab[1]) - 1, int(lab[2]) - 1) for lab in labels]  # from legacy/training
    assert TANGENT_INDEX == expected


def test_tangent_from_vector_is_symmetric():
    y = np.random.default_rng(0).normal(size=(5, 21))
    lt = tangent_from_vector(y)
    assert lt.shape == (6, 6, 5)
    np.testing.assert_array_equal(lt, lt.transpose(1, 0, 2))
    assert lt[0, 3, 2] == y[2, 8]  # C14 for point 2 (legacy: Lt[:,0,3] = Y[:,8])


# ------------------------------------------------------------------ recurrent state handling
class StubAssembly:
    """Minimal stand-in for a fedoo assembly (sv / sv_start dictionaries)."""

    _nlgeom = False

    def __init__(self, n_points):
        self.n_gauss_points = n_points
        self.sv = {"Strain": 0}
        self.sv_start = {}

    def commit(self):  # what fedoo's assembly.set_start does
        self.sv_start = dict(self.sv)

    def restore(self):  # what fedoo's assembly.to_start does
        self.sv = dict(self.sv_start)


def make(weights_dir, mode, n_points=4):
    law = LSTMConstitutiveLaw(weights_dir, state_update=mode)
    asm = StubAssembly(n_points)
    law.initialize(asm, None)
    asm.commit()
    return law, asm


def stress(asm):
    return np.asarray(asm.sv["Stress"]).copy()


def set_strain(asm, value):
    asm.sv["Strain"] = np.full((6, asm.n_gauss_points), value)


def test_consistent_mode_newton_iterations_restart_from_committed_state(weights_dir):
    law, asm = make(weights_dir, "consistent")
    set_strain(asm, 0.01)
    law.update(asm, None)
    s1 = stress(asm)
    law.update(asm, None)  # second Newton iteration, same strain -> same answer
    np.testing.assert_array_equal(stress(asm), s1)


def test_legacy_mode_advances_state_at_every_update(weights_dir):
    law, asm = make(weights_dir, "legacy")
    set_strain(asm, 0.01)
    law.update(asm, None)
    s1 = stress(asm)
    law.update(asm, None)
    assert not np.allclose(stress(asm), s1)  # thesis behaviour: one LSTM step per NR iteration


def test_modes_agree_with_one_update_per_increment(weights_dir):
    legacy, asm_l = make(weights_dir, "legacy")
    consistent, asm_c = make(weights_dir, "consistent")
    for value in (0.002, 0.005, 0.004, 0.008):
        for law, asm in ((legacy, asm_l), (consistent, asm_c)):
            set_strain(asm, value)
            law.update(asm, None)
            law.set_start(asm, None)
            asm.commit()
        np.testing.assert_allclose(stress(asm_l), stress(asm_c), rtol=1e-13)


def test_legacy_to_start_restores_cell_state_only(weights_dir):
    law, asm = make(weights_dir, "legacy")
    set_strain(asm, 0.01)
    law.update(asm, None)
    law.set_start(asm, None)
    saved = [(h.copy(), c.copy()) for h, c in law._state]
    law.update(asm, None)
    law.to_start(asm, None)
    for (h_saved, c_saved), (h, c) in zip(saved, law._state):
        np.testing.assert_array_equal(c, c_saved)  # restored
        assert not np.array_equal(h, h_saved)  # kept from the failed iteration, as in 2022


def test_invalid_mode(weights_dir):
    with pytest.raises(ValueError):
        LSTMConstitutiveLaw(weights_dir, state_update="other")


# ------------------------------------------------------------------------- fedoo 1.x wiring
class LinearElasticNet:
    """Fake 'network' returning linear elasticity, to test the fedoo wiring exactly."""

    n_inputs = 6
    n_lstm_layers = 2

    def __init__(self, kind):
        self.H = isotropic_stiffness()
        self.n_outputs = 6 if kind == "stress" else 21
        self.kind = kind

    def zero_state(self, n):
        return [(np.zeros((n, 1)), np.zeros((n, 1)))] * 2

    def step(self, x, state):
        if self.kind == "stress":
            return x @ self.H.T, state
        y = np.array([self.H[i, j] for i, j in TANGENT_INDEX])
        return np.tile(y, (x.shape[0], 1)), state


def solve_cube(material):
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=4, ny=4, nz=4, elm_type="hex8")
    wf = fd.weakform.StressEquilibrium(material, nlgeom=False)
    asm = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(asm)
    pb.set_nr_criterion("Displacement", tol=1e-8, max_subiter=10)
    left, right = mesh.find_nodes("X", mesh.bounding_box.xmin), mesh.find_nodes("X", mesh.bounding_box.xmax)
    pb.bc.add("Dirichlet", left, "Disp", 0)
    pb.bc.add("Dirichlet", right, "DispY", 0)
    pb.bc.add("Dirichlet", right, "DispX", 0.01)
    pb.nlsolve(dt=0.25, tmax=1, update_dt=False, print_info=0)
    return pb.get_disp(), np.asarray(asm.sv["Stress"])


@pytest.mark.parametrize("mode", ["legacy", "consistent"])
def test_fedoo_wiring_matches_elastic_isotrop(mode):
    law = LSTMConstitutiveLaw.from_networks(LinearElasticNet("stress"), LinearElasticNet("tangent"), mode)
    disp, stress_lstm = solve_cube(law)
    disp_ref, stress_ref = solve_cube(fd.constitutivelaw.ElasticIsotrop(113800.0, 0.342))
    np.testing.assert_allclose(disp, disp_ref, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(stress_lstm, stress_ref, rtol=1e-6, atol=1e-6)


def test_runs_in_fedoo_with_random_networks(weights_dir):
    law = LSTMConstitutiveLaw(weights_dir, state_update="consistent")
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=3, ny=3, nz=3, elm_type="hex8")
    asm = fd.Assembly.create(fd.weakform.StressEquilibrium(law, nlgeom=False), mesh)
    pb = fd.problem.NonLinear(asm)
    pb.set_nr_criterion("Displacement", tol=10.0, max_subiter=3)  # random net: smoke test only
    pb.bc.add("Dirichlet", mesh.find_nodes("X", mesh.bounding_box.xmin), "Disp", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", mesh.bounding_box.xmax), "DispX", 0.01)
    pb.nlsolve(dt=0.25, tmax=1, update_dt=False, print_info=0)
    s = np.asarray(asm.sv["Stress"])
    assert s.shape == (6, asm.n_gauss_points) and np.all(np.isfinite(s))
