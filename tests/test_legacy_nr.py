"""The legacy displacement criterion reproduces legacy fedoo's NewtonRaphsonError."""

import numpy as np

from fe_lstm.fem.legacy_nr import use_legacy_displacement_criterion


class StubProblem:
    def __init__(self):
        self._dof_free = np.array([0, 1, 2])
        self._U = np.array([1.0, -2.0, 0.5, 9.0])  # dof 3 is fixed: ignored
        self._dU = 0
        self._X = np.zeros(4)
        self.predictions = 0

    def elastic_prediction(self):
        self.predictions += 1
        self._dU = np.array([0.1, -0.2, 0.0, 0.0])
        self._X = self._dU.copy()

    def get_X(self):
        return self._X


def test_legacy_criterion_formula():
    pb = use_legacy_displacement_criterion(StubProblem())
    pb.elastic_prediction()
    assert pb.predictions == 1  # original prediction still runs
    assert pb.compute_nr_error() == 1.0  # first check of an increment always fails
    pb._X = np.array([0.01, -0.03, 0.02, 5.0])  # Newton correction
    expected = 0.03 / 2.2  # max|X| / max|U + dU| on free dofs
    assert np.isclose(pb.compute_nr_error(), expected)


def test_err0_reset_at_each_increment():
    pb = use_legacy_displacement_criterion(StubProblem())
    pb.elastic_prediction()
    pb.compute_nr_error()
    pb._U = pb._U * 10
    pb.elastic_prediction()  # new increment
    assert pb.compute_nr_error() == 1.0
    pb._X = np.array([0.0, 0.0, 0.5, 0.0])
    assert np.isclose(pb.compute_nr_error(), 0.5 / 20.2)
