"""Legacy (2022 fedoo) Newton-Raphson "Displacement" convergence criterion for fedoo 1.x.

The two fedoo versions define the "Displacement" error differently:

* legacy fedoo (``Problem_NonLinearStatic.NewtonRaphsonError``)::

      err0  = max |U_total + dU|      (free dofs, computed at the first check of an increment,
                                       which always returns 1 so at least one correction is made)
      error = max |X| / err0          (X = last Newton correction)

  The reference is the *total* displacement, so the criterion gets looser as loading proceeds.

* fedoo 1.x::

      error = ||X||_2 / (||dU||_2 + 0.01 ||U||_2)

  The reference is mainly the *increment* displacement: much stricter for the same tolerance.

With the thesis tolerance (0.05), the FE-LSTM run converges with the legacy definition but not
with the fedoo 1.x one. ``use_legacy_displacement_criterion`` installs the legacy definition on a
``fedoo.problem.NonLinear`` instance so the thesis settings can be reproduced.
"""

from __future__ import annotations

import types

import numpy as np


def _legacy_elastic_prediction(self, *args, **kwargs):
    self._legacy_err0 = None  # legacy: err0 reset at each new increment (elastic prediction)
    return self._legacy_original_elastic_prediction(*args, **kwargs)


def _legacy_compute_nr_error(self):
    dof_free = self._dof_free
    if len(dof_free) == 0:
        return 0
    if self._legacy_err0 is None:
        total = self._U + self._dU
        err0 = 0.0 if np.isscalar(total) else np.max(np.abs(np.asarray(total)[dof_free]))
        self._legacy_err0 = err0 if err0 > 0 else 1.0
        return 1.0
    return np.max(np.abs(np.asarray(self.get_X())[dof_free])) / self._legacy_err0


def use_legacy_displacement_criterion(pb):
    """Make ``pb`` (a fedoo 1.x NonLinear problem) use the 2022 displacement criterion."""
    pb._legacy_err0 = None
    pb._legacy_original_elastic_prediction = pb.elastic_prediction
    pb.elastic_prediction = types.MethodType(_legacy_elastic_prediction, pb)
    pb.compute_nr_error = types.MethodType(_legacy_compute_nr_error, pb)
    return pb
