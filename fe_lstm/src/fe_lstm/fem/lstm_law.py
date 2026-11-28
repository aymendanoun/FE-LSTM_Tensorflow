"""FE-LSTM constitutive law for fedoo 1.x.

Port of ``legacy/fedoo_coupling/fedoo/libConstitutiveLaw/ConstitutiveLaw_LSTM.py`` (old fedoo
API: ``Initialize / Update / NewTimeIncrement / ResetTimeIncrement / GetH``) to the fedoo 1.x
``Mechanical3D`` API (``initialize / update / set_start / to_start / get_tangent_matrix``,
state stored in ``assembly.sv``).

Two LSTM networks run at every Gauss point:

* the *stress* network: strain (6) -> stress (6)
* the *tangent* network: strain (6) -> 21 independent components of the symmetric 6x6
  tangent matrix (6 diagonal terms, then the upper triangle row by row)

Voigt convention (unchanged between the legacy fedoo and fedoo 1.x):
``[e_xx, e_yy, e_zz, g_xy, g_xz, g_yz]`` with engineering shear strains, and stresses
``[s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]``. This is also the column order used for training.

Recurrent state and Newton-Raphson iterations
---------------------------------------------
fedoo calls ``update`` once per Newton-Raphson iteration. Two behaviours are available.
Deliberate change from the thesis code: the default is ``"consistent"``. With ``"legacy"`` the
specimen run fails to converge with fedoo 1.x (increment 77 of 100).

``state_update="legacy"`` (reproduces the thesis code exactly, kept for reference)
    The LSTM state advances at every call to ``update``, i.e. at every Newton-Raphson
    iteration. At the start of an increment only the cell states ``c`` are saved, and only
    they are restored if the increment is restarted; the hidden states ``h`` are not.
    The state also advances once with zero strain at initialization.

``state_update="consistent"`` (default)
    Every Newton-Raphson iteration restarts from the state committed at the end of the
    previous converged increment, so the LSTM advances exactly one step per converged time
    increment (like the training sequences). ``h`` and ``c`` are stored in ``assembly.sv``
    and committed / restored by fedoo together with stress and strain.

The two modes give identical results when every increment converges at the first check
(no Newton correction). They differ otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fedoo.core.mechanical3d import Mechanical3D
from fedoo.util.voigt_tensors import StressTensorList

from fe_lstm.model.numpy_lstm import LSTMSurrogate

# Position in the 6x6 tangent matrix of each of the 21 outputs of the tangent network
# (training labels: C11..C66, then C12, C13, C14, C15, C16, C23, C24, ..., C56).
_DIAG = [(i, i) for i in range(6)]
_UPPER = [(i, j) for i in range(6) for j in range(i + 1, 6)]
TANGENT_INDEX = _DIAG + _UPPER

# Names of the inputs / outputs the law expects, in this order (training stage conventions);
# checked against architecture.json when the network was exported by the training stage.
STRAIN_NAMES = ["Exx", "Eyy", "Ezz", "Exy", "Exz", "Eyz"]
STRESS_NAMES = ["Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"]
TANGENT_NAMES = [f"C{i + 1}{i + 1}" for i in range(6)] + [f"C{i + 1}{j + 1}_m" for i, j in _UPPER]

_STATE_MODES = ("legacy", "consistent")
_TANGENT_MODES = ("network", "finite_difference")
FD_STEP = 1e-6  # strain perturbation for the finite-difference tangent


def tangent_from_vector(y: np.ndarray) -> np.ndarray:
    """Build symmetric tangent matrices, shape (6, 6, n_points), from network outputs (n_points, 21)."""
    n_points = y.shape[0]
    lt = np.zeros((6, 6, n_points))
    for k, (i, j) in enumerate(TANGENT_INDEX):
        lt[i, j] = y[:, k]
        lt[j, i] = y[:, k]
    return lt


def _check_names(net, outputs: list[str], kind: str):
    """Inputs / outputs order of a network exported with architecture.json (no check otherwise)."""
    arch = getattr(net, "architecture", None)
    if not arch:
        return
    if list(arch["inputs"]) != STRAIN_NAMES or list(arch["outputs"]) != outputs:
        raise ValueError(
            f"The {kind} network maps {arch['inputs']} -> {arch['outputs']}; the law needs "
            f"{STRAIN_NAMES} -> {outputs} (see training/configs/training.yaml)."
        )


class LSTMConstitutiveLaw(Mechanical3D):
    """Data-driven constitutive law: LSTM surrogates for stress and tangent matrix.

    Use with :class:`fedoo.weakform.StressEquilibrium` in small strain (``nlgeom=False``),
    as in the original work.

    Parameters
    ----------
    weights_dir : str or Path
        Folder with two subfolders, as produced in the original project:
        ``weights_scaler_stress/`` (with ``scaling_stress.npz``) and
        ``weights_scaler_Lt/`` (with ``scaling_Lt.npz``).
    state_update : {"consistent", "legacy"}
        How the recurrent state is handled across Newton-Raphson iterations
        (see module docstring). Default "consistent".
    tangent : {"finite_difference", "network"}
        ``"finite_difference"`` (default since 2026-11-28): tangent computed from the stress
        network by forward finite differences. It is the derivative of the stress the solver
        uses, so Newton-Raphson converges to the same solution with fewer failures; costs 6
        extra stress evaluations per update; ``weights_scaler_Lt/`` is not needed.
        ``"network"`` (thesis): tangent matrix predicted by the separately trained tangent
        network. The converged FE solution depends only on the stress network either way.
    name : str
        Name of the constitutive law.
    """

    def __init__(
        self,
        weights_dir: str | Path,
        state_update: str = "consistent",
        tangent: str = "finite_difference",
        name: str = "",
    ):
        weights_dir = Path(weights_dir)
        stress_net = LSTMSurrogate(weights_dir / "weights_scaler_stress", "scaling_stress.npz")
        tangent_net = None
        if tangent == "network":
            tangent_net = LSTMSurrogate(weights_dir / "weights_scaler_Lt", "scaling_Lt.npz")
        self._setup(stress_net, tangent_net, state_update, tangent, name)

    @classmethod
    def from_networks(
        cls, stress_net, tangent_net=None, state_update: str = "consistent", tangent: str | None = None,
        name: str = "",
    ):
        """Build the law from network objects (``LSTMSurrogate`` or any object with the same
        ``n_inputs``, ``n_outputs``, ``n_lstm_layers``, ``zero_state`` and ``step``).

        ``tangent=None`` uses ``"network"`` when a tangent network is given, else
        ``"finite_difference"``."""
        if tangent is None:
            tangent = "network" if tangent_net is not None else "finite_difference"
        law = cls.__new__(cls)
        law._setup(stress_net, tangent_net, state_update, tangent, name)
        return law

    def _setup(self, stress_net, tangent_net, state_update, tangent, name):
        if state_update not in _STATE_MODES:
            raise ValueError(f"state_update must be one of {_STATE_MODES}, got {state_update!r}")
        if tangent not in _TANGENT_MODES:
            raise ValueError(f"tangent must be one of {_TANGENT_MODES}, got {tangent!r}")
        Mechanical3D.__init__(self, name=name)
        if stress_net.n_inputs != 6 or stress_net.n_outputs != 6:
            raise ValueError("The stress network must map 6 strain components to 6 stresses.")
        if tangent == "network" and (
            tangent_net is None or tangent_net.n_inputs != 6 or tangent_net.n_outputs != 21
        ):
            raise ValueError("The tangent network must map 6 strain components to 21 terms.")
        _check_names(stress_net, STRESS_NAMES, "stress")
        if tangent == "network":
            _check_names(tangent_net, TANGENT_NAMES, "tangent")
        self.stress_net = stress_net
        self.tangent_net = tangent_net if tangent == "network" else None
        self.tangent = tangent
        # The network tangent (21 terms) is symmetric by construction; the finite-difference
        # tangent is not, and fedoo assembles only one triangle when assume_sym is True.
        self.symmetric_tangent = tangent == "network"
        self.state_update = state_update
        self._state = None
        self._c_start = None
        self.n_update_calls = 0  # diagnostic: number of network evaluations

    # ------------------------------------------------------------------ helpers
    def _run(self, strain: np.ndarray, state_stress, state_tangent):
        """Evaluate both networks for strain of shape (n_points, 6)."""
        stress, new_state_stress = self.stress_net.step(strain, state_stress)
        if self.tangent == "network":
            y_tangent, new_state_tangent = self.tangent_net.step(strain, state_tangent)
            return stress, tangent_from_vector(y_tangent), new_state_stress, new_state_tangent
        # d(stress)/d(strain) of the same step (same incoming recurrent state)
        lt = np.empty((6, 6, strain.shape[0]))
        for k in range(6):
            perturbed = strain.copy()
            perturbed[:, k] += FD_STEP
            stress_k, _ = self.stress_net.step(perturbed, state_stress)
            lt[:, k, :] = ((stress_k - stress) / FD_STEP).T
        return stress, lt, new_state_stress, []

    @staticmethod
    def _store(assembly, stress, tangent):
        assembly.sv["Stress"] = StressTensorList(np.ascontiguousarray(stress.T))
        assembly.sv["TangentMatrix"] = tangent

    # --------------------------------------------------------- fedoo interface
    def initialize(self, assembly, pb):
        if assembly._nlgeom:
            raise NotImplementedError(
                "LSTMConstitutiveLaw was trained and used in small strain: use nlgeom=False."
            )
        n_points = assembly.n_gauss_points
        zero_strain = np.zeros((n_points, 6))
        state_stress = self.stress_net.zero_state(n_points)
        state_tangent = self.tangent_net.zero_state(n_points) if self.tangent_net else []

        if self.state_update == "legacy":
            # legacy: memory_start was saved *before* the initial zero-strain evaluation
            self._c_start = [c.copy() for _, c in state_stress + state_tangent]

        stress, tangent, state_stress, state_tangent = self._run(
            zero_strain, state_stress, state_tangent
        )
        self.n_update_calls = 1

        if self.state_update == "legacy":
            self._state = state_stress + state_tangent  # list of (h, c), 4 layers
        else:
            assembly.sv["LSTM_state_stress"] = state_stress
            assembly.sv["LSTM_state_tangent"] = state_tangent

        self._store(assembly, stress, tangent)
        self.is_initialized = True

    def update(self, assembly, pb):
        strain = assembly.sv["Strain"]
        n_points = assembly.n_gauss_points
        if np.isscalar(strain) and strain == 0:
            strain = np.zeros((n_points, 6))
        else:
            strain = np.asarray(strain, dtype=float).reshape(6, n_points).T

        if self.state_update == "legacy":
            n_s = self.stress_net.n_lstm_layers
            stress, tangent, state_stress, state_tangent = self._run(
                strain, self._state[:n_s], self._state[n_s:]
            )
            self._state = state_stress + state_tangent
        else:
            stress, tangent, state_stress, state_tangent = self._run(
                strain,
                assembly.sv_start["LSTM_state_stress"],
                assembly.sv_start["LSTM_state_tangent"],
            )
            assembly.sv["LSTM_state_stress"] = state_stress
            assembly.sv["LSTM_state_tangent"] = state_tangent

        self.n_update_calls += 1
        self._store(assembly, stress, tangent)

    def set_start(self, assembly, pb):
        # "consistent": fedoo copies assembly.sv to assembly.sv_start, nothing else to do.
        if self.state_update == "legacy":
            self._c_start = [c.copy() for _, c in self._state]

    def to_start(self, assembly, pb):
        # "consistent": fedoo restores assembly.sv from assembly.sv_start.
        if self.state_update == "legacy":
            # legacy restored the cell states only, the hidden states are kept as they are
            self._state = [(h, c.copy()) for (h, _), c in zip(self._state, self._c_start)]

    def get_tangent_matrix(self, assembly, dimension=None):
        return assembly.sv["TangentMatrix"]

    def get_stress(self, assembly=None):
        return assembly.sv["Stress"]

    def reset(self):
        self.is_initialized = False
        self._state = None
        self._c_start = None
        self.n_update_calls = 0
