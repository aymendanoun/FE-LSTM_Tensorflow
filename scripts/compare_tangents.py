"""Compare the tangent network with the finite-difference tangent of the stress network.

Both LSTMs are run side by side along the same strain paths. At each step:

* ``L_net``: 6x6 matrix built from the 21 outputs of the tangent network (``weights_scaler_Lt``)
* ``L_fd``: d(stress)/d(strain) of the stress network for that step, with the incoming
  recurrent state held fixed (central differences), i.e. the tangent Newton actually needs.

Usage::

    python scripts/compare_tangents.py [--weights legacy/fedoo_coupling/weights_scaler_3D]
        [--out results/tangent_comparison]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fe_lstm.fem.lstm_law import tangent_from_vector
from fe_lstm.model.numpy_lstm import LSTMSurrogate

LABELS = ["11", "22", "33", "12", "13", "23"]
N_STEPS = 100  # training sequences have 100 increments


def ramp(peak: np.ndarray, n: int = N_STEPS) -> np.ndarray:
    return np.linspace(0.0, 1.0, n + 1)[1:, None] * peak[None, :]


def load_unload(peak: np.ndarray, n: int = N_STEPS) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    # load to peak over the first half, then unload to -0.6 * peak over the second half
    a = np.where(t < 0.5, 2 * t, 1 - 3.2 * (t - 0.5))
    return a[:, None] * peak[None, :]


def strain_paths() -> dict[str, np.ndarray]:
    e = np.eye(6)
    paths = {
        "uniaxial e11 to 0.04": ramp(0.04 * e[0]),
        "e11 load/unload/reverse": load_unload(0.04 * e[0]),
        "shear g12 to 0.02": ramp(0.02 * e[3]),
        "biaxial e11=e22 to 0.03": ramp(0.03 * (e[0] + e[1])),
        "uniaxial e22 + shear g12": ramp(0.03 * e[1] + 0.015 * e[3]),
    }
    rng = np.random.default_rng(0)
    # smooth random path inside the training box (random walk of increments)
    lim = np.array([0.05, 0.05, 0.05, 0.025, 0.025, 0.025]) * 0.8
    inc = rng.normal(size=(N_STEPS, 6)) * lim / 15
    paths["random walk (seed 0)"] = np.clip(np.cumsum(inc, axis=0), -lim, lim)
    return paths


def fd_tangent(net: LSTMSurrogate, strain: np.ndarray, state, h: float) -> np.ndarray:
    """Central-difference d(stress)/d(strain), shape (6, 6) for one point."""
    lt = np.empty((6, 6))
    for k in range(6):
        dp = strain.copy()
        dm = strain.copy()
        dp[0, k] += h
        dm[0, k] -= h
        sp, _ = net.step(dp, state)
        sm, _ = net.step(dm, state)
        lt[:, k] = (sp - sm)[0] / (2 * h)
    return lt


def run_path(stress_net, tangent_net, path, h):
    s_state = stress_net.zero_state(1)
    t_state = tangent_net.zero_state(1)
    # The law evaluates both nets once at zero strain at initialization; mimic that.
    _, s_state = stress_net.step(np.zeros((1, 6)), s_state)
    _, t_state = tangent_net.step(np.zeros((1, 6)), t_state)
    L_net, L_fd, stress = [], [], []
    for eps in path:
        x = eps[None, :]
        L_fd.append(fd_tangent(stress_net, x, s_state, h))
        sig, s_state = stress_net.step(x, s_state)
        y, t_state = tangent_net.step(x, t_state)
        L_net.append(tangent_from_vector(y)[:, :, 0])
        stress.append(sig[0])
    return np.array(L_net), np.array(L_fd), np.array(stress)


def secant_check(path, stress):
    """Tangent from the stress-strain curve itself (d sigma / d eps along the path)."""
    de = np.diff(np.vstack([np.zeros(6), path]), axis=0)
    ds = np.diff(np.vstack([np.zeros(6), stress]), axis=0)
    return de, ds


def summarise(name, path, L_net, L_fd, stress):
    rel = np.linalg.norm(L_net - L_fd, axis=(1, 2)) / np.linalg.norm(L_fd, axis=(1, 2))
    asym = np.linalg.norm(L_fd - L_fd.transpose(0, 2, 1), axis=(1, 2)) / np.linalg.norm(L_fd, axis=(1, 2))
    # "incremental" check: does L * d_eps predict d_sigma of the stress net along the path?
    de, ds = secant_check(path, stress)
    pred_net = np.einsum("nij,nj->ni", L_net, de)
    pred_fd = np.einsum("nij,nj->ni", L_fd, de)
    nrm = np.linalg.norm(ds[1:], axis=1)
    inc_err_net = np.linalg.norm(pred_net - ds, axis=1)[1:] / nrm
    inc_err_fd = np.linalg.norm(pred_fd - ds, axis=1)[1:] / nrm
    print(f"\n=== {name} ===")
    print(f"  ||L_net - L_fd|| / ||L_fd||  : first {rel[0]:.3f}  median {np.median(rel):.3f}  max {rel.max():.3f}")
    print(f"  asymmetry of L_fd            : median {np.median(asym):.3f}")
    print(f"  |L*de - dsigma|/|dsigma| (median over steps): net {np.median(inc_err_net):.3f}  fd {np.median(inc_err_fd):.3f}")
    for n in (0, len(path) // 2, len(path) - 1):
        print(f"  step {n + 1:3d}  strain = {np.array2string(path[n], precision=4)}")
        print("     diag   net: " + " ".join(f"{v:8.1f}" for v in np.diag(L_net[n])))
        print("     diag   fd : " + " ".join(f"{v:8.1f}" for v in np.diag(L_fd[n])))
        print(f"     L12/L13/L23 net: {L_net[n,0,1]:8.1f} {L_net[n,0,2]:8.1f} {L_net[n,1,2]:8.1f}"
              f"   fd: {L_fd[n,0,1]:8.1f} {L_fd[n,0,2]:8.1f} {L_fd[n,1,2]:8.1f}")
    return rel, inc_err_net, inc_err_fd


def plot(results, out: Path):
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(len(names), 3, figsize=(13, 2.6 * len(names)), squeeze=False)
    comps = [(0, 0), (3, 3), (0, 1)]
    for r, name in enumerate(names):
        path, L_net, L_fd, _ = results[name]
        steps = np.arange(1, len(path) + 1)
        for c, (i, j) in enumerate(comps):
            ax = axes[r, c]
            ax.plot(steps, L_fd[:, i, j], label="finite difference of stress net", color="#1f77b4")
            ax.plot(steps, L_net[:, i, j], "--", label="tangent net", color="#d62728")
            ax.set_title(f"{name}: L{LABELS[i]}{LABELS[j]}", fontsize=9)
            ax.tick_params(labelsize=8)
        axes[r, 0].set_ylabel("MPa")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("increment")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "tangent_net_vs_fd.png", dpi=130)
    print(f"\nFigure: {out / 'tangent_net_vs_fd.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="legacy/fedoo_coupling/weights_scaler_3D")
    p.add_argument("--out", default="results/tangent_comparison")
    p.add_argument("--fd-step", type=float, default=1e-6)
    args = p.parse_args()

    w = Path(args.weights)
    stress_net = LSTMSurrogate(w / "weights_scaler_stress", "scaling_stress.npz")
    tangent_net = LSTMSurrogate(w / "weights_scaler_Lt", "scaling_Lt.npz")
    print(f"stress net: {stress_net.n_inputs} -> {stress_net.n_outputs}, units {stress_net.units}")
    print(f"tangent net: {tangent_net.n_inputs} -> {tangent_net.n_outputs}, units {tangent_net.units}")

    results = {}
    for name, path in strain_paths().items():
        L_net, L_fd, stress = run_path(stress_net, tangent_net, path, args.fd_step)
        summarise(name, path, L_net, L_fd, stress)
        results[name] = (path, L_net, L_fd, stress)
    plot(results, Path(args.out))


if __name__ == "__main__":
    main()
