"""
fh_plotting.py

All Fermi-Hubbard figures. Style mirrors the TFIM project's plotting.py: a
single _finalize() helper, restrained styling, PNG output into PLOT_SAVE_DIR.
Each function takes already-computed data (dicts/arrays) and returns the saved
path, so the run scripts own the physics and this module owns only presentation.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fh_config as cfg

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})


def _ensure_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)


def _finalize(fig, path, save_dir):
    _ensure_dir(save_dir)
    full = os.path.join(save_dir, path)
    fig.tight_layout()
    fig.savefig(full, bbox_inches="tight")
    plt.close(fig)
    return full


# ---------------------------------------------------------------------------
def plot_ground_state_vs_U(gs_by_lattice, save_dir=cfg.PLOT_SAVE_DIR):
    """gs_by_lattice: {"LxxLy": [ground-state dicts across U]}. Plots average
    double occupancy and energy/site vs U/t."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for label, rows in gs_by_lattice.items():
        rows = sorted(rows, key=lambda r: r["U"] / r["t"])
        uovert = [r["U"] / r["t"] for r in rows]
        dbl = [r["avg_double_occupancy"] for r in rows]
        epn = [r["energy_per_site"] for r in rows]
        ax1.plot(uovert, dbl, "o-", label=label)
        ax2.plot(uovert, epn, "s-", label=label)
    ax1.set_xlabel("U/t"); ax1.set_ylabel(r"$\langle n_\uparrow n_\downarrow\rangle$ (avg double occ.)")
    ax1.set_title("Double occupancy vs interaction")
    ax1.axhline(0.25, ls=":", c="grey", lw=1)  # uncorrelated (infinite-T) value
    ax2.set_xlabel("U/t"); ax2.set_ylabel(r"$E/N_{\rm sites}$")
    ax2.set_title("Ground-state energy density")
    for ax in (ax1, ax2):
        ax.legend(fontsize=9)
    return _finalize(fig, "fig1_groundstate_vs_U.png", save_dir)


# ---------------------------------------------------------------------------
def plot_quench_dynamics(ed, trot, shots=None, shot_err=None, order=2,
                         save_dir=cfg.PLOT_SAVE_DIR, fname="fig2_quench_dynamics.png"):
    """Compare ED (exact) vs Trotter (statevector) vs optional shots for the two
    scalar observables over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    t = ed["times"]
    ax1.plot(t, ed["avg_double_occupancy"], "-", c="k", lw=2, label="ED (exact)")
    ax1.plot(t, trot["avg_double_occupancy"], "o--", c="C0", ms=4,
             label=f"Trotter (order {order})")
    ax2.plot(t, ed["staggered_magnetization"], "-", c="k", lw=2, label="ED (exact)")
    ax2.plot(t, trot["staggered_magnetization"], "o--", c="C0", ms=4,
             label=f"Trotter (order {order})")
    if shots is not None:
        st, sD, sM = shots["times"], shots["avg_double_occupancy"], shots["staggered_magnetization"]
        eD = shot_err["avg_double_occupancy"] if shot_err else None
        eM = shot_err["staggered_magnetization"] if shot_err else None
        ax1.errorbar(st, sD, yerr=eD, fmt="s", c="C3", ms=5, capsize=2, label="Emulator shots")
        ax2.errorbar(st, sM, yerr=eM, fmt="s", c="C3", ms=5, capsize=2, label="Emulator shots")
    ax1.set_xlabel("time [1/t]"); ax1.set_ylabel(r"$\langle n_\uparrow n_\downarrow\rangle$")
    ax1.set_title("Double occupancy vs time")
    ax2.set_xlabel("time [1/t]"); ax2.set_ylabel(r"staggered magnetization $m_s$")
    ax2.set_title("Neel order melting")
    for ax in (ax1, ax2):
        ax.legend(fontsize=9)
    return _finalize(fig, fname, save_dir)


# ---------------------------------------------------------------------------
def plot_density_heatmaps(dyn, Lx, Ly, snapshots=(0, None, None), save_dir=cfg.PLOT_SAVE_DIR):
    """Per-site density heatmaps at a few times from an ED/Trotter dynamics dict
    (uses dyn['density_per_site'] shaped (n_sites, steps) and dyn['sites'])."""
    dens = np.asarray(dyn["density_per_site"])
    sites = [tuple(s) for s in dyn["sites"]]
    steps = dens.shape[1]
    idxs = [0, steps // 2, steps - 1]
    fig, axes = plt.subplots(1, len(idxs), figsize=(4 * len(idxs), 3.4))
    if len(idxs) == 1:
        axes = [axes]
    vmin, vmax = dens.min(), dens.max()
    for ax, k in zip(axes, idxs):
        grid = np.full((Ly, Lx), np.nan)
        for si, (x, y) in enumerate(sites):
            grid[y, x] = dens[si, k]
        im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {dyn['times'][k]:.2f}/t")
        ax.set_xticks(range(Lx)); ax.set_yticks(range(Ly))
        for si, (x, y) in enumerate(sites):
            ax.text(x, y, f"{dens[si,k]:.2f}", ha="center", va="center",
                    color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, label=r"$\langle n_i\rangle$")
    fig.suptitle("Per-site particle density during the quench")
    return _finalize(fig, "fig3_density_heatmaps.png", save_dir)


# ---------------------------------------------------------------------------
def plot_dt_convergence(conv, save_dir=cfg.PLOT_SAVE_DIR):
    """conv: {order: {"dt":[...], "dev_D":[...], "dev_M":[...]}}. Log-log
    max %-deviation vs dt for each Trotter order + the 5% pass line."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for order, d in conv.items():
        ax1.loglog(d["dt"], d["dev_D"], "o-", label=f"order {order}")
        ax2.loglog(d["dt"], d["dev_M"], "s-", label=f"order {order}")
    for ax, title in ((ax1, r"double occupancy $\langle n_\uparrow n_\downarrow\rangle$"),
                      (ax2, r"staggered magnetization $m_s$")):
        ax.axhline(5.0, ls="--", c="r", lw=1, label="5% pass bar")
        ax.set_xlabel(r"Trotter step $\delta t$ [1/t]")
        ax.set_ylabel("max deviation from ED [%]")
        ax.set_title(title)
        ax.legend(fontsize=9)
    fig.suptitle("Trotter convergence: halving dt (fixed total time)")
    return _finalize(fig, "fig4_dt_convergence.png", save_dir)


# ---------------------------------------------------------------------------
def plot_ed_vs_circuit(labels, ed_vals, shot_vals, shot_errs, ylabel, title,
                       fname, save_dir=cfg.PLOT_SAVE_DIR):
    """Grouped bar comparison of ED vs emulator-shot observables."""
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 4))
    ax.bar(x - w / 2, ed_vals, w, label="ED (exact)", color="C7")
    ax.bar(x + w / 2, shot_vals, w, yerr=shot_errs, capsize=3, label="Emulator shots", color="C3")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=9)
    return _finalize(fig, fname, save_dir)


# ---------------------------------------------------------------------------
def plot_n_scaling(scaling_rows, save_dir=cfg.PLOT_SAVE_DIR):
    """scaling_rows: list of dicts with keys n_sites, n_qubits, energy_per_site
    (weak & strong). Plots E/N vs lattice size and the qubit budget with the
    26-qubit H2 line and the 32-qubit 4x4 marker."""
    rows = sorted(scaling_rows, key=lambda r: r["n_sites"])
    ns = [r["n_sites"] for r in rows]
    nq = [r["n_qubits"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for key, mk, lab in (("epn_weak", "o-", "U/t=1"), ("epn_strong", "s-", "U/t=8")):
        if all(key in r for r in rows):
            ax1.plot(ns, [r[key] for r in rows], mk, label=lab)
    ax1.set_xlabel(r"$N_{\rm sites}$"); ax1.set_ylabel(r"$E/N_{\rm sites}$")
    ax1.set_title("ED energy density vs size (half-filling)"); ax1.legend(fontsize=9)

    ax2.plot(ns, nq, "o-", c="C0", label="JW qubits (2/site)")
    ax2.axhline(26, ls="--", c="r", lw=1, label="H2 exact-emulator limit (26q)")
    ax2.scatter([16], [32], c="k", marker="X", s=80, zorder=5, label="4x4 = 32q (out of reach)")
    ax2.set_xlabel(r"$N_{\rm sites}$"); ax2.set_ylabel("qubits required")
    ax2.set_title("Qubit budget"); ax2.legend(fontsize=8)
    return _finalize(fig, "fig5_n_scaling.png", save_dir)


# ---------------------------------------------------------------------------
def plot_vqe(vqe_rows, save_dir=cfg.PLOT_SAVE_DIR):
    """vqe_rows: list of run_vqe_local result dicts. Left: convergence history;
    right: VQE vs ED energy bars with %error annotation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for r in vqe_rows:
        hist = r["history"]
        running = np.minimum.accumulate(hist)
        ax1.plot(running, label=f"U/t={r['U']/r['t']:.0f}, p={r['layers']}")
        ax1.axhline(r["energy_ed"], ls=":", lw=1, c="grey")
    ax1.set_xlabel("COBYLA evaluation"); ax1.set_ylabel("running best energy")
    ax1.set_title("VQE convergence (dotted = ED)"); ax1.legend(fontsize=8)

    labels = [f"U/t={r['U']/r['t']:.0f}\np={r['layers']}" for r in vqe_rows]
    x = np.arange(len(vqe_rows)); w = 0.38
    ax2.bar(x - w / 2, [r["energy_ed"] for r in vqe_rows], w, label="ED", color="C7")
    ax2.bar(x + w / 2, [r["energy_vqe"] for r in vqe_rows], w, label="VQE", color="C0")
    for i, r in enumerate(vqe_rows):
        ax2.text(i, min(r["energy_ed"], r["energy_vqe"]),
                 f"{r['error_percent']:.1f}%", ha="center", va="top", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("ground-state energy"); ax2.set_title("VQE vs ED"); ax2.legend(fontsize=9)
    return _finalize(fig, "fig6_vqe.png", save_dir)