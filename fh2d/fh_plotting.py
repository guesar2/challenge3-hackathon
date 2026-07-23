"""
fh_plotting.py

All Fermi-Hubbard figures. Each function takes already-computed data
(dicts/arrays) and returns the saved path, so the run scripts own the physics
and this module owns only presentation.

Figure inventory (deliberately short -- every removed panel was either
redundant, a bar chart, or a first-order-Trotter comparison):

  fig1_groundstate_vs_U.png   double occupancy and TOTAL ground-state energy
                              vs U/t, for 2x2 and 3x4
  fig2_quench_dynamics.png    ED vs 2nd-order Trotter vs emulator shots
  fig3_density_heatmaps.png   per-site density during the stripe quench (3x4)
  fig4_dt_convergence.png     2nd-order Trotter error vs dt (5% bar)
  fig6_vqe.png                VQE convergence history
  fig8_h2_run.png             H2 emulator shots vs ED

Removed on purpose: fig5 (ED energy density vs size + qubit budget),
fig7 (ED vs emulator-shot bar chart), the VQE bar panel, and every
first-order-Trotter curve.
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
    """gs_by_lattice: {"LxxLy": [ground-state dicts across U]}.

    Left  : average double occupancy vs U/t.
    Right : TOTAL ground-state energy E vs U/t (not the energy density).

    Because the two lattices have very different site counts (4 vs 12), the
    total energies live on different scales, so the right panel uses twin y-axes
    -- one per lattice -- rather than squashing both onto one axis. The shapes
    are what matter: E rises roughly linearly in U once double occupancy is
    suppressed, and it does so on both sizes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    labels = list(gs_by_lattice.keys())
    colors = [f"C{i}" for i in range(len(labels))]

    # ---- left: double occupancy ----
    for label, c in zip(labels, colors):
        rows = sorted(gs_by_lattice[label], key=lambda r: r["U"] / r["t"])
        uovert = [r["U"] / r["t"] for r in rows]
        dbl = [r["avg_double_occupancy"] for r in rows]
        ax1.plot(uovert, dbl, "o-", color=c, label=label)
    ax1.axhline(0.25, ls=":", c="grey", lw=1)   # uncorrelated (U=0) value
    ax1.text(0.02, 0.252, "uncorrelated limit 0.25", fontsize=8,
             color="grey", transform=ax1.get_yaxis_transform())
    ax1.set_xlabel("U/t")
    ax1.set_ylabel(r"$\langle n_\uparrow n_\downarrow\rangle$ (avg double occ.)")
    ax1.set_title("Double occupancy vs interaction")
    ax1.legend(fontsize=9)

    # ---- right: TOTAL ground-state energy, one axis per lattice ----
    axes = [ax2]
    if len(labels) > 1:
        axes.append(ax2.twinx())
        axes[1].grid(False)

    handles = []
    for i, (label, c) in enumerate(zip(labels, colors)):
        ax = axes[min(i, len(axes) - 1)]
        rows = sorted(gs_by_lattice[label], key=lambda r: r["U"] / r["t"])
        uovert = [r["U"] / r["t"] for r in rows]
        energy = [r["energy"] for r in rows]
        (h,) = ax.plot(uovert, energy, "s--", color=c,
                       label=f"{label}  ({rows[0].get('n_sites', '?')} sites)"
                       if "n_sites" in rows[0] else label)
        handles.append(h)
        ax.set_ylabel(rf"$E_0$  [{label}]", color=c)
        ax.tick_params(axis="y", labelcolor=c)

    ax2.set_xlabel("U/t")
    ax2.set_title("Ground-state energy vs interaction")
    ax2.legend(handles=handles, fontsize=9, loc="lower right")

    return _finalize(fig, "fig1_groundstate_vs_U.png", save_dir)


# ---------------------------------------------------------------------------
def plot_quench_dynamics(ed, trot, shots=None, shot_err=None, order=2,
                         save_dir=cfg.PLOT_SAVE_DIR, fname="fig2_quench_dynamics.png"):
    """Compare ED (exact) vs second-order Trotter (statevector) vs optional
    shots for the two scalar observables over time."""
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
    # Plain-language title: this panel just shows the magnetic (antiferromagnetic)
    # order of the initial state decaying as the state delocalises.
    ax2.set_title("Loss of magnetic order vs time")
    for ax in (ax1, ax2):
        ax.legend(fontsize=9)
    return _finalize(fig, fname, save_dir)


# ---------------------------------------------------------------------------
def plot_density_heatmaps(dyn, Lx, Ly, n_snapshots=3, U=None, initial_state=None,
                          save_dir=cfg.PLOT_SAVE_DIR, fname="fig3_density_heatmaps.png"):
    """Per-site density heatmaps at a few times from a dynamics dict
    (uses dyn['density_per_site'] shaped (n_sites, steps) and dyn['sites']).

    Works for any Lx x Ly. The colour scale is shared across the snapshots so
    the panels are directly comparable, and it is anchored on the FULL range of
    the run (not per-panel) so a melting stripe visibly flattens.
    """
    dens = np.asarray(dyn["density_per_site"])
    sites = [tuple(s) for s in dyn["sites"]]
    steps = dens.shape[1]
    n_snapshots = max(2, min(n_snapshots, steps))
    idxs = list(np.unique(np.linspace(0, steps - 1, n_snapshots).astype(int)))

    fig, axes = plt.subplots(1, len(idxs), figsize=(3.6 * len(idxs), 3.6))
    if len(idxs) == 1:
        axes = [axes]
    vmin, vmax = float(dens.min()), float(dens.max())
    if abs(vmax - vmin) < 1e-9:          # perfectly uniform: keep a sane range
        vmin, vmax = vmin - 0.1, vmax + 0.1

    im = None
    for ax, k in zip(axes, idxs):
        grid = np.full((Ly, Lx), np.nan)
        for si, (x, y) in enumerate(sites):
            grid[y, x] = dens[si, k]
        im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {dyn['times'][k]:.2f}/t")
        ax.set_xticks(range(Lx)); ax.set_yticks(range(Ly))
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(False)
        for si, (x, y) in enumerate(sites):
            val = dens[si, k]
            ax.text(x, y, f"{val:.2f}", ha="center", va="center",
                    color="w" if val < 0.6 * (vmin + vmax) else "k", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, label=r"$\langle n_i\rangle$")

    sub = f"{Lx}x{Ly} periodic"
    if U is not None:
        sub += f",  U/t = {U:.0f}"
    if initial_state:
        sub += f",  initial state: {initial_state}"
    fig.suptitle(f"Per-site particle density during the quench\n({sub})", fontsize=12)
    _ensure_dir(save_dir)
    full = os.path.join(save_dir, fname)
    fig.savefig(full, bbox_inches="tight")
    plt.close(fig)
    return full


# ---------------------------------------------------------------------------
def plot_dt_convergence(conv, order=2, save_dir=cfg.PLOT_SAVE_DIR):
    """conv: {"dt":[...], "dev_D":[...], "dev_M":[...]} for the single Trotter
    order that is actually used (second). Log-log max %-deviation vs dt plus the
    5% pass line, with the expected O(dt^2) slope drawn for reference."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    dt = np.asarray(conv["dt"], dtype=float)

    for ax, key, mk, title in (
            (ax1, "dev_D", "o-", r"double occupancy $\langle n_\uparrow n_\downarrow\rangle$"),
            (ax2, "dev_M", "s-", r"staggered magnetization $m_s$")):
        dev = np.asarray(conv[key], dtype=float)
        ax.loglog(dt, dev, mk, c="C0", label=f"order {order}")
        # reference slope: second-order Trotter error scales as dt^2
        ref = dev[-1] * (dt / dt[-1]) ** 2
        ax.loglog(dt, ref, ":", c="grey", lw=1.2, label=r"$\propto \delta t^{2}$")
        ax.axhline(5.0, ls="--", c="r", lw=1, label="5% pass bar")
        ax.set_xlabel(r"Trotter step $\delta t$ [1/t]")
        ax.set_ylabel("max deviation from ED [%]")
        ax.set_title(title)
        ax.legend(fontsize=9)
    fig.suptitle(f"Trotter convergence: halving dt at fixed total time "
                 f"(order {order})")
    return _finalize(fig, "fig4_dt_convergence.png", save_dir)


# ---------------------------------------------------------------------------
def plot_vqe(vqe_rows, save_dir=cfg.PLOT_SAVE_DIR):
    """vqe_rows: list of run_vqe_local result dicts. Convergence histories only;
    the dotted horizontal lines are the corresponding exact ED energies, so how
    close each curve settles to its line IS the accuracy statement. (The old
    VQE-vs-ED bar panel was removed.)"""
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    for i, r in enumerate(vqe_rows):
        running = np.minimum.accumulate(r["history"])
        c = f"C{i}"
        ax1.plot(running, color=c,
                 label=f"U/t={r['U']/r['t']:.0f}, p={r['layers']} "
                       f"(err {r['error_percent']:.1f}%)")
        ax1.axhline(r["energy_ed"], ls=":", lw=1, c=c)
    ax1.set_xlabel("COBYLA evaluation")
    ax1.set_ylabel("running best energy")
    ax1.set_title("VQE convergence (dotted = exact ED energy)")
    ax1.legend(fontsize=8)
    return _finalize(fig, "fig6_vqe.png", save_dir)
