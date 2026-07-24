"""
fh_plotting.py

All Fermi-Hubbard figures. Each function takes already-computed data
(dicts/arrays) and returns the saved path, so the run scripts own the physics
and this module owns only presentation.

Figure inventory (deliberately short -- every removed panel was either
redundant, a bar chart, or a first-order-Trotter comparison):

  fig1_groundstate_vs_U.png   double occupancy and TOTAL ground-state energy
                              vs U/t, for 2x2 and 3x4
  fig2_quench_dynamics.png    ED vs 2nd-order Trotter vs noiseless emulator
                              shots vs raw noisy shots vs ZNE-mitigated
  fig3_density_heatmaps.png   per-site map during the quench (see
                              HEATMAP_QUANTITIES / fh_config.HEATMAP_*)
  fig6_vqe.png                VQE convergence history
  fig8_h2_run.png             H2 emulator shots vs ED

Removed on purpose: fig4 (dt-convergence), fig5 (ED energy density vs size +
qubit budget), fig7 (ED vs emulator-shot bar chart), the VQE bar panel, and
every first-order-Trotter curve.
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


# --- per-site heatmap quantities (fig3) ------------------------------------
# key      : which array in the dynamics dict to draw
# cbar     : colour-bar label
# title    : figure title
# symmetric: anchor the colour scale symmetrically about 0 (for signed fields)
HEATMAP_QUANTITIES = {
    "density": {"key": "density_per_site",
                "cbar": r"$\langle n_i \rangle$",
                "title": "Per-site particle density during the quench",
                "cmap": "viridis", "symmetric": False},
    "sz":      {"key": "sz_per_site",
                "cbar": r"$\langle S^z_i \rangle$",
                "title": "Per-site spin density during the quench",
                "cmap": "coolwarm", "symmetric": True},
    "double":  {"key": "double_per_site",
                "cbar": r"$\langle D_i \rangle$",
                "title": "Per-site double occupancy during the quench",
                "cmap": "viridis", "symmetric": False},
}


def _label_colour(frac, cmap):
    """Readable text colour for a cell whose value sits at `frac` of the colour
    scale. viridis is dark at the bottom, coolwarm is dark at BOTH ends."""
    if cmap == "coolwarm":
        return "w" if (frac < 0.15 or frac > 0.85) else "k"
    return "w" if frac < 0.55 else "k"



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
    ax1.set_ylabel(r"$\langle D \rangle$  (avg double occupancy)")
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
                         noisy=None, noisy_err=None, zne=None, zne_err=None,
                         shots_label="Emulator shots (noiseless)",
                         noisy_label="Emulator shots (raw noisy)",
                         zne_label="Noisy + ZNE (error-mitigated)",
                         save_dir=cfg.PLOT_SAVE_DIR, fname="fig2_quench_dynamics.png"):
    """Compare, on the same axes and the same time grid, every level of the
    simulation stack for the two scalar quench observables:

      ED (exact)                  black line   -- expm_multiply reference
      Trotter (order `order`)     blue         -- algorithmic (Trotter) error only
      `shots`                     red squares  -- NOISELESS emulator shots
                                                  (H2-1LE / local statevector
                                                  sampler): + shot noise
      `noisy`                     magenta ^    -- raw shots from the noisy device
                                                  emulator: + device noise
      `zne`                       green o      -- the same noisy shots after
                                                  zero-noise extrapolation

    The three shot-based series all carry error bars: bootstrap standard errors
    for the two measured ones, and the ZNE fit's own propagated standard error
    for the mitigated one. `noisy`/`zne` may be evaluated on a coarser subset of
    the time grid than ED/Trotter (see fh_config.NOISY_TIME_STRIDE) -- each
    series is plotted against its own 'times', so they still line up.

    Everything except `ed` and `trot` is optional: pass nothing extra and this
    reproduces the original three-curve figure, with the noiseless series now
    labelled as such.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    t = ed["times"]
    ax1.plot(t, ed["avg_double_occupancy"], "-", c="k", lw=2, label="ED (exact)")
    ax1.plot(t, trot["avg_double_occupancy"], "o--", c="C0", ms=4,
             label=f"Trotter (order {order}, noiseless)")
    ax2.plot(t, ed["staggered_magnetization"], "-", c="k", lw=2, label="ED (exact)")
    ax2.plot(t, trot["staggered_magnetization"], "o--", c="C0", ms=4,
             label=f"Trotter (order {order}, noiseless)")

    # (series, errors, matplotlib fmt, colour, label) in draw order
    series = [
        (shots, shot_err, "s", "C3", shots_label),
        (noisy, noisy_err, "^", "m", noisy_label),
        (zne, zne_err, "o", "g", zne_label),
    ]
    for data, err, fmt, colour, label in series:
        if data is None:
            continue
        st = data["times"]
        eD = err["avg_double_occupancy"] if err else None
        eM = err["staggered_magnetization"] if err else None
        ax1.errorbar(st, data["avg_double_occupancy"], yerr=eD, fmt=fmt, c=colour,
                     ms=5, capsize=2, label=label)
        ax2.errorbar(st, data["staggered_magnetization"], yerr=eM, fmt=fmt, c=colour,
                     ms=5, capsize=2, label=label)

    ax1.set_xlabel("time [1/t]"); ax1.set_ylabel(r"$\langle D \rangle$")
    ax1.set_title("Double occupancy vs time")
    ax2.set_xlabel("time [1/t]"); ax2.set_ylabel(r"staggered magnetization $m_s$")
    # Plain-language title: this panel just shows the magnetic (antiferromagnetic)
    # order of the initial state decaying as the state delocalises.
    ax2.set_title("Loss of magnetic order vs time")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
    return _finalize(fig, fname, save_dir)


# ---------------------------------------------------------------------------
def plot_density_heatmaps(dyn, Lx, Ly, n_snapshots=3, U=None, initial_state=None,
                          quantity="density",
                          save_dir=cfg.PLOT_SAVE_DIR, fname="fig3_density_heatmaps.png"):
    """Per-site heatmaps at a few times from a dynamics dict.

    `quantity` picks which per-site map is drawn (see HEATMAP_QUANTITIES):
        "density" -> dyn["density_per_site"]   <n_i>
        "sz"      -> dyn["sz_per_site"]        <S^z_i>
        "double"  -> dyn["double_per_site"]    <D_i>

    The colour scale is shared across the snapshots so the panels are directly
    comparable, and it is anchored on the FULL range of the run, not per-panel.
    Layout is constrained so the figure title never lands on the panel titles.
    """
    spec = HEATMAP_QUANTITIES[quantity]
    field = np.asarray(dyn[spec["key"]], dtype=float)
    sites = [tuple(s) for s in dyn["sites"]]
    steps = field.shape[1]
    n_snapshots = max(2, min(n_snapshots, steps))
    idxs = list(np.unique(np.linspace(0, steps - 1, n_snapshots).astype(int)))

    fig, axes = plt.subplots(1, len(idxs), figsize=(3.4 * len(idxs), 3.9),
                             layout="constrained")
    axes = np.atleast_1d(axes).ravel().tolist()

    vmin, vmax = float(field.min()), float(field.max())
    if spec["symmetric"]:
        m = max(abs(vmin), abs(vmax), 1e-9)
        vmin, vmax = -m, m
    elif abs(vmax - vmin) < 1e-9:        # perfectly uniform: keep a sane range
        vmin, vmax = vmin - 0.1, vmax + 0.1

    im = None
    for ax, k in zip(axes, idxs):
        grid = np.full((Ly, Lx), np.nan)
        for si, (x, y) in enumerate(sites):
            grid[y, x] = field[si, k]
        im = ax.imshow(grid, origin="lower", cmap=spec["cmap"], vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {dyn['times'][k]:.2f}/t", fontsize=11)
        ax.set_xticks(range(Lx)); ax.set_yticks(range(Ly))
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(False)
        for si, (x, y) in enumerate(sites):
            val = field[si, k]
            frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            ax.text(x, y, f"{val:.2f}", ha="center", va="center",
                    color=_label_colour(frac, spec["cmap"]), fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, label=spec["cbar"])

    sub = f"{Lx}x{Ly} periodic"
    if U is not None:
        sub += f",  U/t = {U:.0f}"
    if initial_state:
        sub += f",  initial state: {initial_state}"
    fig.suptitle(f"{spec['title']}\n({sub})", fontsize=12)

    _ensure_dir(save_dir)
    full = os.path.join(save_dir, fname)
    fig.savefig(full, bbox_inches="tight")
    plt.close(fig)
    return full


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
