"""
ed_figures.py

Figure generation for the ED / classical-Trotter TFIM pipeline.
Kept separate from exact_diagonalization.py so the physics module stays readable.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_figures(ed_results, trotter_results, config, figures_dir):
    """Write the four challenge figures (png + pdf) to figures_dir."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    figsize = (10, 6)
    N_values = config["N_values"]
    h_values = config["h_over_J_values"]
    dt_values = config["dt_values"]

    # Fig 1: ⟨X⟩ vs h/J (ED ground state).
    # ⟨Z⟩ vanishes by Z2 symmetry in finite ED without a longitudinal field;
    # ⟨X⟩ is the observable that tracks the ferro → para crossover.
    fig, ax = plt.subplots(figsize=figsize)
    for N in N_values:
        mag_x_vals = [ed_results[N][h]["mag_x"] for h in h_values]
        ax.plot(h_values, mag_x_vals, "-o", label=f"N = {N}", markersize=8)
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.7, label="Transición h/J = 1")
    ax.set_xlabel("h / J", fontsize=12)
    ax.set_ylabel(r"$\langle X \rangle$", fontsize=12)
    ax.set_title("Magnetización en X vs. campo transverso (ED)", fontsize=14)
    ax.legend()
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_magnetization_ed.png", dpi=300)
    fig.savefig(figures_dir / "fig1_magnetization_ed.pdf")
    plt.close(fig)
    print("[✓] Figura 1 guardada: fig1_magnetization_ed.png")

    # Fig 2: ⟨Z(t)⟩ ED vs classical Trotter (N=8, h/J=1)
    N, h_over_J = 8, 1.0
    fig, ax = plt.subplots(figsize=figsize)
    ed_data = ed_results[N][h_over_J]
    ax.plot(ed_data["times"], ed_data["mag_z_t"], "k-", linewidth=2, label="ED (exacto)")
    if N in trotter_results and h_over_J in trotter_results[N]:
        for dt in dt_values:
            trot = trotter_results[N][h_over_J][dt]
            ax.plot(trot["times"], trot["mag_z_t"], "--", alpha=0.7,
                    label=f"Trotter Δt = {dt}")
    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel(r"$\langle Z(t) \rangle$", fontsize=12)
    ax.set_title(f"Evolución temporal (N={N}, h/J={h_over_J})", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_time_evolution.png", dpi=300)
    fig.savefig(figures_dir / "fig2_time_evolution.pdf")
    plt.close(fig)
    print("[✓] Figura 2 guardada: fig2_time_evolution.png")

    # Fig 3: Trotter error vs Δt
    fig, ax = plt.subplots(figsize=figsize)
    for N in [6, 8]:
        if N not in trotter_results:
            continue
        for h_over_J in h_values:
            dts, errors = [], []
            for dt in dt_values:
                if dt in trotter_results[N][h_over_J]:
                    dts.append(dt)
                    errors.append(trotter_results[N][h_over_J][dt]["error_vs_ed"])
            ax.loglog(dts, errors, "-o", label=f"N={N}, h/J={h_over_J}")
    ref_dt = np.array(dt_values)
    ax.loglog(ref_dt, ref_dt ** 2 * 0.1, "k--", alpha=0.5, label=r"O(Δt²) referencia")
    ax.set_xlabel("Δt", fontsize=12)
    ax.set_ylabel("Error relativo vs ED", fontsize=12)
    ax.set_title("Convergencia del error de Trotter", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_trotter_convergence.png", dpi=300)
    fig.savefig(figures_dir / "fig3_trotter_convergence.pdf")
    plt.close(fig)
    print("[✓] Figura 3 guardada: fig3_trotter_convergence.png")

    # Fig 4: ZZ correlations vs distance
    N = 8
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, h_over_J in enumerate(h_values):
        corr = np.asarray(ed_results[N][h_over_J]["corr_zz"])
        distances = np.arange(len(corr))
        axes[idx].bar(distances, corr, color="steelblue", edgecolor="black")
        axes[idx].set_xlabel("Distancia |i-j|", fontsize=11)
        axes[idx].set_title(f"h/J = {h_over_J}", fontsize=12)
        axes[idx].set_ylim(-1.1, 1.1)
    axes[0].set_ylabel(r"$\langle Z_i Z_j \rangle$", fontsize=12)
    fig.suptitle(f"Correlaciones espaciales (N={N}, ED)", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_correlations.png", dpi=300)
    fig.savefig(figures_dir / "fig4_correlations.pdf")
    plt.close(fig)
    print("[✓] Figura 4 guardada: fig4_correlations.png")

    print(f"\n[✓] Todas las figuras guardadas en {figures_dir}/")
