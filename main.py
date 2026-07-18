#!/usr/bin/env python3
"""
Quantathon CR 2026 · Challenge 3
Punto de entrada único para reproducir TODAS las figuras y cifras reportadas.

Ejecución:
    python main.py

Este script ejecuta secuencialmente:
    1. Diagonalización exacta (ED) del TFIM para N = {4, 6, 8}
    2. Simulación trotterizada con Qiskit Aer (statevector, sin ruido)
    3. Comparación cuántico vs. clásico
    4. Generación de figuras en /figures/
    5. Exportación de datos en /data/

Autor: [Tu nombre]
Fecha: 2026-07-18
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Asegurar que src/ esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from exact_diagonalization import exact_diagonalization_tfim, evolve_state_ed
from trotter_circuit import trotter_evolution_qiskit
from observables import compute_magnetization, compute_correlations

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

CONFIG = {
    "N_values": [4, 6, 8],           # Tamaños de cadena a simular
    "h_over_J_values": [0.5, 1.0, 2.0],  # Puntos del barrido h/J
    "J": 1.0,                         # Acoplamiento (unidad de energía)
    "dt_values": [0.1, 0.05, 0.025],  # Pasos de Trotter para convergencia
    "t_max": 5.0,                     # Tiempo máximo de evolución
    "boundary": "open",               # Condiciones de frontera: "open" o "periodic"
    "random_seed": 42,
}

# Directorios de salida
DATA_DIR = Path("data")
FIGURES_DIR = Path("figures")
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

np.random.seed(CONFIG["random_seed"])


def run_exact_diagonalization():
    """Fase 1: Línea base clásica — Diagonalización exacta."""
    print("=" * 60)
    print("FASE 1: DIAGONALIZACIÓN EXACTA (ED)")
    print("=" * 60)

    ed_results = {}

    for N in CONFIG["N_values"]:
        print(f"\n--- N = {N} espines ---")
        ed_results[N] = {}

        for h_over_J in CONFIG["h_over_J_values"]:
            h = h_over_J * CONFIG["J"]
            print(f"  h/J = {h_over_J:.1f} (h = {h:.2f})")

            # Estado fundamental
            E0, psi0, H = exact_diagonalization_tfim(
                N=N,
                J=CONFIG["J"],
                h=h,
                boundary=CONFIG["boundary"],
            )

            # Observables en el estado fundamental
            mag_z = compute_magnetization(psi0, N, axis="z")
            mag_x = compute_magnetization(psi0, N, axis="x")
            corr_zz = compute_correlations(psi0, N, op="zz")

            # Evolución temporal desde estado producto |0...0>
            times = np.linspace(0, CONFIG["t_max"], 100)
            mag_z_t = []
            for t in times:
                psi_t = evolve_state_ed(H, psi0, t)
                mag_z_t.append(compute_magnetization(psi_t, N, axis="z"))

            ed_results[N][h_over_J] = {
                "E0": float(E0),
                "mag_z": float(mag_z),
                "mag_x": float(mag_x),
                "corr_zz": corr_zz.tolist(),
                "times": times.tolist(),
                "mag_z_t": np.array(mag_z_t).tolist(),
            }

            print(f"    E0 = {E0:.6f}")
            print(f"    <Z> = {mag_z:.6f}, <X> = {mag_x:.6f}")

    # Guardar resultados
    np.savez(DATA_DIR / "ed_results.npz", **{f"N{k}": v for k, v in ed_results.items()})
    print(f"\n[✓] Resultados ED guardados en {DATA_DIR / 'ed_results.npz'}")

    return ed_results


def run_trotter_simulation(ed_results):
    """Fase 2: Simulación trotterizada con Qiskit (sin ruido)."""
    print("\n" + "=" * 60)
    print("FASE 2: SIMULACIÓN TROTTERIZADA")
    print("=" * 60)

    trotter_results = {}
    comparison = {}

    for N in [6, 8]:  # Comparar contra ED para N=6 (requisito del challenge)
        print(f"\n--- N = {N} espines ---")
        trotter_results[N] = {}
        comparison[N] = {}

        for h_over_J in CONFIG["h_over_J_values"]:
            h = h_over_J * CONFIG["J"]
            print(f"  h/J = {h_over_J:.1f}")

            dt_convergence = {}
            for dt in CONFIG["dt_values"]:
                print(f"    Δt = {dt:.3f} ...", end=" ")

                # Ejecutar evolución trotterizada
                result = trotter_evolution_qiskit(
                    N=N,
                    J=CONFIG["J"],
                    h=h,
                    dt=dt,
                    t_max=CONFIG["t_max"],
                    boundary=CONFIG["boundary"],
                    order=1,  # Primer orden Trotter-Suzuki
                )
                dt_convergence[dt] = result
                print(f"OK (error relativo vs ED: {result['error_vs_ed']:.4f})")

            trotter_results[N][h_over_J] = dt_convergence

            # Comparación contra ED para el Δt más fino
            best_dt = min(dt_convergence.keys())
            best_result = dt_convergence[best_dt]
            ed_ref = ed_results[N][h_over_J]

            comparison[N][h_over_J] = {
                "dt": best_dt,
                "mag_z_error": abs(best_result["mag_z"] - ed_ref["mag_z"]),
                "mag_x_error": abs(best_result["mag_x"] - ed_ref["mag_x"]),
                "relative_error": best_result["error_vs_ed"],
            }

    # Guardar resultados
    np.savez(DATA_DIR / "trotter_results.npz", **trotter_results)
    with open(DATA_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[✓] Resultados Trotter guardados en {DATA_DIR / 'trotter_results.npz'}")
    print(f"[✓] Comparación guardada en {DATA_DIR / 'comparison.json'}")

    return trotter_results, comparison


def generate_figures(ed_results, trotter_results, comparison):
    """Fase 3: Generación de figuras para el informe técnico."""
    print("\n" + "=" * 60)
    print("FASE 3: GENERACIÓN DE FIGURAS")
    print("=" * 60)

    plt.style.use("seaborn-v0_8-whitegrid")
    figsize = (10, 6)

    # ------------------------------------------------------------------
    # Figura 1: Magnetización <Z> vs h/J para diferentes N (ED)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    for N in CONFIG["N_values"]:
        h_vals = []
        mag_z_vals = []
        for h_over_J in CONFIG["h_over_J_values"]:
            h_vals.append(h_over_J)
            mag_z_vals.append(ed_results[N][h_over_J]["mag_z"])
        ax.plot(h_vals, mag_z_vals, "-o", label=f"N = {N}", markersize=8)

    ax.axvline(1.0, color="red", linestyle="--", alpha=0.7, label="Transición h/J = 1")
    ax.set_xlabel("h / J", fontsize=12)
    ax.set_ylabel(r"$\langle Z \rangle$", fontsize=12)
    ax.set_title("Magnetización en Z vs. campo transverso (ED)", fontsize=14)
    ax.legend()
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_magnetization_ed.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig1_magnetization_ed.pdf")
    print("[✓] Figura 1 guardada: fig1_magnetization_ed.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figura 2: Evolución temporal <Z(t)>: Trotter vs ED (h/J = 1.0, N=8)
    # ------------------------------------------------------------------
    N = 8
    h_over_J = 1.0
    fig, ax = plt.subplots(figsize=figsize)

    # ED
    ed_data = ed_results[N][h_over_J]
    ax.plot(ed_data["times"], ed_data["mag_z_t"], "k-", linewidth=2, label="ED (exacto)")

    # Trotter con diferentes Δt
    for dt in CONFIG["dt_values"]:
        if N in trotter_results and h_over_J in trotter_results[N]:
            trot_data = trotter_results[N][h_over_J][dt]
            ax.plot(trot_data["times"], trot_data["mag_z_t"], "--", alpha=0.7,
                    label=f"Trotter Δt = {dt}")

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel(r"$\langle Z(t) \rangle$", fontsize=12)
    ax.set_title(f"Evolución temporal (N={N}, h/J={h_over_J})", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_time_evolution.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig2_time_evolution.pdf")
    print("[✓] Figura 2 guardada: fig2_time_evolution.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figura 3: Error relativo vs Δt (convergencia Trotter)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    for N in [6, 8]:
        if N in trotter_results:
            for h_over_J in CONFIG["h_over_J_values"]:
                dts = []
                errors = []
                for dt in CONFIG["dt_values"]:
                    if dt in trotter_results[N][h_over_J]:
                        dts.append(dt)
                        errors.append(trotter_results[N][h_over_J][dt]["error_vs_ed"])
                ax.loglog(dts, errors, "-o", label=f"N={N}, h/J={h_over_J}")

    # Línea de referencia O(Δt²)
    ref_dt = np.array(CONFIG["dt_values"])
    ref_error = ref_dt**2 * 0.1  # factor arbitrario para visualización
    ax.loglog(ref_dt, ref_error, "k--", alpha=0.5, label="O(Δt²) referencia")

    ax.set_xlabel("Δt", fontsize=12)
    ax.set_ylabel("Error relativo vs ED", fontsize=12)
    ax.set_title("Convergencia del error de Trotter", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_trotter_convergence.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig3_trotter_convergence.pdf")
    print("[✓] Figura 3 guardada: fig3_trotter_convergence.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figura 4: Correlaciones <Z_i Z_j> vs distancia (h/J = 1.0, N=8)
    # ------------------------------------------------------------------
    N = 8
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, h_over_J in enumerate(CONFIG["h_over_J_values"]):
        corr = np.array(ed_results[N][h_over_J]["corr_zz"])
        distances = np.arange(len(corr))
        axes[idx].bar(distances, corr, color="steelblue", edgecolor="black")
        axes[idx].set_xlabel("Distancia |i-j|", fontsize=11)
        axes[idx].set_title(f"h/J = {h_over_J}", fontsize=12)
        axes[idx].set_ylim(-1.1, 1.1)
    axes[0].set_ylabel(r"$\langle Z_i Z_j \rangle$", fontsize=12)
    fig.suptitle(f"Correlaciones espaciales (N={N}, ED)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_correlations.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig4_correlations.pdf")
    print("[✓] Figura 4 guardada: fig4_correlations.png")
    plt.close(fig)

    print(f"\n[✓] Todas las figuras guardadas en {FIGURES_DIR}/")


def print_summary(comparison):
    """Resumen final de resultados."""
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"{'N':<5} {'h/J':<8} {'Δt':<10} {'Error rel.':<15} {'Estado'}")
    print("-" * 60)

    all_pass = True
    for N, h_data in comparison.items():
        for h_over_J, metrics in h_data.items():
            dt = metrics["dt"]
            err = metrics["relative_error"]
            status = "✓ PASS" if err < 0.05 else "✗ FAIL"
            if err >= 0.05:
                all_pass = False
            print(f"{N:<5} {h_over_J:<8.1f} {dt:<10.4f} {err:<15.6f} {status}")

    print("-" * 60)
    if all_pass:
        print("🎉 Todos los casos cumplen el criterio de < 5% de error vs ED!")
    else:
        print("⚠️  Algunos casos exceden el umbral del 5%. Revisar Δt o circuito.")


def main():
    """Pipeline principal de ejecución."""
    print("\n" + "█" * 60)
    print("  QUANTATHON CR 2026 · CHALLENGE 3")
    print("  Simulación TFIM — Punto de entrada único")
    print("█" * 60 + "\n")

    # Fase 1: ED
    ed_results = run_exact_diagonalization()

    # Fase 2: Trotter
    trotter_results, comparison = run_trotter_simulation(ed_results)

    # Fase 3: Figuras
    generate_figures(ed_results, trotter_results, comparison)

    # Resumen
    print_summary(comparison)

    print("\n" + "=" * 60)
    print("EJECUCIÓN COMPLETADA")
    print("=" * 60)
    print(f"Datos:   {DATA_DIR}/")
    print(f"Figuras: {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
