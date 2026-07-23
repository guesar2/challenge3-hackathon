"""
fh_tables.py

The summary tables: one table per U value, each holding <N>, <D> and m_stag with
ED as the reference and the deviations of the other two engines rowed underneath.

WHAT IS BEING COMPARED, AND WHY IT NEEDS TWO REFERENCE ROWS
-----------------------------------------------------------
ED, Trotter and VQE do not all compute the same object, so a single reference
row would be physically meaningless:

  * VQE     produces a variational approximation to the GROUND STATE.
  * Trotter produces an approximation to the time-evolved QUENCH state.
  * ED      can produce both, exactly.

So each table carries two exact ED rows -- the ground state and the quench state
at the final time -- and each approximate engine is compared against the one it
is actually approximating. Mixing them (e.g. comparing a VQE ground state to an
ED quench state) would produce a number with no meaning.

TWO HONEST CAVEATS THAT THE TABLES MAKE VISIBLE
-----------------------------------------------
1. m_stag vanishes IDENTICALLY in the ground state of any finite cluster. The
   Hubbard Hamiltonian is SU(2) symmetric and a finite system cannot
   spontaneously break that symmetry, so <S^z_i> = 0 site by site. A non-zero
   m_stag in the ground-state block would be a bug, not antiferromagnetism. The
   VQE row does show a non-zero value, because its ansatz starts from a Neel
   reference and never restores the symmetry -- that is a real, reportable
   limitation of the ansatz, not of the code. To measure antiferromagnetic order
   in a finite ground state properly you need the staggered structure factor
   S(pi,pi) = (1/N) sum_ij (-1)^(i-j) <S^z_i S^z_j>, which does not vanish.
   The quench block is where m_stag IS meaningful: the initial Neel product
   state breaks the symmetry by hand, and we watch that order decay.

2. At U=0 the half-filled ground state can be an OPEN SHELL. On 2x2 the
   single-particle levels are -2t, 0, 0, +2t, so half filling leaves the zero
   mode partly filled and the ground state is 4-fold degenerate. <D> and m_stag
   are then properties of a degenerate MANIFOLD, not of one vector, so the ED
   row reports the (basis-independent) ensemble average and the table flags the
   degeneracy. A variational method will land on one particular member of the
   manifold, so a non-zero deviation there is expected and is not an error.
"""
from __future__ import annotations

import os

import numpy as np

import fh_config as cfg
from fh_lattice import HubbardLattice
from fh_exact_diagonalization import ed_ground_state, ed_time_evolution
from fh_trotter_simulation import trotter_time_evolution
from fh_vqe import run_vqe_local
import fh_persistence as persistence

OBSERVABLES = (
    ("<N>", "total_particles"),
    ("<D>", "avg_double_occupancy"),
    ("m_stag", "staggered_magnetization"),
)

_ABS_TOL = 1e-6      # below this the reference counts as "zero"


def _deviation(value, reference):
    """Return (text, kind). Percent deviation when the reference is non-zero,
    absolute difference otherwise (so we never divide by ~0 for m_stag)."""
    diff = value - reference
    if abs(reference) > _ABS_TOL:
        return f"{abs(diff) / abs(reference) * 100:8.2f} %", "pct"
    return f"{diff:+8.2e} abs", "abs"


def _collect_for_U(lat, t, U, dt, steps, order, init, vqe_layers, vqe_restarts,
                   vqe_maxiter, seed, verbose=True):
    """Run all three engines at one U and return the raw numbers."""
    if verbose:
        print(f"\n  [U/t={U/t:.0f}] ED ground state ...")
    gs = ed_ground_state(lat, t, U, verbose=verbose)

    if verbose:
        print(f"  [U/t={U/t:.0f}] ED + Trotter quench ...")
    ed_dyn = ed_time_evolution(lat, t, U, dt, steps, initial_state=init)
    tr_dyn = trotter_time_evolution(lat, t, U, dt, steps, initial_state=init,
                                    order=order)

    if verbose:
        print(f"  [U/t={U/t:.0f}] VQE ...")
    vqe = run_vqe_local(lat, t, U, layers=vqe_layers, restarts=vqe_restarts,
                        maxiter=vqe_maxiter, seed=seed, verbose=verbose)

    k = steps - 1
    return {
        "U": U,
        "degeneracy": gs.get("degeneracy", 1),
        "ed_ground": {
            "total_particles": gs["total_particles"],
            "avg_double_occupancy": gs["avg_double_occupancy"],
            "staggered_magnetization": gs["staggered_magnetization"],
        },
        "ed_quench": {
            "total_particles": float(lat.n_sites),   # exactly conserved
            "avg_double_occupancy": float(ed_dyn["avg_double_occupancy"][k]),
            "staggered_magnetization": float(ed_dyn["staggered_magnetization"][k]),
        },
        "trotter_quench": {
            "total_particles": float(lat.n_sites),
            "avg_double_occupancy": float(tr_dyn["avg_double_occupancy"][k]),
            "staggered_magnetization": float(tr_dyn["staggered_magnetization"][k]),
        },
        "vqe_ground": {
            "total_particles": vqe["vqe_particles"],
            "avg_double_occupancy": vqe["vqe_double_occ"],
            "staggered_magnetization": vqe["vqe_m_stag"],
        },
        "vqe_energy": vqe["energy_vqe"],
        "ed_energy": vqe["energy_ed"],
        "vqe_energy_error_percent": vqe["error_percent"],
        "vqe_layers": vqe_layers,
    }


def format_table(row, lat, t, dt, steps, order, init):
    """Render one U's numbers as a fixed-width text table."""
    U = row["U"]
    T = dt * steps
    w = 14
    head = f"{'':34s}" + "".join(f"{name:>{w}s}" for name, _ in OBSERVABLES)
    sep = "-" * len(head)
    lines = []
    lines.append("=" * len(head))
    lines.append(f" U/t = {U/t:.0f}     {lat.Lx}x{lat.Ly} periodic, half filling"
                 f"     (Trotter order {order}, dt={dt}, t={T:.2f}/t, "
                 f"init={init}, VQE p={row['vqe_layers']})")
    lines.append("=" * len(head))
    lines.append(head)
    lines.append(sep)

    def val_row(label, d):
        return f" {label:33s}" + "".join(
            f"{d[key]:>{w}.6f}" for _, key in OBSERVABLES)

    def dev_row(label, d, ref):
        cells = ""
        for _, key in OBSERVABLES:
            txt, _kind = _deviation(d[key], ref[key])
            cells += f"{txt:>{w}s}"
        return f" {label:33s}" + cells

    lines.append(val_row("ED  ground state      (exact)", row["ed_ground"]))
    lines.append(dev_row("   VQE      deviation", row["vqe_ground"], row["ed_ground"]))
    lines.append(sep)
    lines.append(val_row(f"ED  quench @ t={T:.2f}/t  (exact)", row["ed_quench"]))
    lines.append(dev_row("   Trotter  deviation", row["trotter_quench"], row["ed_quench"]))
    lines.append(sep)
    lines.append(f" ground-state energy:  ED = {row['ed_energy']:+.6f}   "
                 f"VQE = {row['vqe_energy']:+.6f}   "
                 f"({row['vqe_energy_error_percent']:.2f} %)")

    notes = []
    if row["degeneracy"] > 1:
        notes.append(f"ED ground state is {row['degeneracy']}-fold degenerate at "
                     f"this U; ED values are the ensemble average over that "
                     f"manifold, so the VQE deviation is not a pure accuracy "
                     f"measure.")
    notes.append("m_stag = 0 in the exact ground state is required by SU(2) "
                 "symmetry on a finite cluster, not a numerical accident; the "
                 "VQE row is non-zero because its Neel-referenced ansatz does "
                 "not restore that symmetry.")
    for n in notes:
        for i, chunk in enumerate(_wrap(n, len(head) - 4)):
            lines.append(("  note: " if i == 0 else "        ") + chunk)
    lines.append("")
    return "\n".join(lines)


def _wrap(text, width):
    words, out, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width:
            out.append(cur); cur = wd
        else:
            cur = (cur + " " + wd).strip()
    if cur:
        out.append(cur)
    return out


def run(U_values=None, save=True, verbose=True):
    """Build and print one summary table per U value."""
    U_values = list(U_values or cfg.U_VALUES)
    Lx, Ly = cfg.TABLE_LATTICE
    lat = HubbardLattice(Lx, Ly)
    t = cfg.T_HOP
    dt, steps = cfg.QUENCH_DT, cfg.QUENCH_STEPS
    order, init = cfg.TROTTER_ORDER, cfg.QUENCH_INITIAL_STATE

    print(f"Summary tables on {lat} -- ED reference, Trotter and VQE deviations")
    rows = []
    for U in U_values:
        rows.append(_collect_for_U(
            lat, t, U, dt, steps, order, init,
            vqe_layers=cfg.TABLE_VQE_LAYERS,
            vqe_restarts=cfg.TABLE_VQE_RESTARTS,
            vqe_maxiter=cfg.VQE_MAXITER_LOCAL,
            seed=cfg.VQE_SEED, verbose=verbose))

    blocks = [format_table(r, lat, t, dt, steps, order, init) for r in rows]
    text = "\n".join(blocks)
    print("\n" + text)

    if save:
        persistence.save_stage_results("summary_tables", {
            "lattice": [Lx, Ly], "t": t, "dt": dt, "steps": steps,
            "order": order, "initial_state": init, "rows": rows})
        os.makedirs(cfg.PLOT_SAVE_DIR, exist_ok=True)
        path = os.path.join(cfg.PLOT_SAVE_DIR, "summary_tables.txt")
        with open(path, "w") as f:
            f.write(text)
        print(f"  wrote {path}")
    return rows


if __name__ == "__main__":
    run()
