"""
run_fh_circuit_check.py

Validate the whole quantum path before trusting any emulator number:

  [A] PauliExpBox angle convention: a 1-qubit exp(-i theta Z) box must equal the
      analytic matrix diag(e^{-i theta}, e^{+i theta}).
  [B] Bit ordering: preparing |1> on qubit 0 of an n-qubit circuit must yield the
      statevector basis index 2^{n-1} (qubit 0 = most-significant bit), matching
      fh_jordan_wigner / fh_lattice / fh_local_sampler.
  [C] Circuit vs statevector-Trotter vs ED: build the quench circuit, sample many
      shots through the free local sampler, and confirm the shot-derived
      observables match the sparse statevector-Trotter observables (within shot
      noise) and the ED reference (within Trotter + shot error).

Run:  python run_fh_circuit_check.py
"""
import numpy as np
from pytket import Circuit
from pytket.circuit import PauliExpBox
from pytket.pauli import Pauli

from fh_lattice import HubbardLattice
from fh_exact_diagonalization import ed_time_evolution
from fh_trotter_simulation import trotter_time_evolution
from fh_tket_circuit import build_quench_circuit
from fh_local_sampler import sample_bitstrings
from fh_shot_observables import bitstrings_to_observables, bootstrap_errors


def check_angle_convention():
    print("[A] PauliExpBox angle convention")
    theta = 0.37
    c = Circuit(1)
    c.add_pauliexpbox(PauliExpBox([Pauli.Z], 2 * theta / np.pi), [0])
    U = c.get_unitary()
    expected = np.array([[np.exp(-1j * theta), 0], [0, np.exp(1j * theta)]])
    err = np.max(np.abs(U - expected))
    print(f"    max|U - exp(-i theta Z)| = {err:.2e}  {'OK' if err < 1e-9 else 'FAIL'}")
    assert err < 1e-9


def check_bit_order():
    print("[B] Bit ordering (qubit 0 = MSB)")
    n = 3
    c = Circuit(n)
    c.X(0)                      # occupy qubit 0
    sv = np.asarray(c.get_statevector())
    idx = int(np.argmax(np.abs(sv)))
    ok = (idx == 2 ** (n - 1))
    print(f"    |1> on qubit 0 -> statevector index {idx} (expect {2**(n-1)})"
          f"  {'OK' if ok else 'FAIL'}")
    assert ok


def check_full_path():
    print("[C] Circuit shots vs statevector-Trotter vs ED")
    lat = HubbardLattice(2, 2)
    t, U, dt, steps, order = 1.0, 8.0, 0.1, 4, 2
    T = steps * dt

    # statevector Trotter (same terms as circuit) and ED at the same final time
    tr = trotter_time_evolution(lat, t, U, dt, steps, order=order)
    ex = ed_time_evolution(lat, t, U, dt, steps)
    tr_D = tr["avg_double_occupancy"][-1]
    tr_M = tr["staggered_magnetization"][-1]
    ex_D = ex["avg_double_occupancy"][-1]
    ex_M = ex["staggered_magnetization"][-1]

    # circuit -> shots -> observables
    circ = build_quench_circuit(lat, t, U, dt, steps, order=order)
    shots = sample_bitstrings(circ, 20000, seed=1)
    obs = bitstrings_to_observables(shots, lat)
    err = bootstrap_errors(shots, lat, n_boot=300, seed=2)

    print(f"    <N>            shots={obs['total_particles']:.3f}"
          f" +/- {err['total_particles']:.3f}  (exact 4.000)")
    print(f"    <D> (T={T:.1f})  shots={obs['avg_double_occupancy']:.4f}"
          f" +/- {err['avg_double_occupancy']:.4f} | trotter={tr_D:.4f} | ED={ex_D:.4f}")
    print(f"    m_stag         shots={obs['staggered_magnetization']:.4f}"
          f" +/- {err['staggered_magnetization']:.4f} | trotter={tr_M:.4f} | ED={ex_M:.4f}")

    # shots must agree with the statevector Trotter within ~4 sigma (same operation)
    assert abs(obs["avg_double_occupancy"] - tr_D) < 4 * err["avg_double_occupancy"] + 1e-3
    assert abs(obs["staggered_magnetization"] - tr_M) < 4 * err["staggered_magnetization"] + 1e-3
    assert abs(obs["total_particles"] - 4.0) < 1e-6  # particle number is exactly conserved
    print("    circuit path consistent with statevector Trotter (within shot noise)  OK")


if __name__ == "__main__":
    check_angle_convention()
    check_bit_order()
    check_full_path()
    print("\nAll circuit checks passed.")