"""
fh_tket_circuit.py

pytket circuit builders for the 2D Fermi-Hubbard model, for execution on
Quantinuum's H2 emulator (via qnexus or the local pytket-quantinuum backend) or
via the free local statevector sampler (fh_local_sampler.py).

Every gate comes from the SAME Jordan-Wigner Pauli-term list used by the
classical statevector Trotter (fh_trotter_simulation.py) and the ED Hamiltonian
(fh_jordan_wigner.py), so the circuit is provably the same operation -- there is
no separate, drift-prone re-derivation of the circuit.

Each Hamiltonian term exp(-i theta P) is realised with a pytket PauliExpBox.
pytket's PauliExpBox(paulis, phase) implements

        exp(-i * (pi/2) * phase * P),

so to get exp(-i theta P) we pass phase = 2*theta/pi. (Verified numerically in
run_fh_circuit_check against exp(-i theta Z).)

All observables the challenge asks for -- density per site, double occupancy,
magnetization -- are DIAGONAL in the computational (Z) basis (n = (1-Z)/2), so a
single Z-basis measure_all() suffices; no extra measurement bases are needed
(unlike the TFIM, whose <X> required an X-basis circuit).
"""
from __future__ import annotations

import math

from pytket import Circuit
from pytket.circuit import PauliExpBox
from pytket.pauli import Pauli

from fh_lattice import HubbardLattice
import fh_jordan_wigner as jw

_PAULI_MAP = {"X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}


def _add_term(circuit: Circuit, pauli_dict, theta):
    """Append exp(-i theta P) for a single Pauli string to `circuit`."""
    if theta == 0.0 or not pauli_dict:
        return
    qubits = sorted(pauli_dict.keys())
    paulis = [_PAULI_MAP[pauli_dict[q]] for q in qubits]
    phase = 2.0 * theta / math.pi  # PauliExpBox uses exp(-i pi/2 * phase * P)
    box = PauliExpBox(paulis, phase)
    circuit.add_pauliexpbox(box, qubits)


def _ordered_terms(lat: HubbardLattice, t: float, U: float):
    """Same ordered (pauli_dict, coeff) list as the statevector Trotter:
    hopping block first, on-site block last."""
    hop, onsite, _ = jw.hubbard_pauli_terms(lat, t, U, mu=0.0, include_constant=False)
    return hop + onsite


def _append_trotter_step(circuit, terms, dt, order):
    if order == 1:
        for pauli, c in terms:
            _add_term(circuit, pauli, c * dt)
    elif order == 2:
        for pauli, c in terms:
            _add_term(circuit, pauli, c * dt / 2)
        for pauli, c in reversed(terms):
            _add_term(circuit, pauli, c * dt / 2)
    else:
        raise ValueError("order must be 1 or 2")


def _prepare_initial_state(circuit, lat, initial_state):
    if initial_state == "neel":
        occ = lat.neel_occupation()
    elif initial_state == "stripe":
        occ = lat.stripe_occupation()
    else:
        occ = [int(b) for b in initial_state]
    for q, b in enumerate(occ):
        if b:
            circuit.X(q)


def build_quench_circuit(lat: HubbardLattice, t, U, dt, steps, initial_state="neel",
                          order=2, measure=True):
    """Trotterised fixed-Hamiltonian quench circuit on 2*n_sites qubits, from a
    product initial state, measured in the Z basis (all requested observables
    are Z-diagonal)."""
    n = lat.n_qubits
    circuit = Circuit(n, n)
    _prepare_initial_state(circuit, lat, initial_state)
    terms = _ordered_terms(lat, t, U)
    for _ in range(steps):
        _append_trotter_step(circuit, terms, dt, order)
    if measure:
        circuit.measure_all()
    return circuit


def build_quench_ansatz_circuit(lat: HubbardLattice, t, U, dt, steps,
                                 initial_state="neel", order=2, decompose_boxes=False):
    """Identical state preparation and Trotter layers to build_quench_circuit,
    but UNMEASURED (no classical register at all).

    Needed for zero-noise extrapolation: qermit's Folding.circuit amplifies noise
    by folding a circuit into C (C^-1 C)^k, which it builds by inverting gates
    (op.dagger). A measurement is not invertible, so folding has to act on this
    bare ansatz, with the Z-basis measurement appended to each already-folded
    copy afterwards (see append_z_measurement and
    fh_qnexus_backend.submit_zne_batch).

    decompose_boxes: run pytket's DecomposeBoxes pass, turning every PauliExpBox
    into primitive gates (CX / Rz / H / ...). REQUIRED before folding -- qermit
    raises "RuntimeError: Box types not supported when folding." otherwise, since
    it cannot dagger a composite op. It is a pure re-expression of the same
    unitary (the pass is part of what qnx.compile() does on the way to H2's
    native gateset), so nothing about the physics changes; it is off by default
    only because the boxed form is more compact for local statevector work.
    """
    n = lat.n_qubits
    circuit = Circuit(n)
    _prepare_initial_state(circuit, lat, initial_state)
    terms = _ordered_terms(lat, t, U)
    for _ in range(steps):
        _append_trotter_step(circuit, terms, dt, order)
    if decompose_boxes:
        from pytket.passes import DecomposeBoxes
        DecomposeBoxes().apply(circuit)
    return circuit


def append_z_measurement(circuit: Circuit):
    """Measure every qubit in the computational (Z) basis, adding the classical
    bits if the circuit has none (the case for build_quench_ansatz_circuit).

    Every observable this project reports -- per-site density, double occupancy,
    staggered magnetization -- is Z-diagonal, so this single basis is sufficient
    and no X-basis companion circuit is needed (unlike the TFIM's <X>).
    """
    circuit.measure_all()
    return circuit


def build_hva_ansatz_circuit(lat: HubbardLattice, t, U, params, initial_state="neel"):
    """Number-conserving, bond/site-resolved Hamiltonian-Variational-Ansatz
    circuit for VQE (no measurement appended).

    Structure (p = len(params) // n_groups layers): each layer applies one shared
    angle per (spin,bond) hop group and one shared angle per site interaction
    group, using the generators from fh_jordan_wigner.hva_generators. Sharing the
    angle within a hop group (its XX and YY halves) keeps particle number
    conserved; independent angles across groups break the Neel reference's
    symmetry so the ground state is reachable. Starts from the half-filling Neel
    product state.

    params layout: for each layer l, [hop angles (one per hop group)] then
    [int angles (one per site)]; concatenated over layers. So
    len(params) = n_layers * (n_hop_groups + n_int_groups).
    """
    n = lat.n_qubits
    hop_groups, int_groups = jw.hva_generators(lat, t, U)
    per_layer = len(hop_groups) + len(int_groups)
    if len(params) % per_layer != 0:
        raise ValueError(f"len(params) must be a multiple of {per_layer}")
    n_layers = len(params) // per_layer

    circuit = Circuit(n)
    _prepare_initial_state(circuit, lat, initial_state)
    k = 0
    for _ in range(n_layers):
        for group in hop_groups:
            theta = params[k]; k += 1
            for pauli, c in group:
                _add_term(circuit, pauli, theta * c)
        for group in int_groups:
            theta = params[k]; k += 1
            for pauli, c in group:
                _add_term(circuit, pauli, theta * c)
    return circuit


def hva_n_params_per_layer(lat: HubbardLattice):
    """Number of variational angles in one HVA layer for this lattice."""
    hop_groups, int_groups = jw.hva_generators(lat, 1.0, 1.0)
    return len(hop_groups) + len(int_groups)