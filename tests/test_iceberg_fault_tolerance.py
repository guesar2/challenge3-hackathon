"""
test_iceberg_fault_tolerance.py

Exhaustive fault-injection verification of the Iceberg code's
initialisation and syndrome-measurement circuits (iceberg_circuits.py),
mirroring the paper's own method (arXiv:2211.06703 Supp. Info / Supp.
Fig. 1): for every single-qubit-gate location, inject every nontrivial
single-qubit Pauli fault; for every two-qubit-gate (CX) location, inject
every nontrivial two-qubit Pauli fault; propagate to the end of the
circuit; and check the ACTUAL fault-tolerance criterion (Gottesman's,
confirmed directly with the Iceberg paper's author): whenever the
circuit's own flags don't fire, the result must be either exactly correct
or guaranteed to flip S_X or S_Z on some *future* check -- not necessarily
caught by this exact round's own ancillas, and not necessarily small. A
weight-3 error that's certain to flip S_Z next time is fine; a weight-2
error that stays silently within the code space forever is not. See
circuit_verification.eventually_detected_or_correct.

(An earlier version of this suite used a stricter "weight <= 1" criterion,
which rejected valid circuits -- kept as matches_ideal_up_to_weight1_pauli
in circuit_verification.py for reference/diagnostics only, not asserted
against here.)

This is the actual correctness authority for iceberg_circuits.py's
init/syndrome-measurement circuits -- there was no access to the paper's
own circuit diagrams (see docs/ICEBERG_QEC_PLAN.md), so these circuits
were built from first principles and are only trusted because they pass
this exhaustive check, the same way the original paper verified its own
(different) circuits.
"""
import numpy as np
import pytest

from iceberg_code import ghz_all_zero_statevector, stabilizer_sx, stabilizer_sz, validate_k
from iceberg_circuits import build_iceberg_init, build_iceberg_syndrome_measurement

from circuit_verification import (
    ONE_QUBIT_PAULI_FAULTS,
    TWO_QUBIT_PAULI_FAULTS,
    eventually_detected_or_correct,
    project,
)

K_VALUES = (2, 4, 6)


def _circuit_from_commands(n_qubits, commands, fault_after_index=None, fault_qubits=(), fault_paulis=()):
    """Rebuild an n_qubits-qubit circuit by replaying an already-fetched
    `commands` list (see module docstring: pytket's get_commands() reorders
    gates by qubit dependency, not insertion order, so it must be fetched
    exactly ONCE per circuit and reused everywhere -- calling it again
    after rebuilding, or on a differently-sized circuit, is not guaranteed
    to return the same order, silently pointing "after_command_index" at
    the wrong gate. Injects the given Pauli fault(s) right after the
    command at fault_after_index (an index into this exact `commands`
    list), if given."""
    from pytket import Circuit as _C

    new_circuit = _C(n_qubits)
    for idx, cmd in enumerate(commands):
        new_circuit.add_gate(cmd.op.type, cmd.op.params, cmd.args)
        if idx == fault_after_index:
            for q, p in zip(fault_qubits, fault_paulis):
                if p == 'x':
                    new_circuit.X(q)
                elif p == 'y':
                    new_circuit.Y(q)
                elif p == 'z':
                    new_circuit.Z(q)
                elif p == 'i':
                    pass
    return new_circuit


def _gate_locations(commands, circuit_for_qubit_lookup, only_qubits=None):
    """Return (index, qubit_indices, arity) for every H and CX command in
    an already-fetched `commands` list. If only_qubits is given, only
    commands touching at least one of those qubit indices are included
    (used to restrict fault injection to syndrome-measurement's own gates
    within a combined init+syndrome-measurement command list, without
    re-deriving indices from a separately-fetched command list -- see
    _circuit_from_commands's docstring for why that would be unsafe)."""
    locations = []
    for idx, cmd in enumerate(commands):
        qubits = [circuit_for_qubit_lookup.qubits.index(q) for q in cmd.args]
        if only_qubits is not None and not any(q in only_qubits for q in qubits):
            continue
        if cmd.op.type.name == 'H':
            locations.append((idx, qubits, 1))
        elif cmd.op.type.name == 'CX':
            locations.append((idx, qubits, 2))
    return locations


@pytest.mark.parametrize("k", K_VALUES)
def test_init_circuit_fault_tolerant(k):
    n = validate_k(k)
    ghz = ghz_all_zero_statevector(k)
    sx_op, sz_op = stabilizer_sx(k), stabilizer_sz(k)

    ideal_circuit, flag = build_iceberg_init(k)
    ideal_full_sv = ideal_circuit.get_statevector()
    prob0, ideal_data_sv = project(ideal_full_sv, n + 1, {flag: 0})
    assert prob0 > 1 - 1e-9  # no-fault case: flag must read 0 with certainty
    assert np.allclose(ideal_data_sv, ghz, atol=1e-9)

    # Fetch get_commands() exactly once and reuse it for both locating
    # gates and injecting faults (see _circuit_from_commands's docstring).
    commands = ideal_circuit.get_commands()

    failures = []
    for idx, qubits, arity in _gate_locations(commands, ideal_circuit):
        fault_set = ONE_QUBIT_PAULI_FAULTS if arity == 1 else TWO_QUBIT_PAULI_FAULTS
        for fault in fault_set:
            fault_paulis = [fault] if arity == 1 else list(fault)
            faulted = _circuit_from_commands(ideal_circuit.n_qubits, commands, idx, qubits, fault_paulis)
            sv = faulted.get_statevector()
            prob_accept, data_sv = project(sv, n + 1, {flag: 0})
            if data_sv is None:
                continue  # flag always fires for this fault -- safely caught
            ok, desc = eventually_detected_or_correct(data_sv, ghz, sx_op, sz_op)
            if not ok:
                failures.append((idx, qubits, fault, desc))

    assert not failures, f"k={k}: undetectable logical errors from faults: {failures[:5]}"


@pytest.mark.parametrize("k", K_VALUES)
def test_syndrome_measurement_fault_tolerant_on_logical_zero(k):
    """Fault-tolerance check for build_iceberg_syndrome_measurement,
    starting from the ideal |0bar> state (prepended, unfaulted): a single
    fault during syndrome extraction must, whenever a1=a2=0, leave the
    data register either exactly as it was, or guaranteed to flip S_X/S_Z
    on a future check (see module docstring -- this does NOT require this
    round's own a1/a2 to be the ones that caught it)."""
    n = validate_k(k)
    from pytket import Circuit as _C

    ghz = ghz_all_zero_statevector(k)
    sx_op, sz_op = stabilizer_sx(k), stabilizer_sz(k)

    # Build the full (n+1)+2-qubit-register circuit by prepending init's
    # own (unfaulted) ladder before syndrome measurement's gates, so
    # everything stays as native gates rather than a black-box state
    # injection. init's own flag qubit (index n) is left idle afterward --
    # its fault tolerance was already checked in test_init_circuit_fault_tolerant.
    init_circuit, init_flag = build_iceberg_init(k)
    syn_circuit, a1, a2 = build_iceberg_syndrome_measurement(k)

    total_qubits = n + 1 + 2  # n data + init's flag + a1,a2
    combined = _C(total_qubits)
    for cmd in init_circuit.get_commands():
        qs = [combined.qubits[init_circuit.qubits.index(q)] for q in cmd.args]
        combined.add_gate(cmd.op.type, cmd.op.params, qs)
    # syn_circuit's own qubits 0..n-1 map to combined's 0..n-1 (data);
    # syn_circuit's n,n+1 (a1,a2) map to combined's n+1,n+2 (shifted by 1
    # past init's flag qubit at index n).
    syn_qubit_map = {i: i for i in range(n)}
    syn_qubit_map.update({n: n + 1, n + 1: n + 2})
    for cmd in syn_circuit.get_commands():
        local_indices = [syn_circuit.qubits.index(q) for q in cmd.args]
        qs = [combined.qubits[syn_qubit_map[i]] for i in local_indices]
        combined.add_gate(cmd.op.type, cmd.op.params, qs)

    combined_init_flag = n
    combined_a1, combined_a2 = n + 1, n + 2
    fixed_accept = {combined_init_flag: 0, combined_a1: 0, combined_a2: 0}

    ideal_full_sv = combined.get_statevector()
    prob_accept, ideal_data_sv = project(ideal_full_sv, total_qubits, fixed_accept)
    assert prob_accept > 1 - 1e-9
    assert np.allclose(ideal_data_sv, ghz, atol=1e-9)

    # Fetch get_commands() exactly once on `combined` itself and reuse it
    # for both locating gates and injecting faults -- computing indices
    # from syn_circuit's own (differently-ordered) command list and
    # reapplying them to combined's is NOT safe, since pytket reorders
    # commands by qubit dependency, not insertion order, and that order
    # depends on the whole circuit's structure, not just a sub-block (see
    # _circuit_from_commands's docstring; this was a real bug caught by
    # inspecting the actual gate landed on for k=6 and finding it was an
    # unrelated init-ladder CX, not the intended syndrome-measurement gate).
    commands = combined.get_commands()
    syndrome_qubits = {combined_a1, combined_a2}

    failures = []
    for idx, qubits, arity in _gate_locations(commands, combined, only_qubits=syndrome_qubits):
        fault_set = ONE_QUBIT_PAULI_FAULTS if arity == 1 else TWO_QUBIT_PAULI_FAULTS
        for fault in fault_set:
            fault_paulis = [fault] if arity == 1 else list(fault)
            faulted = _circuit_from_commands(total_qubits, commands, idx, qubits, fault_paulis)
            sv = faulted.get_statevector()
            prob, data_sv = project(sv, total_qubits, fixed_accept)
            if data_sv is None:
                continue
            ok, desc = eventually_detected_or_correct(data_sv, ghz, sx_op, sz_op)
            if not ok:
                failures.append((idx, qubits, fault, desc))

    assert not failures, f"k={k}: undetectable logical errors from faults: {failures[:5]}"
