"""
iceberg_tfim_circuit.py

Encodes this repo's TFIM quench circuit (tket_circuit.build_quench_circuit)
with the Iceberg [[k+2,k,2]] error-detection code (iceberg_code.py,
iceberg_circuits.py): fault-tolerant initialisation, then for each Trotter
step the RZZ coupling half-step and RX field half-step EACH followed by a
syndrome-measurement round -- splitting every Trotter step into two and
measuring in between, per the paper author's own guidance on how often to
check, rather than only at full-step boundaries -- and finally one more
syndrome round (S_X) plus the destructive Z-basis measurement of every
code-register qubit (Fig. 1(e) of arXiv:2211.06703).

k plays the role N does in tket_circuit.py -- k logical qubits map 1:1
onto the TFIM's N spins (k must be even, matching every N/H2_* value
already used in config.py).

Design implication worth restating (see docs/ICEBERG_QEC_PLAN.md): the
RZZ half-step maps 1:1 onto the same physical qubit pairs the unencoded
circuit uses (Zbar_i Zbar_j = Z_iZ_j, Eq. 6, no t/b involvement, so
different-color edges stay parallel). The RX half-step does not: every
logical Xbar_i involves the *same* shared physical qubit t (Eq. 1), so a
Trotter layer's single parallel Rx layer becomes a sequential ladder of k
MS gates through t. This module also does not use tket_circuit.py's
Rx-fusion optimization (merging a step's trailing Rx with the next step's
leading Rx) -- a syndrome-measurement round sits between every half-step
by design, so there's no adjacent-Rx boundary left to fuse.

Ancillas (a1, a2) are reused across every syndrome round: measured into a
fresh pair of bits in the shared "flags" register each round, then Reset
before the next round touches them -- exactly as the paper's own Fig. 1(d)
describes ("two additional ancilla qubits, which can be reset and reused").

Real-time early exit (early_exit=True, the default): rather than running
every Trotter step to completion regardless and discarding in classical
post-processing only, a 1-bit "discard" register accumulates (via native
pytket classical logic, OR-ing in each round's a1/a2 outcomes -- no Wasm
needed, see docs/ICEBERG_QEC_PLAN.md's note on this) whether any round so
far has flagged an error, and every two-qubit gate from that point on
(remaining Trotter half-steps, remaining syndrome-round extractions) is
wrapped in condition=if_not_bit(discard) so it's skipped once a shot is
already doomed to be discarded -- the runtime saving the paper's own
discussion section mentions ("we expect this to halve the experimental
runtime of a rejected circuit on average"). Measure/Reset ops are left
unconditional (cheap, and keeps the classical bookkeeping simple); only
the two-qubit entangling work is gated.
"""
import math

from pytket import Circuit
from pytket.circuit import if_not_bit

from iceberg_code import validate_k
from iceberg_circuits import (
    build_iceberg_init,
    build_iceberg_syndrome_measurement,
    compile_logical_rx,
    compile_logical_rzz,
)


def _append_circuit(dest, src, qubit_map, condition=None):
    """Replay every command of `src` onto `dest`, remapping src's own
    qubit indices through `qubit_map` (a dict: src local index -> dest
    local index). If `condition` is given, every replayed gate is
    conditioned on it (used for native early-exit gating -- see module
    docstring); pytket's add_gate rejects condition=None outright, so
    it's only passed when not None.
    """
    kwargs = {} if condition is None else {"condition": condition}
    for cmd in src.get_commands():
        local_indices = [src.qubits.index(q) for q in cmd.args]
        qs = [dest.qubits[qubit_map[i]] for i in local_indices]
        dest.add_gate(cmd.op.type, cmd.op.params, qs, **kwargs)


def build_iceberg_quench_circuit(k, color_edges, steps, dt, h_field, J,
                                  initial_state_label=None, early_exit=True):
    """Build a fault-tolerant, Iceberg-encoded pytket Circuit implementing
    `steps` fixed-Hamiltonian Trotter layers on k logical qubits, with a
    syndrome-measurement round after each RZZ half-step and each RX
    half-step (2 rounds per Trotter step), then one final syndrome round
    and a destructive Z-basis measurement of every code-register qubit.

    initial_state_label: optional bitstring (e.g. "0110") preparing a
    computational basis state instead of |0...0> -- applied as logical
    Xbar_i pi-rotations right after initialisation, before any Trotter
    layer (mirrors tket_circuit.build_quench_circuit's parameter of the
    same name).

    early_exit: if True (default), gate every two-qubit operation from
    each syndrome round onward on a running "discard" bit so a shot
    already known to be rejected skips the rest of its entangling work in
    real time, rather than always running to completion and discarding
    only in classical post-processing (iceberg_decode.py) -- see module
    docstring. Set False for a plain post-selection-only circuit (e.g. to
    A/B-compare runtime, or if a target backend doesn't support
    mid-circuit classical-conditioned gates).

    Returns (circuit, metadata). metadata is a dict:
      - "n": k+2 (code-register size)
      - "n_rounds": total syndrome-measurement rounds (2*steps + 1)
      - "data_creg_name": classical register holding the final n-bit
        destructive measurement
      - "flags_creg_name": classical register holding every round's (a1,
        a2) outcomes, concatenated in round order (2 bits per round) --
        feed both registers' per-shot bitstrings into
        iceberg_decode.decode_shots.
      - "discard_creg_name": (only if early_exit) the running 1-bit
        discard flag -- purely informational/for early-exit gating, NOT
        a substitute for iceberg_decode.should_discard's own check on the
        full flags+data registers (an early-exited shot's LATER flag bits
        stay at their reset value of 0 and must not be misread as "no
        error" on their own).
    """
    n = validate_k(k)
    theta_x = -2 * h_field * dt
    theta_zz = -2 * J * dt

    a1, a2 = n, n + 1
    circuit = Circuit(n + 2)

    init_circuit, init_flag = build_iceberg_init(k)
    _append_circuit(circuit, init_circuit, {i: i for i in range(n)} | {n: a1})
    init_flag_creg = circuit.add_c_register("init_flag", 1)
    circuit.Measure(circuit.qubits[a1], init_flag_creg[0])
    circuit.Reset(circuit.qubits[a1])

    if initial_state_label:
        for i, bit in enumerate(initial_state_label):
            if bit == '1':
                compile_logical_rx(circuit, k, i, math.pi)

    n_rounds_total = 2 * steps + 1
    flags_creg = circuit.add_c_register("flags", 2 * n_rounds_total)
    round_counter = [0]

    discard_creg = None
    if early_exit:
        discard_creg = circuit.add_c_register("discard", 1)
        circuit.add_c_copybits([init_flag_creg[0]], [discard_creg[0]])

    def current_condition():
        return if_not_bit(discard_creg[0]) if early_exit else None

    def append_syndrome_round():
        syn_circuit, syn_a1, syn_a2 = build_iceberg_syndrome_measurement(k)
        syn_map = {i: i for i in range(n)} | {syn_a1: a1, syn_a2: a2}
        _append_circuit(circuit, syn_circuit, syn_map, condition=current_condition())
        r = round_counter[0]
        circuit.Measure(circuit.qubits[a1], flags_creg[2 * r])
        circuit.Measure(circuit.qubits[a2], flags_creg[2 * r + 1])
        circuit.Reset(circuit.qubits[a1])
        circuit.Reset(circuit.qubits[a2])
        if early_exit:
            circuit.add_clexpr_from_logicexp(
                discard_creg[0] | flags_creg[2 * r] | flags_creg[2 * r + 1], [discard_creg[0]]
            )
        round_counter[0] += 1

    for _ in range(steps):
        # RZZ half-step -- stays parallel across color classes, same
        # physical pairs as the unencoded circuit (Eq. 6).
        cond = current_condition()
        for edge_list in color_edges:
            for (i, j) in edge_list:
                compile_logical_rzz(circuit, k, i, j, theta_zz, condition=cond)
        append_syndrome_round()

        # RX half-step -- sequential through the shared physical qubit t
        # (Eq. 1); see module docstring.
        cond = current_condition()
        for i in range(k):
            compile_logical_rx(circuit, k, i, theta_x, condition=cond)
        append_syndrome_round()

    # Final syndrome round (Fig. 1(e): extract S_X -- and, redundantly but
    # harmlessly, S_Z again -- before the destructive measurement, since
    # S_X can't be recovered from a Z-basis measurement afterward).
    append_syndrome_round()
    assert round_counter[0] == n_rounds_total

    data_creg = circuit.add_c_register("data", n)
    for i in range(n):
        circuit.Measure(circuit.qubits[i], data_creg[i])

    metadata = {
        "n": n,
        "n_rounds": n_rounds_total,
        "data_creg_name": "data",
        "flags_creg_name": "flags",
    }
    if early_exit:
        metadata["discard_creg_name"] = "discard"
    return circuit, metadata
