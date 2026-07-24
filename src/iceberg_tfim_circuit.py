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
fresh 2-bit register each round ("flags_r0", "flags_r1", ...), then Reset
before the next round touches them -- exactly as the paper's own Fig. 1(d)
describes ("two additional ancilla qubits, which can be reset and reused").
One small register per round rather than a single concatenated "flags"
register spanning all rounds is deliberate, not cosmetic: Quantinuum's
H2-Emulator caps classical register width at 64 bits (a QIR-conversion
limit, `ClassicalRegisterWidthError`) -- discovered the hard way when a
k=8, 30-step, syndrome_every=1 circuit (61 rounds x 2 bits = 122-bit
"flags" register) was rejected server-side on submission. Per-round
registers keep every register at exactly 2 bits regardless of how many
rounds a circuit has.

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
                                  initial_state_label=None, early_exit=True,
                                  syndrome_every=1):
    """Build a fault-tolerant, Iceberg-encoded pytket Circuit implementing
    `steps` fixed-Hamiltonian Trotter layers on k logical qubits, with a
    mid-circuit syndrome-measurement round every `syndrome_every`
    half-steps (a "half-step" is one RZZ layer or one RX layer -- there
    are 2*steps of them total), plus one final syndrome round and a
    destructive Z-basis measurement of every code-register qubit.

    syndrome_every: protection/depth tradeoff knob. 1 (default) measures
    after every half-step, matching the paper author's own guidance and
    this module's original behavior; 2 measures once per full Trotter
    step (after the RX half-step only); larger values measure less often,
    trading protection (a longer window in which an undetected fault
    could accumulate before being caught) for a shallower circuit and
    correspondingly lower discard rate. Concretely (see the k=8, steps=30
    estimate this was built to investigate): syndrome_every=1 there gives
    61 total rounds and ~1700 two-qubit gates -- about 2.6x the paper's
    own deepest real-hardware demonstration -- so a large syndrome_every
    (or fewer steps) is likely necessary for any circuit deep enough to
    have a usable discard rate on real noisy hardware.

    syndrome_every=None selects a separate, sparser mode: exactly one
    mid-circuit round, placed right after Trotter step `steps // 2`
    (0-indexed) finishes both its half-steps, plus the always-present
    final round -- 2 rounds total regardless of `steps`. This isn't
    reachable via any periodic syndrome_every value (a period can't land
    on a single centered checkpoint without also firing elsewhere), so
    it's handled as its own placement rule rather than a large period.
    Cuts round count/depth far more aggressively than any integer
    syndrome_every, at real cost: the single fault-detection guarantee
    (see docs/ICEBERG_QEC_PLAN.md and the fault-tolerance note this
    module's tests rely on) only covers one fault between checks -- with
    two ~steps/2-step-wide windows, two independent faults landing in the
    same window can cancel each other's stabilizer signature and produce
    an undetected logical error, not just a missed discard. Also forfeits
    most of early_exit's runtime benefit, since discard status is mostly
    unknown until partway through the circuit at best.

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
      - "n_rounds": total syndrome-measurement rounds (2*steps //
        syndrome_every, plus 1 for the always-present final round)
      - "data_creg_name": classical register holding the final n-bit
        destructive measurement
      - "flags_creg_names": list of n_rounds 2-bit classical register
        names, one per syndrome round ("flags_r0", "flags_r1", ...) --
        each round's (a1, a2) outcome, kept separate rather than one
        large concatenated register (see module docstring: avoids a
        64-bit-per-register limit on H2-Emulator). Concatenate their
        per-shot bitstrings, in this list's order, before feeding into
        iceberg_decode.decode_shots alongside the data register.
      - "discard_creg_name": (only if early_exit) the running 1-bit
        discard flag -- purely informational/for early-exit gating, NOT
        a substitute for iceberg_decode.should_discard's own check on the
        full flags+data registers (an early-exited shot's LATER flag bits
        stay at their reset value of 0 and must not be misread as "no
        error" on their own).
    """
    if syndrome_every is not None and syndrome_every < 1:
        raise ValueError(f"syndrome_every must be >= 1 or None, got {syndrome_every}")

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

    n_half_steps = 2 * steps
    mid_step_index = steps // 2  # only used when syndrome_every is None
    if syndrome_every is None:
        n_mid_rounds = 1 if steps > 0 else 0
    else:
        n_mid_rounds = n_half_steps // syndrome_every
    n_rounds_total = n_mid_rounds + 1  # always end with one final round
    flags_cregs = [circuit.add_c_register(f"flags_r{r}", 2) for r in range(n_rounds_total)]
    round_counter = [0]
    half_step_counter = [0]

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
        round_creg = flags_cregs[r]
        circuit.Measure(circuit.qubits[a1], round_creg[0])
        circuit.Measure(circuit.qubits[a2], round_creg[1])
        circuit.Reset(circuit.qubits[a1])
        circuit.Reset(circuit.qubits[a2])
        if early_exit:
            circuit.add_clexpr_from_logicexp(
                discard_creg[0] | round_creg[0] | round_creg[1], [discard_creg[0]]
            )
        round_counter[0] += 1

    def after_half_step():
        half_step_counter[0] += 1
        if syndrome_every is not None and half_step_counter[0] % syndrome_every == 0:
            append_syndrome_round()

    for step_idx in range(steps):
        # RZZ half-step -- stays parallel across color classes, same
        # physical pairs as the unencoded circuit (Eq. 6).
        cond = current_condition()
        for edge_list in color_edges:
            for (i, j) in edge_list:
                compile_logical_rzz(circuit, k, i, j, theta_zz, condition=cond)
        after_half_step()

        # RX half-step -- sequential through the shared physical qubit t
        # (Eq. 1); see module docstring.
        cond = current_condition()
        for i in range(k):
            compile_logical_rx(circuit, k, i, theta_x, condition=cond)
        after_half_step()

        if syndrome_every is None and step_idx == mid_step_index:
            append_syndrome_round()

    # Final syndrome round (Fig. 1(e): extract S_X -- and, redundantly but
    # harmlessly, S_Z again -- before the destructive measurement, since
    # S_X can't be recovered from a Z-basis measurement afterward). Always
    # appended regardless of syndrome_every, even if the last half-step
    # already triggered one.
    append_syndrome_round()
    assert round_counter[0] == n_rounds_total

    data_creg = circuit.add_c_register("data", n)
    for i in range(n):
        circuit.Measure(circuit.qubits[i], data_creg[i])

    metadata = {
        "n": n,
        "n_rounds": n_rounds_total,
        "data_creg_name": "data",
        "flags_creg_names": [f"flags_r{r}" for r in range(n_rounds_total)],
    }
    if early_exit:
        metadata["discard_creg_name"] = "discard"
    return circuit, metadata
