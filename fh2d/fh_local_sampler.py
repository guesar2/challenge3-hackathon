"""
fh_local_sampler.py

A free, dependency-light local backend that samples Z-basis measurement
bitstrings from the exact statevector of a pytket circuit. It mirrors the call
signature of qnexus_backend.submit_vqe_batch_job / local_emulator_backend
(a list of circuits in, a list of bitstring-lists out), so the H2 runner and the
VQE driver can be pointed at this sampler for free, instant iteration and then
at the real H2 backends (qnexus / pytket-quantinuum+pecos) with a one-line swap.

Because H2-1LE is itself exact noiseless state-vector emulation (only shot
noise), this sampler reproduces the SAME distribution as an H2-1LE run -- it
just skips the network/queue/quota. It does NOT model physical device noise
(that is the noisy H2-Emulator's job, reachable only through qnexus).

Bit-order convention: pytket's get_statevector() indexes basis states with
qubit 0 as the most-significant bit; the returned bitstrings are ordered
[q0, q1, ..., q_{n-1}] -- the same convention as fh_lattice.occupation_to_label
and fh_jordan_wigner, so no reversal is needed anywhere downstream.
"""
from __future__ import annotations

import numpy as np
from pytket import Circuit
from pytket.circuit import OpType


def _strip_measurements(circuit: Circuit) -> Circuit:
    """Return a copy of `circuit` with measurement (and barrier) ops removed, so
    an exact statevector can be extracted."""
    n = circuit.n_qubits
    out = Circuit(n)
    qubits = circuit.qubits
    index = {q: i for i, q in enumerate(qubits)}
    for cmd in circuit.get_commands():
        if cmd.op.type in (OpType.Measure, OpType.Barrier):
            continue
        qs = [index[q] for q in cmd.qubits]
        # Pass the Op object directly so composite ops (e.g. PauliExpBox) survive.
        out.add_gate(cmd.op, qs)
    return out


def sample_bitstrings(circuit: Circuit, n_shots: int, seed=None):
    """Return a list of Z-basis bitstrings sampled from |amplitudes|^2."""
    bare = _strip_measurements(circuit)
    sv = np.asarray(bare.get_statevector())
    probs = np.abs(sv) ** 2
    probs = probs / probs.sum()
    n = bare.n_qubits
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(probs), size=n_shots, p=probs)
    # qubit 0 is the most-significant bit in pytket's statevector index.
    return [format(i, f"0{n}b") for i in idx]


def submit_batch(circuits, n_shots, device_name="local-sampler", project_name=None,
                 job_name=None, seed=None):
    """Mirror of submit_vqe_batch_job: one bitstring-list per input circuit."""
    return [sample_bitstrings(c, n_shots, seed=seed) for c in circuits]


# Alias so callers can `from fh_local_sampler import submit_vqe_batch_job`
submit_vqe_batch_job = submit_batch