"""
fh_local_noisy_sampler.py

A free, local stand-in for the NOISY Quantinuum H2-Emulator, mirroring the
interface of fh_qnexus_backend.submit_zne_batch so the noisy and ZNE-mitigated
quench curves can be produced (and the whole pipeline tested) without a qnexus
login or any metered quota.

WHAT IT MODELS (and what it does not)
-------------------------------------
fh_local_sampler.py samples the EXACT statevector, i.e. the noiseless H2-1LE
distribution: shot noise only. This module adds two device-like error channels
on top of that same distribution:

  1. Global depolarizing noise whose strength grows with the circuit's
     two-qubit-gate count:

         p_dep = 1 - (1 - eps_2q)^(N_2q * fold_factor)
         P_noisy = (1 - p_dep) * P_ideal + p_dep * Uniform

     Depolarizing noise pulls the state toward the maximally mixed state, which
     does NOT conserve particle number -- exactly like real device noise, and
     the reason the raw noisy <n_up n_dn> drifts toward its maximally mixed
     value 0.25 while m_stag decays to 0.

  2. An independent per-qubit readout bit flip (SPAM), applied AFTER sampling.
     Deliberately NOT scaled by the fold factor: circuit folding amplifies gate
     noise, not state-preparation-and-measurement error, so ZNE structurally
     cannot extrapolate SPAM away. That limitation is real and is reproduced
     here rather than hidden.

FOLDING. The real backend folds each circuit with qermit's Folding.circuit
(C -> C C^-1 C ...), which is the identity as a unitary and only serves to
multiply the gate count -- and hence the accumulated noise -- by the fold
factor. Building the folded circuit locally would therefore change nothing
except runtime, so the fold factor is applied directly to the gate count in
p_dep above. The ideal distribution is fold-independent, as it should be.

This is a PLAUSIBLE stand-in, not Quantinuum's calibrated noise model. Only the
cfg.RUN_ON_H2_EMULATOR = True path (fh_qnexus_backend.submit_zne_batch against
H2-Emulator) is device-accurate; any number quoted as a hardware result must
come from that path. See the LOCAL_NOISE_* comments in fh_config.py.
"""
from __future__ import annotations

import numpy as np
from pytket import Circuit
from pytket.circuit import OpType

import fh_config as cfg
from fh_local_sampler import _strip_measurements
from fh_tket_circuit import build_quench_ansatz_circuit


def count_entangling_gates(circuit: Circuit) -> int:
    """Two-qubit-gate count of `circuit` -- the quantity that sets the
    accumulated gate error, and hence the depolarizing strength below.

    Handles both forms the FH circuits come in:
      * a k-qubit PauliExpBox (or any other box) counts as the 2*(k-1) CNOTs its
        standard CNOT-ladder decomposition costs: a weight-2 ZZ rotation = 2,
        a weight-4 JW hopping string = 6, a single-qubit term = 0;
      * an already-decomposed primitive k-qubit gate counts as k-1, so a plain
        CX counts once rather than twice.

    Checked against the real thing: for 2x2 / order 2 this returns 112 per
    Trotter step, and DecomposeBoxes + n_gates_of_type(OpType.CX) on the same
    circuit gives exactly 112 per step as well, for both forms.
    """
    n2q = 0
    for cmd in circuit.get_commands():
        if cmd.op.type in (OpType.Measure, OpType.Barrier):
            continue
        k = len(cmd.qubits)
        if k < 2:
            continue
        n2q += 2 * (k - 1) if cmd.op.type.name.endswith("Box") else (k - 1)
    return n2q


def sample_bitstrings_noisy(circuit: Circuit, n_shots: int, seed=None, fold_factor=1,
                            two_qubit_error=None, readout_error=None, noise_scale=1.0):
    """Z-basis bitstrings from `circuit` under the depolarizing + readout model
    described in the module docstring. Same return shape as
    fh_local_sampler.sample_bitstrings (qubit 0 first).
    """
    eps2 = cfg.LOCAL_NOISE_TWO_QUBIT_ERROR if two_qubit_error is None else two_qubit_error
    eps_ro = cfg.LOCAL_NOISE_READOUT_ERROR if readout_error is None else readout_error
    scale = 1.0 if noise_scale is None else float(noise_scale)

    bare = _strip_measurements(circuit)
    sv = np.asarray(bare.get_statevector())
    probs = np.abs(sv) ** 2
    probs = probs / probs.sum()
    n = bare.n_qubits

    n2q = count_entangling_gates(bare) * int(fold_factor)
    p_dep = 1.0 - (1.0 - scale * eps2) ** n2q
    probs = (1.0 - p_dep) * probs + p_dep / len(probs)
    probs = probs / probs.sum()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(probs), size=n_shots, p=probs)
    # qubit 0 is the most-significant bit in pytket's statevector index
    bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1).astype(np.int8)

    p_ro = scale * eps_ro
    if p_ro > 0:
        bits ^= (rng.random(bits.shape) < p_ro).astype(np.int8)

    return ["".join(str(int(b)) for b in row) for row in bits]


def submit_zne_batch(lat, t, U, dt, step_counts, fold_factors, n_shots,
                     initial_state="neel", order=2, device_name="local-noisy-sampler",
                     project_name=None, job_name=None, timeout=None,
                     noise_scale=1.0, seed=0,
                     two_qubit_error=None, readout_error=None):
    """Drop-in local replacement for fh_qnexus_backend.submit_zne_batch.

    Same signature, same return shape -- {step_count: {fold_factor: [bitstring,
    ...]}} -- so fh_zne.py can swap between the two with one config flag.
    device_name/project_name/job_name/timeout are accepted and ignored (nothing
    is submitted anywhere); `seed` makes the whole batch reproducible.
    """
    out = {}
    for i, sc in enumerate(step_counts):
        circ = build_quench_ansatz_circuit(lat, t, U, dt, sc,
                                           initial_state=initial_state, order=order)
        out[sc] = {}
        for j, fold in enumerate(fold_factors):
            out[sc][fold] = sample_bitstrings_noisy(
                circ, n_shots, seed=seed + 1000 * i + j, fold_factor=fold,
                two_qubit_error=two_qubit_error, readout_error=readout_error,
                noise_scale=noise_scale,
            )
    return out
