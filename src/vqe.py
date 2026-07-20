"""
vqe.py

VQE ground-state search against the H2 emulator: a hardware-efficient
ansatz (tket_circuit.build_hea_ansatz_circuit) optimized via gradient-free
SciPy COBYLA, following Quantinuum's own batched-variational-experiment
pattern (see docs.quantinuum.com's
"Quantinuum_variational_experiment_with_batching" reference) rather than
literal parameter-shift gradient descent -- with 6*N parameters, per-
parameter gradients would need thousands of real hardware submissions per
optimizer step, which is infeasible even absent quota limits.

pytket's measurement_reduction groups the TFIM Hamiltonian's terms into the
minimum number of commuting measurement circuits (2: all-ZZ in the Z basis,
all-X in the X basis), and qnexus_backend.submit_vqe_batch_job submits both
of an iteration's circuits in a single compile/execute round trip -- the
Nexus-layer equivalent of the reference's start_batch/add_to_batch.
"""
import random

import numpy as np
from pytket.circuit import Qubit
from pytket.partition import PauliPartitionStrat, measurement_reduction
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import expectation_from_shots
from pytket.utils.operators import QubitPauliOperator
from scipy.optimize import minimize

from persistence import save_stage_results
from qnexus_backend import bitstrings_to_observables, bootstrap_observable_errors, submit_vqe_batch_job
from tket_circuit import build_hea_ansatz_circuit


def build_tfim_pauli_operator(N, J, h):
    """Build H = -J * sum_i Zi Zi+1 - h * sum_i Xi as a pytket
    QubitPauliOperator (periodic chain), matching the sign convention of
    pauli_ops.build_tfim_hamiltonian so VQE energies are directly
    comparable to ed_baseline's ground energy.
    """
    terms = {}
    for i in range(N):
        j = (i + 1) % N
        terms[QubitPauliString([Qubit(i), Qubit(j)], [Pauli.Z, Pauli.Z])] = -J
    for i in range(N):
        terms[QubitPauliString([Qubit(i)], [Pauli.X])] = -h
    return QubitPauliOperator(terms)


def _bitstrings_to_shot_array(bitstrings):
    return np.array([[int(bit) for bit in bitstr] for bitstr in bitstrings])


def energy_from_batch(measurement_setup, operator, bitstring_lists):
    """Combine one VQE iteration's batch of measurement-circuit bitstrings
    into a total energy estimate, using measurement_setup's per-term
    MeasurementBitMap (which circuit, which bit columns, sign) to pick out
    each Hamiltonian term's expectation value via
    pytket.utils.expectation_from_shots.
    """
    shot_arrays = [_bitstrings_to_shot_array(bs) for bs in bitstring_lists]

    energy = 0.0
    for term, coeff in operator._dict.items():
        bitmaps = measurement_setup.results[term]
        term_val = 0.0
        for bitmap in bitmaps:
            shot_table = shot_arrays[bitmap.circ_index][:, bitmap.bits]
            val = expectation_from_shots(shot_table)
            if bitmap.invert:
                val = -val
            term_val += val
        term_val /= len(bitmaps)
        energy += complex(coeff).real * term_val
    return energy


def run_vqe_h2(N, h_target, J, n_shots, max_iters, tol, seed, device_name="H2-1LE",
               project_name="ftim-hackathon"):
    """Run one VQE ground-state search (fixed h_target) against the H2
    emulator: COBYLA over the HEA's 6*N parameters, one batched
    Z-basis+X-basis circuit submission per iteration.

    Persists each iteration's raw batch result immediately (before
    computing the energy from it), same crash-safety pattern as the
    quench/adiabatic H2 pipelines.

    Returns {'energy_history', 'final_params', 'final_energy',
    'final_z_rms', 'final_z_err', 'final_mzz', 'final_mzz_err'}.
    """
    operator = build_tfim_pauli_operator(N, J, h_target)
    pauli_strings = list(operator._dict.keys())
    measurement_setup = measurement_reduction(pauli_strings, PauliPartitionStrat.CommutingSets)

    # Which measurement circuit is the Z-basis one (all-ZZ terms) isn't
    # guaranteed to be index 0 by measurement_reduction -- look it up from
    # any pure-Z term's bitmap rather than assuming an ordering.
    z_term = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])
    z_circuit_index = measurement_setup.results[z_term][0].circ_index

    energy_history = []
    raw_by_iteration = {}
    best = {'energy': None, 'bitstring_lists': None}

    def objective(params):
        ansatz = build_hea_ansatz_circuit(N, params)
        circuits = []
        for mc in measurement_setup.measurement_circs:
            full = ansatz.copy()
            full.append(mc)
            circuits.append(full)

        iteration = len(energy_history)
        bitstring_lists = submit_vqe_batch_job(
            circuits, n_shots, device_name=device_name, project_name=project_name,
            job_name=f"tfim-vqe-N{N}-h{h_target:.2f}-iter{iteration}",
        )

        # Persist the raw hardware result immediately -- before computing
        # the energy from it -- so a bug downstream never means
        # resubmitting to recover results that already came back.
        raw_by_iteration[iteration] = bitstring_lists
        save_stage_results("h2_vqe_raw", raw_by_iteration)

        energy = energy_from_batch(measurement_setup, operator, bitstring_lists)
        energy_history.append(energy)

        # Track the best iterate seen so far -- COBYLA's *last* evaluated
        # point is often a trust-region probe, not its incumbent best (its
        # own reported result.fun can be substantially better than
        # energy_history[-1]), so the final point must not be assumed to
        # be the best one.
        if best['energy'] is None or energy < best['energy']:
            best['energy'] = energy
            best['bitstring_lists'] = bitstring_lists

        return energy

    random.seed(seed)
    initial_params = np.array([2 * np.pi * random.uniform(0, 1) for _ in range(6 * N)])

    result = minimize(
        objective, initial_params, method="COBYLA",
        options={"disp": True, "maxiter": max_iters}, tol=tol,
    )

    # Reuse the Z-basis circuit's shots from the best iteration found
    # (already collected as part of that objective evaluation) rather than
    # submitting a new circuit just to read off <Z>/<Zi Zi+1>.
    z_bitstrings = best['bitstring_lists'][z_circuit_index]
    z_rms, mzz = bitstrings_to_observables(z_bitstrings, N)
    z_err, mzz_err = bootstrap_observable_errors(z_bitstrings, N)

    return {
        'energy_history': energy_history,
        'final_params': result.x.tolist(),
        'final_energy': best['energy'],
        'final_z_rms': z_rms, 'final_z_err': z_err,
        'final_mzz': mzz, 'final_mzz_err': mzz_err,
    }
