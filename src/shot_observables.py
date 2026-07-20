"""
shot_observables.py

Converts Z-basis measurement bitstrings (from either qnexus_backend or
local_emulator_backend -- both return the same plain string-of-'0'/'1'
shot format) into TFIM observables. Kept separate from both backend
modules so neither needs to import the other's dependencies (qnexus vs.
pytket-quantinuum) just to reuse this postprocessing.
"""
import numpy as np


def _observables_from_shots(shots, N):
    mz = shots.sum(axis=1)
    z_rms = np.sqrt(np.mean(mz ** 2)) / N
    mzz_per_shot = sum(shots[:, i] * shots[:, (i + 1) % N] for i in range(N))
    mzz = np.mean(mzz_per_shot) / N
    return z_rms, mzz


def bitstrings_to_observables(bitstrings, N):
    """Convert Z-basis measurement bitstrings into (<Z>_rms per site,
    <Zi Zi+1> per bond), matching the convention used by
    pauli_ops.expectation_values for the ED/statevector pipeline (RMS
    magnetization rather than the mean, since <Mz> vanishes exactly for a
    Z2-symmetric state/Hamiltonian even when individual shots don't).
    """
    shots = np.array([[1 - 2 * int(bit) for bit in bitstr] for bitstr in bitstrings])
    return _observables_from_shots(shots, N)


def bootstrap_observable_errors(bitstrings, N, n_boot=1000, seed=0):
    """Shot-noise standard errors for (<Z>_rms, <Zi Zi+1>) via bootstrap
    resampling of the measured shots -- for error bars on hardware-run
    figures. Bootstrap (rather than a closed-form propagation) since both
    observables are nonlinear functions of the per-shot bits.
    """
    rng = np.random.default_rng(seed)
    shots = np.array([[1 - 2 * int(bit) for bit in bitstr] for bitstr in bitstrings])
    n_shots = shots.shape[0]

    z_samples = np.empty(n_boot)
    mzz_samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_shots, size=n_shots)
        z_samples[b], mzz_samples[b] = _observables_from_shots(shots[idx], N)
    return z_samples.std(ddof=1), mzz_samples.std(ddof=1)
