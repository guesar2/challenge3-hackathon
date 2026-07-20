"""
qnexus_backend.py

Submission of the pytket-built TFIM Trotter quench circuit to a
Quantinuum-hosted backend (default: the H2-1LE noiseless-leakage emulator)
via qnexus. Kept fully separate from the local statevector pipeline
(trotter_simulation.py) since this path requires a live qnexus login and
costs against a metered usage quota.

Nothing in this module executes on import. It's only invoked from
ftim_main.py when config.RUN_ON_H2_EMULATOR is explicitly set to True.
"""
import numpy as np
import qnexus as qnx

from circuits import build_chain_color_edges
from tket_circuit import build_quench_circuit


def get_project(project_name: str):
    """Fetch or create the qnexus project jobs will be filed under."""
    return qnx.projects.get_or_create(name=project_name)


def submit_quench_job(N, h_field, J, dt, steps, n_shots, device_name="H2-1LE",
                       initial_state_label=None, mirror=True,
                       project_name="ftim-hackathon", job_name=None):
    """Build, upload, and run one fixed-Hamiltonian Trotter quench circuit on
    a Quantinuum-hosted backend. Blocks until results are returned.

    Costs against the qnexus usage quota -- call only with explicit approval.

    Returns a dict with the raw Z-basis measurement bitstrings (one per
    shot, string of '0'/'1', qubit 0 first, matching pauli_ops's qubit
    ordering) plus enough metadata (qnexus circuit id, job name, device,
    circuit parameters) to trace the result back to the actual hardware run
    later -- callers should persist this dict as-is before postprocessing,
    since resubmitting to recover lost raw data spends quota again.
    """
    color_edges = build_chain_color_edges(N)
    circuit = build_quench_circuit(
        N, color_edges, steps, dt, h_field, J, mirror=mirror,
        initial_state_label=initial_state_label,
    )

    project = get_project(project_name)
    job_name = job_name or f"tfim-quench-N{N}-h{h_field:.2f}-steps{steps}"

    circuit_ref = qnx.circuits.upload(
        circuit=circuit,
        name=job_name,
        project=project,
        description=(f"TFIM Trotter quench: N={N}, h/J={h_field / J:.2f}, "
                      f"dt={dt}, steps={steps}, mirror={mirror}"),
    )

    backend_config = qnx.QuantinuumConfig(device_name=device_name)
    # Our circuit is built from Rx/ZZPhase; H2's native gateset doesn't
    # include a raw Rx (it wants Rz/PhasedX/ZZPhase/...), so it must be
    # rebased via a compile job before execute() will accept it. Compile
    # jobs are classical (tket passes run on Nexus, not the QPU/emulator)
    # and don't cost hardware quota, unlike execute().
    compiled_refs = qnx.compile(
        programs=[circuit_ref],
        backend_config=backend_config,
        name=f"{job_name}-compile",
        project=project,
    )
    compiled_ref = compiled_refs[0]

    results = qnx.execute(
        programs=[compiled_ref],
        n_shots=[n_shots],
        backend_config=backend_config,
        name=job_name,
        project=project,
    )

    shots = results[0].get_shots()
    bitstrings = ["".join(str(bit) for bit in shot) for shot in shots]

    return {
        "bitstrings": bitstrings,
        "circuit_ref_id": str(circuit_ref.id),
        "compiled_circuit_ref_id": str(compiled_ref.id),
        "job_name": job_name,
        "project_name": project_name,
        "device_name": device_name,
        "N": N, "h_field": h_field, "J": J, "dt": dt, "steps": steps,
        "n_shots": n_shots, "mirror": mirror,
        "initial_state_label": initial_state_label or ("0" * N),
    }


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
