"""
fh_qnexus_backend.py

Nexus backend for the Fermi-Hubbard circuits. Uses the same
qnx.compile() / qnx.execute() pattern as the TFIM project's
qnexus_backend.py but has NO dependency on any TFIM module, so it
can live cleanly in the fh2d folder alongside the FH code.

The only public function is submit_vqe_batch_job, which matches the
generic (circuits, n_shots, device_name, project_name, job_name)
interface used by run_fh_h2.py -- the same signature as the TFIM
backend, so the two are drop-in substitutes for each other.

Before running: make sure you are authenticated with Nexus. The
simplest way is to call qnx.login() once in a Python session or
terminal; credentials are then cached locally by the qnexus library.

    import qnexus as qnx
    qnx.login()
"""
import qnexus as qnx


def _get_project(project_name: str):
    """Fetch or create the Nexus project jobs will be filed under."""
    return qnx.projects.get_or_create(name=project_name)


def submit_vqe_batch_job(circuits, n_shots, device_name="H2-1LE",
                          project_name="fermi-hubbard-hackathon", job_name=None):
    """Upload, compile, and execute a batch of pytket circuits on the
    Quantinuum H2 emulator via Nexus, in a single compile + execute call
    (one queue round-trip for all circuits rather than one per circuit).

    Parameters
    ----------
    circuits     : list of pytket.Circuit  -- already contain measure_all()
    n_shots      : int  -- shots per circuit
    device_name  : str  -- "H2-1LE" (noiseless statevector, default) or
                           "H2-Emulator" (real published noise model)
    project_name : str  -- Nexus project to file the job under
    job_name     : str  -- human-readable label shown in the Nexus UI

    Returns
    -------
    list of list[str]  -- one bitstring list per input circuit, same order
    """
    project = _get_project(project_name)
    job_name = job_name or "fh-quench-batch"

    print(f"  [Nexus] uploading {len(circuits)} circuits to project '{project_name}' ...")
    circuit_refs = [
        qnx.circuits.upload(
            circuit=circ,
            name=f"{job_name}-{i}",
            project=project,
            description=f"FH Trotter quench circuit {i + 1}/{len(circuits)}",
        )
        for i, circ in enumerate(circuits)
    ]

    backend_config = qnx.QuantinuumConfig(device_name=device_name)

    print(f"  [Nexus] compiling {len(circuit_refs)} circuits for {device_name} ...")
    compiled_refs = qnx.compile(
        programs=circuit_refs,
        backend_config=backend_config,
        name=f"{job_name}-compile",
        project=project,
    )

    print(f"  [Nexus] executing {len(compiled_refs)} circuits "
          f"({n_shots} shots each) on {device_name} ...")
    results = qnx.execute(
        programs=list(compiled_refs),
        n_shots=[n_shots] * len(compiled_refs),
        backend_config=backend_config,
        name=job_name,
        project=project,
    )

    print(f"  [Nexus] done. Retrieving bitstrings ...")
    return [
        ["".join(str(bit) for bit in shot) for shot in r.get_shots()]
        for r in results
    ]


def _build_zne_circuits(lat, t, U, dt, step_counts, fold_factors,
                        initial_state="neel", order=2):
    """(jobs, circuits) for the ZNE grid: one folded, measured circuit per
    (step_count, fold_factor)."""
    from qermit.zero_noise_extrapolation.zne import Folding

    from fh_tket_circuit import build_quench_ansatz_circuit, append_z_measurement

    jobs = [(sc, fold) for sc in step_counts for fold in fold_factors]
    circuits = []
    for sc, fold in jobs:
        # decompose_boxes=True is REQUIRED here: qermit's Folding.circuit rebuilds
        # the circuit command by command and inverts each op via op.dagger, which
        # it refuses to do for composite ops -- it raises
        #     RuntimeError: Box types not supported when folding.
        # on the PauliExpBoxes every FH circuit is made of. Decomposing into
        # primitive gates (CX / Rz / H / ...) first is exactly what qnx.compile()
        # would do on the way to H2's native gateset anyway, so it changes the
        # gate-level representation and nothing about the unitary (verified: the
        # folded circuit's statevector overlap with the unfolded one is 1.0, and
        # Folding.circuit(c, 3) triples the gate count as intended).
        ansatz = build_quench_ansatz_circuit(lat, t, U, dt, sc,
                                             initial_state=initial_state, order=order,
                                             decompose_boxes=True)
        folded = Folding.circuit(ansatz, fold)[0]
        append_z_measurement(folded)
        circuits.append(folded)
    return jobs, circuits


def start_zne_batch(lat, t, U, dt, step_counts, fold_factors, n_shots,
                    initial_state="neel", order=2, device_name="H2-Emulator",
                    project_name="fermi-hubbard-hackathon", job_name=None,
                    noise_scale=None, seed=None):
    """Upload, compile and SUBMIT (non-blocking) the whole ZNE grid, returning
    immediately via qnx.start_execute_job rather than blocking in qnx.execute.

    Splitting submit into start + collect (the same pattern the TFIM project's
    qnexus_backend.start_quench_batch/collect_quench_batch use) is what makes a
    client-side timeout survivable: the ExecuteJobRef -- and the job's name --
    exist before any waiting happens, so results that the quota has already been
    spent on can be picked up later with collect_zne_batch or resume_zne_batch
    instead of being resubmitted.

    Returns an opaque `pending` dict for collect_zne_batch; 'job_name' inside it
    is what resume_zne_batch needs if the process dies entirely.
    """
    project = _get_project(project_name)
    job_name = job_name or f"fh-zne-{lat.Lx}x{lat.Ly}-U{U:g}"

    jobs, circuits = _build_zne_circuits(lat, t, U, dt, step_counts, fold_factors,
                                         initial_state=initial_state, order=order)

    print(f"  [Nexus] uploading {len(circuits)} folded circuits to '{project_name}' ...")
    circuit_refs = [
        qnx.circuits.upload(
            circuit=circ, name=f"{job_name}-steps{sc}-fold{fold}", project=project,
            description=(f"ZNE-folded FH quench: {lat.Lx}x{lat.Ly}, U/t={U / t:.2f}, "
                         f"dt={dt}, steps={sc}, order={order}, fold={fold}"),
        )
        for (sc, fold), circ in zip(jobs, circuits)
    ]

    config_kwargs = {}
    if noise_scale is not None:
        # Imported lazily so the plain (noiseless) submission path does not need
        # quantinuum-schemas installed.
        from quantinuum_schemas.models.quantinuum_systems_noise import UserErrorParams
        config_kwargs["error_params"] = UserErrorParams(scale=noise_scale)
    backend_config = qnx.QuantinuumConfig(device_name=device_name, **config_kwargs)

    print(f"  [Nexus] compiling {len(circuit_refs)} circuits for {device_name} ...")
    compiled_refs = qnx.compile(
        programs=circuit_refs, backend_config=backend_config,
        name=f"{job_name}-compile", project=project,
    )

    print(f"  [Nexus] submitting job '{job_name}' ({len(compiled_refs)} circuits, "
          f"{n_shots} shots each, noise_scale={noise_scale}) ...")
    execute_job_ref = qnx.start_execute_job(
        programs=list(compiled_refs), n_shots=[n_shots] * len(compiled_refs),
        backend_config=backend_config, name=job_name, project=project,
    )

    return {
        "execute_job_ref": execute_job_ref, "jobs": jobs,
        "step_counts": list(step_counts), "fold_factors": list(fold_factors),
        "job_name": job_name, "project_name": project_name,
        "device_name": device_name, "n_shots": n_shots, "noise_scale": noise_scale,
    }


def _results_to_batch(result_refs, jobs, step_counts, fold_factors):
    """Download shots and reshape into {step_count: {fold_factor: bitstrings}}."""
    bitstrings_by = {}
    for (sc, fold), result_ref in zip(jobs, result_refs):
        result = result_ref.download_result()
        bitstrings_by[(sc, fold)] = [
            "".join(str(bit) for bit in shot) for shot in result.get_shots()
        ]
    return {sc: {fold: bitstrings_by[(sc, fold)] for fold in fold_factors}
            for sc in step_counts}


def collect_zne_batch(pending, timeout=1800.0):
    """Block until `pending` (from start_zne_batch) finishes, then download its
    shots into {step_count: {fold_factor: bitstrings}}.

    A TimeoutError here is client-side only -- the job keeps running on Nexus.
    Re-call this with the same `pending`, or resume_zne_batch(pending["job_name"])
    from a fresh process; do NOT resubmit, the quota is already spent.
    """
    qnx.jobs.wait_for(pending["execute_job_ref"], timeout=timeout)
    result_refs = qnx.jobs.results(pending["execute_job_ref"])
    print(f"  [Nexus] job '{pending['job_name']}' complete. Retrieving bitstrings ...")
    return _results_to_batch(result_refs, pending["jobs"],
                             pending["step_counts"], pending["fold_factors"])


def resume_zne_batch(job_name, step_counts, fold_factors,
                     project_name="fermi-hubbard-hackathon", timeout=1800.0):
    """Look a previously started ZNE job up BY NAME and collect its results --
    the recovery path after a client-side TimeoutError (or any crash) once the
    job itself has finished on Nexus. Costs no additional quota.

    step_counts / fold_factors must be the same lists the job was started with:
    Nexus returns one result per submitted circuit, in submission order, and
    that order is what maps results back onto the (step_count, fold_factor) grid.
    """
    project = _get_project(project_name)
    job = qnx.jobs.get(name=job_name, project=project)
    qnx.jobs.wait_for(job, timeout=timeout)
    result_refs = qnx.jobs.results(job)
    jobs = [(sc, fold) for sc in step_counts for fold in fold_factors]
    print(f"  [Nexus] resumed '{job_name}': {len(result_refs)} results. "
          f"Retrieving bitstrings ...")
    return _results_to_batch(result_refs, jobs, list(step_counts), list(fold_factors))


def submit_zne_batch(lat, t, U, dt, step_counts, fold_factors, n_shots,
                     initial_state="neel", order=2, device_name="H2-Emulator",
                     project_name="fermi-hubbard-hackathon", job_name=None,
                     timeout=1800.0, noise_scale=None, seed=None):
    """Run the Trotter quench on the NOISY H2 emulator at every requested time
    point, each folded by every factor in `fold_factors`, as a single
    compile/execute batch.

    Mirrors the TFIM project's qnexus_backend.submit_zne_batch (same qermit
    Folding.circuit mechanism, same one-round-trip batching), but builds the
    Fermi-Hubbard circuits from fh_tket_circuit and measures in the Z basis only
    -- every FH observable requested here is Z-diagonal.

    step_counts   : Trotter step counts (time = step_count * dt), e.g. [2, 4, 6, 8]
    fold_factors  : ODD integers. Folding.circuit(c, 1) performs zero fold
                    iterations and returns the circuit unchanged, so fold 1 IS
                    the raw-noisy baseline -- it is not submitted separately.
    timeout       : seconds to wait CLIENT-side. Exceeding it raises TimeoutError
                    while the job continues on Nexus -- see collect_zne_batch.
    noise_scale   : optional linear multiplier on the device's published
                    gate/SPAM/crosstalk/dephasing error rates, passed as
                    UserErrorParams(scale=...). Independent of fold_factors:
                    this scales the DEVICE's noise model, folding scales noise
                    via the CIRCUIT. None leaves the model untouched.
    seed          : accepted and ignored -- kept so this function is a drop-in
                    swap for fh_local_noisy_sampler.submit_zne_batch, whose
                    randomness is seedable. Hardware/emulator shots are not.

    Costs against the qnexus usage quota (len(step_counts) * len(fold_factors)
    circuits).

    Returns {step_count: {fold_factor: [bitstring, ...]}}.
    """
    pending = start_zne_batch(
        lat, t, U, dt, step_counts, fold_factors, n_shots,
        initial_state=initial_state, order=order, device_name=device_name,
        project_name=project_name, job_name=job_name, noise_scale=noise_scale,
    )
    return collect_zne_batch(pending, timeout=timeout)
