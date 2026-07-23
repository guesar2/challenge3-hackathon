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