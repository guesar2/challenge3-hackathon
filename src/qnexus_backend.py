"""
qnexus_backend.py

Submission of pytket-built TFIM circuits (quench, adiabatic ramp, VQE) to a
Quantinuum-hosted backend (default: H2-1LE) via qnexus. Kept fully separate
from the local statevector pipeline (trotter_simulation.py) since this path
requires a live qnexus login and costs against a metered usage quota.

H2-1LE is exact noiseless state-vector emulation -- only shot noise, no
physical device noise model at all (docs.quantinuum.com's emulators page:
"Emulator targets ending with LE enable noiseless (state-vector) emulation,
which means that only shot-noise is included in the job result.") -- so
it's the same computation as local_emulator_backend.py's pytket-quantinuum
local Pecos backend, just executed over the network via Nexus instead of on
this machine. Use qnexus (this module) for the real noisy H2-1E emulator or
actual H2-1 hardware, or for a final confirmation run; use
local_emulator_backend for free, instant iteration against H2-1LE while
tuning parameters, since neither costs quota there or changes the result.

submit_quench_batch/submit_adiabatic_batch each submit a whole family of
related circuits (a step-count curve or an h/J sweep) as a single
qnx.compile()/qnx.execute() call rather than one round trip per circuit,
since each qnx.execute() call queues independently on Nexus. VqeSession
does the analogous thing for VQE, but split into a one-time symbolic
compile (__init__) and a per-iteration substitute+execute (submit) --
see its docstring. Quantinuum's own job-batching feature
(QuantinuumConfig's attempt_batching/batch_id, which keeps consecutive
*separate* jobs in one queue session) only applies to real hardware devices,
not emulators like H2-1LE -- consolidating circuits into one execute() call
is the emulator-compatible way to cut down on queue-wait instead.

For quench and adiabatic batches, this builds BOTH Z-basis and X-basis
circuits for each requested step/h-target (since hardware only returns
shots in the measured basis, and the TFIM phase transition requires both
<Z> and <X>). All circuits are submitted in a single compile/execute batch.

Nothing in this module executes on import. It's only invoked from
ftim_main.py when config.RUN_ON_H2_EMULATOR is explicitly set to True.
"""
import qnexus as qnx
from qermit.zero_noise_extrapolation.zne import Folding
from quantinuum_schemas.models.quantinuum_systems_noise import UserErrorParams

from circuits import build_chain_color_edges
from tket_circuit import (
    append_basis_measurement, build_adiabatic_circuit, build_quench_ansatz_circuit,
    build_quench_circuit,
)


def get_project(project_name: str):
    """Fetch or create the qnexus project jobs will be filed under."""
    return qnx.projects.get_or_create(name=project_name)


def submit_quench_batch(N, h_field, J, dt, step_counts, n_shots, device_name="H2-1LE",
                         initial_state_label=None, mirror=True,
                         project_name="ftim-hackathon", job_name=None, timeout=300.0,
                         noise_scale=None):
    """Build, upload, and run a Trotter quench circuit for every step count
    in `step_counts` (e.g. the whole 1..H2_STEPS curve for one h) in a
    single compile job and a single execute job, rather than one round trip
    per step. qnx.compile()/qnx.execute() both natively accept lists of
    circuit refs / shot counts, so this amortizes Nexus queue-wait across
    the whole quench-vs-time curve -- the same trick submit_vqe_batch_job
    uses across an iteration's measurement circuits.

    (Quantinuum's own job-batching feature -- QuantinuumConfig's
    attempt_batching/batch_id -- only applies to real hardware, not
    emulators like H2-1LE, so it can't help here; consolidating multiple
    circuits into one execute() call is the emulator-compatible way to cut
    down on queue-wait.)

    Costs against the qnexus usage quota -- call only with explicit approval.

    Each entry is measured in both the Z and X basis (two circuits per
    step count -- TFIM's <Z> and <X> magnetization both matter for the
    phase-transition signal, and hardware only returns the basis it was
    measured in) so the Z-basis and X-basis circuits for the whole
    step-count curve are still submitted as one compile/execute batch.

    Returns {step_count: result_dict}, each result_dict shaped like the
    single-circuit submission used to (bitstrings + metadata) -- callers
    should persist the whole dict as-is before postprocessing, since
    resubmitting to recover lost raw data spends quota again.

    timeout: seconds qnx.execute() blocks waiting for the whole batch to
    finish (its own default is 300s). Bigger batches -- more step_counts,
    more shots -- take longer on Nexus; a batch that outlives this raises
    TimeoutError from qnx.execute() itself (client-side only, the job
    keeps running server-side), so bump this for large step-count x shots
    sweeps (e.g. a Trotter-depth scan) rather than hitting that.

    noise_scale: optional linear multiplier on H2-Emulator's default
    gate/SPAM/crosstalk/dephasing error rates, passed through as
    UserErrorParams(scale=noise_scale) on the QuantinuumConfig
    (docs.quantinuum.com/systems/user_guide/emulator_user_guide/noise_model.html:
    "A scaling factor can be applied that multiplies all the default or
    supplied error parameters by the scaling rate. [...] a 1 does not
    change the error rates while 0 makes all the errors have a probability
    of 0."). None (default) leaves the device's own default noise model
    untouched. Only meaningful against a noisy device (H2-Emulator) --
    H2-1LE is exact noiseless state-vector emulation and has no error
    model to scale.
    """
    color_edges = build_chain_color_edges(N)
    project = get_project(project_name)
    job_name = job_name or f"tfim-quench-N{N}-h{h_field:.2f}"

    bases = ("z", "x")
    circuits = [
        build_quench_circuit(N, color_edges, steps, dt, h_field, J, mirror=mirror,
                              initial_state_label=initial_state_label, basis=basis)
        for steps in step_counts for basis in bases
    ]
    circuit_refs = [
        qnx.circuits.upload(
            circuit=circ,
            name=f"{job_name}-steps{steps}-{basis}",
            project=project,
            description=(f"TFIM Trotter quench: N={N}, h/J={h_field / J:.2f}, "
                          f"dt={dt}, steps={steps}, mirror={mirror}, basis={basis}"),
        )
        for (steps, basis), circ in zip(((s, b) for s in step_counts for b in bases), circuits)
    ]

    config_kwargs = {}
    if noise_scale is not None:
        config_kwargs["error_params"] = UserErrorParams(scale=noise_scale)
    backend_config = qnx.QuantinuumConfig(device_name=device_name, **config_kwargs)
    # Our circuit is built from Rx/ZZPhase; H2's native gateset doesn't
    # include a raw Rx (it wants Rz/PhasedX/ZZPhase/...), so it must be
    # rebased via a compile job before execute() will accept it. Compile
    # jobs are classical (tket passes run on Nexus, not the QPU/emulator)
    # and don't cost hardware quota, unlike execute().
    compiled_refs = qnx.compile(
        programs=circuit_refs,
        backend_config=backend_config,
        name=f"{job_name}-compile",
        project=project,
    )

    results = qnx.execute(
        programs=list(compiled_refs),
        n_shots=[n_shots] * len(compiled_refs),
        backend_config=backend_config,
        name=job_name,
        project=project,
        timeout=timeout,
    )

    bitstrings_by = {}
    refs_by = {}
    for (steps, basis), circuit_ref, compiled_ref, result in zip(
        ((s, b) for s in step_counts for b in bases), circuit_refs, compiled_refs, results
    ):
        bitstrings_by[(steps, basis)] = ["".join(str(bit) for bit in shot) for shot in result.get_shots()]
        refs_by[(steps, basis)] = (circuit_ref, compiled_ref)

    out = {}
    for steps in step_counts:
        circuit_ref, compiled_ref = refs_by[(steps, "z")]
        circuit_ref_x, compiled_ref_x = refs_by[(steps, "x")]
        out[steps] = {
            "bitstrings": bitstrings_by[(steps, "z")],
            "bitstrings_x": bitstrings_by[(steps, "x")],
            "circuit_ref_id": str(circuit_ref.id),
            "compiled_circuit_ref_id": str(compiled_ref.id),
            "circuit_ref_id_x": str(circuit_ref_x.id),
            "compiled_circuit_ref_id_x": str(compiled_ref_x.id),
            "job_name": job_name,
            "project_name": project_name,
            "device_name": device_name,
            "N": N, "h_field": h_field, "J": J, "dt": dt, "steps": steps,
            "n_shots": n_shots, "mirror": mirror,
            "initial_state_label": initial_state_label or ("0" * N),
            "noise_scale": noise_scale,
        }
    return out


def submit_zne_batch(N, h_field, J, dt, step_counts, fold_factors, n_shots, device_name="H2-Emulator",
                      mirror=True, initial_state_label=None, project_name="ftim-hackathon",
                      job_name=None, timeout=300.0, noise_scale=None, bases=("z", "x")):
    """Build, upload, and run a Trotter quench ansatz for every step count in
    `step_counts` (e.g. the whole 1..H2_STEPS curve, matching
    submit_quench_batch's step_counts convention), each folded by every
    factor in `fold_factors` (qermit's Folding.circuit -- e.g. (1, 3, 5),
    ODD integers only -- see Folding.circuit's own ValueError otherwise) and
    measured in every basis in `bases`, all as a single compile/execute
    batch -- same batching rationale as submit_quench_batch, just with two
    extra axes (fold factor, and optionally a restricted basis set) folded
    into the one round trip.

    fold_factors: noise-scaling multipliers for Folding.circuit. Folding.circuit
    with noise_scaling=1 performs zero fold iterations and returns the plain
    unfolded circuit unchanged (verified directly) -- so fold_factors=1 IS the
    raw-noisy baseline circuit, not a separate submission.

    bases: which measurement bases to include -- defaults to both Z and X
    (needed for <Z>/<Zi Zi+1> and <X> respectively). Pass ("z",) alone when
    only Z-basis observables are needed (e.g. testing with <Zi Zi+1> only),
    to roughly halve the circuit count of a trial run before committing to
    the full sweep.

    Folded circuits are still built from the same native Rx/ZZPhase gate types
    as every other circuit this module submits (Folding.circuit inverts gates
    via their own .dagger, it doesn't introduce new gate types), so they
    compile to H2's native gateset the same way via qnx.compile() below -- no
    local rebase pass is needed here, unlike a design that folds circuits
    outside of a submission batch.

    noise_scale: optional linear multiplier on H2-Emulator's default error
    rates (UserErrorParams(scale=...), same meaning as submit_quench_batch's
    noise_scale) -- independent of fold_factors: this scales the *device's*
    noise model, fold_factors scales noise via *circuit* folding. Two
    different knobs, kept distinctly named on purpose.

    Returns {step_count: {fold_factor: {"bitstrings": [...], "bitstrings_x": [...] (if
    requested)}}}, matching submit_quench_batch's per-step_count shape (one
    level up) so callers can reuse shot_observables.py's
    bitstrings_to_observables/bitstrings_to_mx/bootstrap_* functions
    unchanged at each (step_count, fold_factor) point.
    """
    color_edges = build_chain_color_edges(N)
    project = get_project(project_name)
    job_name = job_name or f"tfim-zne-N{N}-h{h_field:.2f}"

    jobs = [(steps, fold, basis) for steps in step_counts for fold in fold_factors for basis in bases]
    circuits = []
    for steps, fold, basis in jobs:
        ansatz = build_quench_ansatz_circuit(N, color_edges, steps, dt, h_field, J, mirror=mirror,
                                              initial_state_label=initial_state_label)
        folded = Folding.circuit(ansatz, fold)[0]
        append_basis_measurement(folded, N, basis)
        circuits.append(folded)

    circuit_refs = [
        qnx.circuits.upload(
            circuit=circ, name=f"{job_name}-steps{steps}-fold{fold}-{basis}", project=project,
            description=(f"ZNE-folded TFIM quench: N={N}, h/J={h_field / J:.2f}, "
                          f"dt={dt}, steps={steps}, fold={fold}, basis={basis}"),
        )
        for (steps, fold, basis), circ in zip(jobs, circuits)
    ]

    config_kwargs = {}
    if noise_scale is not None:
        config_kwargs["error_params"] = UserErrorParams(scale=noise_scale)
    backend_config = qnx.QuantinuumConfig(device_name=device_name, **config_kwargs)

    compiled_refs = qnx.compile(
        programs=circuit_refs, backend_config=backend_config,
        name=f"{job_name}-compile", project=project,
    )

    results = qnx.execute(
        programs=list(compiled_refs), n_shots=[n_shots] * len(compiled_refs),
        backend_config=backend_config, name=job_name, project=project, timeout=timeout,
    )

    bitstrings_by = {}
    for (steps, fold, basis), result in zip(jobs, results):
        bitstrings_by[(steps, fold, basis)] = [
            "".join(str(bit) for bit in shot) for shot in result.get_shots()
        ]

    out = {}
    for steps in step_counts:
        out[steps] = {}
        for fold in fold_factors:
            entry = {}
            if "z" in bases:
                entry["bitstrings"] = bitstrings_by[(steps, fold, "z")]
            if "x" in bases:
                entry["bitstrings_x"] = bitstrings_by[(steps, fold, "x")]
            out[steps][fold] = entry
    return out


def submit_adiabatic_batch(N, h_targets, J, ramp_steps_by_target, dt_by_target, n_shots, h_init,
                            device_name="H2-1LE", mirror=True, project_name="ftim-hackathon", job_name=None):
    """Build, upload, and run an adiabatic-ramp Trotter circuit (h_init ->
    h_target, J: 0 -> J, starting from |+...+>) for every h in `h_targets`
    in a single compile job and a single execute job.

    ramp_steps_by_target, dt_by_target: sequences of ramp length and Trotter
    step size, one entry each per h_targets entry (not single shared
    values). Two independent things vary per target:
    - Targets closer to h_init need a shorter ramp to reach the same
      |dh/dt| -- see sweep_schedule.steps_for_target for that rate-based
      logic (used to pick ramp_steps_by_target).
    - A target sitting at (or near) the critical point h/J=1 can need a
      finer dt (more steps at fixed total ramp time steps*dt) to converge
      the Trotter approximation itself, independent of the ramp-length
      argument above -- critical slowing down means adiabaticity error
      alone isn't the limiting factor there.

    Same batching rationale as submit_quench_batch -- one round trip for
    the whole h/J sweep instead of one per h -- just built from
    build_adiabatic_circuit instead of build_quench_circuit. Each h_target
    is measured in both the Z and X basis (two circuits per target, since
    the phase-transition signal needs both <Z> and <X> and hardware only
    returns the basis it was measured in), still folded into the same
    single compile/execute batch across the whole sweep.

    Costs against the qnexus usage quota -- call only with explicit approval.

    Returns {h_target: result_dict}, where each result_dict contains:
      - 'bitstrings':  Z-basis measurement shots
      - 'bitstrings_x': X-basis measurement shots
      - metadata (circuit_ref_id, compiled_circuit_ref_id, job_name, etc.)
    """
    color_edges = build_chain_color_edges(N)
    project = get_project(project_name)
    job_name = job_name or f"tfim-adiabatic-N{N}"

    bases = ("z", "x")
    jobs = [
        (h_target, ramp_steps, dt, basis)
        for h_target, ramp_steps, dt in zip(h_targets, ramp_steps_by_target, dt_by_target)
        for basis in bases
    ]
    circuits = [
        build_adiabatic_circuit(N, color_edges, ramp_steps, dt, h_target, J, h_init,
                                 mirror=mirror, basis=basis)
        for h_target, ramp_steps, dt, basis in jobs
    ]
    circuit_refs = [
        qnx.circuits.upload(
            circuit=circ,
            name=f"{job_name}-h{h_target:.2f}-{basis}",
            project=project,
            description=(f"TFIM adiabatic ramp: N={N}, h_init={h_init}, h_target/J={h_target / J:.2f}, "
                          f"dt={dt}, ramp_steps={ramp_steps}, mirror={mirror}, basis={basis}"),
        )
        for (h_target, ramp_steps, dt, basis), circ in zip(jobs, circuits)
    ]

    backend_config = qnx.QuantinuumConfig(device_name=device_name)
    compiled_refs = qnx.compile(
        programs=circuit_refs,
        backend_config=backend_config,
        name=f"{job_name}-compile",
        project=project,
    )

    results = qnx.execute(
        programs=list(compiled_refs),
        n_shots=[n_shots] * len(compiled_refs),
        backend_config=backend_config,
        name=job_name,
        project=project,
    )

    bitstrings_by = {}
    refs_by = {}
    for (h_target, ramp_steps, dt, basis), circuit_ref, compiled_ref, result in zip(
        jobs, circuit_refs, compiled_refs, results
    ):
        bitstrings_by[(h_target, basis)] = ["".join(str(bit) for bit in shot) for shot in result.get_shots()]
        refs_by[(h_target, basis)] = (circuit_ref, compiled_ref)

    out = {}
    for h_target, ramp_steps, dt in zip(h_targets, ramp_steps_by_target, dt_by_target):
        circuit_ref, compiled_ref = refs_by[(h_target, "z")]
        circuit_ref_x, compiled_ref_x = refs_by[(h_target, "x")]
        out[h_target] = {
            "bitstrings": bitstrings_by[(h_target, "z")],
            "bitstrings_x": bitstrings_by[(h_target, "x")],
            "circuit_ref_id": str(circuit_ref.id),
            "compiled_circuit_ref_id": str(compiled_ref.id),
            "circuit_ref_id_x": str(circuit_ref_x.id),
            "compiled_circuit_ref_id_x": str(compiled_ref_x.id),
            "job_name": job_name,
            "project_name": project_name,
            "device_name": device_name,
            "N": N, "h_target": h_target, "J": J, "dt": dt, "ramp_steps": ramp_steps,
            "h_init": h_init, "n_shots": n_shots, "mirror": mirror,
        }
    return out


class VqeSession:
    """One VQE run's compiled circuit family, compiled once (symbolically)
    and re-executed every COBYLA iteration by substituting that iteration's
    parameter values, rather than re-uploading and re-compiling from
    scratch each time (the pattern the old submit_vqe_batch_job used).

    Following Quantinuum's own VQE reference
    (docs.quantinuum.com/nexus/trainings/notebooks/knowledge_articles/vqe_example.html):
    it builds the ansatz once with free (sympy) symbols, compiles that
    symbolic circuit to the backend's native gateset a single time, then
    each iteration does circuit.symbol_substitution(...) on the
    already-compiled circuit before submitting -- gateset rebasing/routing
    doesn't depend on the parameter values for a fixed-structure ansatz
    like HEA or HVA, so redoing it every iteration (as the float-parameter
    version of this module did) was wasted qnx.compile() round trips.

    symbolic_circuits: list of pytket Circuits built with sympy Symbol
    parameters (e.g. ansatz + one measurement-basis group each), matching
    tket_circuit.build_hea_ansatz_circuit/build_hva_ansatz_circuit called
    with Symbols instead of floats.
    """

    def __init__(self, symbolic_circuits, device_name="H2-1LE",
                 project_name="ftim-hackathon", job_name=None):
        self.project = get_project(project_name)
        self.job_name = job_name or "tfim-vqe"
        self.backend_config = qnx.QuantinuumConfig(device_name=device_name)

        circuit_refs = [
            qnx.circuits.upload(
                circuit=circ,
                name=f"{self.job_name}-symbolic-{i}",
                project=self.project,
                description=f"VQE symbolic ansatz+measurement circuit {i}/{len(symbolic_circuits)}",
            )
            for i, circ in enumerate(symbolic_circuits)
        ]
        compiled_refs = qnx.compile(
            programs=circuit_refs,
            backend_config=self.backend_config,
            name=f"{self.job_name}-compile",
            project=self.project,
        )
        # Pull the compiled (native-gateset, routed) circuits back down so
        # symbol_substitution can run locally each iteration without
        # another Nexus compile job.
        self.compiled_circuits = [ref.download_circuit() for ref in compiled_refs]

    def submit(self, symbol_map, n_shots, iteration=0):
        """Substitute one iteration's parameter values into the
        once-compiled circuits and execute -- no recompile.

        symbol_map: {sympy.Symbol: float} for every free symbol used when
        building the symbolic circuits passed to __init__.

        Returns a list of bitstring-lists, one per compiled circuit, in
        the same order as the `symbolic_circuits` passed to __init__.
        """
        substituted = []
        for circ in self.compiled_circuits:
            c = circ.copy()
            c.symbol_substitution(symbol_map)
            substituted.append(c)

        circuit_refs = [
            qnx.circuits.upload(
                circuit=c,
                name=f"{self.job_name}-iter{iteration}-{i}",
                project=self.project,
                description=f"VQE iteration {iteration} circuit {i}/{len(substituted)}",
            )
            for i, c in enumerate(substituted)
        ]
        results = qnx.execute(
            programs=circuit_refs,
            n_shots=[n_shots] * len(circuit_refs),
            backend_config=self.backend_config,
            name=f"{self.job_name}-iter{iteration}",
            project=self.project,
        )

        return [
            ["".join(str(bit) for bit in shot) for shot in r.get_shots()]
            for r in results
        ]