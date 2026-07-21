"""
local_emulator_backend.py

Local counterpart to qnexus_backend.py: runs the same pytket circuits
(tket_circuit.py) against Quantinuum's H2-1LE via pytket-quantinuum's
QuantinuumBackend + pytket-pecos, instead of submitting through qnexus.

H2-1LE is exact noiseless state-vector emulation -- only shot noise, no
physical device noise model at all (see qnexus_backend.py's module
docstring for the source quote) -- so this is the *same* computation
qnexus's H2-1LE submission gives, just executed on this machine: no login,
no network call, no Nexus queue wait, no quota cost. Requires
`pip install "pytket-quantinuum[pecos]"`.

Mirrors submit_quench_batch/submit_adiabatic_batch/submit_vqe_batch_job's
signatures and return shapes (accepting and ignoring the qnexus-only
project_name/job_name kwargs) so callers (run_h2_emulator.py, vqe.py) can
point at either backend with a single import swap. Meant for fast, free
iteration while tuning parameters; qnexus is still the path for the real
noisy H2-1E emulator, actual H2-1 hardware, or a final confirmation run.

For quench and adiabatic batches, this builds BOTH Z-basis and X-basis
circuits for each requested step/h-target (since hardware only returns
shots in the measured basis, and the TFIM phase transition requires both
<Z> and <X>). All circuits are submitted in a single local round trip.
"""
from pytket.extensions.quantinuum import QuantinuumBackend

from circuits import build_chain_color_edges
from tket_circuit import build_adiabatic_circuit, build_quench_circuit


def _get_backend(device_name):
    return QuantinuumBackend(device_name)


def _run_batch(backend, circuits, n_shots):
    """Compile, execute, and collect bitstrings for a list of circuits in
    one local round trip -- the local equivalent of qnexus's compile()/
    execute() batching."""
    compiled = backend.get_compiled_circuits(circuits)
    handles = backend.process_circuits(compiled, n_shots=[n_shots] * len(compiled))
    results = backend.get_results(handles)
    return [
        ["".join(str(bit) for bit in shot) for shot in r.get_shots()]
        for r in results
    ]


def submit_quench_batch(N, h_field, J, dt, step_counts, n_shots, device_name="H2-1LE",
                         initial_state_label=None, mirror=True,
                         project_name=None, job_name=None):
    """Local equivalent of qnexus_backend.submit_quench_batch.

    Returns {step_count: result_dict}, where each result_dict contains:
      - 'bitstrings':  Z-basis measurement shots (for <Z> and <Zi Zi+1>)
      - 'bitstrings_x': X-basis measurement shots (for <X>)
      - metadata (N, h_field, dt, steps, etc.)

    Unlike the qnexus version, there is no hosted job to trace back to,
    hence no qnexus-only job/project tracing fields.
    """
    color_edges = build_chain_color_edges(N)
    bases = ("z", "x")
    circuits = [
        build_quench_circuit(N, color_edges, steps, dt, h_field, J, mirror=mirror,
                              initial_state_label=initial_state_label, basis=basis)
        for steps in step_counts for basis in bases
    ]

    backend = _get_backend(device_name)
    bitstring_lists = _run_batch(backend, circuits, n_shots)
    bitstrings_by = dict(zip(
        ((s, b) for s in step_counts for b in bases), bitstring_lists
    ))

    out = {}
    for steps in step_counts:
        out[steps] = {
            "bitstrings": bitstrings_by[(steps, "z")],
            "bitstrings_x": bitstrings_by[(steps, "x")],
            "device_name": device_name,
            "N": N, "h_field": h_field, "J": J, "dt": dt, "steps": steps,
            "n_shots": n_shots, "mirror": mirror,
            "initial_state_label": initial_state_label or ("0" * N),
        }
    return out


def submit_adiabatic_batch(N, h_targets, J, ramp_steps_by_target, dt_by_target, n_shots, h_init,
                            device_name="H2-1LE", mirror=True, project_name=None, job_name=None):
    """Local equivalent of qnexus_backend.submit_adiabatic_batch.

    Returns {h_target: result_dict}, where each result_dict contains:
      - 'bitstrings':  Z-basis measurement shots (for <Z> and <Zi Zi+1>)
      - 'bitstrings_x': X-basis measurement shots (for <X>)
      - metadata (N, h_target, dt, ramp_steps, etc.)
    """
    color_edges = build_chain_color_edges(N)
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

    backend = _get_backend(device_name)
    bitstring_lists = _run_batch(backend, circuits, n_shots)
    bitstrings_by = dict(zip(
        ((h_target, basis) for h_target, _, _, basis in jobs), bitstring_lists
    ))

    out = {}
    for h_target, ramp_steps, dt in zip(h_targets, ramp_steps_by_target, dt_by_target):
        out[h_target] = {
            "bitstrings": bitstrings_by[(h_target, "z")],
            "bitstrings_x": bitstrings_by[(h_target, "x")],
            "device_name": device_name,
            "N": N, "h_target": h_target, "J": J, "dt": dt, "ramp_steps": ramp_steps,
            "h_init": h_init, "n_shots": n_shots, "mirror": mirror,
        }
    return out


def submit_vqe_batch_job(circuits, n_shots, device_name="H2-1LE", project_name=None, job_name=None):
    """Local equivalent of qnexus_backend.submit_vqe_batch_job -- one
    compile+execute round trip for a VQE iteration's measurement circuits,
    executed locally instead of submitted through qnexus.
    """
    backend = _get_backend(device_name)
    return _run_batch(backend, circuits, n_shots)