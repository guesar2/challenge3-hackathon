"""
run_iceberg_qec.py

Orchestrates a run of the Iceberg-encoded TFIM quench circuit
(iceberg_tfim_circuit.py) against the real noisy H2-Emulator via qnexus,
decodes/post-selects the results (iceberg_decode.py), and reports the
decoded observables alongside the discard rate -- this is what
run_n_scaling.py's run_noisy_stub() docstring says is missing: a
QEC-encoded circuit to actually answer "does encoded error correction
help", not just noisy-device access to an unencoded one.

Gated by config.ICEBERG_RUN_ON_H2_EMULATOR (OFF by default, separate from
config.RUN_ON_H2_EMULATOR -- this is a new capability, not covered by any
prior approval to spend quota). Standalone: `python run_iceberg_qec.py`.
"""
import config
from circuits import build_chain_color_edges
from iceberg_tfim_circuit import build_iceberg_quench_circuit
from iceberg_decode import decode_shots
from persistence import save_stage_results
from shot_observables import bitstrings_to_observables


def compile_check(k=None, device_name=None, steps=None, early_exit=None, syndrome_every=1):
    """Tier-1 check (docs/ICEBERG_QEC_PLAN.md): compile the Iceberg-encoded
    circuit against `device_name` WITHOUT executing it -- a classical
    Nexus-side pass, no hardware queue, no quota cost -- and confirm the
    "data"/"flags_r*" classical registers survive compilation with their
    names and bit order intact (qnexus_backend.submit_iceberg_quench_batch
    assumes this; this function is how to verify it before ever spending
    quota on a real run).

    KNOWN GAP: this does not catch every execute-time-only failure. A
    ClassicalRegisterWidthError (H2-Emulator caps classical register width
    at 64 bits) was only raised from qnx.execute's job status, not from
    qnx.compile -- QIR conversion appears to happen at execute time, not
    compile time -- so a passing compile_check does not guarantee a
    subsequent execute() will succeed. Run compile_check at the actual
    step count intended for execution regardless (it still catches
    gateset/routing problems), but treat a pass as necessary, not
    sufficient.

    Returns True if the compile succeeded and the register assumption
    held; raises otherwise (surfacing exactly what broke).
    """
    import qnexus as qnx

    k = k if k is not None else config.ICEBERG_K
    device_name = device_name or config.ICEBERG_DEVICE_NAME
    steps = steps if steps is not None else config.ICEBERG_STEPS
    early_exit = config.ICEBERG_EARLY_EXIT if early_exit is None else early_exit

    color_edges = build_chain_color_edges(k)
    circuit, meta = build_iceberg_quench_circuit(
        k, color_edges, steps, config.ICEBERG_DT, config.ICEBERG_H, config.J,
        early_exit=early_exit, syndrome_every=syndrome_every,
    )

    project = qnx.projects.get_or_create(name="ftim-hackathon")
    circuit_ref = qnx.circuits.upload(circuit=circuit, name="iceberg-compile-check", project=project)
    backend_config = qnx.QuantinuumConfig(device_name=device_name)
    compiled_ref = qnx.compile(
        programs=[circuit_ref], backend_config=backend_config,
        name="iceberg-compile-check", project=project,
    )[0]
    compiled_circuit = compiled_ref.download_circuit()

    for name in (meta["data_creg_name"], *meta["flags_creg_names"]):
        reg = compiled_circuit.get_c_register(name)
        assert len(reg) == len(circuit.get_c_register(name)), (
            f"register {name!r} changed size across compilation -- "
            f"submit_iceberg_quench_batch's bit-list assumption doesn't hold"
        )

    print(f"Compile check OK: k={k}, steps={steps}, device={device_name}, "
          f"n_gates(compiled)={len(compiled_circuit.get_commands())}")
    return True


def run_iceberg_noisy(k=None, device_name=None, shots=None, steps=None, early_exit=None):
    """Submit the Iceberg-encoded quench circuit to the real noisy
    H2-Emulator, decode the results, and report discard rate + observables.

    Costs against the qnexus usage quota -- gated by
    config.ICEBERG_RUN_ON_H2_EMULATOR; returns a "SKIPPED" status dict
    without submitting anything if that's False (same convention as
    run_n_scaling.run_noisy_stub).
    """
    k = k if k is not None else config.ICEBERG_K
    device_name = device_name or config.ICEBERG_DEVICE_NAME
    shots = shots if shots is not None else config.ICEBERG_SHOTS
    steps = steps if steps is not None else config.ICEBERG_STEPS
    early_exit = config.ICEBERG_EARLY_EXIT if early_exit is None else early_exit

    if not config.ICEBERG_RUN_ON_H2_EMULATOR:
        print("\n[Iceberg QEC] Skipped (config.ICEBERG_RUN_ON_H2_EMULATOR = False).")
        return {
            "k": k, "device": device_name, "shots": shots, "steps": steps,
            "early_exit": early_exit, "discard_rate": None,
            "z_rms": None, "mzz": None, "status": "SKIPPED (ICEBERG_RUN_ON_H2_EMULATOR=False)",
        }

    from qnexus_backend import submit_iceberg_quench_batch

    print("\n" + "=" * 70)
    print(f"RUNNING ICEBERG-ENCODED TFIM QUENCH on {device_name}")
    print(f"(k={k}, h/J={config.ICEBERG_H:.2f}, dt={config.ICEBERG_DT}, steps={steps}, "
          f"shots={shots}, early_exit={early_exit})")
    print("=" * 70)

    batch = submit_iceberg_quench_batch(
        k, config.ICEBERG_H, config.J, config.ICEBERG_DT, [steps], shots,
        device_name=device_name, early_exit=early_exit, project_name=config.H2_PROJECT_NAME,
    )
    entry = batch[steps]
    raw_shots = list(zip(entry["flags_bitstrings"], entry["data_bitstrings"]))
    kept, discard_rate = decode_shots(raw_shots, k)

    result = {
        "k": k, "device": device_name, "shots": shots, "steps": steps,
        "early_exit": early_exit, "discard_rate": discard_rate,
        "n_kept": len(kept), "status": "Completed",
    }
    if kept:
        z_rms, mzz = bitstrings_to_observables(kept, k)
        result["z_rms"] = z_rms
        result["mzz"] = mzz
        print(f"  Discard rate: {discard_rate:.1%} ({len(kept)}/{shots} kept)")
        print(f"  <Z>_rms = {z_rms:.4f}, <Zi Zi+1> = {mzz:.4f}")
    else:
        result["z_rms"] = None
        result["mzz"] = None
        print("  Discard rate: 100% -- no shots survived post-selection.")

    save_stage_results("iceberg_qec", result)
    return result


if __name__ == "__main__":
    run_iceberg_noisy()
