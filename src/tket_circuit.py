"""
tket_circuit.py

Builds the Trotterized TFIM quench circuit as a pytket Circuit, for
submission to the Quantinuum H2 emulator via qnexus. This mirrors the
Rx(theta/2)-Rzz(theta)-Rx(theta/2) layer from circuits.py used by the local
Qiskit/statevector pipeline, but as a single fully-unrolled pytket Circuit
(covering all Trotter steps) with a final Z-basis measurement, since
qnexus/H2 execution returns measurement shots rather than a statevector.

pytket's Circuit.Rx/Rz/ZZPhase take angles in half-turns (multiples of pi),
not radians -- same convention as Qiskit's Operator would give for
Rx(theta_rad) when passing theta_rad / pi (verified: ZZPhase(alpha, i, j) ==
qiskit's rzz(alpha * pi) exactly). pytket has a native ZZPhase gate, so no
CX-Rz-CX decomposition is needed (unlike the earlier Guppy version).
"""
import math

from pytket import Circuit


def build_quench_circuit(N, color_edges, steps, dt, h_field, J, mirror=True,
                          initial_state_label=None):
    """Build a pytket Circuit implementing `steps` fixed-Hamiltonian Trotter
    layers on N qubits, then measure every qubit in the Z basis.

    color_edges: list of edge-color groups (as from circuits.edge_coloring) --
    keeps the generated gate sequence consistent with the Qiskit version.
    initial_state_label: optional bitstring (e.g. "0110") prepared via X
    gates before the Trotter layers; defaults to |0...0>.
    """
    theta_x = -2 * h_field * dt
    theta_zz = -2 * J * dt
    half_x_halfturns = (theta_x / 2 if mirror else theta_x) / math.pi
    zz_halfturns = theta_zz / math.pi

    circuit = Circuit(N, N)

    if initial_state_label:
        for i, bit in enumerate(initial_state_label):
            if bit == '1':
                circuit.X(i)

    for _ in range(steps):
        for i in range(N):
            circuit.Rx(half_x_halfturns, i)
        for edge_list in color_edges:
            for a, b in edge_list:
                circuit.ZZPhase(zz_halfturns, a, b)
        if mirror:
            for i in range(N):
                circuit.Rx(half_x_halfturns, i)

    circuit.measure_all()
    return circuit


def build_hea_ansatz_circuit(N, params):
    """Hardware-efficient ansatz for VQE: RZ-RX-RZ single-qubit rotations,
    one layer of nearest-neighbor CNOTs (open chain), then RZ-RX-RZ again.
    6*N parameters total. Starts from |0...0> (pytket's default); no
    measurement is appended here -- that's added per measurement-basis
    group by the VQE driver (see vqe.py), since different Hamiltonian
    terms need different measurement bases.

    params angles are in half-turns (pytket convention), matching the
    other circuit builders in this module.
    """
    circuit = Circuit(N)

    for i in range(N):
        circuit.Rz(params[i], i)
        circuit.Rx(params[N + i], i)
        circuit.Rz(params[2 * N + i], i)

    for i in range(N - 1):
        circuit.CX(i, i + 1)

    for i in range(N):
        circuit.Rz(params[3 * N + i], i)
        circuit.Rx(params[4 * N + i], i)
        circuit.Rz(params[5 * N + i], i)

    return circuit


def build_adiabatic_circuit(N, color_edges, ramp_steps, dt, h_target, J, h_init, mirror=True):
    """Build a pytket Circuit implementing an adiabatic ramp from h_init to
    h_target (J: 0 -> J) over `ramp_steps` fixed-length layers, starting
    from |+...+>, then measure every qubit in the Z basis.

    Mirrors trotter_simulation.run_adiabatic_exact's schedule (linear ramp
    of h and J with the sweep fraction s_ramp = step/ramp_steps), but as a
    single fully-unrolled pytket Circuit -- unlike build_quench_circuit,
    each layer's angles are recomputed per step since h_eff/J_eff vary
    across the ramp rather than staying fixed.
    """
    circuit = Circuit(N, N)

    for i in range(N):
        circuit.H(i)

    for step in range(1, ramp_steps + 1):
        s_ramp = step / ramp_steps
        h_eff = (1 - s_ramp) * h_init + s_ramp * h_target
        J_eff = s_ramp * J

        theta_x = -2 * h_eff * dt
        theta_zz = -2 * J_eff * dt
        half_x_halfturns = (theta_x / 2 if mirror else theta_x) / math.pi
        zz_halfturns = theta_zz / math.pi

        for i in range(N):
            circuit.Rx(half_x_halfturns, i)
        for edge_list in color_edges:
            for a, b in edge_list:
                circuit.ZZPhase(zz_halfturns, a, b)
        if mirror:
            for i in range(N):
                circuit.Rx(half_x_halfturns, i)

    circuit.measure_all()
    return circuit
