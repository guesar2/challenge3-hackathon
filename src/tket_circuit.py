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
