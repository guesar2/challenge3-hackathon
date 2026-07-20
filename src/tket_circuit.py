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

Boundary Rx fusion (_append_trotter_layers, mirror=True path): when the
symmetric layer Rx(a/2)-ZZPhase-Rx(a/2) repeats, the trailing Rx(a/2) of
one step and the leading Rx(a/2) of the next sit back-to-back on the same
qubit with nothing between them, so they combine exactly via
Rx(a)*Rx(b) = Rx(a+b) into a single gate -- same unitary, no Trotter-error
cost, just fewer single-qubit gates (steps+1 Rx layers instead of 2*steps).
Same idea as the "reduction of gate count using circuit identities" used
for the hopping circuits in Quantinuum's Fermi-Hubbard/pairing-correlations
paper (arXiv:2511.02125, Fig. S8) -- that paper merges fermionic SWAP gates
with interaction gates, which doesn't apply here (TFIM has no SWAP network,
being already nearest-neighbour), but the underlying identity -- merge
adjacent single-qubit rotations at a layer boundary -- carries over
directly to the mirrored Rx layer used here. Verified numerically (unitary
diff ~1e-15) for both the fixed-angle quench circuit and the varying-angle
adiabatic circuit before landing this.
"""
import math

from pytket import Circuit


def _append_trotter_layers(circuit, N, color_edges, layer_angles, mirror):
    """Append len(layer_angles) Trotter layers to `circuit` in place.

    layer_angles: list of (x_angle, zz_angle) per step, in pytket half-turns.
    x_angle is the Rx angle for that step as computed by the caller: already
    halved when mirror=True (it's applied twice per interior boundary), or
    the full single-qubit rotation when mirror=False.

    When mirror=True, adjacent half-turn Rx gates at layer boundaries are
    fused into one Rx via Rx(a)*Rx(b) = Rx(a+b) -- see module docstring.
    When mirror=False there's no boundary Rx pair to fuse (each layer is
    just Rx-then-ZZPhase), so this is just the plain unrolled loop.
    """
    if not mirror:
        for x_angle, zz_angle in layer_angles:
            for i in range(N):
                circuit.Rx(x_angle, i)
            for edge_list in color_edges:
                for a, b in edge_list:
                    circuit.ZZPhase(zz_angle, a, b)
        return

    pending_x = None
    for x_angle, zz_angle in layer_angles:
        merged_x = x_angle if pending_x is None else pending_x + x_angle
        for i in range(N):
            circuit.Rx(merged_x, i)
        for edge_list in color_edges:
            for a, b in edge_list:
                circuit.ZZPhase(zz_angle, a, b)
        pending_x = x_angle
    for i in range(N):
        circuit.Rx(pending_x, i)


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
    x_angle = (theta_x / 2 if mirror else theta_x) / math.pi
    zz_angle = theta_zz / math.pi

    circuit = Circuit(N, N)

    if initial_state_label:
        for i, bit in enumerate(initial_state_label):
            if bit == '1':
                circuit.X(i)

    _append_trotter_layers(circuit, N, color_edges, [(x_angle, zz_angle)] * steps, mirror)

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
    across the ramp rather than staying fixed. Boundary Rx fusion (see
    _append_trotter_layers) still applies even though the angle changes
    step to step -- Rx(a)*Rx(b) = Rx(a+b) regardless of whether a == b.
    """
    circuit = Circuit(N, N)

    for i in range(N):
        circuit.H(i)

    layer_angles = []
    for step in range(1, ramp_steps + 1):
        s_ramp = step / ramp_steps
        h_eff = (1 - s_ramp) * h_init + s_ramp * h_target
        J_eff = s_ramp * J

        theta_x = -2 * h_eff * dt
        theta_zz = -2 * J_eff * dt
        x_angle = (theta_x / 2 if mirror else theta_x) / math.pi
        zz_angle = theta_zz / math.pi
        layer_angles.append((x_angle, zz_angle))

    _append_trotter_layers(circuit, N, color_edges, layer_angles, mirror)

    circuit.measure_all()
    return circuit
