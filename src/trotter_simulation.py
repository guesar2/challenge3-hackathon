"""
trotter_simulation.py

Exact (noiseless) statevector propagation through Trotterized circuits.

"""
import numpy as np
from qiskit.quantum_info import Statevector

from circuits import build_chain_color_edges, build_single_layer_circuit
from pauli_ops import build_collective_observables, expectation_values

DEFAULT_H_INIT = 2.0  # starting transverse field for adiabatic sweeps (deep paramagnetic phase)


def _propagate(N, steps, dt, initial_state: Statevector, schedule_fn, mirror=True):
    """Shared Trotter propagation loop.

    schedule_fn(step, s) -> (h_eff, J_eff) determines the instantaneous
    couplings at each step (s = step/steps is the sweep fraction, unused
    for fixed-Hamiltonian runs).
    """
    color_edges = build_chain_color_edges(N)
    _, Mz_sq_op, Mx_op, Mzz_op = build_collective_observables(N)

    state = initial_state
    z_rms = np.zeros(steps)
    mzz = np.zeros(steps)
    x_exp = np.zeros(steps)

    for step in range(1, steps + 1):
        s = step / steps
        h_eff, J_eff = schedule_fn(step, s)
        theta_x = -2 * h_eff * dt
        theta_zz = -2 * J_eff * dt
        layer = build_single_layer_circuit(N, color_edges, theta_x, theta_zz, mirror=mirror)
        state = state.evolve(layer)
        z_rms[step - 1], mzz[step - 1], x_exp[step - 1] = expectation_values(
            state.data, N, Mz_sq_op, Mx_op, Mzz_op
        )

    times = np.arange(1, steps + 1) * dt
    return times, z_rms, mzz, x_exp


def run_adiabatic_exact(N, ramp_steps, h_target, J_target, dt, mirror=True, h_init=DEFAULT_H_INIT,
                         hold_steps=0):
    """Linearly ramp h: h_init -> h_target (J: 0 -> J_target) over `ramp_steps` steps,
    starting from the |+...+> product state (ground state at h -> infinity).

    If hold_steps > 0, continues propagating for that many extra steps at the
    fixed final (h_target, J_target) after the ramp completes. This lets the
    convergence plot show whether the state has actually settled onto the
    ground state, rather than just happening to reach it right as the ramp ends.
    """
    def schedule(step, s):
        ramp_step = min(step, ramp_steps)
        s_ramp = ramp_step / ramp_steps
        h_eff = (1 - s_ramp) * h_init + s_ramp * h_target
        J_eff = s_ramp * J_target
        return h_eff, J_eff

    initial_state = Statevector.from_label('+' * N)
    total_steps = ramp_steps + hold_steps
    times, z_rms, mzz, x_exp = _propagate(N, total_steps, dt, initial_state, schedule, mirror=mirror)
    return times, z_rms, mzz, x_exp


def run_trotter_fixed_hamiltonian(N, h, J, dt, steps, initial_state_label=None, mirror=True):
    """Evolve under a fixed Hamiltonian H = -J*sum(ZZ) - h*sum(X) from a product state."""
    if initial_state_label is None:
        initial_state_label = '0' * N

    def schedule(step, s):
        return h, J

    initial_state = Statevector.from_label(initial_state_label)
    return _propagate(N, steps, dt, initial_state, schedule, mirror=mirror)
