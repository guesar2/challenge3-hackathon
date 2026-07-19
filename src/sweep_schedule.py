"""
sweep_schedule.py

Scheduling logic for the adiabatic sweep: how many Trotter steps are needed
to go from h_init to a given h_target while keeping |dh/dt| roughly
constant across different targets, plus the driver that runs the sweep
for a list of target fields and collects results.
"""
import numpy as np

from trotter_simulation import DEFAULT_H_INIT, run_adiabatic_exact


def steps_for_target(h_target, dt, rate_ref, h_init=DEFAULT_H_INIT, min_steps=1):
    """Number of Trotter steps so that |h_target - h_init| / (steps*dt) ~= rate_ref."""
    delta_h = abs(h_target - h_init)
    if delta_h == 0:
        return min_steps
    t_total = delta_h / rate_ref
    return max(min_steps, int(round(t_total / dt)))


def run_adiabatic_simulation(N, J, h_values, dt, rate_ref, h_init=DEFAULT_H_INIT, verbose=True):
    """Run the adiabatic sweep for each target h/J, keeping |dh/dt| ~= rate_ref.

    Returns dict: h -> {z_expect, mzz, x_expect, z_final, mzz_final, x_final,
                         time, steps, t_total}.
    """
    trotter_data = {}
    if verbose:
        print(f"\n--- Adiabatic Sweep with dt = {dt:.3f}, fixed rate |dh/dt| = {rate_ref:.4f} "
              f"(exact statevector, no shot noise) ---")

    for h in h_values:
        steps = steps_for_target(h, dt, rate_ref, h_init=h_init)
        t_total = steps * dt
        delta_h = abs(h - h_init)
        actual_rate = delta_h / t_total

        if verbose:
            print(f"\nSimulating target h/J = {h:.1f} ... "
                  f"(|Delta h| = {delta_h:.2f}, steps = {steps}, t_total = {t_total:.2f}, "
                  f"rate = {actual_rate:.4f})")

        z_expect, mzz, x_expect = run_adiabatic_exact(
            N, steps, h, J, dt, mirror=True, h_init=h_init
        )
        trotter_data[h] = {
            'z_expect': z_expect,
            'mzz': mzz,
            'x_expect': x_expect,
            'z_final': z_expect[-1],
            'mzz_final': mzz[-1],
            'x_final': x_expect[-1],
            'time': np.arange(1, steps + 1) * dt,
            'steps': steps,
            't_total': t_total,
        }
    return trotter_data
