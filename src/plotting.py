"""
plotting.py

All matplotlib figure generation, kept separate from the numerics so the
simulation modules stay pure/testable and the plotting code can be swapped
out (e.g. for a notebook, a dashboard, or headless batch runs) without
touching physics code.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _finalize(fig, filename, save_dir):
    """Save the figure if save_dir is given, and only call plt.show() when
    the current backend can actually display something.

    With a non-interactive backend (Agg — the default in headless scripts,
    containers, and some IDE run configurations) plt.show() is a no-op and
    matplotlib emits: "FigureCanvasAgg is non-interactive, and thus cannot
    be shown". That warning is harmless, but calling show() there achieves
    nothing, so we skip it and save to disk instead.
    """
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {path}")

    interactive_backends = {
        'qtagg', 'qt5agg', 'tkagg', 'gtk3agg', 'gtk4agg',
        'macosx', 'wxagg', 'nbagg', 'ipympl', 'widget',
    }
    if matplotlib.get_backend().lower() in interactive_backends:
        plt.show()
    else:
        plt.close(fig)


def plot_adiabatic_convergence(h_values, trotter_data, ed_results, rate_ref, save_dir=None):
    """<Z>, <ZiZi+1>, <X> vs. sweep time, for each target h, with ED reference lines."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    series = [
        ('z_expect', 'mz_rms', r'$\langle Z \rangle$', r'$\langle Z \rangle$ Converging to GS', 'ED <Z>'),
        ('mzz', 'mzz', r'$\langle Z_i Z_{i+1} \rangle$ (avg. per bond)', 'Nearest-Neighbor Spin Correlation', 'ED'),
        ('x_expect', 'mx', r'$\langle X \rangle$', r'$\langle X \rangle$ Converging to GS', 'ED <X>'),
    ]
    for ax, (data_key, ed_key, ylabel, title, ed_label) in zip(axes, series):
        for h in h_values:
            t = trotter_data[h]['time']
            ax.plot(t, trotter_data[h][data_key], 'o-', markersize=3, label=f'Target h/J = {h:.1f}')
            color = ax.lines[-1].get_color()
            ed_val = next(r[ed_key] for r in ed_results if r['h'] == h)
            ax.axhline(y=ed_val, color=color, linestyle='--',
                       alpha=0.6, label=f'{ed_label} = {ed_val:.3f}')
            ramp_end = trotter_data[h].get('ramp_end_time')
            if ramp_end is not None and ramp_end < t[-1]:
                ax.axvline(x=ramp_end, color=color, linestyle=':', alpha=0.4)
        ax.set_xlabel('Adiabatic Sweep Time t (dotted vline = end of ramp, hold follows)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=7)

    _finalize(fig, 'adiabatic_convergence.png', save_dir)
    return fig


def plot_phase_transition(h_values, trotter_data, ed_results, rate_ref, save_dir=None):
    """Final <Z>, <X>, and <Zi Zi+1> vs. target h/J: Trotter vs. ED, marking the critical point."""
    fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(17, 4.5))

    z_final = [trotter_data[h]['z_final'] for h in h_values]
    z_ed = [next(r['mz_rms'] for r in ed_results if r['h'] == h) for h in h_values]
    x_final = [trotter_data[h]['x_final'] for h in h_values]
    x_ed = [next(r['mx'] for r in ed_results if r['h'] == h) for h in h_values]
    mzz_final = [trotter_data[h]['mzz_final'] for h in h_values]
    mzz_ed = [next(r['mzz'] for r in ed_results if r['h'] == h) for h in h_values]

    ax2.plot(h_values, z_final, 'bo-', markersize=8, label=f'Adiabatic (rate={rate_ref:.3f})')
    ax2.plot(h_values, z_ed, 'rs--', markersize=8, label='ED (ground state)')
    ax2.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='Critical h/J=1')
    ax2.set_xlabel('Target h / J')
    ax2.set_ylabel(r'$\langle Z \rangle$')
    ax2.set_title('Quantum Phase Transition (Finite Size) - <Z>')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(h_values, x_final, 'bo-', markersize=8, label=f'Adiabatic (rate={rate_ref:.3f})')
    ax3.plot(h_values, x_ed, 'rs--', markersize=8, label='ED (ground state)')
    ax3.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='Critical h/J=1')
    ax3.set_xlabel('Target h / J')
    ax3.set_ylabel(r'$\langle X \rangle$')
    ax3.set_title('Quantum Phase Transition (Finite Size) - <X>')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(h_values, mzz_final, 'bo-', markersize=8, label=f'Adiabatic (rate={rate_ref:.3f})')
    ax4.plot(h_values, mzz_ed, 'rs--', markersize=8, label='ED (ground state)')
    ax4.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='Critical h/J=1')
    ax4.set_xlabel('Target h / J')
    ax4.set_ylabel(r'$\langle Z_i Z_{i+1} \rangle$')
    ax4.set_title('Quantum Phase Transition (Finite Size) - <Zi Zi+1>')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    _finalize(fig, 'phase_transition.png', save_dir)
    return fig


def plot_dt_convergence(h_values, dt_values, error_data, save_dir=None):
    """Log-log Trotter error vs. dt, per h/J, with an O(dt^2) reference
    line -- the symmetrized Rx(theta/2)-Rzz(theta)-Rx(theta/2) layer is a
    2nd-order Trotter-Suzuki step, so max % deviation from ED at fixed
    total evolution time should fall roughly as dt^2 (halving dt cuts the
    error ~4x) if the circuit is actually converging correctly.

    error_data: dict h -> {'dt_values': [...], 'max_pct_z': [...], 'max_pct_mzz': [...]}
    """
    fig, (ax_z, ax_mzz) = plt.subplots(1, 2, figsize=(12, 5))

    dt_arr = np.array(sorted(dt_values))
    for ax, err_key, ylabel, title in [
        (ax_z, 'max_pct_z', r'Max % deviation in $\langle Z \rangle$', 'Trotter Convergence — <Z>'),
        (ax_mzz, 'max_pct_mzz', r'Max % deviation in $\langle Z_i Z_{i+1} \rangle$', 'Trotter Convergence — <Zi Zi+1>'),
    ]:
        for h in h_values:
            dts = error_data[h]['dt_values']
            errs = error_data[h][err_key]
            ax.loglog(dts, errs, 'o-', markersize=6, label=f'h/J = {h:.1f}')
        # O(dt^2) reference, anchored to the coarsest point's error scale
        ref_scale = max(error_data[h][err_key][0] for h in h_values) / dt_arr[-1] ** 2
        ax.loglog(dt_arr, ref_scale * dt_arr ** 2, 'k--', alpha=0.5, label=r'$O(dt^2)$ reference')
        ax.set_xlabel('Trotter step size dt')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)

    _finalize(fig, 'dt_convergence.png', save_dir)
    return fig


def plot_fixed_hamiltonian_evolution(h_values, evolution_results, ed_results, save_dir=None):
    """<Z> and <ZiZi+1> vs. time for fixed-H evolution from a product state.

    evolution_results: dict h -> {'times_ed','z_ed','mzz_ed','times_trot','z_trot','mzz_trot'}
    """
    fig, axes = plt.subplots(len(h_values), 2, figsize=(12, 4 * len(h_values)))

    for idx, h in enumerate(h_values):
        r = evolution_results[h]
        gs_z = next(res['mz_rms'] for res in ed_results if res['h'] == h)
        gs_mzz = next(res['mzz'] for res in ed_results if res['h'] == h)

        ax1 = axes[idx, 0] if len(h_values) > 1 else axes[0]
        ax1.plot(r['times_ed'], r['z_ed'], 'r-', linewidth=2, label='ED (exact)')
        ax1.plot(r['times_trot'], r['z_trot'], 'bo', markersize=3, label='Trotter')
        ax1.axhline(y=gs_z, color='gray', linestyle=':', alpha=0.5, label='GS ED')
        ax1.set_xlabel('Time t')
        ax1.set_ylabel(r'$\langle Z \rangle$ (RMS per site)')
        ax1.set_title(f'h/J = {h:.1f} (starting from |0...0>)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8)

        ax2 = axes[idx, 1] if len(h_values) > 1 else axes[1]
        ax2.plot(r['times_ed'], r['mzz_ed'], 'r-', linewidth=2, label='ED (exact)')
        ax2.plot(r['times_trot'], r['mzz_trot'], 'bo', markersize=3, label='Trotter')
        ax2.axhline(y=gs_mzz, color='gray', linestyle=':', alpha=0.5, label='GS ED')
        ax2.set_xlabel('Time t')
        ax2.set_ylabel(r'$\langle Z_i Z_{i+1} \rangle$')
        ax2.set_title(f'h/J = {h:.1f} (starting from |0...0>)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8)

    _finalize(fig, 'fixed_hamiltonian_evolution.png', save_dir)
    return fig


def plot_h2_vs_ed_time(h_values, time_series_data, save_dir=None, saved_at=None,
                        filename="h2_vs_ed_time.png"):
    """<Z> and <Zi Zi+1> vs. time for the H2 emulator quench, one row per h/J,
    with the ED exact evolution as a continuous reference curve.

    time_series_data: dict h -> {'times', 'z_h2', 'z_err', 'mzz_h2',
    'mzz_err', 'z_ed', 'mzz_ed'} (see run_h2_emulator.run()). z_err/mzz_err
    are shot-noise standard errors (bootstrap over the raw measured shots --
    see qnexus_backend.bootstrap_observable_errors), shown as error bars so
    the hardware numbers aren't reported without a noise estimate alongside
    them.
    """
    fig, axes = plt.subplots(len(h_values), 2, figsize=(12, 4 * len(h_values)))

    for idx, h in enumerate(h_values):
        r = time_series_data[h]

        ax1 = axes[idx, 0] if len(h_values) > 1 else axes[0]
        ax1.plot(r['times'], r['z_ed'], 'r-', linewidth=2, label='ED (exact)')
        ax1.errorbar(r['times'], r['z_h2'], yerr=r['z_err'], fmt='bo', markersize=6,
                     capsize=4, label='H2 emulator')
        ax1.set_xlabel('Time t')
        ax1.set_ylabel(r'$\langle Z \rangle$ (RMS per site)')
        ax1.set_title(f'h/J = {h:.1f} (starting from |0...0>)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8)

        ax2 = axes[idx, 1] if len(h_values) > 1 else axes[1]
        ax2.plot(r['times'], r['mzz_ed'], 'r-', linewidth=2, label='ED (exact)')
        ax2.errorbar(r['times'], r['mzz_h2'], yerr=r['mzz_err'], fmt='bo', markersize=6,
                     capsize=4, label='H2 emulator')
        ax2.set_xlabel('Time t')
        ax2.set_ylabel(r'$\langle Z_i Z_{i+1} \rangle$')
        ax2.set_title(f'h/J = {h:.1f} (starting from |0...0>)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8)

    title = 'Quantinuum H2 Emulator Quench vs. Exact Diagonalization (error bars: shot noise)'
    if saved_at:
        title += f'\nrun: {saved_at}'
    fig.suptitle(title, fontsize=10)

    _finalize(fig, filename, save_dir)
    return fig


def plot_h2_phase_transition(h_values, h2_data, ed_results, save_dir=None, saved_at=None,
                              method_label="adiabatic", filename="h2_phase_transition.png"):
    """<Z> and <Zi Zi+1> vs. h/J for an H2 ground-state-search protocol
    (adiabatic ramp or VQE), vs. the ED ground state -- the phase-transition
    signal (h/J=1 crossover) as reproduced on hardware, styled like
    plot_phase_transition.

    h2_data: dict h -> {'z_h2', 'z_err', 'mzz_h2', 'mzz_err'} (see
    run_h2_emulator.run_phase_transition()/run_vqe()). z_err/mzz_err are
    shot-noise standard errors (qnexus_backend.bootstrap_observable_errors).

    method_label/filename let callers distinguish which protocol produced
    the data (e.g. "VQE" vs. the default "adiabatic") without duplicating
    this function.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    z_h2 = [h2_data[h]['z_h2'] for h in h_values]
    z_err = [h2_data[h]['z_err'] for h in h_values]
    z_ed = [next(r['mz_rms'] for r in ed_results if r['h'] == h) for h in h_values]
    mzz_h2 = [h2_data[h]['mzz_h2'] for h in h_values]
    mzz_err = [h2_data[h]['mzz_err'] for h in h_values]
    mzz_ed = [next(r['mzz'] for r in ed_results if r['h'] == h) for h in h_values]

    ax1.errorbar(h_values, z_h2, yerr=z_err, fmt='bo', markersize=8, capsize=4,
                 label=f'H2 emulator ({method_label})')
    ax1.plot(h_values, z_ed, 'rs--', markersize=8, label='ED (ground state)')
    ax1.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='Critical h/J=1')
    ax1.set_xlabel('Target h / J')
    ax1.set_ylabel(r'$\langle Z \rangle$')
    ax1.set_title('Quantum Phase Transition (H2 Emulator) — <Z>')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.errorbar(h_values, mzz_h2, yerr=mzz_err, fmt='bo', markersize=8, capsize=4,
                 label=f'H2 emulator ({method_label})')
    ax2.plot(h_values, mzz_ed, 'rs--', markersize=8, label='ED (ground state)')
    ax2.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='Critical h/J=1')
    ax2.set_xlabel('Target h / J')
    ax2.set_ylabel(r'$\langle Z_i Z_{i+1} \rangle$')
    ax2.set_title('Quantum Phase Transition (H2 Emulator) — <Zi Zi+1>')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    method_title = method_label if method_label.isupper() else method_label.title()
    title = f'Quantinuum H2 {method_title} vs. Exact Diagonalization (error bars: shot noise)'
    if saved_at:
        title += f'\nrun: {saved_at}'
    fig.suptitle(title, fontsize=10)

    _finalize(fig, filename, save_dir)
    return fig


def plot_vqe_convergence(h_values, vqe_data, ed_results, save_dir=None, saved_at=None,
                          filename="vqe_convergence.png"):
    """VQE energy vs. COBYLA iteration, one subplot per h/J target, with
    the ED ground energy as a horizontal reference line.

    vqe_data: dict h -> {'energy_history', ...} (see vqe.run_vqe_h2()).
    ed_results ground energies are per-site (ed_baseline's 'energy' field)
    -- multiplied by N here to match the VQE's total-energy convention
    (vqe.build_tfim_pauli_operator is unnormalized, same as
    pauli_ops.build_tfim_hamiltonian).
    """
    fig, axes = plt.subplots(1, len(h_values), figsize=(5 * len(h_values), 4.5))
    if len(h_values) == 1:
        axes = [axes]

    for ax, h in zip(axes, h_values):
        energy_history = vqe_data[h]['energy_history']
        ed_energy_per_site = next(r['energy'] for r in ed_results if r['h'] == h)
        N = len(vqe_data[h]['final_params']) // 6
        ed_energy_total = ed_energy_per_site * N

        ax.plot(range(len(energy_history)), energy_history, 'bo-', markersize=4,
                label='VQE (H2 emulator)')
        ax.axhline(y=ed_energy_total, color='r', linestyle='--', label='ED ground energy')
        ax.set_xlabel('COBYLA iteration')
        ax.set_ylabel('Energy')
        ax.set_title(f'h/J = {h:.1f}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    title = 'VQE Convergence on Quantinuum H2 Emulator'
    if saved_at:
        title += f'\nrun: {saved_at}'
    fig.suptitle(title, fontsize=10)

    _finalize(fig, filename, save_dir)
    return fig