"""
plotting.py

All matplotlib figure generation, kept separate from the numerics so the
simulation modules stay pure/testable and the plotting code can be swapped
out (e.g. for a notebook, a dashboard, or headless batch runs) without
touching physics code.

UNIFIED VERSION:
- Merges plotting.py and plotting_g.py.
- Includes 3-panel H2 plots (Z, X, ZZ) from the _g version, robust to missing X data.
- Includes N-scaling, ED-scaling, and runtime-scaling plots from both sides.
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


# =============================================================================
#  ED SCALING PLOTS (from _g version)
# =============================================================================

def plot_ed_scaling(h_values, ed_results_by_N, save_dir=None, filename="ed_scaling.png"):
    """<X>, <Z>_rms, and <Zi Zi+1> vs. h/J, one line per system size N --
    the observable side of the scaling comparison (run_ed.py runs ED at
    every N in config.ED_EXTRA_N_VALUES and passes them all here).

    ed_results_by_N: dict N -> list of per-h dicts as returned by
    exact_diagonalization.ed_baseline (each {'h', 'mz_rms', 'mx', 'mzz',
    'energy'}).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    for N, ed_results in sorted(ed_results_by_N.items()):
        x_vals = [r['mx'] for r in ed_results]
        z_vals = [r['mz_rms'] for r in ed_results]
        zz_vals = [r['mzz'] for r in ed_results]
        ax1.plot(h_values, x_vals, '-o', markersize=7, label=f'N = {N}')
        ax2.plot(h_values, z_vals, '-o', markersize=7, label=f'N = {N}')
        ax3.plot(h_values, zz_vals, '-o', markersize=7, label=f'N = {N}')

    for ax, ylabel, title in (
        (ax1, r'$\langle X \rangle$', 'Magnetización en X'),
        (ax2, r'$\langle Z \rangle$ (RMS por sitio)', 'Magnetización en Z'),
        (ax3, r'$\langle Z_i Z_{i+1} \rangle$', 'Correlación ZZ'),
    ):
        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='Crítico h/J=1')
        ax.set_xlabel('h / J')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle('ED ground-state observables vs. system size (classical baseline scaling)',
                 fontsize=10)
    _finalize(fig, filename, save_dir)
    return fig


def plot_ed_runtime_scaling(timings, extrapolate_to=None, save_dir=None,
                             filename="ed_runtime_scaling.png"):
    """Wall-clock ED cost (Hamiltonian build + eigsh, one h/J point) vs.
    system size N, log-scale y-axis -- the "honest extrapolation: state
    where classical methods still win" evidence (run_ed.py measures
    `timings` at config.N_RUNTIME_SCALING_VALUES).

    timings: list of {'N', 'dim', 'time_s'} dicts, all actually measured on
    this machine -- see run_ed.py. If extrapolate_to is given (and exceeds
    the largest measured N), a log-linear fit through the measured points
    (exponential-growth assumption, consistent with the 2^N Hilbert space)
    is projected out to it and plotted as a separate, clearly-labeled
    dashed series -- never blended into the "measured" line, since it is
    NOT an actual run.
    """
    N_vals = np.array([t['N'] for t in timings])
    t_vals = np.array([t['time_s'] for t in timings])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.semilogy(N_vals, t_vals, 'o-', color='steelblue', markersize=8,
                label='Measured (this machine)')

    if extrapolate_to and extrapolate_to > N_vals.max():
        fit_slope, fit_intercept = np.polyfit(N_vals, np.log(t_vals), 1)
        N_ext = np.arange(N_vals.max(), extrapolate_to + 1)
        t_ext = np.exp(fit_intercept + fit_slope * N_ext)
        ax.semilogy(N_ext, t_ext, 'o--', color='indianred', markersize=6, alpha=0.7,
                    label='Extrapolated (log-linear fit, NOT run)')

    for seconds, label in ((60, '1 min'), (3600, '1 hour'), (86400, '1 day')):
        if seconds >= t_vals.min():
            ax.axhline(seconds, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.annotate(label, xy=(N_vals.min(), seconds), xytext=(2, 2),
                        textcoords='offset points', fontsize=8, color='gray')

    ax.set_xlabel('N (number of spins)')
    ax.set_ylabel('Wall-clock time (s, log scale)')
    ax.set_title('Classical ED cost vs. system size (h/J = 1, this implementation)')
    ax.set_xticks(N_vals if extrapolate_to is None else np.arange(N_vals.min(), extrapolate_to + 1, 2))
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    _finalize(fig, filename, save_dir)
    return fig


# =============================================================================
#  ADIABATIC & QUENCH PLOTS (identical in both versions)
# =============================================================================

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


# =============================================================================
#  H2 EMULATOR PLOTS (merged: _g version is the base, with 3 panels for X)
# =============================================================================

def plot_h2_vs_ed_time(h_values, time_series_data, save_dir=None, saved_at=None,
                        filename="h2_vs_ed_time.png"):
    """<Z>, <X>, and <Zi Zi+1> vs. time for the H2 emulator quench, one row
    per h/J, with the ED exact evolution as a continuous reference curve.

    time_series_data: dict h -> {'times', 'z_h2', 'z_err', 'x_h2', 'x_err',
    'mzz_h2', 'mzz_err', 'z_ed', 'x_ed', 'mzz_ed'} (see
    run_h2_emulator.run()). *_err are shot-noise standard errors (bootstrap
    over the raw measured shots -- see
    shot_observables.bootstrap_observable_errors), shown as error bars so the
    hardware numbers aren't reported without a noise estimate alongside them.
    """
    fig, axes = plt.subplots(len(h_values), 3, figsize=(17, 4 * len(h_values)))

    for idx, h in enumerate(h_values):
        r = time_series_data[h]
        row = axes[idx] if len(h_values) > 1 else axes

        ax1 = row[0]
        ax1.plot(r['times'], r['z_ed'], 'r-', linewidth=2, label='ED (exact)')
        ax1.errorbar(r['times'], r['z_h2'], yerr=r['z_err'], fmt='bo', markersize=6,
                     capsize=4, label='H2 emulator')
        ax1.set_xlabel('Time t')
        ax1.set_ylabel(r'$\langle Z \rangle$ (RMS per site)')
        ax1.set_title(f'h/J = {h:.1f} (starting from |0...0>)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8)

        ax2 = row[1]
        ax2.plot(r['times'], r['x_ed'], 'r-', linewidth=2, label='ED (exact)')
        ax2.errorbar(r['times'], r['x_h2'], yerr=r['x_err'], fmt='go', markersize=6,
                     capsize=4, label='H2 emulator')
        ax2.set_xlabel('Time t')
        ax2.set_ylabel(r'$\langle X \rangle$ (mean per site)')
        ax2.set_title(f'h/J = {h:.1f} (starting from |0...0>)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8)

        ax3 = row[2]
        ax3.plot(r['times'], r['mzz_ed'], 'r-', linewidth=2, label='ED (exact)')
        ax3.errorbar(r['times'], r['mzz_h2'], yerr=r['mzz_err'], fmt='bo', markersize=6,
                     capsize=4, label='H2 emulator')
        ax3.set_xlabel('Time t')
        ax3.set_ylabel(r'$\langle Z_i Z_{i+1} \rangle$')
        ax3.set_title(f'h/J = {h:.1f} (starting from |0...0>)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=8)

    title = 'Quantinuum H2 Emulator Quench vs. Exact Diagonalization (error bars: shot noise)'
    if saved_at:
        title += f'\nrun: {saved_at}'
    fig.suptitle(title, fontsize=10)

    _finalize(fig, filename, save_dir)
    return fig


def plot_h2_noise_comparison(h_values, noiseless_data, noisy_data, save_dir=None,
                              n=None, filename="h2_noise_comparison.png"):
    """<Z>, <X>, and <Zi Zi+1> vs. time for the H2 emulator quench, overlaying
    ED (exact), noiseless H2-1LE, and noisy H2-Emulator on the same axes so
    the noisy-vs-noiseless gap (hardware noise, with Trotter error cancelled
    since both ran the identical circuit) is visible directly, rather than
    inferred from two separate noisy-vs-ED and noiseless-vs-ED figures.

    noiseless_data / noisy_data: dicts h -> {...} in the same shape as
    run_h2_emulator.run()'s `results` (see plot_h2_vs_ed_time's docstring)
    -- both are expected to come from the same N/h/dt/steps configuration,
    so their 'z_ed'/'x_ed'/'mzz_ed' and 'times' entries coincide; ED is
    plotted once per panel from noiseless_data.
    """
    fig, axes = plt.subplots(len(h_values), 3, figsize=(17, 4 * len(h_values)))

    for idx, h in enumerate(h_values):
        r0 = noiseless_data[h]
        r1 = noisy_data[h]
        row = axes[idx] if len(h_values) > 1 else axes

        ax1 = row[0]
        ax1.plot(r0['times'], r0['z_ed'], 'r-', linewidth=2, label='ED (exact)')
        ax1.errorbar(r0['times'], r0['z_h2'], yerr=r0['z_err'], fmt='bo', markersize=6,
                     capsize=4, label='H2-1LE (noiseless)')
        ax1.errorbar(r1['times'], r1['z_h2'], yerr=r1['z_err'], fmt='m^', markersize=6,
                     capsize=4, label='H2-Emulator (noisy)')
        ax1.set_xlabel('Time t')
        ax1.set_ylabel(r'$\langle Z \rangle$ (RMS per site)')
        ax1.set_title(f'h/J = {h:.1f}' + (f', N={n}' if n is not None else ''))
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8)

        ax2 = row[1]
        ax2.plot(r0['times'], r0['x_ed'], 'r-', linewidth=2, label='ED (exact)')
        ax2.errorbar(r0['times'], r0['x_h2'], yerr=r0['x_err'], fmt='go', markersize=6,
                     capsize=4, label='H2-1LE (noiseless)')
        ax2.errorbar(r1['times'], r1['x_h2'], yerr=r1['x_err'], fmt='m^', markersize=6,
                     capsize=4, label='H2-Emulator (noisy)')
        ax2.set_xlabel('Time t')
        ax2.set_ylabel(r'$\langle X \rangle$ (mean per site)')
        ax2.set_title(f'h/J = {h:.1f}' + (f', N={n}' if n is not None else ''))
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8)

        ax3 = row[2]
        ax3.plot(r0['times'], r0['mzz_ed'], 'r-', linewidth=2, label='ED (exact)')
        ax3.errorbar(r0['times'], r0['mzz_h2'], yerr=r0['mzz_err'], fmt='bo', markersize=6,
                     capsize=4, label='H2-1LE (noiseless)')
        ax3.errorbar(r1['times'], r1['mzz_h2'], yerr=r1['mzz_err'], fmt='m^', markersize=6,
                     capsize=4, label='H2-Emulator (noisy)')
        ax3.set_xlabel('Time t')
        ax3.set_ylabel(r'$\langle Z_i Z_{i+1} \rangle$')
        ax3.set_title(f'h/J = {h:.1f}' + (f', N={n}' if n is not None else ''))
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=8)

    title = 'ED vs. noiseless (H2-1LE) vs. noisy (H2-Emulator)' + (f' -- N={n}' if n is not None else '')
    fig.suptitle(title, fontsize=11)

    _finalize(fig, filename, save_dir)
    return fig


def plot_zne_comparison(h_values, zne_data, save_dir=None, n=None,
                         filename="h2_zne_comparison.png"):
    """<Z>, <X>, and <Zi Zi+1> at one fixed (h, step_count) point per h/J: ED
    (exact) vs. raw-noisy H2-Emulator (qermit Folding.circuit's fold_factor=1,
    with a real bootstrap error bar) vs. ZNE-mitigated (zne_fit.zne_extrapolate's
    zero-noise-limit fit, with its own propagated error bar). Unlike
    plot_h2_noise_comparison, this plots single points per h (not a
    times-series curve), since ZNE here targets one fixed circuit depth --
    see run_zne.run()'s docstring.

    zne_data: dict h -> {'z_ed','x_ed','mzz_ed', 'z_raw','z_raw_err',
    'x_raw','x_raw_err','mzz_raw','mzz_raw_err', 'z_zne','z_zne_err',
    'x_zne','x_zne_err','mzz_zne','mzz_zne_err'}, as produced by
    run_zne.run()'s `results`.
    """
    fig, axes = plt.subplots(len(h_values), 3, figsize=(17, 4 * len(h_values)))

    for idx, h in enumerate(h_values):
        r = zne_data[h]
        row = axes[idx] if len(h_values) > 1 else axes

        panels = (
            (row[0], 'z', r'$\langle Z \rangle$ (RMS per site)'),
            (row[1], 'x', r'$\langle X \rangle$ (mean per site)'),
            (row[2], 'mzz', r'$\langle Z_i Z_{i+1} \rangle$'),
        )
        for ax, key, ylabel in panels:
            ax.axhline(r[f'{key}_ed'], color='r', linewidth=2, label='ED (exact)')
            ax.errorbar([0], [r[f'{key}_raw']], yerr=[r[f'{key}_raw_err']], fmt='m^',
                        markersize=10, capsize=4, label='H2-Emulator (raw noisy)')
            ax.errorbar([1], [r[f'{key}_zne']], yerr=[r[f'{key}_zne_err']], fmt='go',
                        markersize=10, capsize=4, label='ZNE-mitigated')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['raw', 'ZNE'])
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylabel(ylabel)
            ax.set_title(f'h/J = {h:.1f}' + (f', N={n}' if n is not None else ''))
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)

    title = 'ED vs. raw-noisy vs. ZNE-mitigated (H2-Emulator)' + (f' -- N={n}' if n is not None else '')
    fig.suptitle(title, fontsize=11)

    _finalize(fig, filename, save_dir)
    return fig


def plot_h2_phase_transition(h_values, h2_data, ed_results, save_dir=None, saved_at=None,
                              method_label="adiabatic", filename="h2_phase_transition.png"):
    """<Z>, <X>, and <Zi Zi+1> vs. h/J for an H2 ground-state-search protocol
    (adiabatic ramp or VQE), vs. the ED ground state -- the phase-transition
    signal (h/J=1 crossover) as reproduced on hardware, styled like
    plot_phase_transition.

    h2_data: dict h -> {'z_h2', 'z_err', 'x_h2', 'x_err', 'mzz_h2',
    'mzz_err'} (see run_h2_emulator.run_phase_transition()/run_vqe()).
    *_err are shot-noise standard errors
    (shot_observables.bootstrap_observable_errors). h2_data entries without
    'x_h2'/'x_err' (e.g. an older saved run_vqe() result) fall back to NaN
    so the <X> panel just shows gaps rather than raising.

    method_label/filename let callers distinguish which protocol produced
    the data (e.g. "VQE" vs. the default "adiabatic") without duplicating
    this function.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    z_h2 = [h2_data[h]['z_h2'] for h in h_values]
    z_err = [h2_data[h]['z_err'] for h in h_values]
    z_ed = [next(r['mz_rms'] for r in ed_results if r['h'] == h) for h in h_values]
    x_h2 = [h2_data[h].get('x_h2', float('nan')) for h in h_values]
    x_err = [h2_data[h].get('x_err', 0.0) for h in h_values]
    x_ed = [next(r['mx'] for r in ed_results if r['h'] == h) for h in h_values]
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

    ax2.errorbar(h_values, x_h2, yerr=x_err, fmt='go', markersize=8, capsize=4,
                 label=f'H2 emulator ({method_label})')
    ax2.plot(h_values, x_ed, 'rs--', markersize=8, label='ED (ground state)')
    ax2.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='Critical h/J=1')
    ax2.set_xlabel('Target h / J')
    ax2.set_ylabel(r'$\langle X \rangle$')
    ax2.set_title('Quantum Phase Transition (H2 Emulator) — <X>')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.errorbar(h_values, mzz_h2, yerr=mzz_err, fmt='bo', markersize=8, capsize=4,
                 label=f'H2 emulator ({method_label})')
    ax3.plot(h_values, mzz_ed, 'rs--', markersize=8, label='ED (ground state)')
    ax3.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='Critical h/J=1')
    ax3.set_xlabel('Target h / J')
    ax3.set_ylabel(r'$\langle Z_i Z_{i+1} \rangle$')
    ax3.set_title('Quantum Phase Transition (H2 Emulator) — <Zi Zi+1>')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

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


# =============================================================================
#  N-SCALING PLOT (from original version)
# =============================================================================

def plot_n_scaling(scaling_data, ed_max_N, save_dir=None):
    """System-size ('does it break down for many spins?') scan: Trotter vs.
    ED accuracy, circuit cost, and wall-clock runtime, all vs. N.

    scaling_data: dict N -> {
        'max_pct_z', 'max_pct_mzz'  (None where N > ed_max_N, i.e. no dense
            ED reference was computed at that size),
        'depth', 'gate_count'       (from the Trotter circuit, all N),
        'trotter_runtime_s'         (wall-clock time for the Trotter run, all N),
        'ed_runtime_s'              (wall-clock time for the ED reference, None above ed_max_N),
    }

    Two panels:
      (1) Trotter error vs. ED (only where ED was computed) -- shows the
          circuit stays accurate as N grows, for as long as ED can check it.
      (2) circuit depth/gate count and wall-clock runtime (Trotter vs. ED)
          vs. N, log-scale on the y-axis -- shows *where* cost actually
          grows: gently for the Trotter circuit, exponentially for dense ED,
          with a vertical line marking where the ED reference stops.
    """
    N_values = sorted(scaling_data.keys())
    fig, (ax_err, ax_cost) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: accuracy vs N (only where an ED reference exists)
    ed_N = [N for N in N_values if scaling_data[N]['max_pct_z'] is not None]
    if ed_N:
        pct_z = [scaling_data[N]['max_pct_z'] for N in ed_N]
        pct_mzz = [scaling_data[N]['max_pct_mzz'] for N in ed_N]
        ax_err.plot(ed_N, pct_z, 'o-', label=r'Max % dev. $\langle Z \rangle$')
        ax_err.plot(ed_N, pct_mzz, 's-', label=r'Max % dev. $\langle Z_i Z_{i+1} \rangle$')
        ax_err.axhline(y=5.0, color='red', linestyle='--', alpha=0.6, label='5% challenge threshold')
    ax_err.set_xlabel('N (number of spins)')
    ax_err.set_ylabel('Max % deviation, Trotter vs. ED')
    ax_err.set_title('Trotter Accuracy vs. System Size')
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(fontsize=8)

    # Panel 2: cost vs N, log-scale (circuit cost stays cheap; dense ED
    # runtime/memory is what actually breaks down)
    depth = [scaling_data[N]['depth'] for N in N_values]
    gate_count = [scaling_data[N]['gate_count'] for N in N_values]
    trot_time = [scaling_data[N]['trotter_runtime_s'] for N in N_values]

    ax_cost.semilogy(N_values, depth, 'o-', label='Trotter circuit depth')
    ax_cost.semilogy(N_values, gate_count, 's-', label='Trotter gate count')
    ax_cost.semilogy(N_values, trot_time, '^-', label='Trotter runtime (s)')

    ed_time_N = [N for N in N_values if scaling_data[N]['ed_runtime_s'] is not None]
    if ed_time_N:
        ed_time = [scaling_data[N]['ed_runtime_s'] for N in ed_time_N]
        ax_cost.semilogy(ed_time_N, ed_time, 'v-', color='red', label='Dense ED runtime (s)')
    ax_cost.axvline(x=ed_max_N, color='gray', linestyle=':', alpha=0.7,
                     label=f'ED reference stops (N={ed_max_N})')
    ax_cost.set_xlabel('N (number of spins)')
    ax_cost.set_ylabel('Cost (log scale)')
    ax_cost.set_title('Circuit Cost & Runtime vs. System Size')
    ax_cost.grid(True, which='both', alpha=0.3)
    ax_cost.legend(fontsize=8)

    _finalize(fig, 'n_scaling.png', save_dir)
    return fig