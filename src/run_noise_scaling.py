"""
run_noise_scaling.py

First diagnostic step towards QEC / error mitigation: characterize how
Quantinuum H2-Emulator's real noise model (gate/SPAM/crosstalk/dephasing)
actually scales with spin-chain size N, before writing any mitigation or
QEC-encoding code.

For each N in NOISE_SCALING_N_VALUES, submits the *same* quench circuit
(config.H2_H_VALUES x steps 1..config.H2_STEPS) to both H2-1LE (noiseless,
run(noisy=False)) and H2-Emulator (noisy, run(noisy=True)) via qnexus --
per project decision, both go through the cloud rather than the local
pytket-pecos emulator (local is slow in practice and Nexus quota is
currently unlimited), so every call below uses local=False.

Comparing noisy vs. ED conflates Trotter error with hardware noise; this
script isolates hardware noise by also comparing noisy vs. noiseless at the
same N (same circuit, so Trotter error cancels) via
plotting.plot_h2_noise_comparison, in addition to the existing noisy/
noiseless-vs-ED figures each run() call already produces.

Each run() call is gated by config.RUN_ON_H2_EMULATOR and costs qnexus
quota (2 devices x len(H2_H_VALUES) h-values x H2_STEPS steps x 2 bases
circuits per N) -- see this module's docstring / the plan for the rough
job count before running.
"""
import config
from plotting import plot_h2_noise_comparison
from run_h2_emulator import run

# Spins to scan. N=4 matches the historical default (config.H2_N); 6 and 8
# extend it -- 8 also matches the PDF's "good enough" noiseless-Trotter
# benchmark size, and ED stays cheap (2^8 = 256 states) at all three.
NOISE_SCALING_N_VALUES = (4, 6, 8)


def run_noise_scaling(n_values=NOISE_SCALING_N_VALUES):
    summary = []
    for n in n_values:
        print("\n" + "#" * 60)
        print(f"# NOISE SCALING: N={n}")
        print("#" * 60)

        noiseless_results = run(local=False, noisy=False, n=n)
        noisy_results = run(local=False, noisy=True, n=n)
        if noiseless_results is None or noisy_results is None:
            print(f"N={n}: skipped (config.RUN_ON_H2_EMULATOR is False) -- "
                  "enable it in config.py to actually submit to qnexus.")
            continue

        filename = f"h2_noise_comparison_N{n}.png"
        plot_h2_noise_comparison(
            config.H2_H_VALUES, noiseless_results, noisy_results,
            save_dir=config.PLOT_SAVE_DIR, n=n, filename=filename,
        )

        for h in config.H2_H_VALUES:
            r0, r1 = noiseless_results[h], noisy_results[h]
            trotter_pct = max(
                abs(a - b) / abs(b) * 100 if b != 0 else 0
                for a, b in zip(r0['z_h2'], r0['z_ed'])
            )
            noise_pct = max(
                abs(a - b) / abs(b) * 100 if b != 0 else 0
                for a, b in zip(r1['z_h2'], r0['z_h2'])
            )
            summary.append({
                'n': n, 'h': h,
                'trotter_error_pct_z': trotter_pct,
                'hardware_noise_pct_z': noise_pct,
            })
            print(f"  N={n}, h/J={h:.2f}: Trotter error (noiseless vs ED) "
                  f"= {trotter_pct:.2f}%, hardware noise (noisy vs noiseless) "
                  f"= {noise_pct:.2f}%")

    print("\nSummary (max % deviation in <Z>, per N/h):")
    for row in summary:
        print(f"  N={row['n']}, h/J={row['h']:.2f}: "
              f"Trotter={row['trotter_error_pct_z']:.2f}%, "
              f"Hardware noise={row['hardware_noise_pct_z']:.2f}%")
    return summary


if __name__ == "__main__":
    run_noise_scaling()
