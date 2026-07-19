"""
reporting.py

Text/console reporting: side-by-side comparison of Trotter final-state
observables against the exact ground state.
"""


def print_comparison_table(ed_results, trotter_data, dt_label):
    """Print a table comparing adiabatic Trotter final state vs. ED ground state."""
    print("\n" + "=" * 104)
    print(f"COMPARISON: {dt_label} Adiabatic Final State vs. ED (ground state)")
    print("Total sweep time now scales with |h_target - h_init| to keep dh/dt constant.")
    print("=" * 104)
    print(f"{'h/J':^6} | {'t_total':^8} | {'Trotter <Z>':^12} | {'ED <Z>':^10} | {'%Diff Z':^9} | "
          f"{'Trotter <X>':^12} | {'ED <X>':^10} | {'%Diff X':^9}")
    print("-" * 104)

    for r in ed_results:
        h_val = r['h']
        t_total = trotter_data[h_val]['t_total']

        trot_z = trotter_data[h_val]['z_final']
        ed_z = r['mz_rms']
        pct_z = (abs(trot_z - ed_z) / ed_z) * 100 if ed_z != 0 else 0

        trot_x = trotter_data[h_val]['x_final']
        ed_x = r['mx']
        pct_x = (abs(trot_x - ed_x) / abs(ed_x)) * 100 if ed_x != 0 else 0

        print(f"{h_val:^6.1f} | {t_total:^8.2f} | {trot_z:^12.6f} | {ed_z:^10.6f} | {pct_z:^8.2f}% | "
              f"{trot_x:^12.6f} | {ed_x:^10.6f} | {pct_x:^8.2f}%")
    print("=" * 104)
