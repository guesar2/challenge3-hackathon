"""
run_ed.py

Capstone section 1/4: exact diagonalization (ED) baseline for the TFIM.
Ground-state observables (<Z>, <X>, <Zi Zi+1>, energy/N) across
config.H_VALUES at config.N, via sparse eigsh.

Standalone: `python run_ed.py` (from src/, or with src/ on sys.path).
Purely classical -- no Trotter simulation, no qnexus -- so it's the fastest
section to check in isolation.
"""
import config
from exact_diagonalization import ed_baseline
from persistence import save_stage_results


def run():
    print("=" * 60)
    print("STAGE 1/4: EXACT DIAGONALIZATION (ED) BASELINE")
    print("=" * 60)
    ed_results = ed_baseline(config.N, config.H_VALUES, J=config.J)
    save_stage_results("ed", {"N": config.N, "J": config.J, "ed_results": ed_results})
    return ed_results


if __name__ == "__main__":
    run()
