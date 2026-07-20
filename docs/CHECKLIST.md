# Challenge 3 (TFIM) — Full-Marks Checklist

Derived from `docs/hackathon.docx`. Ordered by rubric weight, highest first.
Two rubrics apply: the **general submission rubric** (all challenges, 100%
total) and the **Challenge 3–specific rubric** (4-point scale per criterion).
Status notes reflect the repo as of this session.

---

## 0. Deal-breakers (checked before anything else is scored)

- [ ] **Public GitHub repo** with all code, `requirements.txt`, a single
      entry-point script/notebook that reproduces every reported figure and
      number, and a `README.md`.
      ⚠️ **Currently broken**: `python main.py` at the repo root imports
      `trotter_circuit` / `observables`, which are empty 0-byte stub files —
      it crashes immediately. The real, working entry point is
      `src/ftim_main.py`. **You must fix this before submitting** — either
      make `main.py` actually call into `src/ftim_main.py:main()`, or
      rename/replace it so graders running "the one script" get real output.
      Failing "runs clean from a fresh clone" causes deductions **across
      every rubric criterion** (explicit in the doc).
- [ ] Repo is actually public (`git remote -v` shows
      `github.com/guesar2/challenge3-hackathon` — confirm visibility).
- [ ] Fresh-clone smoke test: `git clone`, `python -m venv`, `pip install -r
      requirements.txt`, run the entry point, confirm figures/numbers appear
      with no manual fixes.
- [ ] Honest limitations section present (explicitly called out as
      **mandatory** in the technical report requirement).

---

## 1. Implementación cuántica — 30% (Intento 10% / Buena ejecución 10% / Ejecución en hardware real 10%)

Highest-weighted single line item. Split into three thirds:

- [ ] **Intento (10%)** — TFIM Hamiltonian correctly built, Trotter circuit
      exists and runs. *Already have this*: `src/pauli_ops.py` +
      `src/circuits.py` + `src/trotter_simulation.py`.
- [ ] **Buena ejecución (10%)** — clean, documented, reproducible code;
      correct results. Trotter step size documented (`config.py` has
      `ADIABATIC_DT`, `QUENCH_DT` — make sure the report states these
      explicitly, not just the code).
- [ ] **Ejecución en hardware cuántico real (10%)** — ⚠️ **Not done yet.**
      Everything currently runs as exact `Statevector` simulation
      (noiseless, no hardware/emulator backend touched). The doc repeatedly
      references the **Quantinuum H2 emulator** (up to 26 qubits, exact
      treatment) as the available platform. You need at least one run
      through H2 (or a Quantinuum-accessible backend) for a small chain,
      with results reported alongside the statevector/ED baseline. This is
      worth as much as the entire "good execution" bucket — **do not skip
      it**; without it you cap out at 20/30 on this line alone.

## 2. Explicación — 20%

- [ ] Team can give a coherent technical explanation of how the code works
      end to end (Hamiltonian → Trotter layer → propagation → observables →
      comparison). This is a presentation/defense criterion, not a file —
      prepare to walk through `src/pauli_ops.py`, `src/circuits.py`,
      `src/trotter_simulation.py`, `src/sweep_schedule.py` and explain the
      physics + design choices (e.g., why the edge coloring, why symmetrized
      Rx-Rzz-Rx, why the adiabatic schedule keeps `|dh/dt|` constant).

## 3. Comparación y escalado — 20%

- [ ] Direct quantum-vs-classical comparison **on the same problem
      instance** (have this: Trotter vs. ED for matching N, h).
- [ ] Scaling across **2 or more problem sizes**. ⚠️ Current `config.py` has
      a single `N = 6`. The doc's own problem statement asks to sweep `N`
      from 4–20 and asks "does it degrade for many spins?" — **run at least
      N ∈ {4, 6, 8} (or more) and plot how Trotter error vs. ED grows with
      N**, not just one fixed size.
- [ ] Honest extrapolation: state where classical methods still win
      (explicitly expected — see Limitaciones honestas item below).
- [ ] Classical baseline comparison should be against the **strongest
      available classical method**, cited from a published source (ties
      into "Línea base clásica" below — cite Sachdev, Suzuki 1976, or
      similar for what "state of the art" ED/DMRG achieves at these sizes).

## 4. Línea base clásica — 15%

- [ ] ED baseline with valid, clearly cited performance reference from a
      published source. *Partially have this*: `src/exact_diagonalization.py`
      does sparse ED via `eigsh`, and `ed_baseline` covers `h/J` sweeps.
      Need explicit citation in the report (Suzuki 1976; Sachdev; Ebadi et
      al. 2021 — already listed in README references, make sure they land
      in the technical report too, not just the README).
- [ ] ED run for the specific required grid: `h/J ∈ {0.5, 1.0, 2.0}` at
      **N = 8** (doc's "Línea base clásica" section explicitly asks for
      N=8 for `⟨Z⟩` and `⟨Zᵢ Zᵢ₊₁⟩`; separately, N=6 is required for the
      quantum-vs-ED comparison). Current `config.py` default (`N=6`,
      `H_VALUES=(0.5,1.0,1.5)`) does **not** match either required grid —
      `1.5` should be `2.0`, and you need an N=8 ED run as well as N=6.

## 5. Reproducibilidad — 10%

- [ ] Runs from a clean environment via `requirements.txt` — verify no
      unpinned/missing deps (check `qiskit-aer`, `networkx` are actually
      listed and match what's imported: `src/circuits.py` uses `networkx`
      and `qiskit`; `src/trotter_simulation.py` uses
      `qiskit.quantum_info.Statevector`).
- [ ] **Single entry point reproduces every reported figure/number.**
      Tied to the Section 0 blocker above — fix `main.py` or make it
      unambiguous which script is "the" entry point, and make sure it
      generates *everything* cited in the technical report (not a subset).
- [ ] `pytest tests/` currently collects **0 tests** — `test_hamiltonian.py`
      and `test_trotter.py` are empty. Not explicitly graded by name, but
      the README promises Hermiticity / norm-conservation / Trotter
      convergence / ED-vs-Qiskit tests — an empty test suite undermines the
      "rigor and honesty" the judges are told to reward. Fill these in.

## 6. Impacto en los ODS — 5%

- [ ] Specific SDG sub-goal identified (not just "SDG 7 relates to energy") —
      state precisely which sub-target (e.g., SDG 7.a — enhanced
      international cooperation for clean-energy research/technology).
- [ ] Explicit causal chain from TFIM simulation → real-world outcome
      (e.g., "strongly-correlated electron physics → high-Tc superconductor
      design → lossless transmission → SDG 7"). README already sketches
      this table; port it into the technical report with the causal chain
      made explicit, not just a table of associations.
- [ ] 2+ additional SDGs considered (README covers 7, 9, 12, 13 — keep all
      four, make sure the causal chain is explicit for at least the primary
      one).

---

## Challenge-3–specific rubric (4-point scale: Excelente/Bueno/Necesita mejorar)

- [ ] **Corrección física** — observables match ED within 5% (noiseless).
      Verify `main()` in `src/ftim_main.py` actually prints/confirms this
      (it does: "Max deviation in ⟨Z⟩/⟨Zᵢ Zᵢ₊₁⟩" — keep this in the final
      report with numbers, not just console output).
- [ ] **Implementación del circuito** — efficient Trotter circuit in the
      chosen SDK **with error analysis**. Have the circuit; need a
      convergence-vs-`dt` plot/analysis (halve `dt`, show convergence) per
      the doc's "Errores comunes" guidance — confirm this exists or add it
      (`figures/` currently has `adiabatic_convergence.png`,
      `fixed_hamiltonian_evolution.png`, `phase_transition.png` — check
      whether any of these actually shows Trotter-step convergence, e.g.
      halving dt, or whether that analysis is still missing).
- [ ] **Extensiones opcionales** (optional but scored) — none attempted yet.
      Cheapest high-value options given time budget:
      - VQE ground-state estimate (explicitly mentioned as optional in the
        TFIM problem statement).
      - Noise mitigation (ZNE) or QEC ([[4,2,2]] or Iceberg) — doc says QEC
        is "viable for small spin chains" for this challenge specifically.
      - Fermi-Hubbard 2D extension (bigger lift — only pursue if time
        allows after core items are solid; "Ejecución sobre ambición" is
        explicit judge guidance, i.e. a flawless core beats an incomplete
        extension).
- [ ] **Presentación y claridad** — clear figures, reproducible code.
      `docs/PRESENTATION.md` is currently **empty** — this is a required
      deliverable (5-minute slide presentation) and isn't started.
- [ ] **Conexiones con los ODS** — explicit link to a specific material
      property (superconducting gap, magnetic ordering) tied to lossless
      transmission or storage, not a general mention.

---

## Required deliverables status (all must exist per submission requirements)

| Deliverable | Required | Status |
|---|---|---|
| Public GitHub repo, code + `requirements.txt` + single entry point + `README.md` | Yes | Repo exists; entry point **broken** (see §0) |
| Technical report (PDF, ≤8 pages): problem statement, classical baseline, quantum implementation summary, results with error bars, mandatory limitations section | Yes | `docs/TECHNICAL_REPORT.md` is **empty** (0 lines) |
| 5-minute slide presentation | Yes | `docs/PRESENTATION.md` is **empty** (0 lines) |
| SDK reflection, ≤200 words (what worked, what didn't, what was missing) | Yes | `docs/SDK_REFLECTION.md` is **empty** (0 lines) |

These three docs being empty is currently the single biggest gap relative to
"full marks" — the rubric can't be scored well on Explicación (20%),
Presentación (challenge-specific), or Reproducibilidad without them, no
matter how good the code is.

---

## Judge red flags to explicitly avoid (from "Orientación para los jueces")

- [ ] Do **not** claim "quantum advantage" without a scaling comparison —
      the doc explicitly warns against this, and repeatedly stresses that
      classical ED/DMRG beat quantum devices for N ≤ 50 here.
- [ ] Do not omit the limitations section (mandatory).
- [ ] Do not cherry-pick the best run — report the full `dt`/N sweep, not
      just the best-converged case.
- [ ] If hardware results are reported, include noise analysis alongside
      them — don't report hardware numbers without discussing noise.
- [ ] Code must run clean from a fresh environment — this is checked
      explicitly and penalizes *every* criterion if it fails.

---

## Suggested order of work (given current repo state)

1. Fix `main.py` / entry-point ambiguity (blocks reproducibility + everything else).
2. Fix `config.py` grid to match required `h/J ∈ {0.5, 1.0, 2.0}` at N=8 (ED) and N=6 (quantum comparison); add a multi-N scaling run (≥2 sizes, ideally 3+).
3. Add a Trotter `dt`-convergence plot (halve `dt`, confirm convergence) if not already present in `figures/`.
4. Write `docs/TECHNICAL_REPORT.md`, `docs/PRESENTATION.md`, `docs/SDK_REFLECTION.md` — currently all empty, all required.
5. Get at least one run on the Quantinuum H2 emulator (10% of the grade is gated on this alone).
6. Fill in `tests/test_hamiltonian.py` and `tests/test_trotter.py` per the README's own claims.
7. If time remains: pick one optional extension (VQE ground state, ZNE, or QEC) rather than spreading across several — the doc rewards a polished core over an ambitious-but-incomplete extension.
