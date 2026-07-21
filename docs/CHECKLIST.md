# Challenge 3 (TFIM) — Full-Marks Checklist

Derived from `docs/hackathon.docx`. Ordered by rubric weight, highest first.
Two rubrics apply: the **general submission rubric** (all challenges, 100%
total) and the **Challenge 3–specific rubric** (4-point scale per criterion).
Status notes reflect the repo as of this session (2026-07-21, updated
after adding X-basis magnetization measurement to the H2 adiabatic
phase-transition sweep, fixing an under-converged adiabatic ramp at
h/J=0.5 that this new ⟨X⟩ measurement exposed, and re-confirming on a
real qnexus run).

---

## 0. Deal-breakers (checked before anything else is scored)

- [x] **`main.py` fixed.** It now inserts `src/` onto `sys.path` and calls
      `ftim_main.main()` — verified working, no crash. (The old broken
      version importing `trotter_circuit`/`observables` is gone.)
- [x] Repo is actually public (`git remote -v` shows
      `github.com/guesar2/challenge3-hackathon`)
- [ ] Fresh-clone smoke test: `git clone`, `python -m venv`, `pip install -r
      requirements.txt`, run the entry point, confirm figures/numbers appear
      with no manual fixes. **Not yet a true fresh clone**, but `python
      main.py` was run end-to-end in the existing checkout this session
      (all 5 stages, including a real qnexus H2 submission) and completed
      clean, exit 0, all figures regenerated. Still do the actual
      fresh-clone version once before final submission.
- [ ] Honest limitations section present (mandatory) — still blocked on
      `docs/TECHNICAL_REPORT.md` being empty (see §"Required deliverables"
      below).

---

## 1. Implementación cuántica — 30% (Intento 10% / Buena ejecución 10% / Ejecución en hardware real 10%)

- [x] **Intento (10%)** — TFIM Hamiltonian + Trotter circuit, working.
      `src/pauli_ops.py` + `src/circuits.py` + `src/trotter_simulation.py`.
- [x] **Buena ejecución (10%)** — clean, documented, reproducible code.
      `config.py` documents `ADIABATIC_DT`/`QUENCH_DT` inline with
      reasoning; make sure these values land explicitly in the technical
      report text too, not just code comments.
- [x] **Ejecución en hardware cuántico real (10%)** — **done.**
      `config.RUN_ON_H2_EMULATOR = True`, and real (non-local, qnexus)
      results exist on disk: `data/h2_emulator_latest.json` /
      `h2_emulator_raw_latest.json` (quench/adiabatic, saved
      2026-07-20T03:18 UTC) and `data/h2_vqe_latest.json` +
      ~120 `h2_vqe_raw_*.json` snapshots (VQE, saved ~05:05–06:11 UTC same
      day) — both against the real Quantinuum H2-1LE emulator via qnexus,
      not `local_emulator_backend`. Figures `h2_vs_ed.png`,
      `h2_vs_ed_time.png`, `h2_phase_transition.png`,
      `vqe_convergence.png`, `h2_phase_transition_vqe.png` reflect this.
      ⚠️ Note: the real qnexus VQE run predates this session's ansatz work
      (see §Extensiones below) — it used the old 6·N-parameter
      hardware-efficient ansatz at `H2_VQE_MAX_ITERS=15`, which COBYLA
      needs ~500 evaluations to actually converge (measured), so those
      real-hardware VQE numbers are **not yet converged**. A follow-up real
      run with the new HVA ansatz (converges in ~40 evaluations
      noiselessly) would fix this cheaply, but costs real qnexus quota —
      get explicit go-ahead before submitting one.
      A fresh full `python main.py` run this session re-confirmed the
      quench H2 path too: max deviation vs. ED at h/J=0.5/1.0/2.0
      was 2.68%/3.49%/4.01% in `⟨Z⟩` and 7.04%/9.52%/**17.79%** in
      `⟨Zᵢ Zᵢ₊₁⟩` — the `⟨Zᵢ Zᵢ₊₁⟩` numbers exceed the challenge's <5%
      target at every h/J on real hardware (expected: H2-1LE's shot noise
      adds on top of whatever local-Trotter bias already exists — see the
      next bullet on the *local* (noiseless) version of this same gap).
      **This quench-pipeline gap (`H2_DT=0.1`, `H2_STEPS=5`) was not
      touched this session** — see the ⚠️ below.
      - **New this session: ⟨X⟩ magnetization added to the H2 phase-transition
        sweep** (`run_h2_emulator.run_phase_transition`), which previously
        only measured ⟨Z⟩/⟨Zᵢ Zᵢ₊₁⟩. Each h/J point now submits a paired
        Z-basis and X-basis circuit (H gate before measurement), batched
        into the same qnexus compile/execute call, so this **~doubles the
        qnexus quota cost per adiabatic-sweep run** going forward — noted
        here explicitly since it's a recurring cost, not a one-time one.
        `⟨X⟩` uses a plain shot mean (`shot_observables.bitstrings_to_mx`),
        not the RMS formula used for `⟨Z⟩` — `⟨X⟩` doesn't vanish by Z₂
        symmetry the way `⟨Z⟩` does, since the `-h·ΣX` field polarizes the
        ground state along +X.
      - Adding `⟨X⟩` immediately surfaced a real bug: h/J=0.5's ramp
        (H_INIT=4.0 → 0.5) passes *through* the gapless h/J=1 point
        mid-ramp, which the existing critical-slowing-down special case
        only handled for a target *landing* on h/J=1, not one transiting
        it. First real qnexus run: `⟨X⟩` at h/J=0.5 was **21.6% off ED**
        (⟨Z⟩/⟨Zᵢ Zᵢ₊₁⟩ passable at ~1.4%/1.4% but not exercised strongly by
        that particular deviation). Fixed by adding
        `config.H2_ADIABATIC_TRANSIT_TIME_FACTOR=4` (400 ramp steps for a
        transiting target, vs. 200 for a target landing exactly on h/J=1),
        tuned via a free local sweep (200/300/400/500 steps → 9.00%/2.34%/
        0.43%/3.22% ⟨X⟩ deviation; 500's rise is shot-noise scatter, not a
        trend). Re-confirmed on a second real qnexus run — full sweep now:

        | h/J | ⟨Z⟩ | ⟨X⟩ | ⟨Zᵢ Zᵢ₊₁⟩ |
        |---|---|---|---|
        | 0.5 | 0.03% | 6.11% | 0.02% |
        | 1.0 | 0.80% | 1.08% | 1.34% |
        | 2.0 | 1.72% | 0.45% | 3.97% |

        8/9 values under the 5% target; the one exception (⟨X⟩ at
        h/J=0.5, 6.11%) is shot noise at `H2_ADIABATIC_SHOTS=2000` — it
        bounced between 0.03%–9% across repeated local runs at the same
        step count, consistent with statistical scatter rather than a
        systematic bias. Bumping shots further would tighten it at
        proportionally more quota cost — not done, pending explicit
        go-ahead. `figures/h2_phase_transition.png` regenerated with a
        third `⟨X⟩` panel (`plot_h2_phase_transition` in `src/plotting.py`).

## 2. Explicación — 20%

- [ ] Team can give a coherent technical explanation end to end
      (Hamiltonian → Trotter layer → propagation → observables →
      comparison). Presentation/defense criterion, not a file — prepare to
      walk through `src/pauli_ops.py`, `src/circuits.py`,
      `src/trotter_simulation.py`, `src/sweep_schedule.py`, and now also
      `src/tket_circuit.py`'s HVA ansatz and `src/vqe.py`'s COBYLA loop.

## 3. Comparación y escalado — 20%

- [x] Direct quantum-vs-classical comparison on the same problem instance
      (Trotter vs. ED for matching N, h) — have this.
- [ ] Scaling across **2+ problem sizes**. ⚠️ **Still not wired into the
      live pipeline.** `config.py` runs a single `N=6` throughout
      (`H2_ADIABATIC_N=6`, `H2_VQE_N=6`). `src/ed_figures.py` *does*
      contain an N∈{6,8} sweep and an N=8 baseline section, but it is
      **dead code** — nothing imports it (`ftim_main.py`, `run_ed.py`,
      etc. don't call it; last touched only in the original "Implement
      FTIM simulation" commit) — so it produces no figures today. Either
      wire it into `run_ed.py`/`ftim_main.py` and actually run it, or
      write a small equivalent that runs and is checked in as a real
      figure.
- [ ] Honest extrapolation: state where classical methods still win.
- [ ] Classical baseline comparison cited from a published source (see
      Línea base clásica below — README already has the citations, port
      them into the technical report).

## 4. Línea base clásica — 15%

- [x] ED baseline with valid mechanism: `src/exact_diagonalization.py`
      (sparse `eigsh`), `ed_baseline` covers `h/J` sweeps.
- [x] Citations present in README (`Suzuki 1976`, `Sachdev`,
      `Ebadi et al. 2021`) — **still need to confirm these actually land in
      `docs/TECHNICAL_REPORT.md`**, not just the README, since the report
      is currently empty (see §0).
- [ ] ED run for the specific required grid: `h/J ∈ {0.5, 1.0, 2.0}` at
      **N=8** specifically (doc's "Línea base clásica" section). Current
      `config.H_VALUES = (0.5, 1.0, 2.0)` now matches the h/J grid
      (previously had a stray `1.5` — **fixed**), but `config.N = 6`
      throughout — no N=8 ED run in the live pipeline (same dead-code gap
      as the scaling item above; `ed_figures.py` has an N=8 section that
      never runs).

## 5. Reproducibilidad — 10%

- [x] `requirements.txt` — has `qiskit-aer`, `networkx`, `pytket-quantinuum`,
      `pytket_pecos`, `qnexus`, etc.; matches what's imported.
- [x] **Single entry point** (`main.py` → `ftim_main.main()`) now runs
      **5 stages**: ED, adiabatic, quench, `dt`-convergence (new this
      session), and H2 emulator quench/adiabatic (gated by
      `config.RUN_ON_H2_EMULATOR`, currently `True`, so it runs by
      default). Verified this session: a full `python main.py` run
      completed clean end-to-end, including a real qnexus H2 submission,
      regenerating every figure. **VQE is still not included** —
      `run_vqe()` is only invoked via `python run_h2_emulator.py --vqe` /
      `--vqe --local`, never from `ftim_main.main()`. If the report cites
      VQE numbers, spell out that extra command, since `main.py` alone
      won't reproduce them.
- [ ] `pytest tests/` still collects **0 tests** —
      `tests/test_hamiltonian.py` and `tests/test_trotter.py` are still
      0 bytes. Not fixed this session. README still promises
      Hermiticity/norm-conservation/Trotter-convergence/ED-vs-Qiskit
      tests that don't exist.

## 6. Impacto en los ODS — 5%

- [x] README has an ODS/SDG table (`## Conexión con los ODS`) covering
      7/9/12/13 with per-SDG connections and citations.
- [ ] Confirm the causal chain (TFIM → strongly-correlated materials →
      real-world outcome → SDG) is made *explicit*, not just tabular
      associations, and that it's **ported into
      `docs/TECHNICAL_REPORT.md`** (currently empty, so it's nowhere yet
      in the report itself).

---

## Challenge-3–specific rubric (4-point scale: Excelente/Bueno/Necesita mejorar)

- [x] **Corrección física** — **fixed this session.** `config.QUENCH_DT`
      was `0.05`, giving max deviation in `⟨Zᵢ Zᵢ₊₁⟩` vs. ED of 0.29%
      (h/J=0.5), 3.12% (h/J=1.0), **8.46% (h/J=2.0)** — out of spec at
      h/J=2.0. Halved to `QUENCH_DT=0.025` (with `QUENCH_STEPS` doubled to
      800 to keep total evolution time fixed at 20.0), as predicted by the
      dt-convergence sweep's O(dt²) scaling. Re-ran `run_quench.py`:
      max deviation in `⟨Zᵢ Zᵢ₊₁⟩` is now 0.07% / 0.77% / **2.10%** across
      h/J=0.5/1.0/2.0 — comfortably under 5% everywhere. Re-ran
      `run_dt_convergence.py` too, confirming the O(dt²) scaling still
      holds at the new base `dt` (error ratios still 4.00x at the new
      finest step sizes). `figures/fixed_hamiltonian_evolution.png` and
      `figures/dt_convergence.png` regenerated.
- [x] **Implementación del circuito con error analysis** — **done this
      session.** `src/run_dt_convergence.py` (new) sweeps
      `dt ∈ {0.2, 0.1, 0.05, 0.025, 0.0125}` at fixed total evolution time,
      compares local Trotter vs. ED at each, and confirms clean O(dt²)
      scaling: error ratios of **4.00x, 4.00x, 4.01x** between the two
      finest `dt` values across h/J=0.5/1.0/2.0 (theoretical expectation
      for the symmetrized 2nd-order Rx/Rzz/Rx step is exactly 4x per
      halving). Plotted in the new `figures/dt_convergence.png`
      (log-log, per-h/J, with an `O(dt²)` reference line — confirmed
      visually parallel to the data). Wired into `ftim_main.main()` as
      stage 4/5, so it now runs via plain `python main.py` — no longer
      dead code. (`src/ed_figures.py`'s original `fig3_trotter_convergence`
      is still dead/unused, but is now superseded by this live
      equivalent.)
- [x] **Extensiones opcionales** — **VQE ground-state search implemented
      and substantially improved this session**:
      - `src/vqe.py::run_vqe_h2` now supports two ansätze via
        `ansatz="hea"|"hva"` — the original 6·N-parameter
        hardware-efficient ansatz, and a new Hamiltonian Variational
        Ansatz (`src/tket_circuit.py::build_hva_ansatz_circuit`, `p`
        layers of (ZZPhase, Rx) mirroring the TFIM's own term structure,
        just `2p` parameters).
      - Noiseless test: HVA hits the *exact* ED ground energy at
        h/J∈{0.5,1.0,2.0} with `p=4` (8 params), where the HEA only
        reached ~2-3% of ED even after ~500 COBYLA evaluations.
      - Against the real (shot-noise) local emulator, HVA's COBYLA
        self-converges in **~36–46 evaluations** vs. hundreds needed for
        the HEA — matching the convergence speed of comparable
        gradient-based VQE demos (cross-checked against PennyLane's
        `tutorial_quantum_phase_transitions`, which uses exact analytic
        gradients rather than gradient-free COBYLA on noisy shots —
        that's the mechanism for their faster convergence, not a
        different ansatz or system size).
      - `config.H2_VQE_ANSATZ="hva"`, `H2_VQE_P=4` now the default; local
        run gets a generous free budget (`H2_VQE_SHOTS_LOCAL=4000`,
        `H2_VQE_MAX_ITERS_LOCAL=500`, COBYLA self-terminates well before
        the cap); qnexus keeps the small quota-safe `H2_VQE_MAX_ITERS=15`
        (flagged in `config.py` as below what HVA needs to fully converge
        on a real run — intentionally left alone pending explicit
        go-ahead to spend more quota).
      - Latest local run (h/J=0.5/1.0/2.0): energy within
        8.3%/1.1%/0.4% of ED; figures at `figures/vqe_convergence_local.png`,
        `figures/h2_phase_transition_vqe_local.png`.
      - Other optional extensions (ZNE/QEC, Fermi-Hubbard 2D) — still not
        attempted; not needed given VQE is now solid — "ejecución sobre
        ambición" per judge guidance.
- [ ] **Presentación y claridad** — `docs/PRESENTATION.md` still **empty**
      (0 lines). Required deliverable, not started.
- [ ] **Conexiones con los ODS** — needs a specific material-property tie
      (superconducting gap, magnetic ordering) → lossless
      transmission/storage, ported into the technical report, not just the
      README table.

---

## Required deliverables status (all must exist per submission requirements)

| Deliverable | Required | Status |
|---|---|---|
| Public GitHub repo, code + `requirements.txt` + single entry point + `README.md` | Yes | Repo exists, `main.py` **fixed and working**; visibility not re-verified this session |
| Technical report (PDF, ≤8 pages) | Yes | `docs/TECHNICAL_REPORT.md` still **empty** (0 lines) |
| 5-minute slide presentation | Yes | `docs/PRESENTATION.md` still **empty** (0 lines) |
| SDK reflection, ≤200 words | Yes | `docs/SDK_REFLECTION.md` still **empty** (0 lines) |

These three docs being empty is still the single biggest gap relative to
"full marks" — the code/results are in much better shape than the docs
(main.py fixed, real H2 hardware runs exist for both quench/adiabatic and
VQE, VQE itself substantially improved with the HVA ansatz), but none of
that is written up anywhere gradeable yet.

---

## Judge red flags to explicitly avoid (from "Orientación para los jueces")

- [ ] Do **not** claim "quantum advantage" without a scaling comparison —
      doubly true now that the N-scaling figure is dead code, not a live
      result — don't cite it as done until `ed_figures.py` (or an
      equivalent) is actually wired in and run.
- [ ] Do not omit the limitations section (mandatory, still blocked on the
      technical report).
- [ ] Do not cherry-pick the best run — the real qnexus VQE data
      (~120 raw snapshots) spans an under-converged HEA config; if citing
      VQE numbers, report the full picture (energy history, not just a
      best-iterate cherry pick) — `vqe.py` already tracks
      `energy_history` for this reason.
- [ ] Do not claim "<5% deviation" as a blanket pass without qualification.
      **Fixed locally** this session (`QUENCH_DT` halved to 0.025, h/J=2.0
      `⟨Zᵢ Zᵢ₊₁⟩` deviation now 2.10%, all h/J under 5%). **Still not true
      on real H2 hardware** — the qnexus quench run uses a separate config
      (`H2_DT=0.1`, `H2_STEPS=5`, deliberately shallow to limit hardware
      cost) and showed 17.79% at h/J=2.0; that number wasn't touched by
      this fix. Report the local vs. hardware numbers separately, and
      either accept/discuss the h/J=2.0 hardware shortfall in the
      limitations section, or reduce `H2_DT` too (costs more real qnexus
      quota per submission — ask before doing that).
- [ ] Hardware results (H2 emulator, both quench/adiabatic and VQE) are
      now real — make sure the report discusses that H2-1LE is *exact
      noiseless emulation with only shot noise*, not a physical noise
      model (see `qnexus_backend.py`'s docstring), so "hardware run" claims
      are accurately scoped.
- [ ] Code must run clean from a fresh environment — do the fresh-clone
      smoke test (§0) before submitting; not done this session.

---

## Suggested order of work (given current repo state)

1. ~~Fix `main.py` / entry-point ambiguity.~~ **Done.**
2. ~~Fix `config.py`'s `h/J` grid to `{0.5, 1.0, 2.0}`.~~ **Done.** Still
   need: an actual N=8 ED run and a ≥2-size scaling comparison wired into
   something that runs (currently only exists as dead code in
   `ed_figures.py`).
3. ~~Add a live `dt`-halving Trotter convergence plot.~~ **Done** —
   `src/run_dt_convergence.py`, wired into `ftim_main.main()` as stage 4/5,
   confirms O(dt²) scaling. **Also fixed the follow-up it surfaced**:
   `QUENCH_DT` halved (0.05→0.025, `QUENCH_STEPS` doubled to keep total
   time fixed) — h/J=2.0's `⟨Zᵢ Zᵢ₊₁⟩` deviation is now 2.10%, comfortably
   under the <5% target at every h/J (locally; the real-hardware qnexus
   number at `H2_DT=0.1` is untouched — see judge red flags above).
4. **Write `docs/TECHNICAL_REPORT.md`, `docs/PRESENTATION.md`,
   `docs/SDK_REFLECTION.md`** — all still empty, all required, now the
   single highest-leverage remaining gap since the code/results side is
   in good shape.
5. ~~Get at least one run on the Quantinuum H2 emulator.~~ **Done** for
   both quench/adiabatic and VQE (via qnexus, real non-local runs on
   disk). Optional follow-up: a fresh qnexus VQE run with the new HVA
   ansatz once `H2_VQE_MAX_ITERS` is bumped (costs quota — ask first).
6. Fill in `tests/test_hamiltonian.py` and `tests/test_trotter.py` — still
   both empty.
7. ~~Pick one optional extension.~~ **Done — VQE**, now with a
   Hamiltonian Variational Ansatz that converges far faster and more
   accurately than the original hardware-efficient ansatz. No need to
   pursue further extensions (ZNE/QEC/Fermi-Hubbard) given the "ejecución
   sobre ambición" guidance — better to spend remaining time on the
   docs (item 4) than a second extension.
