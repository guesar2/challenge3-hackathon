# Iceberg Code (arXiv:2211.06703) — Implementation Plan

## Why

`run_n_scaling.py`'s noisy section is currently a placeholder
(`run_noisy_stub()`), and its own docstring says explicitly: it's blocked on
"a QEC-encoded circuit that doesn't exist in this project yet." This plan
closes that gap using the Iceberg code (Self, Benedetti & Amaro,
*Protecting Expressive Circuits with a Quantum Error Detection Code*,
arXiv:2211.06703), applied directly to this repo's existing TFIM
quench/adiabatic/VQE circuits, run against Quantinuum's H2-Emulator (noisy)
via `qnexus`, with real-time discard decided by native pytket classical
control flow.

Per your decisions: (1) integrate into the existing TFIM/VQE circuits rather
than build a standalone demo, (2) decide discard in real time rather than
pure post-hoc post-selection, (3) validate everything against `qnexus`
directly (H2-1LE for cheap functional checks, H2-Emulator for the real noisy
demonstration) rather than the free local emulator, since the whole point is
to see a real noisy backend's errors get caught — a noiseless run trivially
discards nothing.

**Decided against the Wasm QEC decoder toolkit**: initial research proposed
using it for the real-time discard decision, but pulling the actual
Quantinuum toolkit page (verbatim code blocks, not a paraphrase) confirmed
that `pytket`'s native classical control flow — `if_bit`/`if_not_bit`,
`reg_eq`/`reg_lt`/etc., bitwise `&`/`|`/`^` on `BitRegister`s, and the
`condition=` kwarg on gates (`pytket.circuit`, see the tket manual's
classical-and-conditional-operations section) — already expresses and acts
on the Iceberg code's discard rule (`a1 OR a2 OR parity(S_Z) != 0`, a simple
boolean combination) with no Wasm module, no Rust/C toolchain, and no
`cargo`/`clang`/`cmake` setup required. Wasm exists for decoders needing
genuine custom computation (lookup tables, cross-round counters, MWPM) —
none of which this code's discard rule needs. Per your instruction, it's
dropped entirely from this plan.

## Code recap

The Iceberg code is a `[[k+2, k, 2]]` CSS stabilizer code: `k` logical
qubits `[k] = {1,...,k}` map onto `n = k+2` physical qubits
`[n] = [k] ∪ {t, b}`, plus 2 reusable ancillas (`a1`, `a2`) for syndrome
extraction — `k+4` physical qubits total, constant overhead.

- Stabilizers: `S_X = ⊗_{i∈[n]} X_i`, `S_Z = ⊗_{i∈[n]} Z_i`.
- Logical single-qubit ops: `X̄_i = X_i X_t`, `Z̄_i = Z_i Z_b` for each `i∈[k]`.
- Every two-qubit **physical** operator `σ_i σ_j` (`i,j∈[n]`) is *some*
  logical operator — including global operators on `k` or `k−1` logical
  qubits compiled onto a single physical pair (paper Eqs. 1–12, e.g.
  `⊗_{j∈[k]} X̄_j = X_t X_b`). This is the "iceberg" naming: small physical
  footprint, big logical operator hiding underneath.
- Universal logical gate set: `exp(-iθ P^σ_{ij}/2)` for any physical pair
  `i,j`, compiled non-fault-tolerantly as **one MS gate + up to four
  single-qubit Cliffords** — ideal for Quantinuum's all-to-all
  trapped-ion connectivity.
- Init, syndrome measurement, and final measurement circuits *are*
  fault-tolerant (no single fault → undetected logical error), using the
  `ABBB...BA` CNOT-ordering flagged-circuit construction from Chao &
  Reichardt (arXiv:1705.02329, their Ref. [31]) adapted to these
  stabilizers. **Gate order in these circuits is safety-critical** — the
  paper states this explicitly.
- Decoding is trivial by construction: read `a1`, `a2`, and the
  reconstructed `S_Z` parity; if any is `-1`, discard the shot. No MWPM,
  no lookup table.

## Design implication specific to this repo's TFIM circuits

This is the one non-obvious cost worth flagging up front, because it
changes the expected circuit depth/2-qubit-gate count materially:

- The TFIM ZZ-coupling layer maps **1:1**: `Z̄_iZ̄_j = Z_iZ_j` (Eq. 6), so
  each `ZZPhase` in `tket_circuit.py`'s edge-colored layer compiles to one
  MS gate on the *same* physical pair, and different-color edges stay
  parallel (none of them touch `t`/`b`). Depth here barely changes.
- The TFIM transverse-field layer does **not**: each logical `X̄_i` is
  `X_i X_t` — every one of the `k` single-qubit `Rx` gates in one Trotter
  layer becomes a two-qubit MS gate through the *same shared* physical
  qubit `t`. A trapped ion can only be in one two-qubit gate at a time, so
  what was one parallel `Rx` layer becomes a **sequential ladder of `k` MS
  gates through `t`** per Trotter step. This is exactly what the paper's
  own QV encoding notes: "the compiled physical circuit does no longer
  present the parallel form of the logical circuit." Expect the
  transverse-field term to dominate 2-qubit gate count and depth growth
  with `N`, not the coupling term.
- `k` must be even for the code's demonstrated sign conventions (Eq. 12)
  — conveniently, every existing `N`/`k` used in `config.py` already is
  (`H2_N=4`, `H2_ADIABATIC_N=6`, `H2_VQE_N=6`, `N=6`).
- TFIM's Trotter circuit only ever needs `RX_i` and `RZZ_ij` logical
  gates — no `RY`, no global operators — so the universal-gate-set
  implementation needed for *this* integration is a strict subset of the
  paper's full set. (Still worth implementing the rest for the
  fault-tolerance test suite, see below.)

## Phase 0 — primary sources to pull before writing circuit code

Do **not** reverse-engineer the fault-tolerant init/syndrome circuits from
the main-text prose alone (I only have the paper's *text*, not the Fig. 1
circuit diagrams, and the paper explicitly warns CNOT order is
safety-critical). Pull these first:

1. **Chao & Reichardt, arXiv:1705.02329** ("Quantum error correction with
   only two extra qubits") — the literal flagged GHZ-prep and
   `ABBB...BA` syndrome-extraction circuits the Iceberg paper adapts.
2. **Zenodo 10.5281/zenodo.8318683** — the Iceberg paper's own research
   data release. Likely contains the actual pytket circuit-construction
   code used for the experiments — ground truth beats re-derivation.

Already confirmed (verbatim, from the tket manual's
classical-and-conditional-operations page): the native conditional API
this plan relies on for real-time discard —
`from pytket.circuit import if_bit, if_not_bit, reg_eq, reg_lt, ...`,
gate-level `condition=` kwarg, and bitwise `&`/`|`/`^` on `BitRegister`s
and their bits — is real, current pytket API. No further toolkit research
needed on that front.

## File layout

New modules, following this repo's existing one-purpose-per-file
convention (`tket_circuit.py`, `qnexus_backend.py`, `config.py`, etc.):

```
src/
├── iceberg_code.py           # stabilizers, logical-operator table (Eqs 1-12),
│                             #   k_to_n(k), even-k validation — pure Python/
│                             #   numpy, no pytket dependency, so Phase 1 tests
│                             #   run without ever touching qnexus
├── iceberg_circuits.py       # pytket circuit builders: build_iceberg_init(k),
│                             #   build_iceberg_syndrome_measurement(k),
│                             #   build_iceberg_measurement(k), and the
│                             #   universal-gate compilers: compile_logical_rx,
│                             #   compile_logical_rz, compile_logical_rxx,
│                             #   compile_logical_ryy, compile_logical_rzz,
│                             #   compile_logical_global (MS + <=4 Cliffords each)
├── iceberg_tfim_circuit.py   # build_iceberg_quench_circuit / _adiabatic_circuit:
│                             #   same signature shape as tket_circuit.py's
│                             #   builders, but emits init -> [layer;
│                             #   syndrome-measurement every M layers] -> final
│                             #   measurement, using iceberg_circuits.py
├── iceberg_decode.py         # classical post-processing: reconstruct S_Z
│                             #   parity from raw shots, per-shot discard mask,
│                             #   decode logical Z_i bits, discard-rate stats
│                             #   (mirrors paper Fig 2c/3b) — feeds into the
│                             #   existing shot_observables.py pipeline. Also
│                             #   builds the native discard BitRegister/
│                             #   condition= expression shared with the
│                             #   real-time early-exit gating in
│                             #   iceberg_tfim_circuit.py, so the mid-circuit
│                             #   condition and the final post-processing
│                             #   check are provably the same boolean rule.
└── run_iceberg_qec.py         # orchestration: wires into run_n_scaling.py's
                                #   run_noisy_stub() replacement; also exposes
                                #   a standalone small-k survival-probability/
                                #   discard-rate sweep (paper-style validation)

tests/
└── test_iceberg_code.py       # stabilizer (anti)commutation, Eqs (1)-(12)
                               #   identities, GHZ-init statevector check,
                               #   fault-tolerance-by-injection (mirrors Supp.
                               #   Fig. 1: inject every single-qubit Pauli
                               #   after every gate location in init/syndrome/
                               #   measurement circuits, assert zero undetected
                               #   logical errors) — numpy/qiskit statevector,
                               #   free, no qnexus
```

`config.py` additions (mirroring existing `H2_*` naming):

```python
ICEBERG_K = 4                    # even; reuse H2_N-scale values as k grows
ICEBERG_SYNDROME_EVERY = 1        # Trotter layers between syndrome rounds
ICEBERG_DEVICE_NAME = "H2-Emulator"   # noisy — no local-emulator fallback here
ICEBERG_SHOTS = 200
ICEBERG_EARLY_EXIT = True        # native condition= gating skips remaining
                                  # gates once a1/a2/S_Z flags an error;
                                  # False -> run to completion, post-select
                                  # only (useful as an A/B comparison of the
                                  # runtime saving from early exit)
```

## Testing strategy — offline first, qnexus only once proven correct

1. **Tier 0 (free, offline, numpy/qiskit `Statevector`)**: stabilizer
   algebra, all 12 logical-operator identities, GHZ-init correctness,
   and — most important — the fault-tolerance-by-injection test that
   reproduces the paper's own Supp. Fig. 1 method (inject every
   single-qubit Pauli after every gate in init/syndrome/measurement,
   propagate, confirm zero undetected logical errors). Run at a modest
   `k` (4–6), not the paper's `k=16`, to keep CI runtime sane — the
   argument for fault-tolerance doesn't depend on `k`.
2. **Tier 1 (qnexus, compile-only, no quota cost)**: `qnx.compile()` the
   full encoded circuit against `ICEBERG_DEVICE_NAME` and confirm it
   routes/rebases without error — this is a classical Nexus-side pass,
   not hardware-queued.
3. **Tier 2 (qnexus, quota-costing, explicit approval gate)**: one small
   pilot — lowest `k`, shallow depth, few shots — end-to-end on
   `H2-Emulator`, checking the discard rate and decoded survival
   probability are sane (near-1 at shallow depth) before scaling up.
4. **Tier 3**: the real sweep — `run_iceberg_qec.py` feeding
   `run_n_scaling.py`'s noisy section across `N_SCALING_NOISY_MAX`,
   replacing `run_noisy_stub()`.

Every qnexus-costing step (Tier 2 and 3) requires your explicit
go-ahead before submission, same as this repo's existing
`RUN_ON_H2_EMULATOR` convention.

## Real-time early exit, without Wasm

The mechanism that actually saves runtime (paper: "a conditional exit
during runtime... we expect this to halve the experimental runtime of a
rejected circuit") is pytket's native gate-level `condition=` /
`if_bit`/`if_not_bit` conditional execution, confirmed via the tket
manual (verbatim code, see Phase 0). After each syndrome-measurement
round, compute a `discard` bit (`a1 | a2 | reduce_xor(partial S_Z
bits)`, all native `BitRegister` bitwise ops) and wrap every subsequent
gate in `condition=if_not_bit(discard)`. `iceberg_decode.py`'s discard
rule is defined once and shared between this mid-circuit gating and the
final post-processing check, so they can't drift apart.

## Outcome (implemented on `iceberg-qec-impl`)

Everything below Tier 1 (Tiers 2/3, the actual qnexus submission) is
built and gated behind `config.ICEBERG_RUN_ON_H2_EMULATOR` (off by
default) — nothing has been submitted to qnexus; no quota spent.

Two corrections worth recording, since both came from real dead ends
during implementation (see `[[iceberg-qec-fault-tolerance-criterion]]`
and `[[pytket-get-commands-reordering]]` in memory for the durable
version):

1. **The fault-tolerance test criterion was wrong at first.** A single
   fault propagating to a large (weight > 1) physical error is *not*
   automatically a violation — the actual (Gottesman) criterion, confirmed
   directly with the paper's author, is that whenever a round's own flags
   don't fire, the result must be either exactly correct or *guaranteed*
   to flip `S_X`/`S_Z` on some future check. An overly strict "weight ≤ 1"
   version of the test rejected the paper's own correct `ABBB...BA`
   construction, sending real effort into unnecessary workarounds (extra
   flag qubits, full ladder duplication) before the mistake was caught.
2. **A `pytket` gotcha independently corrupted the same tests**:
   `Circuit.get_commands()` reorders gates by qubit dependency, not
   insertion order, so an index computed from one circuit's command list
   is not valid against a different (e.g. merged) circuit's. Fixed by
   fetching `get_commands()` exactly once per circuit and reusing that
   same list everywhere.

With both fixed, the paper's actual minimal 2-ancilla `ABBB...BA`
syndrome-measurement circuit — not a heavier workaround — passes
exhaustive fault injection at k=2,4,6.

Final architecture: `iceberg_code.py` (stabilizer/logical-operator
algebra, Tier 0), `iceberg_circuits.py` (init, `ABBB...BA` syndrome
measurement, RX/RZZ/RXX/RYY compilers), `iceberg_decode.py` (post-hoc
discard + logical-bit decode), `iceberg_tfim_circuit.py` (TFIM
integration — each Trotter step split into its RZZ and RX half-steps,
with a syndrome-measurement round after *each* half, per the paper
author's own guidance, rather than only at full-step boundaries), native
`condition=if_not_bit(discard)` early exit (no Wasm — dropped per your
instruction once the native API was confirmed sufficient),
`run_iceberg_qec.py` + `qnexus_backend.submit_iceberg_quench_batch` wired
into `run_n_scaling.py`, replacing the old placeholder framing.

Verified for free, before ever touching qnexus: the full encoded circuit
compiles to H2's native gateset and runs correctly on the local
(no-login, no-quota) `H2-1LE` backend, with exactly 0% discard rate (as
expected — no noise source at all) and physically sensible TFIM
observables, for both `early_exit=True` and `False`.

## Milestone checklist

- [x] Phase 0: research (Chao–Reichardt confirmed the stabilizer/logical
      structure independently; native conditional API confirmed;
      Zenodo data checked, no source code, only plot CSVs)
- [x] `iceberg_code.py` + Tier 0 tests passing (stabilizers, logical ops,
      fault-tolerance-by-injection) — 47 tests, k=2,4,6
- [x] `iceberg_circuits.py`: init/syndrome/measurement circuits + RX/RZZ/
      RXX/RYY compilers, verified against Tier 0 tests
- [x] `iceberg_tfim_circuit.py`: encoded quench circuit with native
      conditional early exit, verified via free local `H2-1LE` run
- [ ] Tier 2 pilot run on qnexus (explicit approval needed) — sanity-check
      discard rate/survival probability against real noise
- [x] `run_iceberg_qec.py` wired into `run_n_scaling.py`, replacing
      `run_noisy_stub()`'s placeholder framing
- [ ] Update `README.md`'s "Limitaciones honestas" #1 and
      `docs/TECHNICAL_REPORT.md` once real (Tier 2/3) qnexus data exists
