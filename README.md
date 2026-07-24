# Quantathon CR 2026 · Challenge 3

## Simulación de materiales para dispositivos de energía de próxima generación

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **ODS:** 7 · 9 · 12 · 13 | **Dificultad:** Avanzado | **Plataforma:** Quantinuum H2 Emulator

---

## Descripción del proyecto

Este repositorio contiene la implementación completa de la simulación del **Modelo de Ising de Campo Transverso (TFIM)** en 1D mediante evolución temporal trotterizada, con comparación sistemática contra diagonalización exacta (ED). El objetivo es detectar la transición de fase cuántica en `h/J = 1` y evaluar la fidelidad de la simulación cuántica frente a métodos clásicos.

### Modelo principal: TFIM 1D

```
H = -J Σ⟨i,j⟩ ZᵢZⱼ  -  h Σᵢ Xᵢ
```

- **J**: acoplamiento ZZ entre vecinos (ferromagnético)
- **h**: campo magnético transverso (fluctuaciones cuánticas)
- **Transición de fase cuántica**: `h/J = 1` (límite termodinámico)

### Extensión opcional: Fermi-Hubbard 2D

Codificación fermiónica (Jordan-Wigner / Bravyi-Kitaev) para redes pequeñas (4×4, 4×6), con análisis de doble ocupación y densidad por sitio.

---

## Estructura del repositorio

```
quantathon-challenge3/
├── README.md                          ← Este archivo
├── LICENSE                            ← MIT
├── requirements.txt                   ← Dependencias
├── main.py                            ← Punto de entrada único: corre el pipeline TFIM
│                                          (ftim_main.main()) y luego la extensión Fermi-Hubbard 2D
│                                          (fh_main.main()) en secuencia
│
├── src/
│   ├── ftim_main.py                   ← Orquestador: ejecuta las 11 secciones TFIM (run_*.py) en secuencia
│   ├── run_ed.py                      ← Sección 1/11 — línea base ED, ejecutable de forma independiente
│   ├── run_adiabatic.py               ← Sección 2/11 — barrido adiabático trotterizado (statevector local)
│   ├── run_quench.py                  ← Sección 3/11 — evolución "quench" ED vs. Trotter local
│   ├── run_dt_convergence.py          ← Sección 4/11 — barrido de dt a tiempo fijo, confirma O(Δt²)
│   ├── run_n_scaling.py               ← Sección 5/11 — escaneo Trotter-vs-ED de ruptura, N=4..20
│   ├── run_quantum_advantage.py       ← Sección 6/11 — costo clásico (ED) vs. costo del circuito Trotter,
│   │                                     puramente clásico, sin cuota
│   ├── run_h2_emulator.py             ← Sección 7/11 — quench contra el emulador Quantinuum H2 vía qnexus
│   │                                     (gateado por config.RUN_ON_H2_EMULATOR — consume cuota;
│   │                                      actualmente en True en config.py, ver nota más abajo)
│   ├── run_zne.py / zne_fit.py        ← Sección 8/11 — Zero-Noise Extrapolation contra el H2-Emulator real
│   │                                     (qermit Folding.circuit); mismo gate que la sección 7, cuota adicional
│   ├── run_iceberg_qec.py             ← Sección 9/11 — corrida piloto Iceberg vs. H2-Emulator real
│   │                                     (gateado por config.ICEBERG_RUN_ON_H2_EMULATOR, False por defecto —
│   │                                      no-op sin cuota a menos que se active)
│   ├── run_noise_scaling.py           ← Sección 10/11 — caracterización del ruido real de H2-Emulator vs. N
│   │                                     (hasta 26 qubits) y profundidad — sección de mayor costo de cuota
│   ├── plot_iceberg_comparison.py     ← Sección 11/11 — regenera la comparación Iceberg vs. ED/ZNE desde
│   │                                     datos ya guardados por las secciones 9 y 8, sin cuota
│   ├── iceberg_code.py, iceberg_circuits.py,
│   │   iceberg_tfim_circuit.py, iceberg_decode.py
│   │                                  ← Código de detección de errores Iceberg [[k+2,k,2]] (arXiv:2211.06703):
│   │                                     álgebra de estabilizadores, circuitos pytket, codificación del circuito
│   │                                     de quench TFIM, y decodificación clásica de los shots (usado por las
│   │                                     secciones 9 y 11 de arriba)
│   ├── config.py                      ← Parámetros de simulación (N, J, H_VALUES, dt, H2_*, H2_ADIABATIC_*, H2_VQE_*, etc.)
│   ├── pauli_ops.py                   ← Operadores de Pauli + construcción del Hamiltoniano TFIM
│   ├── exact_diagonalization.py       ← Línea base ED (estado fundamental y evolución exacta)
│   ├── circuits.py                    ← Circuitos de Trotter para el simulador local (Qiskit, edge coloring, capa simétrica)
│   ├── tket_circuit.py                ← Circuitos para hardware (pytket): quench (ZZPhase), barrido adiabático y ansatz HEA para VQE
│   │                                     (con fusión de puertas Rx en los bordes de capa Trotter -- ver Metodología)
│   ├── qnexus_backend.py              ← Envío/ejecución de circuitos pytket en Quantinuum vía qnexus (quench, adiabático, batch VQE)
│   │                                     -- consume cuota; usar para el emulador H2-1E con ruido real, hardware H2-1, o la corrida final de confirmación
│   ├── local_emulator_backend.py      ← Misma interfaz que qnexus_backend.py pero contra el emulador H2-1LE vía pytket-quantinuum + pytket-pecos,
│   │                                     ejecutado localmente (sin login, sin red, sin cuota) -- ver Metodología: por qué es la MISMA computación
│   ├── shot_observables.py            ← bitstrings_to_observables / bootstrap_observable_errors, compartido por ambos backends H2
│   ├── vqe.py                         ← Búsqueda del estado fundamental por VQE (ansatz hardware-efficient + COBYLA) contra el emulador H2
│   ├── trotter_simulation.py          ← Propagación del statevector (barrido adiabático y quench)
│   ├── sweep_schedule.py              ← Lógica de número de pasos para el barrido adiabático
│   ├── plotting.py / ed_figures.py    ← Generación de figuras (matplotlib), incluye las variantes H2/VQE
│   ├── plot_h2_comparison.py          ← Script auxiliar para regenerar gráficos H2 vs. ED a partir de datos ya guardados en data/
│   ├── reporting.py                   ← Tabla comparativa Trotter vs. ED en consola
│   ├── persistence.py                 ← Guardado/carga de resultados por etapa en data/ (JSON con timestamp + puntero *_latest.json)
│   └── fermi_hubbard.py               ← Modelo Fermi-Hubbard 2D (JW, disperso) — implementación previa, no
│                                          importada por ningún otro módulo de src/; superada por el paquete
│                                          completo fh2d_src/ (no confundir uno con el otro)
│
├── notebooks/
│   ├── 01_exact_diagonalization.ipynb ← Vacío — pendiente
│   ├── 02_trotter_simulation.ipynb    ← Vacío — pendiente
│   ├── 03_phase_transition.ipynb      ← Vacío — pendiente
│   ├── 04_optional_FermiHubbard.ipynb ← Extensión Fermi-Hubbard 2D (contenido real)
│   ├── 04_optional_fermi_hubbard.ipynb ← Vacío — duplicado de nombre del anterior, pendiente de limpieza
│   ├── TFIM_LíneaBase_SinRuido.ipynb  ← Desarrollo interactivo del TFIM sin ruido (notebook de trabajo, contenido real)
│   └── pauli_ops.py                   ← Copia de trabajo de src/pauli_ops.py para desarrollo interactivo en notebooks
│
├── figures/                           ← Figuras generadas al ejecutar `python main.py` desde la raíz
│
├── fh2d_src/                           ← Extensión opcional: Fermi-Hubbard 2D (paquete propio,
│   └── fh_main.py                        sin código compartido con src/; `python main.py` la
│                                          ejecuta también, después del pipeline TFIM — ver más abajo)
│
└── docs/
    ├── TECHNICAL_REPORT.md            ← Informe técnico (máx. 8 pp.)
    ├── PRESENTATION.md                ← Guión de la presentación (5 min) — pendiente
    ├── SDK_REFLECTION.md              ← Reflexión sobre SDK (≤200 palabras) — pendiente
    ├── PLAN.md                        ← Plan de trabajo del hackathon (nota interna)
    ├── ICEBERG_QEC_PLAN.md            ← Plan de diseño del código Iceberg QEC (nota interna)
    └── CHECKLIST.md                   ← Checklist de entregables (nota interna)
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/quantathon-challenge3.git
cd quantathon-challenge3
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate        # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota sobre el emulador H2:** `qnexus` y `pytket` (usados por `run_h2_emulator.py`
> y `vqe.py`) ya están en `requirements.txt`. Ese script requiere además una
> sesión de `qnexus` iniciada (login contra Quantinuum Nexus) y **consume una
> cuota de uso medida** — está pensado para permanecer desactivado por defecto
> (`config.RUN_ON_H2_EMULATOR = False`); el resto del pipeline no lo necesita.
> ⚠️ En este repo `config.py` actualmente tiene `RUN_ON_H2_EMULATOR = True` —
> revisa y ajusta ese valor antes de correr `main.py` / `python
> src/run_h2_emulator.py` si no querés consumir cuota.

---

## Uso

### Ejecución completa (punto de entrada único)

```bash
source venv/bin/activate
python main.py                  # desde la raíz del repo — reproduce todas las figuras/cifras
```

Este script orquesta el pipeline TFIM (`src/ftim_main.py`, once secciones) y luego la extensión Fermi-Hubbard 2D (`fh2d_src/fh_main.py`) en secuencia:
1. `run_ed.py` — `ed_baseline`: observables del estado fundamental (ED) para `config.H_VALUES`
2. `run_adiabatic.py` — `run_adiabatic_simulation`: barrido adiabático trotterizado (statevector local, Qiskit) desde `config.H_INIT` hasta cada `h` objetivo, comparado contra ED
3. `run_quench.py` — evolución "quench" desde el estado producto `|0...0⟩`, comparando `ed_time_evolution_exact` vs. `run_trotter_fixed_hamiltonian` (objetivo: <5% de desviación en ⟨Z⟩ y ⟨ZᵢZᵢ₊₁⟩)
4. `run_dt_convergence.py` — barrido de `dt` a tiempo total fijo, confirma la convergencia O(Δt²) del Trotter de segundo orden (`figures/dt_convergence.png`)
5. `run_n_scaling.py` — escaneo Trotter-vs-ED de ruptura, N=4..20 (`figures/n_scaling.png`)
6. `run_quantum_advantage.py` — costo clásico medido (ED) vs. costo del circuito Trotter (puertas/profundidad vs. N) — puramente clásico, sin cuota (`figures/quantum_advantage_scaling.png`)
7. `run_h2_emulator.py` — el mismo circuito de quench, construido en pytket, enviado al emulador Quantinuum H2 vía `qnexus` — gateado por `config.RUN_ON_H2_EMULATOR`, **actualmente `True`** (ver nota de cuota arriba: esta sección consume cuota real en cada corrida a menos que se ponga en `False`)
8. `run_zne.py` — Zero-Noise Extrapolation contra el H2-Emulator real (qermit `Folding.circuit` + `src/zne_fit.py`) — gateado por el mismo `config.RUN_ON_H2_EMULATOR`; consume cuota real adicional en cada corrida
9. `run_iceberg_qec.py` — corrida piloto del código de detección de errores Iceberg [[k+2,k,2]] (`src/iceberg_*.py`) contra el H2-Emulator real — gateado por `config.ICEBERG_RUN_ON_H2_EMULATOR`, **`False` por defecto** (no-op sin cuota a menos que se active)
10. `run_noise_scaling.py` — caracteriza cómo escala el modelo de ruido real de H2-Emulator con N (hasta 26 qubits) y con profundidad de circuito — gateado por `config.RUN_ON_H2_EMULATOR`; **la sección de mayor costo de cuota del pipeline** (dos corridas H2 por cada N)
11. `plot_iceberg_comparison.py` — regenera la comparación Iceberg vs. ED/ZNE a partir de los datos guardados por la sección 9 (`data/iceberg_qec_latest.json`) y la sección 8 (`data/h2_zne_latest.json`) — sin cuota; sin datos de la sección 9 (p. ej. con `ICEBERG_RUN_ON_H2_EMULATOR = False`, el valor por defecto) simplemente imprime que no encontró nada y no genera figura, sin fallar
12. `fh2d_src/fh_main.py` — extensión opcional Fermi-Hubbard 2D (paquete separado, sin código compartido con `src/`) — ver más abajo; su propio `fh2d_src/fh_config.py` también tiene `RUN_ON_H2_EMULATOR = True` por defecto, así que también consume cuota real

### Ejecución de una sola sección

Cada sección es independiente — calcula su propia línea base ED y no depende de que las demás se hayan ejecutado antes — así que se puede verificar una sin correr el resto:

```bash
python src/run_ed.py                 # solo línea base ED (rápido, puramente clásico)
python src/run_adiabatic.py          # solo el barrido adiabático
python src/run_quench.py             # solo la evolución "quench"
python src/run_dt_convergence.py     # solo el barrido de convergencia dt
python src/run_n_scaling.py          # solo el escaneo Trotter-vs-ED, N=4..20
python src/run_quantum_advantage.py  # puramente clásico, sin cuota
python src/run_h2_emulator.py        # consume cuota real (config.RUN_ON_H2_EMULATOR = True por defecto)
python src/run_zne.py                # consume cuota real (mismo gate que run_h2_emulator.py)
python src/run_iceberg_qec.py        # no-op salvo config.ICEBERG_RUN_ON_H2_EMULATOR = True — consume cuota si se activa
python src/run_noise_scaling.py      # consume cuota real — la corrida más cara del pipeline (N hasta 26)
python src/plot_iceberg_comparison.py # sin cuota — requiere datos previos de run_iceberg_qec.py y run_zne.py
```

### Funciones adicionales del emulador H2 (fuera de `main.py`)

`run_h2_emulator.py` expone tres entradas independientes, todas gateadas por
`config.RUN_ON_H2_EMULATOR` y cada una con su propio costo de cuota — solo
`run()` (quench) corre desde `main.py`/`ftim_main.py`; las otras dos se
invocan explícitamente:

```bash
python src/run_h2_emulator.py                      # run(): quench vs. tiempo (figures/h2_vs_ed_time.png)
python src/run_h2_emulator.py --phase-transition    # run_phase_transition(): barrido adiabático en hardware (figures/h2_phase_transition.png)
python src/run_h2_emulator.py --vqe                 # run_vqe(): VQE (ansatz hardware-efficient + COBYLA) por cada h/J
                                                     #   (figures/vqe_convergence.png, figures/h2_phase_transition_vqe.png)
```

`run_vqe()` (en `src/vqe.py`) busca el estado fundamental directamente vía
optimización COBYLA sobre un ansatz hardware-efficient de `6·N` parámetros,
agrupando las mediciones del Hamiltoniano TFIM en 2 circuitos (base Z para
ZZ, base X para X) mediante `pytket.partition.measurement_reduction` y
enviando ambos en un mismo batch por iteración — evita el problema de
transiciones diabáticas cerca de `h/J=1` que sí afecta a
`run_phase_transition()` (rampa adiabática de longitud fija en hardware).
`load_last_run()` recupera los últimos resultados guardados (quench,
adiabático y VQE, crudos y procesados) sin gastar cuota.

**Ejecución local (gratis, sin cuota) — `--local`:** las tres funciones
aceptan `local=True` (o el flag `--local` combinado con cualquiera de los
anteriores, p. ej. `python src/run_h2_emulator.py --phase-transition
--local`), que las redirige a `local_emulator_backend.py` en vez de
`qnexus_backend.py`. H2-1LE es emulación exacta de vector de estado sin
ningún modelo de ruido físico — solo ruido de muestreo (shot noise); ver
la nota de Metodología más abajo — así que correr localmente vía
`pytket-quantinuum`+`pytket-pecos` (`pip install
"pytket-quantinuum[pecos]"`) da exactamente la misma computación que
enviarla por qnexus, sin login, sin espera de cola y sin costo. Los
resultados se guardan bajo un sufijo `_local` en `data/` y `figures/` para
no sobrescribir nunca una corrida real de qnexus. Pensado para iterar
rápido y gratis sobre parámetros (pasos de Trotter, dt, shots) antes de
gastar cuota confirmando una vez en qnexus.

### Ejecución por notebooks (desarrollo interactivo)

```bash
jupyter notebook notebooks/
```

| Notebook | Contenido | Estado |
|----------|-----------|--------|
| `01_exact_diagonalization.ipynb` | Línea base ED, validación de observables | Vacío — pendiente |
| `02_trotter_simulation.ipynb` | Construcción de circuitos, análisis de error Trotter | Vacío — pendiente |
| `03_phase_transition.ipynb` | Barrido h/J, gráficos de magnetización | Vacío — pendiente |
| `TFIM_LíneaBase_SinRuido.ipynb` | Desarrollo interactivo del TFIM sin ruido (equivalente de trabajo a 01-03) | Contenido real |
| `04_optional_FermiHubbard.ipynb` | Extensión Fermi-Hubbard 2D (opcional) | Contenido real |
| `04_optional_fermi_hubbard.ipynb` | — | Vacío, nombre duplicado del anterior — pendiente de limpieza |

> El pipeline principal (`python main.py`) no depende de ningún notebook —
> son material de desarrollo interactivo, no parte del entregable ejecutable.

---

## Metodología

### Fase 1: Línea base clásica (Exact Diagonalization)

- **Herramienta:** `scipy.sparse.linalg.eigsh` (matrices sparse)
- **Escalado:** O(2^N) — límite práctico N ≈ 20 en laptop
- **Observables calculados:**
  - Energía del estado fundamental
  - Magnetización ⟨Z⟩ y ⟨X⟩
  - Correlaciones ⟨ZᵢZⱼ⟩
  - Evolución temporal completa desde estado producto

### Fase 2: Simulación cuántica (Trotter-Suzuki)

- **Descomposición de primer orden:**
  ```
  e^(-iHt) ≈ [e^(-iH_ZZ Δt) · e^(-iH_X Δt)]^(t/Δt)
  ```
- **Descomposición de segundo orden (opcional):**
  ```
  e^(-iHt) ≈ [e^(-iH_X Δt/2) · e^(-iH_ZZ Δt) · e^(-iH_X Δt/2)]^(t/Δt)
  ```
- **Verificación de convergencia:** reducir Δt a la mitad y confirmar estabilidad de observables
- **SDK:** Qiskit (simulador local `Statevector`, sin ruido) → pytket + `qnexus` (emulador Quantinuum H2)
- **Reducción de puertas en el circuito de hardware (pytket):** en la capa
  simetrizada `Rx(θ/2) — ZZPhase — Rx(θ/2)` (`mirror=True`), la `Rx(θ/2)`
  final de un paso de Trotter y la `Rx(θ/2)` inicial del siguiente actúan
  sobre el mismo qubit sin nada entremedio, así que se fusionan de forma
  exacta vía `Rx(a)·Rx(b) = Rx(a+b)` en una sola puerta — mismo unitario,
  sin costo de error de Trotter, solo menos puertas de un qubit (`steps+1`
  capas de Rx en vez de `2·steps`). Implementado en
  `tket_circuit._append_trotter_layers` y verificado numéricamente
  (diferencia de unitarios ~1e-15) antes de aplicarlo a
  `build_quench_circuit`/`build_adiabatic_circuit`. En el barrido adiabático
  de N=6 usado para el emulador H2 esto redujo el conteo total de puertas
  ~33% (p. ej. h/J=0.5 con 100 pasos: 1812 → 1218 puertas, profundidad 402 →
  303) sin cambiar ninguna observable. Inspirado por la técnica de
  "reducción de conteo de puertas mediante identidades de circuito" de la
  Ref. [6] (arXiv:2511.02125, Fig. S8) — ahí fusionan puertas SWAP
  fermiónicas con puertas de interacción para el hopping de Fermi-Hubbard,
  lo cual no aplica directamente aquí (TFIM ya es a primeros vecinos, sin
  red de SWAPs), pero la misma identidad de fusión de rotaciones adyacentes
  de un qubit sí se traslada directamente a esta capa Trotter simetrizada.
- **H2-1LE es emulación exacta de vector de estado, no "sin ruido de
  leakage":** la documentación de Quantinuum es explícita — *"Emulator
  targets ending with LE enable noiseless (state-vector) emulation, which
  means that only shot-noise is included in the job result."* Sin modelo
  de ruido físico alguno (ni de compuerta, ni de leakage, ni de medición),
  solo ruido de muestreo finito. Por eso `local_emulator_backend.py`
  (pytket-quantinuum + pytket-pecos, corriendo en esta máquina) da
  exactamente la misma física que enviar el circuito por `qnexus` al mismo
  H2-1LE — permitiendo iterar gratis sobre parámetros con la certeza de
  que el resultado final por qnexus será equivalente.

### Fase 3: Comparación y análisis de error

- Métrica: desviación relativa |⟨O⟩_quantum − ⟨O⟩_ED| / |⟨O⟩_ED|
- **Objetivo:** < 5% para N = 8 sin ruido
- Documentar error sistemático de Trotter vs. error estocástico de hardware
- **Caso de estudio — punto crítico h/J=1 en el barrido adiabático de H2:**
  con una rampa de longitud fija por h/J (`config.H2_ADIABATIC_MAX_STEPS`),
  h/J=1 mostraba un sesgo sistemático de ⟨ZᵢZᵢ₊₁⟩ (~6.5% frente a ED,
  confirmado a 14-17σ con 20 000 shots gratis vía el backend local — nada
  que ver con ruido de muestreo). Un experimento aislando resolución
  Trotter (`dt`) de tiempo total de rampa (`T = pasos·dt`) — barato de
  hacer localmente — mostró que afinar `dt` a `T` fijo apenas cambiaba el
  sesgo, mientras que duplicar `T` a `dt` fijo lo redujo de ~6.5% a ~2.1%:
  ralentización crítica de manual — el teorema adiabático exige más
  *tiempo* según se cierra el gap en `h/J=1`, no más resolución temporal.
  `config.H2_ADIABATIC_CRITICAL_TIME_FACTOR` implementa esto: h/J=1 usa
  `H2_ADIABATIC_MAX_STEPS · H2_ADIABATIC_CRITICAL_TIME_FACTOR` pasos al
  mismo `dt` que el resto de los objetivos, no un `dt` más fino.

---

## Resultados esperados

### Figuras principales

Generadas por `python main.py` (o por `run_adiabatic.py` / `run_quench.py`
individualmente) en `figures/` (o `src/figures/` si se ejecuta directamente
desde `src/`):

| Figura | Descripción | Generada por | Archivo |
|--------|-------------|---------------|---------|
| Convergencia adiabática | ⟨Z⟩, ⟨ZᵢZᵢ₊₁⟩, ⟨X⟩ vs. tiempo de barrido para cada `h` objetivo, con línea de referencia ED y marca del fin de la rampa | `run_adiabatic.py` | `figures/adiabatic_convergence.png` |
| Transición de fase | ⟨Z⟩, ⟨X⟩, ⟨ZᵢZᵢ₊₁⟩ finales vs. h/J objetivo — Trotter vs. ED, con la línea crítica h/J=1 | `run_adiabatic.py` | `figures/phase_transition.png` |
| Evolución "quench" | ⟨Z⟩ y ⟨ZᵢZᵢ₊₁⟩ vs. tiempo, ED vs. Trotter local, para cada `h` en `config.H_VALUES` | `run_quench.py` | `figures/fixed_hamiltonian_evolution.png` |
| Quench en H2 vs. ED | ⟨Z⟩ y ⟨ZᵢZᵢ₊₁⟩ vs. tiempo, emulador H2 (con barras de error bootstrap) vs. ED, para cada `h` en `config.H2_H_VALUES` | `run_h2_emulator.py` (`run()`) | `figures/h2_vs_ed_time.png` |
| Transición de fase en H2 (barrido adiabático) | ⟨Z⟩/⟨ZᵢZᵢ₊₁⟩ finales vs. h/J tras una rampa adiabática de longitud fija ejecutada en el emulador H2, vs. ED | `run_h2_emulator.py` (`run_phase_transition()`) | `figures/h2_phase_transition.png` |
| Convergencia de VQE | Energía / ⟨Z⟩ / ⟨ZᵢZᵢ₊₁⟩ del ansatz hardware-efficient vs. iteración de COBYLA, comparado con el estado fundamental ED | `run_h2_emulator.py` (`run_vqe()`, vía `vqe.py`) | `figures/vqe_convergence.png` |
| Transición de fase vía VQE | ⟨Z⟩/⟨ZᵢZᵢ₊₁⟩ del estado fundamental hallado por VQE vs. h/J objetivo, vs. ED | `run_h2_emulator.py` (`run_vqe()`) | `figures/h2_phase_transition_vqe.png` |

`plot_h2_comparison.py` es un script auxiliar independiente para regenerar
las figuras H2/ED/VQE a partir de los datos ya guardados en `data/` (vía
`persistence.load_latest`), sin volver a someter jobs a qnexus.

`src/ed_figures.py` contiene generación de figuras adicional (magnetización
multi-N, convergencia de Trotter vs. Δt, correlaciones vs. distancia) pero
actualmente **no está conectada al pipeline** (`run_*.py` no la invoca) —
está disponible para extender el análisis pero no forma parte de la
ejecución por defecto.

### Datos de salida

Cada sección guarda sus resultados en `data/` como JSON vía
`src/persistence.py` — un archivo permanente con timestamp UTC
(`<sección>_<timestamp>.json`) y un puntero `<sección>_latest.json` que se
sobrescribe en cada corrida:

| Sección | Archivo |
|---------|---------|
| `run_ed.py` | `data/ed_latest.json` |
| `run_adiabatic.py` | `data/adiabatic_latest.json` |
| `run_quench.py` | `data/quench_latest.json` |
| `run_h2_emulator.py` (`run()`, quench) | `data/h2_emulator_raw_latest.json` (bitstrings crudos + metadatos, guardados **antes** de cualquier post-procesamiento) y `data/h2_emulator_latest.json` (observables ya calculados vs. ED) |
| `run_h2_emulator.py` (`run_phase_transition()`, adiabático) | `data/h2_adiabatic_raw_latest.json` y `data/h2_adiabatic_latest.json` |
| `run_h2_emulator.py` (`run_vqe()`) | `data/h2_vqe_raw_latest.json` y `data/h2_vqe_latest.json` |

El guardado de `h2_emulator_raw` ocurre inmediatamente después de que cada
job de qnexus devuelve resultados — antes de convertir bitstrings a
observables — para que un error en el post-procesamiento nunca obligue a
reenviar el circuito (y volver a gastar cuota) para recuperar datos que el
hardware ya devolvió.

---

## Conexión con los ODS

| ODS | Conexión directa con el proyecto |
|-----|----------------------------------|
| **7** (Energía Limpia) | TFIM → comprensión de magnetismo cuántico → superconductores de alta Tc → transmisión sin pérdidas |
| **9** (Industria e Innovación) | Simuladores cuánticos como nueva clase de instrumento científico; validación de hardware H2 |
| **12** (Producción Responsable) | Diseño de catalizadores (fijación de N₂) mediante simulación de Hubbard |
| **13** (Acción Climática) | Materiales para baterías de próxima generación y electrocatalizadores de captura de CO₂ |

> *"La simulación precisa del Fermi-Hubbard podría guiar el diseño de superconductores de cuprato, acortando el descubrimiento de materiales de décadas a años."*

---

## Limitaciones honestas

1. **Sin ventaja cuántica establecida:** Para N ≤ 50, métodos clásicos (ED, DMRG, tensor networks) superan a los dispositivos cuánticos actuales.
2. **Profundidad de circuito:** Más de ~50 compuertas en hardware actual están dominadas por ruido.
3. **Efectos de tamaño finito:** La transición de fase en `h/J = 1` es nítida solo en el límite termodinámico; para N = 6–8 se observa redondeo.
4. **Error de Trotter:** Error sistemático O(Δt²) para primer orden; requiere Δt pequeño o descomposición de segundo orden.
5. **Extensión Fermi-Hubbard:** Codificación J-W escala como O(N) en profundidad de circuito para interacciones no locales.

---

## Extensión opcional: Fermi-Hubbard 2D (`fh2d_src/`)

`fh2d_src/` es un paquete separado y autocontenido — no comparte código con
`src/` — que implementa el modelo de Fermi-Hubbard 2D (codificación
Jordan-Wigner, ED, dinámica de Trotter, VQE) como demostración adicional
más allá del TFIM 1D requerido. `python main.py` en la raíz ejecuta esta
extensión automáticamente después del pipeline TFIM (sección 5 de "Uso"
arriba); también puede ejecutarse sola con su propio punto de entrada:

```bash
cd fh2d_src
python fh_main.py               # pipeline completo de la extensión
python fh_main.py --quick       # salta VQE y las tablas resumen
```

Ver el docstring de `fh_main.py` para el resto de las banderas
(`--no-vqe`, `--no-shots`, `--no-noise`, `--no-heatmap`) — estas solo
aplican a la ejecución independiente; `python main.py` siempre corre el
pipeline completo de la extensión.

---

## Referencias

1. Sachdev, S. *Quantum Phase Transitions* (2nd ed.). Cambridge University Press, 2011. Cap. 1.
2. Suzuki, M. "Generalized Trotter's formula and systematic approximants of exponential operators." *J. Math. Phys.* 17, 1976.
3. Ebadi, S. et al. "Quantum phases of matter on a 256-atom programmable quantum simulator." *Nature* 595, 2021.
4. Bravyi, S. & Kitaev, A. "Fermionic quantum computation." *Ann. Phys.* 298, 2002.
5. Jordan, P. & Wigner, E. "Über das Paulische Äquivalenzverbot." *Z. Phys.* 47, 1928.
6. "Superconducting pairing correlations on a trapped-ion quantum computer." arXiv:2511.02125v3, 2025. [arxiv.org/abs/2511.02125](https://arxiv.org/abs/2511.02125)

---

## Autores

- **Equipo Challenge 3** — Física / Ciencia de Materiales / Computación Cuántica
- Hackathon Quantathon CR 2026 · Challenge 3

## Licencia

MIT License — ver [LICENSE](LICENSE) para detalles.

---

> **Nota para jueces:** Todo el código es reproducible desde un entorno limpio usando `requirements.txt`. El script `main.py` en la raíz del repositorio es el único punto de entrada necesario para regenerar todas las figuras y cifras reportadas de las once secciones del pipeline TFIM (ED, adiabático, quench, convergencia dt, escaneo N, costo clásico-vs-cuántico, emulador H2, ZNE, piloto Iceberg QEC, escaneo de ruido, comparación Iceberg) y de la extensión Fermi-Hubbard 2D (`fh2d_src/`). Las secciones 7, 8 y 10 (emulador H2, ZNE y escaneo de ruido) requieren `config.RUN_ON_H2_EMULATOR = True` y una sesión de `qnexus`, ya que consumen cuota de uso medida — **actualmente activado por defecto**, y la sección 10 en particular es la de mayor costo (dos corridas H2 por cada N hasta 26 qubits). La sección 9 (Iceberg QEC) tiene su propio interruptor separado, `config.ICEBERG_RUN_ON_H2_EMULATOR`, **desactivado por defecto**. El barrido adiabático en hardware (`--phase-transition`) y el VQE (`--vqe`) de `run_h2_emulator.py` son entradas adicionales que se invocan aparte de `main.py` por costo extra de cuota (ver "Funciones adicionales del emulador H2" arriba). La extensión Fermi-Hubbard 2D también tiene su propio interruptor de cuota, `fh2d_src/fh_config.py`'s `RUN_ON_H2_EMULATOR` (activado por defecto, igual que el de `src/config.py`) — su corrida de emulador H2 (incluyendo VQE) consume cuota real de `qnexus` cuando `python main.py` la ejecuta.
