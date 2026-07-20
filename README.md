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
├── requirements.txt                   ← Dependencias
├── main.py                            ← Punto de entrada único (inserta src/ en sys.path y llama a ftim_main.main())
│
├── src/
│   ├── ftim_main.py                   ← Orquestador: ejecuta las 4 secciones (run_*.py) en secuencia
│   ├── run_ed.py                      ← Sección 1/4 — línea base ED, ejecutable de forma independiente
│   ├── run_adiabatic.py               ← Sección 2/4 — barrido adiabático trotterizado (statevector local)
│   ├── run_quench.py                  ← Sección 3/4 — evolución "quench" ED vs. Trotter local
│   ├── run_h2_emulator.py             ← Sección 4/4 — envío del circuito al emulador Quantinuum H2 vía qnexus
│   │                                     (no-op salvo que config.RUN_ON_H2_EMULATOR = True — consume cuota)
│   ├── config.py                      ← Parámetros de simulación (N, J, H_VALUES, dt, H2_*, etc.)
│   ├── pauli_ops.py                   ← Operadores de Pauli + construcción del Hamiltoniano TFIM
│   ├── exact_diagonalization.py       ← Línea base ED (estado fundamental y evolución exacta)
│   ├── circuits.py                    ← Circuitos de Trotter para el simulador local (Qiskit, edge coloring, capa simétrica)
│   ├── tket_circuit.py                ← Circuito de Trotter para hardware (pytket, gate ZZPhase nativo)
│   ├── qnexus_backend.py              ← Envío/ejecución del circuito pytket en Quantinuum vía qnexus
│   ├── trotter_simulation.py          ← Propagación del statevector (barrido adiabático y quench)
│   ├── sweep_schedule.py              ← Lógica de número de pasos para el barrido adiabático
│   ├── plotting.py / ed_figures.py    ← Generación de figuras (matplotlib)
│   ├── reporting.py                   ← Tabla comparativa Trotter vs. ED en consola
│   ├── figures/                       ← Figuras generadas por defecto al ejecutar desde src/
│   └── trotter_circuit.py, observables.py,
│       tfim_hamiltonian.py, fermi_hubbard.py
│                                       ← Placeholders vacíos (0 bytes), refactorizados hacia
│                                         los módulos de arriba; no importar desde aquí
│
├── notebooks/
│   ├── 01_exact_diagonalization.ipynb ← Línea base clásica
│   ├── 02_trotter_simulation.ipynb    ← Simulación trotterizada
│   ├── 03_phase_transition.ipynb        ← Barrido h/J y transición de fase
│   └── 04_optional_fermi_hubbard.ipynb ← Extensión Fermi-Hubbard 2D
│
├── tests/
│   ├── test_hamiltonian.py            ← Vacío — pendiente (ver Tests)
│   └── test_trotter.py               ← Vacío — pendiente (ver Tests)
│
├── figures/                           ← Figuras generadas al ejecutar `python main.py` desde la raíz
│
└── docs/
    ├── TECHNICAL_REPORT.md            ← Borrador del informe técnico (máx. 8 pp.)
    ├── PRESENTATION.md                ← Guión de la presentación (5 min)
    ├── SDK_REFLECTION.md              ← Reflexión sobre SDK (≤200 palabras)
    ├── PLAN.md                        ← Plan de trabajo del hackathon
    └── CHECKLIST.md                   ← Checklist de entregables


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

> **Nota sobre el emulador H2:** `qnexus` y `pytket` (usados por `run_h2_emulator.py`)
> ya están en `requirements.txt`. Ese script requiere además una sesión de
> `qnexus` iniciada (login contra Quantinuum Nexus) y **consume una cuota de
> uso medida** — está desactivado por defecto (`config.RUN_ON_H2_EMULATOR =
> False`); el resto del pipeline no lo necesita.

---

## Uso

### Ejecución completa (punto de entrada único)

```bash
source venv/bin/activate
python main.py                  # desde la raíz del repo — reproduce todas las figuras/cifras
# equivalente:
python src/ftim_main.py         # ejecutar desde src/, o con src/ en sys.path
```

Este script orquesta las cuatro secciones del pipeline en secuencia, con parámetros centralizados en `src/config.py`:
1. `run_ed.py` — `ed_baseline`: observables del estado fundamental (ED) para `config.H_VALUES`
2. `run_adiabatic.py` — `run_adiabatic_simulation`: barrido adiabático trotterizado (statevector local, Qiskit) desde `config.H_INIT` hasta cada `h` objetivo, comparado contra ED
3. `run_quench.py` — evolución "quench" desde el estado producto `|0...0⟩`, comparando `ed_time_evolution_exact` vs. `run_trotter_fixed_hamiltonian` (objetivo: <5% de desviación en ⟨Z⟩ y ⟨ZᵢZᵢ₊₁⟩)
4. `run_h2_emulator.py` — el mismo circuito de quench, construido en pytket, enviado al emulador Quantinuum H2 vía `qnexus` — **desactivado por defecto**; solo corre si `config.RUN_ON_H2_EMULATOR = True`

### Ejecución de una sola sección

Cada sección es independiente — calcula su propia línea base ED y no depende de que las demás se hayan ejecutado antes — así que se puede verificar una sin correr el resto:

```bash
cd src
python run_ed.py            # solo línea base ED (rápido, puramente clásico)
python run_adiabatic.py     # solo el barrido adiabático
python run_quench.py        # solo la evolución "quench"
python run_h2_emulator.py   # no-op salvo que config.RUN_ON_H2_EMULATOR = True
```

### Ejecución por notebooks (desarrollo interactivo)

```bash
jupyter notebook notebooks/
```

| Notebook | Contenido | Tiempo estimado |
|----------|-----------|-----------------|
| `01_exact_diagonalization.ipynb` | Línea base ED, validación de observables | ~10 min |
| `02_trotter_simulation.ipynb` | Construcción de circuitos, análisis de error Trotter | ~15 min |
| `03_phase_transition.ipynb` | Barrido h/J, gráficos de magnetización | ~10 min |
| `04_optional_fermi_hubbard.ipynb` | Extensión Fermi-Hubbard 2D (opcional) | ~20 min |

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

### Fase 3: Comparación y análisis de error

- Métrica: desviación relativa |⟨O⟩_quantum − ⟨O⟩_ED| / |⟨O⟩_ED|
- **Objetivo:** < 5% para N = 8 sin ruido
- Documentar error sistemático de Trotter vs. error estocástico de hardware

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
| `run_h2_emulator.py` | `data/h2_emulator_raw_latest.json` (bitstrings crudos + metadatos, guardados **antes** de cualquier post-procesamiento) y `data/h2_emulator_latest.json` (observables ya calculados vs. ED) |

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

## Tests

```bash
pytest tests/ -v
```

> **Estado actual:** `tests/test_hamiltonian.py` y `tests/test_trotter.py`
> están vacíos (0 bytes), por lo que `pytest tests/` recolecta 0 tests. Los
> tests descritos abajo son los que faltan implementar:
- Hermiticidad del Hamiltoniano TFIM
- Conservación de la norma en evolución trotterizada
- Convergencia de Trotter al reducir Δt
- Coincidencia ED vs. Qiskit statevector para N = 4

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

- **[Tu nombre]** — Física / Ciencia de Materiales / Computación Cuántica
- Hackathon Quantathon CR 2026 · Challenge 3

## Licencia

MIT License — ver [LICENSE](LICENSE) para detalles.

---

> **Nota para jueces:** Todo el código es reproducible desde un entorno limpio usando `requirements.txt`. El script `main.py` en la raíz del repositorio es el único punto de entrada necesario para regenerar todas las figuras y cifras reportadas (secciones 1–3; la sección 4, emulador Quantinuum H2, requiere `config.RUN_ON_H2_EMULATOR = True` y una sesión de `qnexus`, ya que consume cuota de uso medida).
