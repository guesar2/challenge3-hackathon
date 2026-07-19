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
├── main.py                            ← Punto de entrada único
│
├── src/
│   ├── ftim_main.py                   ← Punto de entrada real (orquestación de las 3 etapas)
│   ├── config.py                      ← Parámetros de simulación (N, J, H_VALUES, dt, etc.)
│   ├── pauli_ops.py                   ← Operadores de Pauli + construcción del Hamiltoniano TFIM
│   ├── exact_diagonalization.py       ← Línea base ED (estado fundamental y evolución exacta)
│   ├── circuits.py                    ← Circuitos de Trotter (Qiskit, edge coloring, capa simétrica)
│   ├── trotter_simulation.py          ← Propagación del statevector (barrido adiabático y quench)
│   ├── sweep_schedule.py              ← Lógica de número de pasos para el barrido adiabático
│   ├── plotting.py / ed_figures.py    ← Generación de figuras (matplotlib)
│   ├── reporting.py                   ← Tabla comparativa Trotter vs. ED en consola
│   ├── figures/                       ← Figuras generadas por defecto
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
├── data/                              ← Resultados numéricos (.npz, .json)
├── figures/                           ← Figuras generadas (.png, .pdf)
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

> **Nota sobre Guppy:** Si se utiliza el SDK nativo de Quantinuum, instalar por separado siguiendo la documentación oficial de [Quantinuum](https://www.quantinuum.com/).

---

## Uso

### Ejecución completa (punto de entrada real)

```bash
source venv/bin/activate
python src/ftim_main.py         # ejecutar desde src/, o con src/ en sys.path
```

Este script orquesta las tres etapas del pipeline, con parámetros centralizados en `src/config.py`:
1. `ed_baseline` — observables del estado fundamental (ED) para `config.H_VALUES`
2. `run_adiabatic_simulation` — barrido adiabático trotterizado desde `config.H_INIT` hasta cada `h` objetivo, comparado contra ED
3. Evolución "quench" desde el estado producto `|0...0⟩`, comparando `ed_time_evolution_exact` vs. `run_trotter_fixed_hamiltonian` (objetivo: <5% de desviación en ⟨Z⟩ y ⟨ZᵢZᵢ₊₁⟩)

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
- **SDK:** Qiskit (simulador local sin ruido) → Guppy (emulador H2)

### Fase 3: Comparación y análisis de error

- Métrica: desviación relativa |⟨O⟩_quantum − ⟨O⟩_ED| / |⟨O⟩_ED|
- **Objetivo:** < 5% para N = 8 sin ruido
- Documentar error sistemático de Trotter vs. error estocástico de hardware

---

## Resultados esperados

### Figuras principales

| Figura | Descripción | Archivo |
|--------|-------------|---------|
| Fig. 1 | Magnetización ⟨Z⟩ vs. h/J para N = {4, 6, 8} (ED) | `figures/fig1_magnetization_ed.png` |
| Fig. 2 | Evolución temporal ⟨Z(t)⟩: Trotter vs. ED para h/J = 1.0 | `figures/fig2_time_evolution.png` |
| Fig. 3 | Error relativo Trotter vs. Δt (convergencia) | `figures/fig3_trotter_convergence.png` |
| Fig. 4 | Correlaciones ⟨ZᵢZⱼ⟩ vs. distancia para h/J ∈ {0.5, 1.0, 2.0} | `figures/fig4_correlations.png` |
| Fig. 5 | (Opcional) Fermi-Hubbard: doble ocupación vs. U/t | `figures/fig5_fermi_hubbard.png` |

### Datos de salida

- `data/ed_results.npz` — Resultados de diagonalización exacta
- `data/trotter_results.npz` — Resultados de simulación trotterizada
- `data/comparison.json` — Tabla de comparación cuántico vs. clásico

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

---

## Autores

- **[Tu nombre]** — Física / Ciencia de Materiales / Computación Cuántica
- Hackathon Quantathon CR 2026 · Challenge 3

## Licencia

MIT License — ver [LICENSE](LICENSE) para detalles.

---

> **Nota para jueces:** Todo el código es reproducible desde un entorno limpio usando `requirements.txt`. El script `main.py` es el único punto de entrada necesario para regenerar todas las figuras y cifras reportadas.
