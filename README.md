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
│   ├── tfim_hamiltonian.py            ← Construcción del Hamiltoniano TFIM
│   ├── exact_diagonalization.py       ← Diagonalización exacta con SciPy
│   ├── trotter_circuit.py             ← Circuitos de Trotter (Qiskit / Guppy)
│   ├── observables.py                 ← Cálculo de ⟨Z⟩, ⟨X⟩, ⟨ZᵢZⱼ⟩
│   └── fermi_hubbard.py               ← Extensión opcional (mapeo J-W, B-K)
│
├── notebooks/
│   ├── 01_exact_diagonalization.ipynb ← Línea base clásica
│   ├── 02_trotter_simulation.ipynb    ← Simulación trotterizada
│   ├── 03_phase_transition.ipynb        ← Barrido h/J y transición de fase
│   └── 04_optional_fermi_hubbard.ipynb ← Extensión Fermi-Hubbard 2D
│
├── tests/
│   ├── test_hamiltonian.py            ← Tests unitarios del Hamiltoniano
│   └── test_trotter.py               ← Tests de convergencia Trotter
│
├── data/                              ← Resultados numéricos (.npz, .json)
├── figures/                           ← Figuras generadas (.png, .pdf)
│
└── docs/
    ├── TECHNICAL_REPORT.md            ← Borrador del informe técnico (máx. 8 pp.)
    ├── PRESENTATION.md                ← Guión de la presentación (5 min)
    └── SDK_REFLECTION.md              ← Reflexión sobre SDK (≤200 palabras)
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

### Ejecución completa (punto de entrada único)

```bash
python main.py
```

Este script reproduce **todas las figuras y cifras reportadas** en el informe técnico:
1. Diagonalización exacta para N = {4, 6, 8} y h/J ∈ {0.5, 1.0, 2.0}
2. Simulación trotterizada con pasos Δt = {0.1, 0.05, 0.025}
3. Gráficos de magnetización ⟨Z⟩, ⟨X⟩ vs. tiempo
4. Barrido de h/J y detección de la transición de fase
5. Comparación cuántico vs. clásico con barras de error

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

Tests incluidos:
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
