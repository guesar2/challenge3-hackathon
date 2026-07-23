# Simulación digital del modelo de Ising de campo transverso en Quantinuum H2

**Equipo:** Quantum in Silico  
**Challenge 3 - Quantathon CR 2026**  
**Fecha:** 23 de julio de 2026

## Resumen

Se presenta una implementación del modelo de Ising unidimensional con campo transverso (TFIM), validada contra diagonalización exacta (ED) y ejecutada mediante circuitos Suzuki-Trotter de segundo orden. El repositorio combina una línea base dispersa, preparación adiabática, dinámica de quench, barridos de convergencia, emulación Quantinuum H2 y una extensión Fermi-Hubbard.

Para N=6, la preparación adiabática reproduce los observables de estado base con desviaciones finales inferiores a 0.17 %. En dinámica hasta T=20, la configuración por defecto dt=0.05 alcanza un error máximo de 8.46 % en CZZ para h/J=2; el barrido demuestra que dt=0.025 reduce ese error a 2.10 %. Se analizan resultados almacenados de H2-1LE y H2-Emulator con incertidumbre de shots, y se documenta una implementación exploratoria de extrapolación a ruido cero (ZNE). No se afirma ventaja cuántica.

## 1. Modelo y observables

El Hamiltoniano periódico implementado es

\[
H=-J\sum_{i=0}^{N-1}Z_iZ_{i+1}-h\sum_{i=0}^{N-1}X_i,\qquad Z_N\equiv Z_0.
\]

Se usa J=1 y se estudian h/J = 0.5, 1.0 y 2.0. Los observables principales son

\[
M_z^{\mathrm{RMS}}=\frac{1}{N}\sqrt{\left\langle\left(\sum_iZ_i\right)^2\right\rangle},
\]

\[
M_x=\frac{1}{N}\left\langle\sum_iX_i\right\rangle,
\qquad
C_{ZZ}=\frac{1}{N}\left\langle\sum_iZ_iZ_{i+1}\right\rangle.
\]

El primer observable no es la magnetización longitudinal firmada, sino su magnitud RMS.

## 2. Metodología

### 2.1 Diagonalización exacta

El Hamiltoniano y los observables se construyen como matrices dispersas CSR. El estado base se obtiene con un solucionador disperso. Para los tamaños auditados, la dinámica exacta usa la descomposición espectral del Hamiltoniano.

Los resultados ED de N=6 y N=8 se recalcularon durante la auditoría y coincidieron con los JSON almacenados.

### 2.2 Suzuki-Trotter

La evolución se separa en HZZ y HX. Un paso de segundo orden es

\[
U_2(\Delta t)=e^{-iH_X\Delta t/2}e^{-iH_{ZZ}\Delta t}e^{-iH_X\Delta t/2}
+\mathcal{O}(\Delta t^3).
\]

Las interacciones ZZ se agrupan mediante coloración de aristas de la cadena periódica. El código pytket fusiona rotaciones X contiguas entre pasos y construye circuitos de medición en bases Z y X.

### 2.3 Protocolos

- **Preparación adiabática:** comienza en h=4 y reduce el campo hasta el objetivo.
- **Quench:** comienza en el estado producto |00...0> y evoluciona con un Hamiltoniano fijo.
- **Nexus:** sube, compila y ejecuta circuitos pytket en H2-1LE o H2-Emulator.
- **ZNE:** genera circuitos plegados con factores 1, 3 y 5 y extrapola el observable al límite de ruido cero.

## 3. Resultados TFIM

### 3.1 Baseline ED

| N | h/J | Mz RMS | Mx | CZZ | E0/N |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.5 | 0.969914 | 0.265198 | 0.931517 | -1.064116 |
| 6 | 1.0 | 0.813220 | 0.643951 | 0.643951 | -1.287901 |
| 6 | 2.0 | 0.554173 | 0.931517 | 0.265198 | -2.128232 |
| 8 | 0.5 | 0.969303 | 0.260062 | 0.933604 | -1.063635 |
| 8 | 1.0 | 0.785827 | 0.640729 | 0.640729 | -1.281458 |
| 8 | 2.0 | 0.481939 | 0.933604 | 0.260062 | -2.127271 |

Al aumentar h/J, Mz RMS y CZZ decrecen, mientras Mx crece. La competencia más fuerte se observa alrededor de h/J=1.

![Transición de fase](figures/phase_transition.png)

### 3.2 Preparación adiabática

Para N=6, dt=0.02 y tasa de referencia |dh/dt|=0.022, la mayor desviación final entre los tres observables y los tres campos es 0.160 %, correspondiente a Mx en h/J=0.5. El protocolo valida la preparación local, aunque su gran cantidad de pasos no lo convierte en un circuito NISQ práctico.

### 3.3 Convergencia del quench

| h/J | error CZZ, dt=0.05 | error CZZ, dt=0.025 |
|---:|---:|---:|
| 0.5 | 0.290 % | 0.073 % |
| 1.0 | 3.117 % | 0.774 % |
| 2.0 | 8.463 % | 2.100 % |

La reducción de dt de 0.05 a 0.025 disminuye el error aproximadamente por un factor cuatro, compatible con convergencia global de segundo orden. La configuración dt=0.025 es la recomendada para cumplir el umbral de 5 % en los casos auditados.

![Convergencia temporal](figures/dt_convergence.png)

## 4. Quantinuum H2

### 4.1 H2-1LE

La corrida almacenada utiliza N=4, dt=0.1, cinco pasos y 200 shots. En el punto final t=0.5, varios observables son estadísticamente compatibles con ED, aunque la baja estadística genera desviaciones visibles. H2-1LE valida el flujo Nexus y el muestreo, no un modelo de ruido físico.

![H2-1LE](figures/h2_final_observables.png)

### 4.2 H2-Emulator

Para h/J=1, dt=0.05, 20 pasos y 200 shots:

| N | error máximo Mz RMS | error máximo CZZ |
|---:|---:|---:|
| 4 | 1.421 % | 4.702 % |
| 6 | 3.988 % | 7.430 % |
| 8 | 0.217 % | 3.433 % |

La no monotonicidad y la baja cantidad de shots no permiten inferir una ley de escalado.

![H2 ruidoso](figures/h2_noisy_scaling.png)

### 4.3 ZNE

El código implementa folding con Qermit y ajuste polinómico ponderado. Existe una figura para N=8, pero no se encontró un JSON inequívoco con valores crudos, incertidumbres, coeficientes del ajuste y estimación extrapolada. Por ello, ZNE se reporta como resultado exploratorio.

![ZNE exploratorio](figures/zne_exploratory.png)

## 5. Escalado

El conteo por capa crece como 3N y la profundidad permanece entre 4 y 5 para N=4-20. Los valores son por capa, no por circuito completo. Para 20 pasos, el statevector local pasa de aproximadamente 0.013 s en N=4 a 17.0 s en N=20. La evolución ED densa alcanza 7.64 s en N=12 y no se ejecuta para tamaños mayores en este estudio.

![Escalado](figures/scaling_summary.png)

Estos datos no demuestran ventaja cuántica. Una comparación rigurosa exige igualar precisión, preparación, observables, compilación, shots y coste extremo a extremo, además de usar los mejores métodos clásicos aplicables.

## 6. Extensión Fermi-Hubbard

La extensión opcional implementa una red 2x2 espínful de ocho qubits a media ocupación. La doble ocupación del estado base disminuye de 0.25 en U/t=0 a 0.0325 en U/t=8. Para el quench en U/t=8, dt=0.05 produce desviaciones máximas de 2.17 % en doble ocupación y 0.75 % en magnetización escalonada.

Una corrida H2-1LE de 500 shots obtiene al final D=0.0715 ± 0.0058 frente a 0.0747 de ED, y Ms=0.401 ± 0.0078 frente a 0.386. El VQE de tres capas presenta errores de energía de 5.37 % en U/t=1 y 28.0 % en U/t=8.

![Fermi-Hubbard](figures/fermi_hubbard_summary.png)

## 7. Reproducibilidad y limitaciones

Los principales bloqueadores son:

1. `requirements.txt` no contiene las dependencias reales del proyecto.
2. Los archivos de pruebas están vacíos.
3. Nexus está activado por defecto, aunque el comentario dice lo contrario.
4. `main.py` no reproduce automáticamente VQE, ZNE ni Fermi-Hubbard.
5. La rama de Trotter de primer orden duplica la rotación X.
6. Los documentos de reporte, presentación y reflexión SDK estaban vacíos.
7. Faltan datos crudos completos para ZNE y varios resultados VQE.

La compilación sintáctica fue exitosa y la ED fue repetida. Las corridas Qiskit, pytket y Nexus no se repitieron en el entorno de auditoría por falta de dependencias disponibles y credenciales/cuota.

## 8. Conclusiones

El repositorio resuelve el núcleo científico del reto y contiene evidencia suficiente para una entrega sólida. La preparación adiabática local coincide estrechamente con ED; el estudio temporal establece dt=0.025 como configuración de referencia; y los resultados H2 demuestran una ruta de ejecución completa hasta ocho qubits. La extensión Fermi-Hubbard añade amplitud técnica, aunque el VQE en acoplamiento fuerte permanece abierto.

Antes de publicar deben corregirse los bloqueadores de reproducibilidad, guardarse los datos crudos de ZNE/VQE y evitarse afirmaciones de ventaja cuántica que no estén sustentadas por una comparación extremo a extremo.

## Referencias

1. P. Pfeuty, *The one-dimensional Ising model with a transverse field*, Annals of Physics 57, 79-90 (1970).
2. M. Suzuki, *Generalized Trotter's formula and systematic approximants*, Communications in Mathematical Physics 51, 183-190 (1976).
3. K. Temme, S. Bravyi y J. Gambetta, *Error mitigation for short-depth quantum circuits*, Physical Review Letters 119, 180509 (2017).
4. *Qermit: a quantum error mitigation toolkit*, Quantum 7, 1059 (2023).
5. Documentación oficial de Quantinuum Nexus y emuladores H2, consultada en julio de 2026.
