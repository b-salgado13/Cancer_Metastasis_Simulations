# Simulation of Cancer Cell Metastasis

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey)
![GitHub last commit](https://img.shields.io/github/last-commit/b-salgado13/Cancer_Metastasis_Simulations)

This repository collects computational models developed to simulate the growth of tumors and the emergence of metastatic cells using stochastic lattice-based simulations coupled to diffusion-reaction equations.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Description of the Model](#description-of-the-model)
3. [Mathematical Model](#mathematical-model)
    1. [Spatial Representation](#1-spatial-representation)
    2. [Oxygen Field Dynamics](#2-oxygen-field-dynamics)
    3. [Cellular Oxygen Consumption](#3-cellular-oxygen-consumption)
    4. [Hypoxia Ratio](#4-hypoxia-ratio)
    5. [Cell Fate Probabilities](#5-cell-fate-probabilities)
    6. [Cell Phenotypes](#6-cell-phenotypes)
    7. [Angiogenic Switch](#7-angiogenic-switch)
    8. [Hypoxia and Necrosis](#8-hypoxia-and-necrosis)
    9. [Metastasis Mechanism](#9-metastasis-mechanism)
    10. [Division-Death Ratio](#10-division-death-ratio)
4. [Model Assumptions and Limitations](#model-assumptions-and-limitations)
5. [Relation to Statistical Physics and Renormalization Group](#relation-to-statistical-physics-and-renormalization-group)
6. [Installation & Usage](#installation--usage)
7. [Example Output](#example-output)
8. [References](#references)

---

## Project Overview

This code is part of a research project carried out by [Bruno Salgado](https://brunosalgado.website/) under the supervision of [Dr. Pere Masjuan](https://orcid.org/0000-0002-8276-413X) at the Institut de Física d'Altes Energies ([IFAE](https://www.ifae.es/es/)) as part of the [Master of Multidisciplinary Research in Experimental Sciences](https://www.upf.edu/web/mmres/) at Universitat Pompeu Fabra.

---

## Description of the Model

The model describes tumor growth on a **3-dimensional cubic lattice** where each lattice site may be empty or occupied by a tumor cell.

The simulation combines three key components:

1. Agent-based tumor cell dynamics
2. Continuous diffusion fields for oxygen and signaling molecules
3. Stochastic rules governing cell fate decisions

This hybrid modeling approach is common in **computational oncology**, where discrete cells interact with continuous biochemical fields.

### Biological Motivation

Tumor growth is strongly regulated by **oxygen availability**.

When tumors grow beyond the diffusion limit of oxygen (~100–200 µm), the inner regions become **hypoxic**, which leads to:

* Increased cell death
* Reduced proliferation
* Activation of **angiogenesis** (formation of new blood vessels)

If hypoxia persists, cells undergo **necrosis**, producing the characteristic **necrotic core** observed in many solid tumors.

This model aims to reproduce these phenomena using a simplified mechanistic description.

---

## Mathematical Model

### 1. Spatial Representation

The tumor grows on a 3D lattice of size:

$$L \times L \times L$$

Each lattice site contains either:

* an empty space
* a living tumor cell
* a necrotic cell

Cells interact with their **18 nearest neighbors** (first and second order neighbors).

---

### 2. Oxygen Field Dynamics

The oxygen concentration field

$$O(\vec{x},t)$$

evolves through **diffusion and cellular consumption**.

The continuous dynamics are approximated numerically using **finite differences**.

#### Diffusion equation

$$\frac{\partial O}{\partial t}=D_O \nabla^2 O - Q(O)$$

where

* $D_O$ : oxygen diffusion coefficient
* $Q(O)$ : oxygen consumption by tumor cells

---

### 3. Cellular Oxygen Consumption

Cells consume oxygen following **Michaelis–Menten kinetics**, a standard model for metabolic uptake:

$$Q(O) = V_{\max} \frac{O}{K_M + O}$$

where

* $V_{\max}$ : maximum oxygen uptake rate
* $K_M$ : half-saturation constant

This form captures the biological fact that oxygen consumption **saturates at high concentrations**.

Only **living cells consume oxygen**, while necrotic cells do not.

---

### 4. Hypoxia Ratio

A useful quantity derived from the oxygen concentration is the **hypoxia ratio**

$$C(\vec{x},t) = 1 - \frac{O(\vec{x},t)}{O_{\max}}$$

Properties:

| Oxygen level | Hypoxia ratio |
| ------------ | ------------- |
| High oxygen  | $C \approx 0$ |
| Low oxygen   | $C \approx 1$ |

This quantity directly modulates **cell division and death probabilities**.

---

### 5. Cell Fate Probabilities

Each simulation step, every cell may:

* divide
* die
* remain unchanged

These processes are **stochastic** and depend on the local hypoxia level.

#### Death probability

$$d = \alpha C$$

where

* $\alpha$ controls the maximum death rate.

Hypoxic regions therefore experience higher mortality.

---

#### Division probability

$$b = \beta(1 + \gamma - C)$$

where

* $\beta$ controls the proliferation rate
* $\gamma$ is the **phenotype parameter**

---

### 6. Cell Phenotypes

Each cell belongs to one of two phenotypes:

#### Condensing cells

$$\gamma > 0$$

Characteristics:

* Higher proliferation
* Higher turnover
* Compact tumor morphology

---

#### Non-condensing cells

$$\gamma < 0$$

Characteristics:

* Slower growth
* More diffuse tumor structure

This mechanism models **evolutionary trade-offs in tumor populations**.

---

### 7. Angiogenic Switch

Tumors initially grow using only **diffusion-limited oxygen**.

When the population exceeds a threshold

$$N_A$$

cells begin producing a **pro-angiogenic factor**

$$\phi(\vec{x},t)$$

which diffuses according to

$$\frac{\partial \phi}{\partial t}=D_\phi \nabla^2 \phi + S_\phi$$

The factor increases local oxygen supply:

$$O \leftarrow O + \Delta \phi$$

This represents the **angiogenic switch**, a hallmark of cancer progression.

---

### 8. Hypoxia and Necrosis

Cells respond to oxygen depletion through two thresholds:

#### Hypoxia threshold

$$O < O_{hypoxia}$$

Effects:

* reduced division
* secretion of angiogenic factors

---

#### Necrotic threshold

$$O < O_{necrosis}$$

If this condition persists for several time steps, cells become **necrotic**.

Necrotic cells:

* stop consuming oxygen
* eventually get removed (simulating immune clearance)

This produces a **necrotic tumor core**.

---

### 9. Metastasis Mechanism

Metastasis is modeled as a **mechanical detachment process**.

When a cell attempts division into an occupied site:

1. The daughter cell performs a **biased random walk** outward.
2. The walk continues until an empty location is found.
3. If the last occupied position has **only one neighbor**, the daughter cell **detaches**.

This event is recorded as a **metastatic event**.

The outward bias models **mechanical pressure pushing cells toward the tumor surface**.

---

### 10. Division-Death Ratio

An important diagnostic quantity in the simulation is

$$R = \frac{b}{d}$$

which measures the **balance between growth and mortality**.

* $R > 1$: tumor expansion
* $R < 1$: tumor shrinkage

This ratio can act as an **effective order parameter** for tumor growth regimes.

---

## Model Assumptions and Limitations

Like all mathematical models of biological systems, this simulation relies on simplifying assumptions that allow the system to be computationally tractable while preserving the essential mechanisms of tumor growth.

### Spatial discretization

Space is represented as a cubic lattice. Each lattice site can host at most one cell. While real tissues are continuous and deformable, lattice-based models capture key spatial interactions with relatively low computational cost.

### Simplified metabolism

Oxygen consumption is modeled using Michaelis–Menten kinetics. In reality, tumor metabolism involves multiple pathways (glycolysis, oxidative phosphorylation, lactate production), but oxygen uptake provides a reasonable first-order approximation of metabolic stress.

### Local microenvironment

Cells interact only with their immediate neighbors and the local oxygen concentration. Long-range biochemical signaling and immune system interactions are not explicitly modeled.

### Angiogenesis approximation

The formation of new blood vessels is represented through a diffusing pro-angiogenic factor that restores oxygen supply. The detailed vascular architecture and blood flow dynamics are not explicitly simulated.

### Mechanical interactions

Mechanical pressure inside the tumor is approximated through biased random walks during attempted cell division. This is a simplified representation of mechanical stresses that occur in real tumors.

### Stochastic dynamics

Cell division and death are probabilistic processes. This reflects the inherent stochasticity of biological systems but also means that individual simulation runs may produce different outcomes.

### Scale limitations

The model focuses on **mesoscopic tumor growth dynamics** rather than molecular-scale biochemical networks or whole-organ tumor development.

Despite these simplifications, the model captures several emergent phenomena observed in solid tumors:

- hypoxic tumor cores
- necrotic regions
- angiogenic switching
- spatial heterogeneity
- metastatic cell detachment

---

## Relation to Statistical Physics and Renormalization Group

The tumor growth model implemented in this repository can be interpreted within the broader framework of **non-equilibrium statistical physics**.

Tumor growth can be viewed as a stochastic birth–death process on a spatial lattice coupled to diffusive fields. Systems of this type often exhibit emergent macroscopic behavior that is largely independent of microscopic details.

### Universality in tumor growth

Several studies have suggested that tumor growth may belong to universality classes similar to those appearing in statistical physics models such as:

- reaction–diffusion systems
- branching processes
- directed percolation

In these systems, large-scale behavior is controlled by a small number of effective parameters.

### Effective order parameters

In this simulation, the ratio

$$R = \frac{b}{d}$$

acts as an effective control parameter governing tumor expansion or collapse.

Values of $R > 1$ correspond to net growth, while $R < 1$ leads to decay. Near the transition region, fluctuations become important and the system may display scale-dependent dynamics.

### Connection to renormalization group ideas

From a renormalization group perspective, microscopic rules governing cell behavior (division probabilities, metabolic rates, and diffusion parameters) flow toward effective macroscopic dynamics that determine tumor morphology and metastatic potential.

Parameter sweeps in this simulation allow exploration of how the system behaves under variations of these microscopic parameters, providing insight into possible **universality classes of tumor growth dynamics**.

Understanding these large-scale behaviors may help identify robust features of tumor evolution that remain invariant under changes in biological details.

---

## Installation & Usage
### Python Version
- Python 3.7 or higher recommended
- Check your version: `python --version` or `python3 --version`

### Required Libraries
Most libraries are built-in, but you'll need to install:
- Libraries: `numpy`, `matplotlib`, `mpl_toolkits`, `concurrent` and `PyOpenGL`
#### Setup
1. Clone repository:
```bash
git clone https://github.com/b-salgado13/Cancer_Metastasis_Simulations.git
cd Cancer_Metastasis_Simulations
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Model
To execute the main simulation and generate plots for the Querétaro data **Run from terminal**:
   ```bash
   cd Cancer_Metastasis_Simulations
   python "Cancer Metastasis Full python.py"
   ```
   or
   ```bash
   cd Cancer_Metastasis_Simulations
   python3 "Cancer Metastasis Full python.py"
   ```

---

## Example Output

With the following initial parameters:

```python
L     = 40          # grid side length (nodes)
GAMMA = 0.1         # condensing factor (+ for condensing, - for non-condensing)
N_A   = 500         # cell count at which angiogenic switch turns on
D_OX  = 2.0         # oxygen diffusion coefficient
D_CH  = 1000.0      # pro-angiogenic factor diffusion coefficient
DELTA = 0.05        # fraction of O_MAX restored per unit phi per step
N_OX  = 100         # oxygen diffusion time steps per simulation step
N_CH  = 50          # chemokine diffusion time steps per simulation step
DT    = 0.0001      # diffusion time step
DX    = 1.0         # lattice spacing

# ── Oxygen metabolism (Michaelis-Menten kinetics) ────────────────────────────
O_MAX = 1.0         # maximum oxygen concentration (normalised)
V_MAX = 0.17        # maximum cellular oxygen uptake rate per step
K_M   = 0.1         # Michaelis-Menten half-saturation constant

# ── Necrotic threshold 
O_HYPOXIA = 0.15     # Cells with O < O_HYPOXIA become hypoxic and start producing pro-angiogenic factors
O_NECROSIS = 0.05   # Cells with O < O_NECROSIS become necrotic and stop consuming O)
NECROSIS_DELAY = 4  # Slow death under hypoxia
NECROTIC_CLEAR_RATE = 0.001  # Fraction of necrotic cells cleared per step (simulate immune clearance)

# Tunable intrinsic parameters (try different combos!)
ALPHA = 0.3         # resistance factor (max death probability)
BETA  = 0.7         # growth factor (max division probability)

MAX_SIM_STEPS = 40    # simulation time steps
SEED    = 42
```

The execution of the `` returns in the terminal de following results:

```bash
============================================================
3D Tumor Growth Simulation
Parameters: α=0.3, β=0.7, L=40
Angiogenic switch at N=500 cells
============================================================
  t=  1 | N=    2 | meta=  0 | <b>=0.535 | <d>=0.041 | angio=off
  t=  6 | N=    6 | meta=  0 | <b>=0.456 | <d>=0.095 | angio=off
  t= 11 | N=   21 | meta=  0 | <b>=0.403 | <d>=0.126 | angio=off
  t= 16 | N=   57 | meta=  2 | <b>=0.414 | <d>=0.120 | angio=off
  t= 21 | N=  126 | meta=  3 | <b>=0.390 | <d>=0.131 | angio=off
  t= 26 | N=  328 | meta= 10 | <b>=0.387 | <d>=0.136 | angio=off
  [t=29] Angiogenic switch ON  (N=546)
  t= 31 | N=  825 | meta= 15 | <b>=0.405 | <d>=0.124 | angio=ON
  t= 36 | N= 4235 | meta=218 | <b>=0.580 | <d>=0.051 | angio=ON
  t= 40 | N=25377 | meta= 37 | <b>=0.639 | <d>=0.027 | angio=ON

Final population      : 25377 cells
Total metastatic events: 1049
Angiogenic switch triggered: True
```

Along with a plot for the general description of important parameters of the tumor evolution, namely:
* Tumor population
* Number of metastatic events
* Division and deaths probabilities
* Mean hypoxia ratio
* 3D tumor morphology

![Tumor results](example-outputs\tumor_results.png)

A second plot that shows:
* Oxygen concentration heat map
* Pro-angiogenic factor heat map

![Oxygen field](example-outputs\tumor_diffusion.png)

The results of the parameter sweep executed through parallel computation are still under development.

---
## References
1. Terradellas Igual, A. (2019). *Fractal dynamics and cancer growth.* (Master Thesis, Universitat Pompeu Fabra). Not published.
2. Ojwang', A.M.E., Bazargan, S., Johnson, J.O., Pilon-Thomas, S. & Rejniak, K.A. (2024) *Histology-guided mathematical model of tumor oxygenation.* bioRxiv [Preprint]. doi: 10.1101/2024.03.05.583363.

---

## Citation

If you use this code in academic work, please cite:

Salgado, B. (2026).
**Simulation of Tumor Growth and Metastasis.**  
Master of Multidisciplinary Research in Experimental Sciences (MMRES), Universitat Pompeu Fabra.

Repository:
https://github.com/b-salgado13/Cancer_Metastasis_Simulations