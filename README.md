<div align="center">

# Phase-Field-Crystal

**A Modular 2D/3D Phase Field Crystal Simulation Framework**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.20.0-orange)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%E2%89%A51.7.0-green)](https://scipy.org/)
[![Taichi](https://img.shields.io/badge/Taichi-%E2%89%A51.0-purple)](https://taichi-lang.org/)
[![MPI](https://img.shields.io/badge/MPI-FFTW3-red)](https://www.open-mpi.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

[English](README.md) | [中文](README.zh.md)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Theoretical Background](#theoretical-background)
- [Lattice Types](#lattice-types)
- [Code Documentation](#code-documentation)
- [Examples](#examples)
- [Output](#output)
- [Author & Citation](#author--citation)
- [Changelog](#changelog)
- [License](#license)

---

## Overview

The **Phase-Field-Crystal (PFC)** simulation framework is a modular, object-oriented Python toolkit designed for computational materials science research. It enables simulation of crystal growth, grain boundary evolution, defect dynamics, and elastic properties at atomic length scales while operating on diffusive time scales.

This framework implements the standard PFC model, the **eXtended Phase-Field Crystal (XPFC)** model in multiple formulations, and other extensions (binary alloys, 3D BCC/FCC/SC lattices, Swift-Hohenberg model) using an efficient **semi-implicit Fourier spectral method**. The XPFC implementations include a dual-amplitude (two-mode) differential-operator formulation, a correlation-function (C₂) based formulation with Taichi acceleration, and a C++/MPI amplitude-representation solver — enabling quantitative studies of solid-liquid coexistence, interface energies, and dislocation energetics. The modular architecture separates core solvers, analysis tools, visualization, and I/O into independent mixin classes, allowing flexible composition and easy extension.

### Key Capabilities

| Category | Features |
|----------|----------|
| **Core Simulation** | Spectral semi-implicit PFC solver; Multiple 2D/3D lattice types; Pure material & binary alloy models |
| **XPFC Extensions** | Dual-amplitude (two-mode) PFC engine; Correlation-function C₂ XPFC solver (Taichi-accelerated); C++/MPI amplitude-representation solver; Flat & circular interface energy calculation; Dislocation energy computation |
| **Analysis** | Defect detection (5/7-fold), Voronoi tessellation, ψ₆ orientational order, grain boundary identification, structure factor analysis |
| **Elasticity** | Energy-strain curves, stress-strain relations, elastic constant fitting |
| **Visualization** | Real-time field rendering, MP4/GIF video generation, 3D isosurface rendering (PyVista), Tkinter GUI with interactive controls |

---

## Features

### Core Simulation
- **Spectral Semi-Implicit PFC Solver** — Efficient Fourier-space numerical integration with unconditional stability for linear terms
- **Multiple Lattice Types** — Hexagonal, Square, Triangle (2D); BCC, FCC, SC (3D)
- **Pure Material & Binary Alloy** — Single-component and two-phase field coupling with Cahn-Hilliard concentration dynamics
- **3D PFC Simulation** — Full 3D BCC lattice support with volumetric visualization and elastic constant calculation
- **Swift-Hohenberg Model** — Canonical pattern-forming PDE for stripe/hexagonal pattern studies

### XPFC (eXtended PFC) Extensions
- **Dual-Amplitude (Two-Mode) PFC Engine** — Product-form linear operator `L = [r + (1+∇²)²][(q²+∇²)²/q⁴]` with cubic nonlinearity; supports square and triangular crystal phases controlled by the `sigma` parameter
- **Correlation-Function XPFC Solver** — Direct correlation function `C₂(k)` constructed from multi-Gaussian peaks on reciprocal lattice sites; linear operator `L(k) = 1 − C₂(k)`; Taichi CPU-accelerated with single-crystal and polycrystal initial conditions
- **C++/MPI Amplitude-Representation Solver** — High-performance XPFC square-lattice solver using amplitude expansion (density + two amplitude fields), MPI parallelization with FFTW3, and dislocation initialization
- **Interface Energy Calculation** — Flat solid-liquid interface energy `γ_SL` and circular nucleus interface energy with effective radius `R_eff`
- **Dislocation Energy Calculation** — Edge dislocation energy `E_dis` via half-crystal shift, computed for both square and triangular phases across a sigma series
- **Tkinter GUI Interface** — Interactive main window for model selection, parameter configuration, sigma-series batch runs, live visualization, and data export

### Analysis Tools
- **Defect Density Analysis** — 5-fold/7-fold coordination defect detection via local peak analysis
- **Voronoi Tessellation** — Topological grain structure analysis with periodic boundary conditions
- **ψ₆ Orientational Order Parameter** — Bond-orientational order quantification for hexagonal lattices
- **Grain Boundary Detection** — D-parameter based identification of grain boundary atoms
- **Structure Factor Analysis** — Reciprocal space diffraction patterns (2D and 3D)
- **Elastic Energy Calculation** — Free energy under applied strain with reference state management
- **Stress-Strain Curves** — Mechanical property extraction via numerical differentiation

### Visualization
- **Real-time Field Visualization** — Density and concentration field rendering with Matplotlib
- **Video Recording** — Automatic MP4 (ffmpeg) or GIF (Pillow) generation of evolution dynamics
- **3D Isosurface Rendering** — Volumetric visualization with PyVista (with Matplotlib fallback)
- **Interactive XPFC GUI** — Tkinter-based interface with slider controls, play/pause, step navigation, and comprehensive result plots
- **Comprehensive Post-processing** — Full suite of analysis plots (energy, defects, grain size, etc.)

---

## Project Structure

> **Note on branching:** The repository is organized into feature branches. Standard PFC modules reside on `core/`, `analysis/`, `config/`, `io/`, and `runner/` branches; the DualXPFC extension lives on the `DualXPFC` branch. The `main` branch serves as the landing page.

```
Phase-Field-Crystal/
│
├── README.md              # English documentation (this file)
├── README.zh.md           # Chinese documentation
│
├── Core Modules (branch: core)
│   ├── pfc_base.py        # Base class: grid setup, k-space construction, simulation logs
│   ├── pfc_pure.py        # Pure material PFC solver (hexagon/square/triangle lattices)
│   ├── pfc_binary.py      # Binary alloy PFC solver (dual-field: density + concentration)
│   ├── pfc_3d.py          # 3D PFC solver (BCC/FCC/SC lattices with PyVista visualization)
│   └── sh_model.py        # Swift-Hohenberg pattern formation solver
│
├── Analysis Modules (branch: analysis)
│   ├── pfc_analysis.py    # Microstructure analysis: atom detection, coordination, ψ6, Voronoi, defects
│   ├── pfc_elastic.py     # Elastic property calculations: strain application, energy curves, fitting
│   └── pfc_plot.py        # Visualization toolkit: fields, structure factors, defects, grain boundaries
│
├── IO Module (branch: io)
│   └── pfc_io.py          # Video recording: in-memory frame caching, ffmpeg MP4 synthesis
│
├── Configuration (branch: config)
│   └── config.py          # Interactive console menu for parameter input with validation
│
├── Run Scripts (branch: runner)
│   ├── run_pure.py        # Standard pure material PFC simulation with interactive config
│   ├── run_binary.py      # Binary alloy simulation with 4 preset modes
│   ├── run_elastic.py     # Elastic constant calculation via energy-strain quadratic fitting
│   └── run_3d.py          # 3D PFC simulation with 4 modes (pure/elastic/alloy/sweep)
│
└── DualXPFC Extension (branch: DualXPFC)
    ├── __init__.py
    ├── pfc_core.py        # Dual-amplitude (two-mode) PFC core engine (DualPFCConfig, DualPFCEngine)
    ├── xpfc_square.py     # Correlation-function C₂ XPFC square solver (Taichi-accelerated)
    ├── main.py            # Tkinter GUI main interface (3 models: flat/round interface, dislocation)
    ├── part1_flat_interface.py   # Model 1: Flat solid-liquid interface energy γ_SL
    ├── part2_round_interface.py  # Model 2: Circular nucleus interface energy γ_SL, R_eff
    ├── part3_dislocation.py      # Model 3: Edge dislocation energy E_dis
    ├── visualization.py   # Interactive visualization module (sliders, play/pause, result plots)
    ├── quick_test.py      # Quick verification test for the dual-amplitude engine
    ├── xpfcSqAmpDis.cpp   # C++/MPI/FFTW3 amplitude-representation XPFC solver (with dislocations)
    └── xpfcSqAmpDis.in    # Input parameter file for the C++ solver
```

---

## Installation

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.20.0 | Core numerical arrays and FFT |
| `scipy` | ≥1.7.0 | FFT operations, spatial algorithms |
| `matplotlib` | ≥3.4.0 | 2D plotting and visualization |
| `taichi` | ≥1.0.0 | XPFC square solver CPU acceleration (DualXPFC) |
| `pyvista` | ≥0.32.0 | 3D isosurface rendering (optional) |
| `scikit-image` | ≥0.18.0 | 3D peak detection (optional) |
| `ffmpeg` | system | Video generation (optional, system package) |
| `MPI` + `FFTW3` | system | C++ amplitude-representation solver (DualXPFC, optional) |

### Setup

```bash
# Clone the repository
git clone https://github.com/RichardWANG-1010/Phase-Field-Crystal.git
cd Phase-Field-Crystal

# Install core dependencies
pip install numpy scipy matplotlib

# Install XPFC dependency (DualXPFC branch)
pip install taichi

# Optional: 3D visualization
pip install pyvista scikit-image

# Optional: Video generation (install via system package manager)
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: download from https://ffmpeg.org/download.html

# Optional: C++/MPI solver dependencies (DualXPFC)
# Ubuntu/Debian: sudo apt-get install libopenmpi-dev libfftw3-dev
# Compile: mpicxx -O3 -o xpfcSqAmpDis xpfcSqAmpDis.cpp -lfftw3_mpi -lfftw3 -lm
```

---

## Quick Start

### 1. Pure Material PFC Simulation

```bash
python run_pure.py
```

Launches an interactive menu to configure simulation parameters, then runs the simulation and generates analysis plots.

### 2. Elastic Constant Calculation

```bash
python run_elastic.py
```

Calculates the elastic modulus by applying uniaxial strain and fitting the energy-strain curve with a quadratic polynomial.

### 3. Binary Alloy Simulation

```bash
python run_binary.py
```

Simulates spinodal decomposition and phase separation in binary alloys. Supports 4 modes: Quick, Interactive, High-Resolution, and Phase Separation.

### 4. 3D PFC Simulation

```bash
python run_3d.py
```

Runs full 3D PFC simulation with BCC lattice support. Choose from: pure material, elastic constant calculation, binary alloy, or parameter sweep.

### 5. Swift-Hohenberg Model

```bash
python sh_model.py
```

Runs the Swift-Hohenberg pattern formation solver directly.

### 6. DualXPFC — GUI Interface

```bash
# Switch to the DualXPFC branch first
git checkout DualXPFC

# Launch the Tkinter GUI (3 models: flat interface, round interface, dislocation)
python main.py
```

Opens an interactive GUI to select from three research models, configure parameters, run sigma-series batch simulations, and visualize results.

### 7. DualXPFC — Correlation-Function Square Solver

```bash
python xpfc_square.py
```

Launches an interactive console menu to choose single-crystal or polycrystal mode, then runs the Taichi-accelerated XPFC square-lattice simulation with real-time video recording.

### 8. DualXPFC — Quick Test

```bash
python quick_test.py
```

Runs a fast verification suite (64×64 grid, 100 steps) to confirm the dual-amplitude engine, all three interface/dislocation models, and energy computation work correctly.

### 9. DualXPFC — C++/MPI Amplitude Solver

```bash
# Compile
mpicxx -O3 -o xpfcSqAmpDis xpfcSqAmpDis.cpp -lfftw3_mpi -lfftw3 -lm

# Run (configure parameters in xpfcSqAmpDis.in)
mpirun -np 4 ./xpfcSqAmpDis < xpfcSqAmpDis.in
```

High-performance amplitude-representation XPFC solver with MPI parallelization, supporting 1024×1024 grids and dislocation initialization.

---

## Theoretical Background

### PFC Model

The Phase Field Crystal model describes crystalline structures at atomic length scales while operating on diffusive time scales, making it ideal for studying grain growth, boundary migration, and dislocation dynamics.

#### Free Energy Functional

```
F = ∫ [φ/2 · (r + (1 + ∇²)²) φ + φ⁴/4] dr
```

Where:
- `φ` — Dimensionless density field
- `r` — Reduced temperature parameter (controls undercooling)
- `∇²` — Laplacian operator

#### Dynamic Equation (Conserved Dynamics)

```
∂φ/∂t = ∇² · δF/δφ
```

Solved via semi-implicit Fourier spectral method for numerical stability and efficiency.

### Dual-Amplitude (Two-Mode) PFC Model

The dual-amplitude PFC formulation generalizes the standard PFC model by introducing a **product-form linear operator** that couples two characteristic wave numbers, together with a **cubic nonlinearity**. This enables coexistence between square and triangular crystal phases controlled by a single `sigma` parameter, and provides more accurate interface and defect energetics.

#### Free Energy Functional

```
F = ∫ [ 1/2 ψ · L · ψ  −  τ/3 · ψ³  +  1/4 · ψ⁴ ] dV
```

#### Linear Operator (Two-Mode, Product Form)

```
L = [ r + (1 + ∇²)² ] · [ (q² + ∇²)² / q⁴ ]
```

In Fourier space:

```
L(k) = [ r + (1 − k²)² ] · [ (q² − k²)² / q⁴ ]
```

Where:
- `ψ` — Dimensionless density field
- `r` — Reduced temperature, mapped from `sigma` via `r = −0.4 + 0.4·sigma`
- `τ` — Cubic coefficient (default `τ = 1.0`)
- `q` — Wave-number ratio (default `q = √3`, corresponding to the second reciprocal shell of a triangular lattice)
- `∇²` — Laplacian operator

#### Dynamic Equation (Conserved Dynamics)

```
∂ψ/∂t = ∇² · δF/δψ = ∇² · [ Lψ − τψ² + ψ³ ]
```

#### Semi-Implicit Time Stepping

```
ψ_k(t+dt) = [ ψ_k(t) − dt·k²·(−τψ² + ψ³)_k ] / [ 1 + dt·k²·L(k) ]
```

#### Phase Selection by Sigma

| Sigma Range | Crystal Phase |
|-------------|---------------|
| `sigma < 0` (sigma₁, sigma₂) | Square lattice |
| `sigma ≥ 0` (sigma₃, sigma₄, sigma₅) | Triangular (hexagonal) lattice |

### Correlation-Function (C₂) XPFC Model

The correlation-function XPFC formulation constructs the linear operator from a **direct correlation function `C₂(k)`** built as a superposition of Gaussian peaks centered on reciprocal lattice sites. This approach provides direct control over peak positions, widths, and weights — governing elasticity, anisotropy, and interface widths.

#### Free Energy Functional

```
F = ∫ [ 1/2 n · (1 − C₂) · n  −  η/6 · n³  +  χ/12 · n⁴ ] dV
```

#### Direct Correlation Function

```
C₂(k) = σ · Σ_i  w_i · exp( −|k − G_i|² / (2α_i²) )
```

Where `G_i` are reciprocal lattice vectors of the square lattice:
- First shell: `(±q, 0)`, `(0, ±q)` with `q = 1.0`
- Second shell: `(±q, ±q)` with `q = √2`

And:
- `n` — Dimensionless density field
- `σ` — Overall correlation strength (melting temperature control)
- `w_i`, `α_i` — Weight and width of each Gaussian peak
- `η` — Cubic coefficient (`−n³/6` term)
- `χ` — Quartic coefficient (`n⁴/12` term)

#### Linear Operator

```
L(k) = 1 − C₂(k)
```

#### Dynamic Equation & Semi-Implicit Stepping

```
∂n/∂t = ∇² · [ (1−C₂)n − η/2·n² + χ/3·n³ ]

n_k(t+dt) = [ n_k(t) − dt·k²·(−η/2·n² + χ/3·n³)_k ] / [ 1 + dt·k²·(1−C₂(k)) ]
```

Implemented with **Taichi** CPU kernels for the nonlinear term and k-space update, achieving ~8× speedup over pure NumPy on 8 threads.

### C++ Amplitude-Representation XPFC Model

The C++ solver uses an **amplitude expansion** of the density field — decomposing it into an average density `n₀` plus complex amplitude fields modulating each reciprocal lattice mode. This representation is numerically efficient for large systems (1024×1024 and above) and naturally incorporates dislocation nucleation via amplitude noise.

#### Field Decomposition

```
n(r) = n₀ + Re[ A(r)·e^{iG₁·r} + B(r)·e^{iG₂·r} + ... ]
```

Where `A`, `B` are slowly-varying complex amplitude fields with independent mobilities `M_A`, `M_B`.

#### Implementation
- **Parallelization**: MPI domain decomposition with FFTW3-MPI transposed FFTs
- **Grid**: Up to 1024×1024, `dx = 0.25`, `dt = 1.0`
- **Correlation function**: k-zero mode + first peak (w₁) + second peak (w₂ = √2)
- **Dislocation**: Controlled via `dislNoiseAmp` parameter in the input file
- **Parameters**: All runtime parameters read from `xpfcSqAmpDis.in`

### Binary Alloy Model

Extended PFC model with two conserved fields: total density `φ` and concentration `c`.

```
F = F_PFC[φ] + F_CH[c] + F_coupling[φ, c]
```

Where `F_CH` is the Cahn-Hilliard free energy and `F_coupling` describes the interaction between density and concentration fields.

---

## Lattice Types

### 2D Lattices

| Lattice | Dispersion Operator `L(k)` | Crystal Structure |
|---------|---------------------------|-------------------|
| **Hexagon** | `L(k) = (1 - k²)² + r` | Triangular lattice |
| **Square** | `L(k) = (1-kx²)²(1-ky²)² + r` | Square lattice |
| **Triangle** | `L(k) = (1 - kx² - kx·ky + ky²)² + r` | Oblique lattice |

### 3D Lattices

| Lattice | Description | Notes |
|---------|-------------|-------|
| **BCC** | Body-Centered Cubic | Standard PFC (Chapter 8, Provatas & Elder) |
| **FCC** | Face-Centered Cubic | Requires additional stabilization terms |
| **SC** | Simple Cubic | Basic cubic symmetry |

### XPFC-Supported Lattices

| Formulation | Supported Lattices | Control Parameter |
|-------------|-------------------|-------------------|
| Dual-amplitude PFC (`pfc_core.py`) | Square, Triangular | `sigma` (low → square, high → triangular) |
| Correlation-function XPFC (`xpfc_square.py`) | Square | `sigma`, peak weights/widths in `C₂(k)` |
| C++ amplitude solver (`xpfcSqAmpDis.cpp`) | Square | `xpfcSqAmpDis.in` input file |

---

## Code Documentation

### Core Modules

#### `pfc_base.py` — Numerical Infrastructure Base Class

The foundation of all PFC solvers. Provides:
- **Grid Setup**: `N×N` (2D) or `N³` (3D) spatial discretization with physical domain size `L`
- **K-Space Construction**: Pre-computed Fourier wave vectors `kx, ky, kz` and squared magnitude `k2 = kx² + ky² + kz²` using `numpy.fft.fftfreq`
- **Simulation Logs**: Arrays for energy, mass, defect density, grain size, and structure factor peaks
- **Analysis Cache**: Storage for detected atom positions, neighbor lists, and ψ₆ values
- **Video Config**: Frame directory setup and recording flags

**Key Method**: `_build_kspace()` — Constructs reciprocal space grids with correct FFT frequency ordering and matrix-style indexing.

---

#### `pfc_pure.py` — Pure Material PFC Solver

Implements the standard single-component PFC model via **multiple inheritance** (mixin pattern):

```python
class PurePFCSolver(PFCBase, PFCAnalysis, PFCPlot, PFCIO, PFCElastic, PFCAdvancedAnalysis):
```

**Capabilities**:
- **Semi-implicit spectral time stepping**: Linear terms implicit, nonlinear `φ³` explicit
- **Lattice-specific dispersion operators**: Real-time calculation of `L(k)` for hexagon/square/triangle
- **Mass conservation enforcement**: Mean density correction after each step
- **Energy computation**: Parseval's theorem for efficient k-space integration
- **Structure factor**: `S(k) = |φ̃(k)|²` with `fftshift` for centered diffraction patterns

**Key Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | 128 | Grid resolution |
| `L` | 64.0 | Physical domain size |
| `r` | -0.25 | Temperature parameter (negative favors crystalline phase) |
| `M` | 1.0 | Mobility coefficient |
| `phi0` | -0.25 | Average density |
| `lattice_type` | `"hexagon"` | Crystal symmetry |

---

#### `pfc_binary.py` — Binary Alloy PFC Solver

Dual conserved-field solver coupling density `φ` and concentration `c`:

**Free Energy**:
```
F = F_PFC[φ] + F_CH[c] + F_coupling[φ, c]
F_PFC = 0.5·φ·L(k)·φ + 0.25·φ⁴
F_CH = 0.5·c·(r_c - ∇²)·c + 0.25·u_c·c⁴
F_coupling = α·c·φ² + β·c·φ
```

**Evolution Equations** (conserved Cahn-Hilliard dynamics):
```
∂φ/∂t = M_φ·∇²·(δF/δφ)
∂c/∂t = M_c·∇²·(δF/δc)
```

**Features**:
- Simultaneous semi-implicit update of both fields in Fourier space
- Concentration physical constraint: `0 ≤ c ≤ 1` via clipping
- Dual-field video capture (density + concentration side-by-side)
- Extended post-processing: concentration evolution, overlay fields, coupling energy
- Concentration structure factor analysis

---

#### `pfc_3d.py` — 3D PFC Solver

Full 3D extension supporting BCC, FCC, and SC lattices. Organized into 4 mixin classes:

| Class | Purpose |
|-------|---------|
| `PFCBase3D` | 3D grid, `N³` k-space (`kx, ky, kz`), volume calculations |
| `PFCAnalysis3D` | 3D structure factor, atom detection via `peak_local_max`, 2D slicing |
| `PFCPlot3D` | 3D isosurfaces (PyVista + Matplotlib fallback), orthogonal slice panels |
| `PFCElastic3D` | 3D strain tensors, uniaxial strain with Poisson effect, bulk modulus |

---

#### `sh_model.py` — Swift-Hohenberg Solver

Standalone pattern formation solver for the canonical Swift-Hohenberg equation:
```
∂ψ/∂t = ε·ψ - (q₀² + ∇²)²·ψ - ψ³
```

With embedded video recording, pre-computed linear operator, and automatic ffmpeg/GIF fallback.

---

### DualXPFC Extension Modules

#### `pfc_core.py` — Dual-Amplitude (Two-Mode) PFC Core Engine

The foundational engine for all DualXPFC research models. Implements the two-mode product-form linear operator with cubic nonlinearity.

**Classes**:

| Class | Purpose |
|-------|---------|
| `DualPFCConfig` | Configuration container: grid (`Nx, Ny, Lx, Ly`), model parameters (`sigma, tau, q, k0`), time stepping (`dt, n_steps`), initial conditions (`amplitude, noise, mean_density`) |
| `DualPFCEngine` | Simulation engine: k-space operator construction, time evolution, energy computation, structure factor |

**Key Features**:
- **Sigma-to-r mapping**: `r = −0.4 + 0.4·sigma` (linear, adjustable for phase diagram tuning)
- **Two-mode linear operator**: `L(k) = [r + (1−k²)²] · [(q²−k²)²/q⁴]` precomputed in Fourier space
- **Crystal initial conditions**: `square_crystal()` (cos(k₀x)+cos(k₀y)) and `triangular_crystal()` (one-mode + two-mode cosine superposition)
- **Semi-implicit stepping**: `denom = 1 + dt·k²·L(k)`, nonlinear `−τψ² + ψ³` computed in real space
- **Energy methods**: `total_energy()`, `energy_density()`, `chemical_potential()`, `bulk_energy()` for square/triangular/liquid phases
- **Structure factor**: `S(k) = |ψ̃(k)|²` with `fftshift`

**Key Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Nx, Ny` | 256, 256 | Grid resolution |
| `Lx, Ly` | `Nx·π/4` | Physical domain size (~8 grid points per 2π period) |
| `sigma` | 0.0 | Control parameter (mapped to `r`; low → square, high → triangular) |
| `tau` | 1.0 | Cubic coefficient |
| `q` | √3 | Wave-number ratio (second shell of triangular lattice) |
| `dt` | 0.5 | Time step |
| `n_steps` | 2000 | Iteration count |
| `amplitude` | 0.3 | Crystal initial amplitude |

---

#### `xpfc_square.py` — Correlation-Function C₂ XPFC Square Solver

Taichi-accelerated standalone XPFC solver for square lattices, using a direct correlation function `C₂(k)` constructed from multi-Gaussian peaks.

**Class**: `XPFC_Square_CPU`

**Key Features**:
- **Correlation function construction**: `C₂(k)` built from 8 reciprocal lattice sites (first shell `q=1.0`, second shell `q=√2`), each a Gaussian with configurable width `α` and weight `w`
- **Linear operator**: `L(k) = 1 − C₂(k)`, with `C₂(0,0) = 0.3` fixed
- **Taichi kernels**: `compute_nl_kernel()` (nonlinear `−η/2·n² + χ/3·n³`), `compute_step_kernel()` (k-space semi-implicit update), `enforce_density_kernel()`, `clip_field_kernel()`
- **Initialization modes**: `initialize_single_crystal()` (perfect periodic lattice) and `initialize_polycrystal()` (random noise + 5 random-phase seeds for multi-domain nucleation)
- **Built-in video encoder**: `VideoEncoder` class using ffmpeg pipe with H.264, configurable CRF/preset/resolution
- **Final analysis plot**: Density field + log structure factor + C₂(k) cross-section

**Key Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | 256 | Grid resolution |
| `L` | 64.0 | Physical domain size |
| `sigma` | 1.5 | Correlation strength (melting temperature) |
| `eta` | 1.0 | Cubic coefficient (`−n³/6` term) |
| `chi` | 1.0 | Quartic coefficient (`n⁴/12` term) |
| `dt` | 0.01 | Time step |
| `n0` | -0.25 | Average density |
| `steps` | 50000 | Total iterations |

---

#### `main.py` — Tkinter GUI Main Interface

Interactive graphical front-end that unifies all three DualXPFC research models.

**Class**: `PFCMainApp`

**Features**:
- **Model selection**: Radio buttons for Model 1 (flat interface), Model 2 (round interface), Model 3 (dislocation)
- **Parameter panel**: Grid-based entry fields for `Nx, Ny, dt, n_steps, tau, q, amplitude, mean_density`, plus comma-separated `sigma` values
- **Run modes**: Single sigma or full sigma-series batch; optional live visualization
- **Background threading**: Simulations run in a daemon thread to keep the GUI responsive; Stop button support
- **Result viewing**: `plot_result_comprehensive_with_initial()` (initial + final field, energy density, structure factor, energy convergence) and `plot_dislocation_comparison_with_initial()`
- **Sigma-series plot**: Interface energy `γ_SL` or dislocation energy `E_dis` vs. sigma
- **Data export**: `simulation_results.txt` (human-readable) + `result_N_psi.npy` (NumPy arrays)

---

#### `part1_flat_interface.py` — Flat Interface Energy (Model 1)

**Class**: `FlatInterfaceModel`

Computes the solid-liquid **flat interface energy** `γ_SL` by constructing a half-solid/half-liquid domain with a smooth `tanh` transition, relaxing to equilibrium, and subtracting bulk energies.

```
γ_SL = (E_total − E_solid·V_s/V − E_liquid·V_l/V) · L / A_interface
```
where `A_interface = Ly` (interface normal along x).

- sigma₁, sigma₂ → square crystal solid phase
- sigma₃–sigma₅ → triangular crystal solid phase

---

#### `part2_round_interface.py` — Circular Interface Energy (Model 2)

**Class**: `RoundInterfaceModel`

Computes the **circular nucleus interface energy** by embedding a circular solid seed (default radius = min(Lx,Ly)/4) in a liquid matrix, relaxing via nucleus growth, and extracting `γ_SL` and effective radius `R_eff`.

```
γ_SL = (E_total − E_solid·V_s/V − E_liquid·V_l/V) / (2π·R)
```

---

#### `part3_dislocation.py` — Dislocation Energy (Model 3)

**Class**: `DislocationModel`

Computes the **edge dislocation energy** `E_dis` by creating a perfect crystal, then shifting the right half (`x > Lx/2`) upward by half a lattice unit to form an edge dislocation, relaxing, and subtracting the perfect-crystal energy.

```
E_dis = E_total(dislocated) − E_perfect(perfect crystal)
```

- 256×256 domain
- Triangular phase: all 5 sigma values
- Square phase: sigma₁, sigma₂
- Total: 7 simulations per full series

---

#### `visualization.py` — Interactive Visualization Module

**Class**: `PFCVisualizer`

Provides comprehensive plotting for all three DualXPFC models:
- **Density field**, **energy density**, **structure factor** (log scale), **energy convergence**
- **Interactive controls**: Matplotlib `Slider`, `Button`, `CheckButtons` with TkAgg backend
- **`plot_result_comprehensive_with_initial()`**: 6-panel comparison (initial field, final field, initial/final structure factor, energy density, energy convergence)
- **`plot_dislocation_comparison_with_initial()`**: Perfect crystal vs. dislocated crystal side-by-side with difference field
- **`plot_sigma_series()`**: `γ_SL` or `E_dis` vs. sigma with error bars
- **Chinese font support**: Auto-detects Microsoft YaHei, SimHei, etc.

---

#### `quick_test.py` — Quick Verification Test

Runs a fast end-to-end test suite on a 64×64 grid (100 steps) to verify:
1. Core engine instantiation and k-space operator construction
2. Square and triangular crystal initial conditions
3. Flat interface model build + run + `γ_SL` computation
4. Round interface model build + run + `γ_SL`, `R_eff`
5. Dislocation model prepare + run + `E_dis`
6. Energy monotonic decrease and mass conservation

---

#### `xpfcSqAmpDis.cpp` — C++/MPI Amplitude-Representation Solver

High-performance XPFC square-lattice solver using amplitude expansion with MPI parallelization.

**Architecture**:
- **MPI domain decomposition**: 1D slab decomposition with FFTW3-MPI transposed FFTs
- **Fields**: Average density `n` + two complex amplitude fields `A`, `B` (modulating first and second reciprocal shells)
- **Mobilities**: Independent `M_n₀`, `M_A`, `M_B` for density and amplitudes
- **Correlation function**: k-zero mode (HSq0, wSq0) + first peak (kSq1, wSq1, sigMSq1) + second peak (kSq2, wSq2, sigMSq2)
- **Dislocation nucleation**: `dislNoiseAmp` controls amplitude noise for dislocation generation
- **Restart capability**: `restartFlag` + `restartTime` for checkpoint/resume

**Typical parameters** (from `xpfcSqAmpDis.in`):
- 1024×1024 grid, `dx = 0.25`, `dt = 1.0`, 1,000,001 iterations
- `wSq1 = 1.0`, `wSq2 = √2`, `sigMSq1 ≈ 0.2026`, `sigMSq2 ≈ 0.1013`
- `η = 1.4` (cubic), `χ = 1.0` (quartic)
- `dislNoiseAmp = 0.3`

---

### Analysis Modules

#### `pfc_analysis.py` — Microstructure & Defect Analysis

Comprehensive mixin providing quantitative microstructure characterization:
- **Atom Detection**: `detect_atoms()` via `peak_local_max`; `build_neighbors()` via periodic KD-Tree; `coordination_numbers()`
- **Topological Analysis**: `voronoi_analysis()`, `compute_psi6()`, `global_psi6()`, `grain_boundary_parameter()` (D-parameter)
- **Defect Analysis**: `analyze_defects()` (5/7-fold density + approximate grain size), `defect_statistics()`

---

#### `pfc_elastic.py` — Elastic Property Calculations

Mixin for mechanical property extraction:
- **Strain Application**: `apply_strain(eps)` rescales domain and rebuilds k-space; `save_reference_state()`
- **Energy-Strain Analysis**: `elastic_energy_curve()`, `fit_elastic_constant()` (quadratic fit → `C = d²F/dε²`), `compute_stress()`

---

#### `pfc_plot.py` — Visualization Toolkit

Extensive plotting mixin (20+ methods): field plots, structure factor, Voronoi, defects, detected atoms, grain boundaries, ψ₆ order, grain orientation, evolution curves, elastic curves, and full pipelines (`postprocess()`, `analyze_psi6()`).

---

### IO Module

#### `pfc_io.py` — Video Recording

Mixin for simulation video generation with in-memory PNG frame caching (zero disk I/O during simulation) and ffmpeg H.264 MP4 synthesis (CRF=18).

---

### Configuration

#### `config.py` — Interactive Parameter Menu

Console-based interactive interface with lattice selection (numbered menu), validated float/int input with defaults, parameter packaging, and confirmation step.

---

### Run Scripts

#### `run_pure.py` — Pure Material Simulation Runner
Entry point for standard PFC simulations: interactive config → `PurePFCSolver` → main loop → `postprocess()` → `analyze_psi6()` → vacancy detection.

#### `run_binary.py` — Binary Alloy Simulation Runner
4 operational modes: Quick (N=512), Interactive, High-Resolution (N=1024, T=5000), Phase Separation. CLI: `--mode`, `--interactive`, `--video`.

#### `run_elastic.py` — Elastic Constant Calculator
Equilibrate (T=1500) → save reference → 13 strain points (`linspace(-0.03, 0.03, 13)`) → relax 2000 steps each → quadratic fit → `C = 2a`.

#### `run_3d.py` — 3D Simulation Runner
4 modes: pure (BCC growth), elastic (3D constants), alloy (3D binary), sweep (r-parameter scan). Memory: N=128 (~200MB), N=256 (~1.5GB, recommended), N=512 (~12GB).

---

## Examples

### Example 1: Grain Growth Simulation (Standard PFC)

```python
from pfc_pure import PurePFCSolver

solver = PurePFCSolver(
    N=256, L=128.0, r=-0.25, dt=0.05, T=2000.0,
    lattice_type="hexagon"
)
solver.run(sample_interval=10)
solver.postprocess()
solver.analyze_psi6()
```

### Example 2: DualXPFC — Flat Interface Energy

```python
from pfc_core import DualPFCConfig, DualPFCEngine
from part1_flat_interface import FlatInterfaceModel

# sigma=0.2 → triangular phase; sigma=-0.2 → square phase
cfg = DualPFCConfig(Nx=256, Ny=256, sigma=0.2, tau=1.0,
                     q=np.sqrt(3), dt=0.5, n_steps=2000)

model = FlatInterfaceModel(cfg, crystal_type='triangular')
model.build_initial_condition()   # half solid, half liquid with tanh transition
model.run()                        # relax to equilibrium
gamma = model.compute_interface_energy()
print(f"Flat interface energy gamma_SL = {gamma:.6f}")
```

### Example 3: DualXPFC — Dislocation Energy

```python
from pfc_core import DualPFCConfig
from part3_dislocation import DislocationModel

cfg = DualPFCConfig(Nx=256, Ny=256, sigma=0.0, n_steps=2000)
model = DislocationModel(cfg, crystal_type='triangular')
model.prepare_and_run()            # perfect crystal → shift half → relax
print(f"Dislocation energy E_dis = {model.E_dis:.6f}")
```

### Example 4: DualXPFC — Correlation-Function Square Solver

```python
from xpfc_square import XPFC_Square_CPU
import numpy as np

sim = XPFC_Square_CPU(
    N=256, L=64.0, sigma=1.5, eta=1.0, chi=1.0,
    dt=0.01, n0=-0.25, noise_amp=0.03
)
sim.initialize_polycrystal(seed=True, seed_amp=0.15)
for i in range(50000):
    sim.step()
    if i % 1000 == 0:
        print(f"Step {i}: <n>={sim.n.mean():.4f}, amp={sim.n.max()-sim.n.min():.4f}")
```

### Example 5: DualXPFC — Core Engine Direct Usage

```python
from pfc_core import DualPFCConfig, DualPFCEngine
import numpy as np

cfg = DualPFCConfig(Nx=128, Ny=128, sigma=0.0, tau=1.0,
                     q=np.sqrt(3), dt=0.5, n_steps=500)
engine = DualPFCEngine(cfg)

X, Y = engine.get_coordinate_grids()
psi = engine.triangular_crystal(X, Y) + 0.01 * np.random.randn(128, 128)

psi, energies = engine.relax(psi, n_steps=500)
print(f"Final energy: {energies[-1][1]:.4f}")
```

---

## Output

### Standard PFC Generated Files

```
result/
├── pfc_simulation.mp4      # Evolution video
├── energy_evolution.png    # Energy curve
├── density_field.png       # Final density field
├── structure_factor.png    # Structure factor (diffraction pattern)
├── voronoi_analysis.png    # Voronoi tessellation
├── defect_analysis.png     # Defect visualization (5/7-fold)
├── defect_density.png      # Defect density evolution
├── grain_size.png          # Grain size evolution
├── psi6_order.png          # Bond-orientational order
└── grain_boundary.png      # Grain boundary atoms
```

### DualXPFC Generated Files

```
DualXPFC/
├── xpfc_output/
│   ├── xpfc_single_crystal.mp4   # Single-crystal evolution video
│   ├── xpfc_polycrystal.mp4      # Polycrystal evolution video
│   └── final_single.png / final_poly.png  # Final analysis (field + S(k) + C2)
├── simulation_results.txt         # GUI-exported results (gamma_SL, E_dis, sigma series)
├── result_1_psi.npy ...           # NumPy final field arrays
└── xpfcSqAmpDis/                  # C++ solver output (amplitude fields, density, restart)
```

### Key Observables

| Observable | Description |
|------------|-------------|
| Free Energy | System free energy evolution (should decrease monotonically) |
| Density Field | Spatial atomic density distribution `φ(r)` or `ψ(r)` |
| Structure Factor | Reciprocal space diffraction pattern `S(k) = |φ̃(k)|²` |
| Defect Density | Fraction of 5/7-fold coordination atoms |
| Grain Size | Approximate average grain diameter from defect density |
| ψ₆ Order | Bond orientational order quantification |
| Elastic Constant | Young's modulus from energy-strain quadratic fit |
| **γ_SL (Flat)** | Flat solid-liquid interface energy (DualXPFC Model 1) |
| **γ_SL, R_eff (Round)** | Circular nucleus interface energy and effective radius (DualXPFC Model 2) |
| **E_dis** | Edge dislocation energy (DualXPFC Model 3) |
| **C₂(k)** | Direct correlation function cross-section (correlation-function XPFC) |
| **Amplitude fields** | Complex amplitude A(r), B(r) (C++ amplitude solver) |

---

## Author & Citation

**Jinpeng Wang (王锦鹏)**
Department of Aviation Engineering, The Hong Kong Polytechnic University
Mitacs Intern @ McMaster University

If you use this framework in your research, please cite appropriately.

### References

- Elder, K. R., & Grant, M. (2004). Modeling elastic and plastic deformations in nonequilibrium processing using phase field crystals. *Physical Review E*, 70(5), 051605.
- Provatas, N., & Elder, K. (2010). Phase-Field Methods in Materials Science and Engineering. Wiley-VCH.
- Berry, J., Elder, K. R., & Grant, M. (2008). Phase-field crystal modeling of eutectic solidification. *Physical Review Letters*, 100(4), 045705.
- Greenwood, M., Rottler, J., & Provatas, N. (2010). Free energy of crystal-liquid interfaces in the phase-field crystal method. *Physical Review E*, 81(6), 061601.
- Athreya, P., et al. (2007). Diffusive atomistic dynamics of edge dislocations in two dimensions. *Physical Review E*, 75(2), 021603.

---

## Changelog

### v1.2.0
- **Added DualXPFC extension** on the `DualXPFC` branch with three research models:
  - Flat solid-liquid interface energy calculation (`part1_flat_interface.py`)
  - Circular nucleus interface energy with effective radius (`part2_round_interface.py`)
  - Edge dislocation energy computation for square and triangular phases (`part3_dislocation.py`)
- **Added dual-amplitude (two-mode) PFC core engine** (`pfc_core.py`) with product-form linear operator `L = [r+(1+∇²)²][(q²+∇²)²/q⁴]`, cubic nonlinearity, and sigma-controlled square/triangular phase selection
- **Added correlation-function C₂ XPFC square solver** (`xpfc_square.py`) with Taichi CPU acceleration, multi-Gaussian peak construction, single/polycrystal modes, and built-in ffmpeg video encoding
- **Added C++/MPI amplitude-representation solver** (`xpfcSqAmpDis.cpp` + `xpfcSqAmpDis.in`) with FFTW3-MPI parallelization, amplitude expansion, and dislocation initialization
- **Added Tkinter GUI main interface** (`main.py`) with model selection, parameter panel, sigma-series batch runs, live visualization, and data export
- **Added interactive visualization module** (`visualization.py`) with sliders, play/pause, comprehensive result comparison plots, and Chinese font support
- **Added quick verification test** (`quick_test.py`) for end-to-end engine and model validation

### v1.1.0
- Added 3D PFC simulation (BCC lattice) with PyVista isosurface rendering
- Added Swift-Hohenberg model with embedded video recording
- Added 3D elastic constant calculation with Poisson effect
- Improved video generation with automatic GIF fallback
- Added binary alloy phase separation modes (quick/interactive/high-res)

### v1.0.0
- Initial release with 2D PFC framework
- Hexagonal, square, triangle lattice support
- Pure material and binary alloy models
- Full analysis and visualization toolkit
- Defect detection, Voronoi analysis, ψ₆ order parameter
- Elastic energy and stress-strain calculations

---

## License

This project is for academic research purposes. Please cite appropriately if used in publications.

*Last updated: 2026-09*
