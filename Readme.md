<div align="center">

# Phase-Field-Crystal

**A Modular 2D/3D Phase Field Crystal Simulation Framework**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.20.0-orange)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%E2%89%A51.7.0-green)](https://scipy.org/)
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

This framework implements the standard PFC model and its extensions (binary alloys, 3D BCC/FCC/SC lattices, Swift-Hohenberg model) using an efficient **semi-implicit Fourier spectral method**. The modular architecture separates core solvers, analysis tools, visualization, and I/O into independent mixin classes, allowing flexible composition and easy extension.

### Key Capabilities

| Category | Features |
|----------|----------|
| **Core Simulation** | Spectral semi-implicit PFC solver; Multiple 2D/3D lattice types; Pure material & binary alloy models |
| **Analysis** | Defect detection (5/7-fold), Voronoi tessellation, ψ₆ orientational order, grain boundary identification, structure factor analysis |
| **Elasticity** | Energy-strain curves, stress-strain relations, elastic constant fitting |
| **Visualization** | Real-time field rendering, MP4/GIF video generation, 3D isosurface rendering (PyVista) |

---

## Features

### Core Simulation
- **Spectral Semi-Implicit PFC Solver** — Efficient Fourier-space numerical integration with unconditional stability for linear terms
- **Multiple Lattice Types** — Hexagonal, Square, Triangle (2D); BCC, FCC, SC (3D)
- **Pure Material & Binary Alloy** — Single-component and two-phase field coupling with Cahn-Hilliard concentration dynamics
- **3D PFC Simulation** — Full 3D BCC lattice support with volumetric visualization and elastic constant calculation
- **Swift-Hohenberg Model** — Canonical pattern-forming PDE for stripe/hexagonal pattern studies

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
- **Comprehensive Post-processing** — Full suite of analysis plots (energy, defects, grain size, etc.)

---

## Project Structure

```
Phase-Field-Crystal/
│
├── README.md              # English documentation (this file)
├── README.zh.md           # Chinese documentation
│
├── Core Modules
│   ├── pfc_base.py        # Base class: grid setup, k-space construction, simulation logs
│   ├── pfc_pure.py        # Pure material PFC solver (hexagon/square/triangle lattices)
│   ├── pfc_binary.py      # Binary alloy PFC solver (dual-field: density + concentration)
│   ├── pfc_3d.py          # 3D PFC solver (BCC/FCC/SC lattices with PyVista visualization)
│   └── sh_model.py        # Swift-Hohenberg pattern formation solver
│
├── Analysis Modules
│   ├── pfc_analysis.py    # Microstructure analysis: atom detection, coordination, ψ6, Voronoi, defects
│   ├── pfc_elastic.py     # Elastic property calculations: strain application, energy curves, fitting
│   └── pfc_plot.py        # Visualization toolkit: fields, structure factors, defects, grain boundaries
│
├── IO Module
│   └── pfc_io.py          # Video recording: in-memory frame caching, ffmpeg MP4 synthesis
│
├── Configuration
│   └── config.py          # Interactive console menu for parameter input with validation
│
└── Run Scripts
    ├── run_pure.py        # Standard pure material PFC simulation with interactive config
    ├── run_binary.py      # Binary alloy simulation with 4 preset modes (quick/interactive/high-res/phase-separation)
    ├── run_elastic.py     # Elastic constant calculation via energy-strain quadratic fitting
    └── run_3d.py          # 3D PFC simulation with 4 modes (pure/elastic/alloy/sweep)
```

---

## Installation

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.20.0 | Core numerical arrays and FFT |
| `scipy` | ≥1.7.0 | FFT operations, spatial algorithms |
| `matplotlib` | ≥3.4.0 | 2D plotting and visualization |
| `pyvista` | ≥0.32.0 | 3D isosurface rendering (optional) |
| `scikit-image` | ≥0.18.0 | 3D peak detection (optional) |
| `ffmpeg` | system | Video generation (optional, system package) |

### Setup

```bash
# Clone the repository
git clone https://github.com/RichardWANG-1010/Phase-Field-Crystal.git
cd Phase-Field-Crystal

# Install core dependencies
pip install numpy scipy matplotlib

# Optional: 3D visualization
pip install pyvista scikit-image

# Optional: Video generation (install via system package manager)
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: download from https://ffmpeg.org/download.html
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
| `lattice_type` | "hexagon" | Crystal symmetry |

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

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `c0` | 0.3 | Initial average concentration |
| `M_c` | 0.1 | Concentration mobility |
| `alpha` | 0.1 | Segregation coupling strength |
| `beta` | 0.0 | Direct coupling strength |
| `r_c` | -0.5 | Cahn-Hilliard parameter |

---

#### `pfc_3d.py` — 3D PFC Solver

Full 3D extension supporting BCC, FCC, and SC lattices. Organized into 4 mixin classes:

| Class | Purpose |
|-------|---------|
| `PFCBase3D` | 3D grid, `N³` k-space (`kx, ky, kz`), volume calculations |
| `PFCAnalysis3D` | 3D structure factor, atom detection via `peak_local_max`, 2D slicing |
| `PFCPlot3D` | 3D isosurfaces (PyVista + Matplotlib fallback), orthogonal slice panels |
| `PFCElastic3D` | 3D strain tensors, uniaxial strain with Poisson effect, bulk modulus |

**Key Features**:
- **BCC dispersion**: `L(k) = (1 - |k|²)² + r` (standard PFC, Chapter 8)
- **FCC stabilization**: Additional `α·(kx²ky² + ky²kz² + kz²kx²)` term
- **3D atom detection**: `skimage.feature.peak_local_max` with `min_distance=5`
- **Isosurface rendering**: PyVista `UniformGrid` + `contour()` with automatic fallback to Matplotlib `marching_cubes`
- **Elastic constants**: Full 3×3 strain tensor support with `fit_elastic_constant_3d()`

---

#### `sh_model.py` — Swift-Hohenberg Solver

Standalone pattern formation solver for the canonical Swift-Hohenberg equation:

```
∂ψ/∂t = ε·ψ - (q₀² + ∇²)²·ψ - ψ³
```

**Design**:
- Inherits `PFCBase` for grid/k-space reuse
- Embedded video recording (independent of `pfc_io.py`)
- Pre-computed linear operator for efficiency: `L = -(ε - (q₀² - k²)²)`
- Automatic ffmpeg/GIF fallback for video generation
- Power spectrum analysis with unstable ring annotation

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon` | 0.3 | Control parameter (pattern threshold) |
| `q0` | 1.0 | Characteristic wavenumber |
| `psi_clip` | 2.0 | Field clipping for numerical stability |

---

### Analysis Modules

#### `pfc_analysis.py` — Microstructure & Defect Analysis

Comprehensive mixin providing quantitative microstructure characterization:

**Atom Detection**:
- `detect_atoms()` — Local peak detection using `skimage.feature.peak_local_max` with `min_distance=7` and `threshold_rel=0.5`
- `build_neighbors()` — Periodic boundary KD-Tree (`cKDTree` with `boxsize`) for neighbor lists
- `coordination_numbers()` — Count nearest neighbors (perfect hexagonal = 6)

**Topological Analysis**:
- `voronoi_analysis()` — `scipy.spatial.Voronoi` tessellation with boundary atom filtering
- `compute_psi6()` — Complex bond-orientational order parameter: `ψ6 = ⟨exp(6i·θ)⟩`
- `global_psi6()` — Local average `|ψ6|` vs. global `|⟨ψ6⟩|` (sensitive to grain boundaries)
- `grain_boundary_parameter()` — D-parameter: mean squared ψ6 difference between neighbors

**Defect Analysis**:
- `analyze_defects()` — 5/7-fold defect density and approximate grain size: `D ≈ √(A/N_defect)`
- `defect_statistics()` — Console output of coordination number distribution

---

#### `pfc_elastic.py` — Elastic Property Calculations

Mixin for mechanical property extraction:

**Strain Application**:
- `apply_strain(eps)` — Rescales domain `L = L0·(1+ε)` and rebuilds k-space
- `save_reference_state()` — Stores equilibrium `phi_ref`, `L_ref`, `dx_ref`

**Energy-Strain Analysis**:
- `elastic_energy_curve(strain_list, relax_steps=2000)` — For each strain: reset → apply → relax → measure energy
- `fit_elastic_constant(strain, energy)` — Quadratic fit `F(ε) = a·ε² + b·ε + c`; extracts `C = 2a = d²F/dε²` and residual strain `ε_r = -b/(2a)`
- `compute_stress(strain, energy)` — Numerical differentiation `σ = dF/dε` via `np.gradient`

---

#### `pfc_plot.py` — Visualization Toolkit

Extensive plotting mixin (20+ methods) for simulation results:

**Field Plots**:
- `plot_field()` — 2D density heatmap (`coolwarm`)
- `plot_structure_factor()` — Log-scaled diffraction pattern (`inferno`)

**Microstructure Plots**:
- `plot_voronoi()` — Voronoi cell topology
- `plot_defects()` — 5-fold (red) / 7-fold (blue) overlay on density field
- `plot_detected_atoms()` — Black scatter on density background
- `plot_grain_boundary_atoms()` — D-parameter thresholding (85th percentile)

**Order Parameter Plots**:
- `plot_psi6()` — Atom coloring by `|ψ6|` magnitude (`viridis`)
- `plot_grain_orientation()` — Atom coloring by `arg(ψ6)` phase (`hsv`)
- `plot_psi6_field()` — Interpolated continuous order field

**Evolution Plots**:
- `plot_energy()`, `plot_defect_density()`, `plot_grain_size()`, `plot_structure_peak()`

**Elastic Plots**:
- `plot_elastic_curve()` — Energy-strain with quadratic fit
- `plot_stress_strain()` — Stress-strain curve

**Pipelines**:
- `postprocess()` — Full 10-plot analysis suite
- `analyze_psi6()` — Complete ψ6 analysis (4 plots + metrics)

---

### IO Module

#### `pfc_io.py` — Video Recording

Mixin for simulation video generation:

**Features**:
- **In-memory caching**: Frames stored as PNG bytes in RAM (`frame_cache` list), zero disk I/O during simulation
- **ffmpeg synthesis**: H.264 MP4 with CRF=18 (high quality), automatic temp file cleanup
- **Configurable**: FPS, figure size, output filename

**Key Methods**:
- `initialize_io()` — Setup recording parameters
- `capture_frame()` — Convert current `phi` field to memory buffer (overridable by subclasses)
- `frames_to_video()` — Synthesize cached frames via ffmpeg subprocess

---

### Configuration

#### `config.py` — Interactive Parameter Menu

Console-based interactive interface for simulation setup:

**Features**:
- **Lattice selection**: Numbered menu (1=hexagon, 2=square, 3=triangle) with input validation
- **Validated input**: `get_float_input()` and `get_int_input()` with default values and type checking
- **Parameter packaging**: Returns structured dictionary `{"solver": {...}, "lattice_type": ...}`
- **Confirmation step**: User review before simulation start

**Parameters Configured**:
`N`, `L`, `r`, `M`, `dt`, `T`, `phi0`, `noise_amp`, `lattice_type`

---

### Run Scripts

#### `run_pure.py` — Pure Material Simulation Runner

Entry point for standard PFC simulations:
1. Calls `input_pfc_parameters()` for interactive config
2. Instantiates `PurePFCSolver` with user parameters
3. Runs main simulation loop + advanced analysis
4. Executes `postprocess()` (10-plot suite)
5. Runs `analyze_psi6()` (full orientational order analysis)
6. Vacancy detection and diffusion animation

#### `run_binary.py` — Binary Alloy Simulation Runner

Flexible entry point with 4 operational modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Quick** | `N=512`, pre-configured standard parameters | Fast preview |
| **Interactive** | Custom all parameters via CLI | Research tuning |
| **High-Resolution** | `N=1024`, extended `T=5000` | Fine microstructure |
| **Phase Separation** | Optimized for spinodal decomposition | Alloy dynamics |

**CLI Arguments**:
```bash
python run_binary.py --mode quick        # Quick mode
python run_binary.py --interactive       # Interactive mode
python run_binary.py --mode quick --video # With video recording
```

#### `run_elastic.py` — Elastic Constant Calculator

Systematic elastic modulus measurement:
1. Equilibrates `PurePFCSolver` (`T=1500`)
2. Saves reference state
3. Applies 13 strain values: `linspace(-0.03, 0.03, 13)`
4. Relaxes 2000 steps per strain point
5. Fits `F(ε) = a·ε² + b·ε + c` → extracts `C = 2a`
6. Plots energy-strain and stress-strain curves

#### `run_3d.py` — 3D Simulation Runner

Interactive mode selection for 3D simulations:

| Mode | Function | Reference |
|------|----------|-----------|
| `pure` | 3D BCC crystal growth | Chapter 8 |
| `elastic` | 3D elastic constant calculation | Section 8.5 |
| `alloy` | 3D binary alloy (simplified demo) | Chapter 9 |
| `sweep` | Parameter scan across `r` values | Stability analysis |

**Memory Notes**:
- `N=128`: ~200MB, fast per step
- `N=256`: ~1.5GB, ~8× slower (recommended)
- `N=512`: ~12GB, ~64× slower (workstation only)

---

## Examples

### Example 1: Grain Growth Simulation

```python
from pfc_pure import PurePFCSolver

# Initialize solver
solver = PurePFCSolver(
    N=256,              # Grid size
    L=128.0,            # Physical domain size
    r=-0.25,            # Temperature parameter
    dt=0.05,            # Time step
    T=2000.0,           # Total simulation time
    lattice_type="hexagon"
)

# Run simulation
solver.run(sample_interval=10)

# Post-processing
solver.postprocess()
solver.analyze_psi6()
```

### Example 2: Elastic Constant Calculation

```python
from pfc_pure import PurePFCSolver
import numpy as np

# Initialize and equilibrate
solver = PurePFCSolver(N=256, L=128, r=-0.35, T=1500)
solver.run()
solver.save_reference_state()

# Apply strain and compute energy
strain = np.linspace(-0.03, 0.03, 13)
energy, phi_list = solver.elastic_energy_curve(strain)

# Fit elastic constant
C, eps_r, coef, _, _ = solver.fit_elastic_constant(strain, energy)
print(f"Elastic constant C = {C:.6e}")
```

### Example 3: Binary Alloy Phase Separation

```python
from pfc_binary import BinaryPFCSolver

solver = BinaryPFCSolver(
    N=256,
    L=128.0,
    r=-0.25,
    c0=0.3,          # Initial concentration
    alpha=0.1,       # Coupling strength
    lattice_type="hexagon"
)

solver.run()
solver.postprocess()
```

---

## Output

### Generated Files

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
├── grain_boundary.png      # Grain boundary atoms
└── ...
```

### Key Observables

| Observable | Description |
|------------|-------------|
| **Free Energy** | System free energy evolution (should decrease monotonically) |
| **Density Field** | Spatial atomic density distribution `φ(r)` |
| **Structure Factor** | Reciprocal space diffraction pattern `S(k) = |φ̃(k)|²` |
| **Defect Density** | Fraction of 5/7-fold coordination atoms |
| **Grain Size** | Approximate average grain diameter from defect density |
| **ψ₆ Order** | Bond orientational order (`|ψ6| = 1` perfect, `0` disordered) |
| **Elastic Constant** | Young's modulus from energy-strain quadratic fit |

---

## Author & Citation

**Jinpeng Wang (王锦鹏)**

- Department of Material Engineering, The Hong Kong Polytechnic University
- Mitacs Intern @ McMaster University

If you use this framework in your research, please cite appropriately.

### References

1. Elder, K. R., & Grant, M. (2004). Modeling elastic and plastic deformations in nonequilibrium processing using phase field crystals. *Physical Review E*, 70(5), 051605.
2. Provatas, N., & Elder, K. (2010). *Phase-Field Methods in Materials Science and Engineering*. Wiley-VCH.
3. Berry, J., Elder, K. R., & Grant, M. (2008). Phase-field crystal modeling of eutectic solidification. *Physical Review Letters*, 100(4), 045705.

---

## Changelog

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

---

*Last updated: 2026-06*
