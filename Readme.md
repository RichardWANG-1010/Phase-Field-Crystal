# Phase-Field-Crystal / 相场晶体模拟框架

**English | [中文](#中文说明)**

A modular 2D/3D Phase Field Crystal (PFC) simulation framework written in Python, designed for materials science research on crystal growth, grain boundary evolution, and elastic properties.

一个用 Python 编写的模块化二维/三维相场晶体（PFC）模拟框架，用于材料科学领域的晶体生长、晶界演化和弹性性质研究。

---

## 📋 Table of Contents / 目录

- [Features / 功能特性](#features--功能特性)
- [Project Structure / 项目结构](#project-structure--项目结构)
- [Installation / 安装依赖](#installation--安装依赖)
- [Quick Start / 快速开始](#quick-start--快速开始)
- [Theoretical Background / 理论背景](#theoretical-background--理论背景)
- [Lattice Types / 晶格类型](#lattice-types--晶格类型)
- [Examples / 示例](#examples--示例)
- [Output / 输出说明](#output--输出说明)
- [Author / 作者](#author--作者)

---

## Features / 功能特性

### 🔬 Core Simulation / 核心模拟
- **Spectral semi-implicit PFC solver** - Efficient Fourier-space numerical method
  **谱方法半隐式PFC求解器** - 高效的傅里叶空间数值方法
- **Multiple lattice types** - Hexagonal, Square, Triangle (2D) & BCC, FCC, SC (3D)
  **多种晶格类型** - 六角、正方、三角（二维）及体心立方、面心立方、简单立方（三维）
- **Pure material & binary alloy** - Single-component and two-phase field coupling
  **纯材料与二元合金** - 单组分和双相场耦合模型
- **3D PFC simulation** - Full 3D BCC lattice support with volumetric visualization
  **三维PFC模拟** - 完整的三维BCC晶格支持及体可视化

### 📊 Analysis Tools / 分析工具
- **Defect density analysis** - 5-fold/7-fold coordination defect detection
  **缺陷密度分析** - 5重/7重配位缺陷检测
- **Voronoi tessellation** - Topological analysis of grain structure
  **Voronoi剖分** - 晶粒结构拓扑分析
- **ψ₆ orientational order parameter** - Bond-orientational order quantification
  **ψ₆取向序参量** - 键取向有序度量化
- **Grain boundary detection** - Identify grain boundary atoms via D-parameter
  **晶界检测** - 通过D参数识别晶界原子
- **Structure factor analysis** - Diffraction pattern in reciprocal space
  **结构因子分析** - 倒易空间衍射图案
- **Elastic energy calculation** - Free energy under applied strain
  **弹性能量计算** - 应变下的自由能计算
- **Stress-strain curve calculation** - Mechanical property extraction
  **应力-应变曲线计算** - 力学性质提取

### 🎨 Visualization / 可视化
- **Real-time field visualization** - Density and concentration field rendering
  **实时场可视化** - 密度场和浓度场渲染
- **Video recording** - Automatic MP4/GIF video generation of evolution
  **视频录制** - 自动生成演化过程的MP4/GIF视频
- **3D isosurface rendering** - Volumetric visualization with PyVista
  **三维等值面渲染** - 使用PyVista进行体可视化
- **Comprehensive post-processing** - Full suite of analysis plots
  **全面后处理** - 全套分析图表

---

## Project Structure / 项目结构

```
Phase-Field-Crystal/
├── Readme.md              # This file / 本说明文档
├── config.py              # User interface menu / 用户交互菜单
│
├── Core Modules / 核心模块
│   ├── pfc_base.py        # Core numerical infrastructure / 数值计算核心
│   ├── pfc_pure.py        # Pure material PFC solver / 纯材料PFC求解器
│   ├── pfc_binary.py      # Binary alloy PFC solver / 二元合金PFC求解器
│   ├── pfc_3d.py          # 3D PFC solver / 三维PFC求解器
│   └── sh_model.py        # Swift-Hohenberg model / Swift-Hohenberg模型
│
├── Analysis Modules / 分析模块
│   ├── pfc_analysis.py    # Defect and microstructure analysis / 缺陷与微观结构分析
│   ├── pfc_elastic.py     # Elasticity calculations / 弹性计算
│   └── pfc_plot.py        # Visualization tools / 可视化工具
│
├── IO Module / 输入输出模块
│   └── pfc_io.py          # Video recording and output / 视频录制与输出
│
└── Run Scripts / 运行脚本
    ├── run_pure.py        # Run standard PFC simulation / 运行标准PFC模拟
    ├── run_binary.py      # Run binary alloy simulation / 运行二元合金模拟
    ├── run_elastic.py     # Run elastic constant calculation / 运行弹性常数计算
    └── run_3d.py          # Run 3D PFC simulation / 运行三维PFC模拟
```

---

## Installation / 安装依赖

### Requirements / 依赖项

```bash
# Core numerical libraries / 核心数值库
numpy >= 1.20.0
scipy >= 1.7.0

# Visualization / 可视化
matplotlib >= 3.4.0

# Optional - 3D visualization / 可选 - 三维可视化
pyvista >= 0.32.0
scikit-image >= 0.18.0

# Optional - Video generation / 可选 - 视频生成
ffmpeg (system package)
```

### Install / 安装

```bash
# Clone the repository / 克隆仓库
git clone https://github.com/RichardWANG-1010/Phase-Field-Crystal.git
cd Phase-Field-Crystal

# Install Python dependencies / 安装Python依赖
pip install numpy scipy matplotlib

# Optional: for 3D visualization / 可选：用于三维可视化
pip install pyvista scikit-image

# Optional: for video generation / 可选：用于视频生成
# Install ffmpeg from your system package manager
# 从系统包管理器安装ffmpeg
```

---

## Quick Start / 快速开始

### 1. Pure Material PFC Simulation / 纯材料PFC模拟

```bash
python run_pure.py
```

This will launch an interactive menu to configure simulation parameters, then run the simulation and generate analysis plots.

这将启动交互式菜单配置模拟参数，然后运行模拟并生成分析图表。

### 2. Elastic Constant Calculation / 弹性常数计算

```bash
python run_elastic.py
```

Calculates elastic modulus by applying uniaxial strain and fitting energy-strain curve.

通过施加单轴应变并拟合能量-应变曲线来计算弹性模量。

### 3. Binary Alloy Simulation / 二元合金模拟

```bash
python run_binary.py
```

Simulates spinodal decomposition and phase separation in binary alloys.

模拟二元合金中的旋节线分解和相分离过程。

### 4. 3D PFC Simulation / 三维PFC模拟

```bash
python run_3d.py
```

Runs full 3D PFC simulation with BCC lattice support.

运行支持BCC晶格的完整三维PFC模拟。

---

## Theoretical Background / 理论背景

### PFC Model / PFC模型

The Phase Field Crystal (PFC) model describes crystalline structures at atomic length scales while operating on diffusive time scales.

相场晶体（PFC）模型在原子长度尺度上描述晶体结构，同时在扩散时间尺度上运行。

#### Free Energy Functional / 自由能泛函

```
F = ∫ [φ/2 · (r + (1 + ∇²)²) φ + φ⁴/4] dr
```

Where:
- `φ` - Dimensionless density field / 无量纲密度场
- `r` - Reduced temperature parameter / 约化温度参数
- `∇²` - Laplacian operator / 拉普拉斯算子

#### Dynamic Equation / 动力学方程

```
∂φ/∂t = ∇² · δF/δφ
```

Conserved dynamics (mass conservation) solved via semi-implicit Fourier spectral method.

守恒动力学（质量守恒）通过半隐式傅里叶谱方法求解。

### Binary Alloy Model / 二元合金模型

Extended PFC model with two conserved fields: total density `φ` and concentration `c`.

扩展的PFC模型，包含两个守恒场：总密度 `φ` 和浓度 `c`。

```
F = F_PFC[φ] + F_CH[c] + F_coupling[φ, c]
```

---

## Lattice Types / 晶格类型

### 2D Lattices / 二维晶格

| Lattice / 晶格 | Dispersion Operator / 色散算子 | Structure / 结构 |
|---------------|-------------------------------|-----------------|
| **Hexagon / 六角** | `L(k) = (1 - k²)² + r` | Triangular lattice / 三角晶格 |
| **Square / 正方** | `L(k) = (1-kx²)²(1-ky²)² + r` | Square lattice / 正方晶格 |
| **Triangle / 三角** | `L(k) = (1 - kx² - kx·ky + ky²)² + r` | Oblique lattice / 斜晶格 |

### 3D Lattices / 三维晶格

| Lattice / 晶格 | Description / 说明 |
|---------------|-------------------|
| **BCC** | Body-Centered Cubic / 体心立方 |
| **FCC** | Face-Centered Cubic / 面心立方 |
| **SC** | Simple Cubic / 简单立方 |

---

## Examples / 示例

### Example 1: Grain Growth Simulation / 示例1：晶粒生长模拟

```python
from pfc_pure import PurePFCSolver

# Initialize solver / 初始化求解器
solver = PurePFCSolver(
    N=256,           # Grid size / 网格尺寸
    L=128.0,         # Physical domain size / 物理域尺寸
    r=-0.25,         # Temperature parameter / 温度参数
    dt=0.05,         # Time step / 时间步长
    T=2000.0,        # Total simulation time / 总模拟时间
    lattice_type="hexagon"  # Lattice type / 晶格类型
)

# Run simulation / 运行模拟
solver.run(sample_interval=10)

# Post-processing / 后处理
solver.postprocess()
solver.analyze_psi6()
```

### Example 2: Elastic Constant Calculation / 示例2：弹性常数计算

```python
from pfc_pure import PurePFCSolver
import numpy as np

# Initialize and equilibrate / 初始化并弛豫
solver = PurePFCSolver(N=256, L=128, r=-0.35, T=1500)
solver.run()
solver.save_reference_state()

# Apply strain and compute energy / 施加应变并计算能量
strain = np.linspace(-0.03, 0.03, 13)
energy, phi_list = solver.elastic_energy_curve(strain)

# Fit elastic constant / 拟合弹性常数
C, eps_r, coef, _, _ = solver.fit_elastic_constant(strain, energy)
print(f"Elastic constant C = {C:.6e}")
```

### Example 3: Binary Alloy Phase Separation / 示例3：二元合金相分离

```python
from pfc_binary import BinaryPFCSolver

solver = BinaryPFCSolver(
    N=256,
    L=128.0,
    r=-0.25,
    c0=0.3,          # Initial concentration / 初始浓度
    alpha=0.1,       # Coupling strength / 耦合强度
    lattice_type="hexagon"
)

solver.run()
solver.postprocess()
```

---

## Output / 输出说明

### Generated Files / 生成文件

```
result/
├── pfc_simulation.mp4      # Evolution video / 演化视频
├── energy_evolution.png    # Energy curve / 能量曲线
├── density_field.png       # Final density field / 最终密度场
├── structure_factor.png    # Structure factor / 结构因子
├── voronoi_analysis.png    # Voronoi diagram / Voronoi图
├── defect_analysis.png     # Defect visualization / 缺陷可视化
├── defect_density.png      # Defect density curve / 缺陷密度曲线
├── grain_size.png          # Grain size curve / 晶粒尺寸曲线
└── ...
```

### Key Observables / 关键观测量

- **Free Energy / 自由能** - System free energy evolution
- **Density Field / 密度场** - Spatial distribution of atomic density
- **Structure Factor / 结构因子** - Reciprocal space diffraction pattern
- **Defect Density / 缺陷密度** - Concentration of 5/7-fold defects
- **Grain Size / 晶粒尺寸** - Average grain diameter estimation
- **ψ₆ Order Parameter / ψ₆序参量** - Bond orientational order measure
- **Elastic Constant / 弹性常数** - Young's modulus from strain-energy fit

---

## Author / 作者

**Jinpeng Wang (王金鹏)**

- Department of Material Engineering / 材料工程系
- The Hong Kong Polytechnic University / 香港理工大学
- Mitacs Intern @ McMaster University / Mitacs实习生 @ 麦克马斯特大学

---

## 📚 References / 参考文献

1. Elder, K. R., & Grant, M. (2004). Modeling elastic and plastic deformations in nonequilibrium processing using phase field crystals. *Physical Review E*, 70(5), 051605.

2. Provatas, N., & Elder, K. (2010). *Phase-Field Methods in Materials Science and Engineering*. Wiley-VCH.

3. Berry, J., Elder, K. R., & Grant, M. (2008). Phase-field crystal modeling of eutectic solidification. *Physical Review Letters*, 100(4), 045705.

---

## 📝 Changelog / 更新日志

### v1.0.0
- Initial release with 2D PFC framework
- 初始版本，包含二维PFC框架
- Hexagonal, square, triangle lattice support
- 支持六角、正方、三角晶格
- Pure material and binary alloy models
- 纯材料和二元合金模型
- Full analysis and visualization toolkit
- 完整的分析和可视化工具包

### v1.1.0
- Added 3D PFC simulation (BCC lattice)
- 新增三维PFC模拟（BCC晶格）
- Added Swift-Hohenberg model
- 新增Swift-Hohenberg模型
- Improved video generation with GIF fallback
- 改进视频生成，增加GIF降级方案

---

## 📄 License / 许可证

This project is for academic research purposes. Please cite appropriately if used in publications.

本项目用于学术研究目的。如在出版物中使用，请适当引用。

---

## 中文说明

### 项目简介

**相场晶体模拟框架**是一个基于 Python 的模块化计算材料科学工具包，用于模拟晶体生长、晶粒演化、缺陷运动和弹性性质等材料科学问题。

相场晶体（PFC）模型是一种介观尺度的模拟方法，能够在原子级分辨率下捕捉晶体结构，同时在扩散时间尺度上运行，非常适合研究晶粒生长、晶界迁移、位错运动等现象。

### 主要功能

- ✅ **多种晶格类型**：支持六角、正方、三角（二维）和BCC、FCC、SC（三维）晶格
- ✅ **纯材料与二元合金**：单组分和双组分相场耦合模型
- ✅ **谱方法求解**：高效的半隐式傅里叶空间数值方法
- ✅ **缺陷分析**：5重/7重配位缺陷检测、晶界识别
- ✅ **结构表征**：Voronoi剖分、ψ₆取向序参量、结构因子
- ✅ **弹性计算**：应变-能量曲线、应力-应变关系、弹性常数拟合
- ✅ **三维模拟**：完整的三维PFC模拟及体可视化
- ✅ **可视化输出**：实时场显示、视频录制、全套后处理图表

### 快速开始

```bash
# 运行标准纯材料PFC模拟
python run_pure.py

# 运行弹性常数计算
python run_elastic.py

# 运行二元合金模拟
python run_binary.py

# 运行三维PFC模拟
python run_3d.py
```

### 技术特点

- **模块化设计**：基础类、求解器、分析、可视化各模块独立，易于扩展
- **面向对象**：采用多继承架构，功能组合灵活
- **中英文注释**：所有代码均配有中英文双语注释，便于学习和使用
- **跨平台**：纯Python实现，支持Windows/Linux/macOS

---

*Last updated / 最后更新: 2026-06*
