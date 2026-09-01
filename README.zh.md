<div align="center">

# Phase-Field-Crystal

**模块化二维/三维相场晶体模拟框架**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.20.0-orange)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%E2%89%A51.7.0-green)](https://scipy.org/)
[![Taichi](https://img.shields.io/badge/Taichi-%E2%89%A51.0-purple)](https://taichi-lang.org/)
[![MPI](https://img.shields.io/badge/MPI-FFTW3-red)](https://www.open-mpi.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

[English](README.md) | [中文](README.zh.md)

</div>

---

## 目录

- [概述](#概述)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [理论背景](#理论背景)
- [晶格类型](#晶格类型)
- [代码文档](#代码文档)
- [使用示例](#使用示例)
- [输出说明](#输出说明)
- [作者与引用](#作者与引用)
- [更新日志](#更新日志)
- [许可证](#许可证)

---

## 概述

**Phase-Field-Crystal (PFC)** 模拟框架是一款面向计算材料科学研究的模块化、面向对象Python工具包，能够在原子长度尺度上模拟晶体生长、晶界演化、缺陷动力学和弹性性质，同时在扩散时间尺度上运行。

本框架实现了标准PFC模型、多种公式的**扩展相场晶体（XPFC）模型**，以及其他扩展模型（二元合金、三维BCC/FCC/SC晶格、Swift-Hohenberg模型），采用高效的**半隐式傅里叶谱方法**求解。XPFC实现包括：双振幅（双模）微分算子公式、基于相关函数（C₂）的Taichi加速公式，以及C++/MPI振幅表述求解器——能够对固液共存、界面能和位错能量进行定量研究。模块化架构将核心求解器、分析工具、可视化和输入输出模块分离为独立的混入类（mixin class），支持灵活组合与便捷扩展。

### 核心能力

| 类别 | 功能 |
|------|------|
| **核心模拟** | 谱方法半隐式PFC求解器；多种2D/3D晶格类型；纯物质与二元合金模型 |
| **XPFC扩展** | 双振幅（双模）PFC引擎；相关函数C₂ XPFC求解器（Taichi加速）；C++/MPI振幅表述求解器；平直与圆形界面能计算；位错能计算 |
| **分析工具** | 缺陷检测（5/7配位）、Voronoi剖分、ψ₆取向序、晶界识别、结构因子分析 |
| **弹性计算** | 能量-应变曲线、应力-应变关系、弹性常数拟合 |
| **可视化** | 实时场渲染、MP4/GIF视频生成、三维等值面渲染（PyVista）、带交互控件的Tkinter GUI |

---

## 功能特性

### 核心模拟
- **谱方法半隐式PFC求解器** — 高效的傅里叶空间数值积分，线性项无条件稳定
- **多种晶格类型** — 六角、正方、三角（2D）；体心立方、面心立方、简单立方（3D）
- **纯物质与二元合金** — 单组元和双相场耦合，结合Cahn-Hilliard浓度动力学
- **三维PFC模拟** — 完整的三维BCC晶格支持，包含体可视化和弹性常数计算
- **Swift-Hohenberg模型** — 经典的斑图形成偏微分方程，用于条纹/六角斑图研究

### XPFC（扩展相场晶体）扩展
- **双振幅（双模）PFC引擎** — 乘积形式线性算子 `L = [r + (1+∇²)²][(q²+∇²)²/q⁴]`，含三次非线性项；通过 `sigma` 参数控制正方相与三角相的选择
- **相关函数XPFC求解器** — 由倒格点上的多高斯峰叠加构建直接相关函数 `C₂(k)`；线性算子 `L(k) = 1 − C₂(k)`；Taichi CPU加速，支持单晶和多晶初始条件
- **C++/MPI振幅表述求解器** — 高性能XPFC正方晶格求解器，采用振幅展开（密度场+两个振幅场），MPI并行化（FFTW3），支持位错初始化
- **界面能计算** — 平直固液界面能 `γ_SL` 和圆形晶核界面能（含有效半径 `R_eff`）
- **位错能计算** — 通过半晶体位移创建刃位错，计算位错能 `E_dis`，覆盖正方和三角相的sigma系列
- **Tkinter图形界面** — 交互式主窗口，支持模型选择、参数配置、sigma系列批量运行、实时可视化和数据导出

### 分析工具
- **缺陷密度分析** — 通过局部峰值分析检测5配位/7配位缺陷
- **Voronoi剖分** — 周期性边界条件下的拓扑晶粒结构分析
- **ψ₆取向序参数** — 六角晶格的键取向序定量分析
- **晶界检测** — 基于D参数的晶界原子识别
- **结构因子分析** — 倒易空间衍射图案（2D和3D）
- **弹性能量计算** — 施加应变下的自由能计算，支持参考态管理
- **应力-应变曲线** — 通过数值微分提取力学性质

### 可视化
- **实时场可视化** — 使用Matplotlib渲染密度场和浓度场
- **视频录制** — 自动生成演化过程的MP4（ffmpeg）或GIF（Pillow）视频
- **三维等值面渲染** — 使用PyVista进行体可视化（Matplotlib作为备选方案）
- **交互式XPFC图形界面** — 基于Tkinter的界面，含滑块控件、播放/暂停、单步导航和综合结果图
- **全面后处理** — 完整的分析图集（能量、缺陷、晶粒尺寸等）

---

## 项目结构

> **分支说明：** 本仓库按功能分支组织。标准PFC模块位于 `core/`、`analysis/`、`config/`、`io/` 和 `runner/` 分支；DualXPFC扩展位于 `DualXPFC` 分支。`main` 分支作为项目首页。

```
Phase-Field-Crystal/
│
├── README.md              # 英文文档
├── README.zh.md           # 中文文档（本文件）
│
├── Core Modules (分支: core)
│   ├── pfc_base.py        # 基类：网格设置、k空间构建、模拟日志
│   ├── pfc_pure.py        # 纯物质PFC求解器（六角/正方/三角晶格）
│   ├── pfc_binary.py      # 二元合金PFC求解器（双场：密度+浓度）
│   ├── pfc_3d.py          # 三维PFC求解器（BCC/FCC/SC晶格，PyVista可视化）
│   └── sh_model.py        # Swift-Hohenberg斑图形成求解器
│
├── Analysis Modules (分支: analysis)
│   ├── pfc_analysis.py    # 微观结构分析：原子检测、配位数、ψ6、Voronoi、缺陷
│   ├── pfc_elastic.py     # 弹性性质计算：应变施加、能量曲线、拟合
│   └── pfc_plot.py        # 可视化工具包：场、结构因子、缺陷、晶界
│
├── IO Module (分支: io)
│   └── pfc_io.py          # 视频录制：内存帧缓存、ffmpeg MP4合成
│
├── Configuration (分支: config)
│   └── config.py          # 交互式控制台参数输入菜单，带输入验证
│
├── Run Scripts (分支: runner)
│   ├── run_pure.py        # 标准纯物质PFC模拟，交互式配置
│   ├── run_binary.py      # 二元合金模拟，4种预设模式
│   ├── run_elastic.py     # 弹性常数计算，通过能量-应变二次拟合
│   └── run_3d.py          # 三维PFC模拟，4种模式（纯物质/弹性/合金/参数扫描）
│
└── DualXPFC Extension (分支: DualXPFC)
    ├── __init__.py
    ├── pfc_core.py        # 双振幅（双模）PFC核心引擎（DualPFCConfig, DualPFCEngine）
    ├── xpfc_square.py     # 相关函数C₂ XPFC正方求解器（Taichi加速）
    ├── main.py            # Tkinter GUI主界面（3个模型：平直/圆形界面、位错）
    ├── part1_flat_interface.py   # 模型1：平直固液界面能 γ_SL
    ├── part2_round_interface.py  # 模型2：圆形晶核界面能 γ_SL, R_eff
    ├── part3_dislocation.py      # 模型3：刃位错能 E_dis
    ├── visualization.py   # 交互式可视化模块（滑块、播放/暂停、结果图）
    ├── quick_test.py      # 双振幅引擎快速验证测试
    ├── xpfcSqAmpDis.cpp   # C++/MPI/FFTW3振幅表述XPFC求解器（含位错）
    └── xpfcSqAmpDis.in    # C++求解器输入参数文件
```

---

## 安装说明

### 依赖要求

| 软件包 | 版本要求 | 用途 |
|--------|----------|------|
| `numpy` | ≥1.20.0 | 核心数值数组与FFT计算 |
| `scipy` | ≥1.7.0 | FFT运算、空间算法 |
| `matplotlib` | ≥3.4.0 | 二维绘图与可视化 |
| `taichi` | ≥1.0.0 | XPFC正方求解器CPU加速（DualXPFC） |
| `pyvista` | ≥0.32.0 | 三维等值面渲染（可选） |
| `scikit-image` | ≥0.18.0 | 三维峰值检测（可选） |
| `ffmpeg` | 系统级 | 视频生成（可选，系统软件包） |
| `MPI` + `FFTW3` | 系统级 | C++振幅表述求解器（DualXPFC，可选） |

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/RichardWANG-1010/Phase-Field-Crystal.git
cd Phase-Field-Crystal

# 安装核心依赖
pip install numpy scipy matplotlib

# 安装XPFC依赖（DualXPFC分支）
pip install taichi

# 可选：三维可视化支持
pip install pyvista scikit-image

# 可选：视频生成（通过系统包管理器安装）
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: 从 https://ffmpeg.org/download.html 下载

# 可选：C++/MPI求解器依赖（DualXPFC）
# Ubuntu/Debian: sudo apt-get install libopenmpi-dev libfftw3-dev
# 编译: mpicxx -O3 -o xpfcSqAmpDis xpfcSqAmpDis.cpp -lfftw3_mpi -lfftw3 -lm
```

---

## 快速开始

### 1. 纯物质PFC模拟

```bash
python run_pure.py
```

启动交互式菜单配置模拟参数，运行模拟并生成分析图表。

### 2. 弹性常数计算

```bash
python run_elastic.py
```

通过施加单轴应变并对能量-应变曲线进行二次多项式拟合，计算弹性模量。

### 3. 二元合金模拟

```bash
python run_binary.py
```

模拟二元合金的旋节分解和相分离过程，支持4种模式：快速模式、交互模式、高分辨率模式和相分离模式。

### 4. 三维PFC模拟

```bash
python run_3d.py
```

运行完整的三维BCC晶格PFC模拟，可选模式：纯物质、弹性常数计算、二元合金、参数扫描。

### 5. Swift-Hohenberg模型

```bash
python sh_model.py
```

直接运行Swift-Hohenberg斑图形成求解器。

### 6. DualXPFC — 图形界面

```bash
# 先切换到DualXPFC分支
git checkout DualXPFC

# 启动Tkinter GUI（3个模型：平直界面、圆形界面、位错）
python main.py
```

打开交互式图形界面，可选择三个研究模型、配置参数、运行sigma系列批量模拟并可视化结果。

### 7. DualXPFC — 相关函数正方求解器

```bash
python xpfc_square.py
```

启动交互式控制台菜单，选择单晶或多晶模式，运行Taichi加速的XPFC正方晶格模拟，支持实时视频录制。

### 8. DualXPFC — 快速测试

```bash
python quick_test.py
```

运行快速验证套件（64×64网格，100步），确认双振幅引擎、三个界面/位错模型以及能量计算均正常工作。

### 9. DualXPFC — C++/MPI振幅求解器

```bash
# 编译
mpicxx -O3 -o xpfcSqAmpDis xpfcSqAmpDis.cpp -lfftw3_mpi -lfftw3 -lm

# 运行（在xpfcSqAmpDis.in中配置参数）
mpirun -np 4 ./xpfcSqAmpDis < xpfcSqAmpDis.in
```

高性能振幅表述XPFC求解器，MPI并行化，支持1024×1024网格和位错初始化。

---

## 理论背景

### PFC模型

相场晶体模型在原子长度尺度上描述晶体结构，同时在扩散时间尺度上运行，非常适合研究晶粒生长、晶界迁移和位错动力学。

#### 自由能泛函

```
F = ∫ [φ/2 · (r + (1 + ∇²)²) φ + φ⁴/4] dr
```

其中：
- `φ` — 无量纲密度场
- `r` — 约化温度参数（控制过冷度）
- `∇²` — 拉普拉斯算子

#### 动力学方程（守恒动力学）

```
∂φ/∂t = ∇² · δF/δφ
```

采用半隐式傅里叶谱方法求解，保证数值稳定性与效率。

### 双振幅（双模）PFC模型

双振幅PFC公式通过引入耦合两个特征波数的**乘积形式线性算子**，结合**三次非线性项**，推广了标准PFC模型。这使得正方相与三角相能够通过单一 `sigma` 参数控制共存，并提供更准确的界面和缺陷能量。

#### 自由能泛函

```
F = ∫ [ 1/2 ψ · L · ψ  −  τ/3 · ψ³  +  1/4 · ψ⁴ ] dV
```

#### 线性算子（双模，乘积形式）

```
L = [ r + (1 + ∇²)² ] · [ (q² + ∇²)² / q⁴ ]
```

在傅里叶空间中：

```
L(k) = [ r + (1 − k²)² ] · [ (q² − k²)² / q⁴ ]
```

其中：
- `ψ` — 无量纲密度场
- `r` — 约化温度，由 `sigma` 映射得到：`r = −0.4 + 0.4·sigma`
- `τ` — 三次项系数（默认 `τ = 1.0`）
- `q` — 波数比（默认 `q = √3`，对应三角晶格的第二倒易壳层）
- `∇²` — 拉普拉斯算子

#### 动力学方程（守恒动力学）

```
∂ψ/∂t = ∇² · δF/δψ = ∇² · [ Lψ − τψ² + ψ³ ]
```

#### 半隐式时间步进

```
ψ_k(t+dt) = [ ψ_k(t) − dt·k²·(−τψ² + ψ³)_k ] / [ 1 + dt·k²·L(k) ]
```

#### Sigma控制的相选择

| Sigma范围 | 晶体相 |
|-----------|--------|
| `sigma < 0`（sigma₁, sigma₂） | 正方晶格 |
| `sigma ≥ 0`（sigma₃, sigma₄, sigma₅） | 三角（六角）晶格 |

### 相关函数（C₂）XPFC模型

相关函数XPFC公式由构建在倒格点上的**直接相关函数 `C₂(k)`**（多高斯峰叠加）构造线性算子。这种方法可以直接控制峰的位置、宽度和权重——从而调控弹性、各向异性和界面宽度。

#### 自由能泛函

```
F = ∫ [ 1/2 n · (1 − C₂) · n  −  η/6 · n³  +  χ/12 · n⁴ ] dV
```

#### 直接相关函数

```
C₂(k) = σ · Σ_i  w_i · exp( −|k − G_i|² / (2α_i²) )
```

其中 `G_i` 是正方晶格的倒易矢量：
- 第一壳层：`(±q, 0)`, `(0, ±q)`，`q = 1.0`
- 第二壳层：`(±q, ±q)`，`q = √2`

以及：
- `n` — 无量纲密度场
- `σ` — 整体相关强度（熔化温度控制）
- `w_i`, `α_i` — 每个高斯峰的权重和宽度
- `η` — 三次项系数（`−n³/6` 项）
- `χ` — 四次项系数（`n⁴/12` 项）

#### 线性算子

```
L(k) = 1 − C₂(k)
```

#### 动力学方程与半隐式步进

```
∂n/∂t = ∇² · [ (1−C₂)n − η/2·n² + χ/3·n³ ]

n_k(t+dt) = [ n_k(t) − dt·k²·(−η/2·n² + χ/3·n³)_k ] / [ 1 + dt·k²·(1−C₂(k)) ]
```

使用 **Taichi** CPU核函数实现非线性项和k空间更新，在8线程上相比纯NumPy实现约8倍加速。

### C++振幅表述XPFC模型

C++求解器使用密度场的**振幅展开**——将其分解为平均密度 `n₀` 加上调制各倒易晶格模式的复振幅场。这种表述对于大系统（1024×1024及以上）数值高效，并且通过振幅噪声自然地融入位错形核。

#### 场分解

```
n(r) = n₀ + Re[ A(r)·e^{iG₁·r} + B(r)·e^{iG₂·r} + ... ]
```

其中 `A`, `B` 是缓变复振幅场，具有独立的迁移率 `M_A`, `M_B`。

#### 实现
- **并行化**：MPI域分解，使用FFTW3-MPI转置FFT
- **网格**：最高1024×1024，`dx = 0.25`，`dt = 1.0`
- **相关函数**：k零模 + 第一峰（w₁）+ 第二峰（w₂ = √2）
- **位错**：通过输入文件中的 `dislNoiseAmp` 参数控制
- **参数**：所有运行时参数从 `xpfcSqAmpDis.in` 读取

### 二元合金模型

扩展的PFC模型，包含两个守恒场：总密度`φ`和浓度`c`。

```
F = F_PFC[φ] + F_CH[c] + F_coupling[φ, c]
```

其中`F_CH`为Cahn-Hilliard自由能，`F_coupling`描述密度场与浓度场之间的相互作用。

---

## 晶格类型

### 二维晶格

| 晶格 | 色散算子 `L(k)` | 晶体结构 |
|------|----------------|----------|
| **六角** | `L(k) = (1 - k²)² + r` | 三角晶格 |
| **正方** | `L(k) = (1-kx²)²(1-ky²)² + r` | 正方晶格 |
| **三角** | `L(k) = (1 - kx² - kx·ky + ky²)² + r` | 斜方晶格 |

### 三维晶格

| 晶格 | 全称 | 说明 |
|------|------|------|
| **BCC** | 体心立方 | 标准PFC模型（第8章，Provatas & Elder） |
| **FCC** | 面心立方 | 需要额外的稳定化项 |
| **SC** | 简单立方 | 基本立方对称性 |

### XPFC支持的晶格

| 公式 | 支持晶格 | 控制参数 |
|------|----------|----------|
| 双振幅PFC（`pfc_core.py`） | 正方、三角 | `sigma`（低→正方，高→三角） |
| 相关函数XPFC（`xpfc_square.py`） | 正方 | `sigma`，`C₂(k)`中的峰权重/宽度 |
| C++振幅求解器（`xpfcSqAmpDis.cpp`） | 正方 | `xpfcSqAmpDis.in`输入文件 |

---

## 代码文档

### 核心模块

#### `pfc_base.py` — 数值基础类

所有PFC求解器的基础，提供以下功能：
- **网格设置**：`N×N`（二维）或`N³`（三维）空间离散，物理域尺寸为`L`
- **k空间构建**：使用`numpy.fft.fftfreq`预先计算傅里叶波矢`kx, ky, kz`和模方`k2 = kx² + ky² + kz²`
- **模拟日志**：存储能量、质量、缺陷密度、晶粒尺寸和结构因子峰值的数组
- **分析缓存**：存储检测到的原子位置、邻居列表和ψ₆值
- **视频配置**：帧目录设置和录制标志

**核心方法**：`_build_kspace()` — 构建具有正确FFT频率顺序和矩阵式索引的倒易空间网格。

---

#### `pfc_pure.py` — 纯物质PFC求解器

通过**多继承**（mixin模式）实现标准单组元PFC模型：

```python
class PurePFCSolver(PFCBase, PFCAnalysis, PFCPlot, PFCIO, PFCElastic, PFCAdvancedAnalysis):
```

**功能特性**：
- **半隐式谱时间步进**：线性项隐式求解，非线性`φ³`项显式求解
- **晶格特异性色散算子**：实时计算六角/正方/三角晶格的`L(k)`
- **质量守恒强制**：每步后修正平均密度
- **能量计算**：利用Parseval定理实现高效k空间积分
- **结构因子**：`S(k) = |φ̃(k)|²`，通过`fftshift`获得中心衍射图案

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N` | 128 | 网格分辨率 |
| `L` | 64.0 | 物理域尺寸 |
| `r` | -0.25 | 温度参数（负值利于晶相） |
| `M` | 1.0 | 迁移率系数 |
| `phi0` | -0.25 | 平均密度 |
| `lattice_type` | `"hexagon"` | 晶体对称性 |

---

#### `pfc_binary.py` — 二元合金PFC求解器

耦合密度场`φ`和浓度场`c`的双守恒场求解器：

**自由能**：
```
F = F_PFC[φ] + F_CH[c] + F_coupling[φ, c]
F_PFC = 0.5·φ·L(k)·φ + 0.25·φ⁴
F_CH = 0.5·c·(r_c - ∇²)·c + 0.25·u_c·c⁴
F_coupling = α·c·φ² + β·c·φ
```

**演化方程**（守恒Cahn-Hilliard动力学）：
```
∂φ/∂t = M_φ·∇²·(δF/δφ)
∂c/∂t = M_c·∇²·(δF/δc)
```

**功能特性**：
- 傅里叶空间中双场同时半隐式更新
- 浓度物理约束：通过截断保证`0 ≤ c ≤ 1`
- 双场视频录制（密度场与浓度场并排显示）
- 扩展后处理：浓度演化、场叠加、耦合能
- 浓度结构因子分析

---

#### `pfc_3d.py` — 三维PFC求解器

完整的三维扩展，支持BCC、FCC和SC晶格，分为4个mixin类：

| 类 | 用途 |
|----|------|
| `PFCBase3D` | 三维网格、`N³` k空间（`kx, ky, kz`）、体积计算 |
| `PFCAnalysis3D` | 三维结构因子、基于`peak_local_max`的原子检测、二维切片 |
| `PFCPlot3D` | 三维等值面（PyVista + Matplotlib备选）、正交切片面板 |
| `PFCElastic3D` | 三维应变张量、考虑泊松效应的单轴应变、体积模量 |

---

#### `sh_model.py` — Swift-Hohenberg求解器

独立的斑图形成求解器，用于经典的Swift-Hohenberg方程：
```
∂ψ/∂t = ε·ψ - (q₀² + ∇²)²·ψ - ψ³
```

内置视频录制、预计算线性算子、自动ffmpeg/GIF回退。

---

### DualXPFC扩展模块

#### `pfc_core.py` — 双振幅（双模）PFC核心引擎

所有DualXPFC研究模型的基础引擎。实现双模乘积形式线性算子与三次非线性项。

**类**：

| 类 | 用途 |
|----|------|
| `DualPFCConfig` | 配置容器：网格（`Nx, Ny, Lx, Ly`）、模型参数（`sigma, tau, q, k0`）、时间步进（`dt, n_steps`）、初始条件（`amplitude, noise, mean_density`） |
| `DualPFCEngine` | 模拟引擎：k空间算子构建、时间演化、能量计算、结构因子 |

**核心特性**：
- **Sigma到r的映射**：`r = −0.4 + 0.4·sigma`（线性，可根据相图调整）
- **双模线性算子**：`L(k) = [r + (1−k²)²] · [(q²−k²)²/q⁴]`，在傅里叶空间预计算
- **晶体初始条件**：`square_crystal()`（cos(k₀x)+cos(k₀y)）和 `triangular_crystal()`（一模+二模余弦叠加）
- **半隐式步进**：`denom = 1 + dt·k²·L(k)`，非线性项`−τψ² + ψ³`在实空间计算
- **能量方法**：`total_energy()`、`energy_density()`、`chemical_potential()`、`bulk_energy()`（正方/三角/液相）
- **结构因子**：`S(k) = |ψ̃(k)|²`，带`fftshift`

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `Nx, Ny` | 256, 256 | 网格分辨率 |
| `Lx, Ly` | `Nx·π/4` | 物理域尺寸（每个2π周期约8个网格点） |
| `sigma` | 0.0 | 控制参数（映射到`r`；低→正方，高→三角） |
| `tau` | 1.0 | 三次项系数 |
| `q` | √3 | 波数比（三角晶格第二壳层） |
| `dt` | 0.5 | 时间步长 |
| `n_steps` | 2000 | 迭代步数 |
| `amplitude` | 0.3 | 晶体初始振幅 |

---

#### `xpfc_square.py` — 相关函数C₂ XPFC正方求解器

Taichi加速的独立XPFC正方晶格求解器，使用由多高斯峰构建的直接相关函数`C₂(k)`。

**类**：`XPFC_Square_CPU`

**核心特性**：
- **相关函数构建**：`C₂(k)`由8个倒格点（第一壳层`q=1.0`，第二壳层`q=√2`）构成，每个点为一个高斯峰，宽度`α`和权重`w`可配置
- **线性算子**：`L(k) = 1 − C₂(k)`，其中`C₂(0,0) = 0.3`固定
- **Taichi核函数**：`compute_nl_kernel()`（非线性`−η/2·n² + χ/3·n³`）、`compute_step_kernel()`（k空间半隐式更新）、`enforce_density_kernel()`、`clip_field_kernel()`
- **初始化模式**：`initialize_single_crystal()`（完美周期晶格）和`initialize_polycrystal()`（随机噪声+5个随机相位种子促进多畴形核）
- **内置视频编码器**：`VideoEncoder`类，使用ffmpeg管道H.264编码，CRF/预设/分辨率可配置
- **最终分析图**：密度场+对数结构因子+C₂(k)截面图

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N` | 256 | 网格分辨率 |
| `L` | 64.0 | 物理域尺寸 |
| `sigma` | 1.5 | 相关强度（熔化温度） |
| `eta` | 1.0 | 三次项系数（`−n³/6`项） |
| `chi` | 1.0 | 四次项系数（`n⁴/12`项） |
| `dt` | 0.01 | 时间步长 |
| `n0` | -0.25 | 平均密度 |
| `steps` | 50000 | 总迭代步数 |

---

#### `main.py` — Tkinter GUI主界面

整合所有三个DualXPFC研究模型的交互式图形前端。

**类**：`PFCMainApp`

**功能特性**：
- **模型选择**：单选按钮选择模型1（平直界面）、模型2（圆形界面）、模型3（位错）
- **参数面板**：网格布局输入框，配置`Nx, Ny, dt, n_steps, tau, q, amplitude, mean_density`，以及逗号分隔的`sigma`值
- **运行模式**：单sigma或完整sigma系列批量；可选实时可视化
- **后台线程**：模拟在守护线程中运行以保持GUI响应；支持停止按钮
- **结果查看**：`plot_result_comprehensive_with_initial()`（初始+最终场、能量密度、结构因子、能量收敛）和`plot_dislocation_comparison_with_initial()`
- **Sigma系列图**：界面能`γ_SL`或位错能`E_dis`随sigma变化
- **数据导出**：`simulation_results.txt`（人类可读）+ `result_N_psi.npy`（NumPy数组）

---

#### `part1_flat_interface.py` — 平直界面能（模型1）

**类**：`FlatInterfaceModel`

通过构建半固相/半液相域（平滑`tanh`过渡），弛豫到平衡态，再减去体能量，计算固液**平直界面能**`γ_SL`。

```
γ_SL = (E_total − E_solid·V_s/V − E_liquid·V_l/V) · L / A_interface
```
其中`A_interface = Ly`（界面法向沿x方向）。

- sigma₁, sigma₂ → 正方晶体固相
- sigma₃–sigma₅ → 三角晶体固相

---

#### `part2_round_interface.py` — 圆形界面能（模型2）

**类**：`RoundInterfaceModel`

通过在液相基体中嵌入圆形固相晶核（默认半径=min(Lx,Ly)/4），经核生长弛豫，提取`γ_SL`和有效半径`R_eff`。

```
γ_SL = (E_total − E_solid·V_s/V − E_liquid·V_l/V) / (2π·R)
```

---

#### `part3_dislocation.py` — 位错能（模型3）

**类**：`DislocationModel`

通过创建完美晶体，然后将右半部分（`x > Lx/2`）向上移动半个晶格单位形成刃位错，弛豫后减去完美晶体能量，计算**刃位错能**`E_dis`。

```
E_dis = E_total(dislocated) − E_perfect(perfect crystal)
```

- 256×256域
- 三角相：全部5个sigma值
- 正方相：sigma₁, sigma₂
- 完整系列共7组模拟

---

#### `visualization.py` — 交互式可视化模块

**类**：`PFCVisualizer`

为所有三个DualXPFC模型提供综合绘图：
- **密度场**、**能量密度**、**结构因子**（对数刻度）、**能量收敛曲线**
- **交互控件**：Matplotlib `Slider`、`Button`、`CheckButtons`，使用TkAgg后端
- **`plot_result_comprehensive_with_initial()`**：6面板对比图（初始场、最终场、初始/最终结构因子、能量密度、能量收敛）
- **`plot_dislocation_comparison_with_initial()`**：完美晶体与位错晶体并排对比，含差值场
- **`plot_sigma_series()`**：`γ_SL`或`E_dis`随sigma变化，含误差棒
- **中文字体支持**：自动检测Microsoft YaHei、SimHei等字体

---

#### `quick_test.py` — 快速验证测试

在64×64网格上运行快速端到端测试套件（100步），验证：
1. 核心引擎实例化和k空间算子构建
2. 正方和三角晶体初始条件
3. 平直界面模型构建+运行+`γ_SL`计算
4. 圆形界面模型构建+运行+`γ_SL`, `R_eff`
5. 位错模型准备+运行+`E_dis`
6. 能量单调递减和质量守恒

---

#### `xpfcSqAmpDis.cpp` — C++/MPI振幅表述求解器

高性能XPFC正方晶格求解器，使用振幅展开和MPI并行化。

**架构**：
- **MPI域分解**：1D带状分解，使用FFTW3-MPI转置FFT
- **场**：平均密度`n` + 两个复振幅场`A`, `B`（调制第一和第二倒易壳层）
- **迁移率**：密度和振幅具有独立的`M_n₀`, `M_A`, `M_B`
- **相关函数**：k零模（HSq0, wSq0）+ 第一峰（kSq1, wSq1, sigMSq1）+ 第二峰（kSq2, wSq2, sigMSq2）
- **位错形核**：`dislNoiseAmp`控制振幅噪声以生成位错
- **重启功能**：`restartFlag` + `restartTime`支持检查点/恢复

**典型参数**（来自`xpfcSqAmpDis.in`）：
- 1024×1024网格，`dx = 0.25`，`dt = 1.0`，1,000,001步迭代
- `wSq1 = 1.0`，`wSq2 = √2`，`sigMSq1 ≈ 0.2026`，`sigMSq2 ≈ 0.1013`
- `η = 1.4`（三次项），`χ = 1.0`（四次项）
- `dislNoiseAmp = 0.3`

---

### 分析模块

#### `pfc_analysis.py` — 微观结构与缺陷分析

全面的mixin类，提供定量微观结构表征：
- **原子检测**：`detect_atoms()`（`peak_local_max`）、`build_neighbors()`（周期性KD-Tree）、`coordination_numbers()`
- **拓扑分析**：`voronoi_analysis()`、`compute_psi6()`、`global_psi6()`、`grain_boundary_parameter()`（D参数）
- **缺陷分析**：`analyze_defects()`（5/7配位密度+近似晶粒尺寸）、`defect_statistics()`

---

#### `pfc_elastic.py` — 弹性性质计算

用于提取力学性质的mixin类：
- **应变施加**：`apply_strain(eps)`重新缩放域并重建k空间；`save_reference_state()`
- **能量-应变分析**：`elastic_energy_curve()`、`fit_elastic_constant()`（二次拟合→`C = d²F/dε²`）、`compute_stress()`

---

#### `pfc_plot.py` — 可视化工具包

丰富的绘图mixin类（20+方法）：场图、结构因子、Voronoi、缺陷、检测原子、晶界、ψ₆序、晶粒取向、演化曲线、弹性曲线，以及完整工作流（`postprocess()`、`analyze_psi6()`）。

---

### 输入输出模块

#### `pfc_io.py` — 视频录制

用于生成模拟视频的mixin类：内存PNG帧缓存（模拟期间零磁盘IO）和ffmpeg H.264 MP4合成（CRF=18）。

---

### 配置模块

#### `config.py` — 交互式参数菜单

基于控制台的交互式界面：晶格选择（编号菜单）、带默认值的验证型浮点/整数输入、参数封装、确认步骤。

---

### 运行脚本

#### `run_pure.py` — 纯物质模拟运行脚本
标准PFC模拟入口：交互式配置→`PurePFCSolver`→主循环→`postprocess()`→`analyze_psi6()`→空位检测。

#### `run_binary.py` — 二元合金模拟运行脚本
4种运行模式：快速（N=512）、交互、高分辨率（N=1024, T=5000）、相分离。CLI：`--mode`、`--interactive`、`--video`。

#### `run_elastic.py` — 弹性常数计算脚本
平衡（T=1500）→保存参考态→13个应变点（`linspace(-0.03, 0.03, 13)`）→每点弛豫2000步→二次拟合→`C = 2a`。

#### `run_3d.py` — 三维模拟运行脚本
4种模式：纯物质（BCC生长）、弹性（三维常数）、合金（三维二元）、扫描（r参数扫描）。内存：N=128（~200MB）、N=256（~1.5GB，推荐）、N=512（~12GB）。

---

## 使用示例

### 示例1：晶粒生长模拟（标准PFC）

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

### 示例2：DualXPFC — 平直界面能

```python
from pfc_core import DualPFCConfig, DualPFCEngine
from part1_flat_interface import FlatInterfaceModel
import numpy as np

# sigma=0.2 → 三角相；sigma=-0.2 → 正方相
cfg = DualPFCConfig(Nx=256, Ny=256, sigma=0.2, tau=1.0,
                     q=np.sqrt(3), dt=0.5, n_steps=2000)

model = FlatInterfaceModel(cfg, crystal_type='triangular')
model.build_initial_condition()   # 半固相半液相，tanh过渡
model.run()                        # 弛豫到平衡
gamma = model.compute_interface_energy()
print(f"平直界面能 gamma_SL = {gamma:.6f}")
```

### 示例3：DualXPFC — 位错能

```python
from pfc_core import DualPFCConfig
from part3_dislocation import DislocationModel

cfg = DualPFCConfig(Nx=256, Ny=256, sigma=0.0, n_steps=2000)
model = DislocationModel(cfg, crystal_type='triangular')
model.prepare_and_run()            # 完美晶体→位移半区→弛豫
print(f"位错能 E_dis = {model.E_dis:.6f}")
```

### 示例4：DualXPFC — 相关函数正方求解器

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

### 示例5：DualXPFC — 核心引擎直接使用

```python
from pfc_core import DualPFCConfig, DualPFCEngine
import numpy as np

cfg = DualPFCConfig(Nx=128, Ny=128, sigma=0.0, tau=1.0,
                     q=np.sqrt(3), dt=0.5, n_steps=500)
engine = DualPFCEngine(cfg)

X, Y = engine.get_coordinate_grids()
psi = engine.triangular_crystal(X, Y) + 0.01 * np.random.randn(128, 128)

psi, energies = engine.relax(psi, n_steps=500)
print(f"最终能量: {energies[-1][1]:.4f}")
```

---

## 输出说明

### 标准PFC生成文件

```
result/
├── pfc_simulation.mp4      # 演化视频
├── energy_evolution.png    # 能量曲线
├── density_field.png       # 最终密度场
├── structure_factor.png    # 结构因子（衍射图案）
├── voronoi_analysis.png    # Voronoi剖分
├── defect_analysis.png     # 缺陷可视化（5/7配位）
├── defect_density.png      # 缺陷密度演化
├── grain_size.png          # 晶粒尺寸演化
├── psi6_order.png          # 键取向序
└── grain_boundary.png      # 晶界原子
```

### DualXPFC生成文件

```
DualXPFC/
├── xpfc_output/
│   ├── xpfc_single_crystal.mp4   # 单晶演化视频
│   ├── xpfc_polycrystal.mp4      # 多晶演化视频
│   └── final_single.png / final_poly.png  # 最终分析（场+S(k)+C2）
├── simulation_results.txt         # GUI导出的结果（gamma_SL, E_dis, sigma系列）
├── result_1_psi.npy ...           # NumPy最终场数组
└── xpfcSqAmpDis/                  # C++求解器输出（振幅场、密度、重启文件）
```

### 关键观测量

| 观测量 | 说明 |
|--------|------|
| 自由能 | 系统自由能演化（应单调递减） |
| 密度场 | 空间原子密度分布`φ(r)`或`ψ(r)` |
| 结构因子 | 倒易空间衍射图案`S(k) = |φ̃(k)|²` |
| 缺陷密度 | 5/7配位原子的比例 |
| 晶粒尺寸 | 由缺陷密度估算的平均晶粒直径 |
| ψ₆序参数 | 键取向序定量表征 |
| 弹性常数 | 由能量-应变二次拟合得到的杨氏模量 |
| **γ_SL（平直）** | 平直固液界面能（DualXPFC模型1） |
| **γ_SL, R_eff（圆形）** | 圆形晶核界面能和有效半径（DualXPFC模型2） |
| **E_dis** | 刃位错能（DualXPFC模型3） |
| **C₂(k)** | 直接相关函数截面（相关函数XPFC） |
| **振幅场** | 复振幅A(r), B(r)（C++振幅求解器） |

---

## 作者与引用

**王锦鹏 (Jinpeng Wang)**
香港理工大学 航空工程系
麦克马斯特大学 Mitacs实习生

如果您在研究中使用了本框架，请合理引用。

### 参考文献

- Elder, K. R., & Grant, M. (2004). Modeling elastic and plastic deformations in nonequilibrium processing using phase field crystals. *Physical Review E*, 70(5), 051605.
- Provatas, N., & Elder, K. (2010). Phase-Field Methods in Materials Science and Engineering. Wiley-VCH.
- Berry, J., Elder, K. R., & Grant, M. (2008). Phase-field crystal modeling of eutectic solidification. *Physical Review Letters*, 100(4), 045705.
- Greenwood, M., Rottler, J., & Provatas, N. (2010). Free energy of crystal-liquid interfaces in the phase-field crystal method. *Physical Review E*, 81(6), 061601.
- Athreya, P., et al. (2007). Diffusive atomistic dynamics of edge dislocations in two dimensions. *Physical Review E*, 75(2), 021603.

---

## 更新日志

### v1.2.0
- **在`DualXPFC`分支新增DualXPFC扩展**，包含三个研究模型：
  - 平直固液界面能计算（`part1_flat_interface.py`）
  - 圆形晶核界面能（含有效半径）（`part2_round_interface.py`）
  - 正方和三角相的刃位错能计算（`part3_dislocation.py`）
- **新增双振幅（双模）PFC核心引擎**（`pfc_core.py`），乘积形式线性算子`L = [r+(1+∇²)²][(q²+∇²)²/q⁴]`，三次非线性项，sigma控制的正方/三角相选择
- **新增相关函数C₂ XPFC正方求解器**（`xpfc_square.py`），Taichi CPU加速，多高斯峰构建，单晶/多晶模式，内置ffmpeg视频编码
- **新增C++/MPI振幅表述求解器**（`xpfcSqAmpDis.cpp` + `xpfcSqAmpDis.in`），FFTW3-MPI并行化，振幅展开，位错初始化
- **新增Tkinter GUI主界面**（`main.py`），模型选择、参数面板、sigma系列批量运行、实时可视化、数据导出
- **新增交互式可视化模块**（`visualization.py`），滑块、播放/暂停、综合结果对比图、中文字体支持
- **新增快速验证测试**（`quick_test.py`），端到端引擎和模型验证

### v1.1.0
- 新增三维PFC模拟（BCC晶格）与PyVista等值面渲染
- 新增Swift-Hohenberg模型，内置视频录制
- 新增考虑泊松效应的三维弹性常数计算
- 改进视频生成，支持自动GIF回退
- 新增二元合金相分离模式（快速/交互/高分辨）

### v1.0.0
- 首次发布，包含二维PFC框架
- 支持六角、正方、三角晶格
- 纯物质与二元合金模型
- 完整的分析与可视化工具包
- 缺陷检测、Voronoi分析、ψ₆序参数
- 弹性能量与应力-应变计算

---

## 许可证

本项目用于学术研究目的。如用于发表，请合理引用。

*最后更新：2026年09月*
