<div align="center">

# Phase-Field-Crystal

**模块化二维/三维相场晶体模拟框架**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.20.0-orange)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%E2%89%A51.7.0-green)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

[English](README.md) | [中文](README.zh.md)

</div>

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [安装依赖](#安装依赖)
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

## 项目简介

**相场晶体（Phase-Field-Crystal, PFC）** 模拟框架是一个面向计算材料科学研究的模块化、面向对象 Python 工具包。它能够在原子长度尺度上描述晶体结构，同时在扩散时间尺度上运行，非常适合研究晶粒生长、晶界迁移、位错运动、缺陷演化以及弹性性质等材料科学问题。

本框架实现了标准 PFC 模型及其扩展（二元合金、三维 BCC/FCC/SC 晶格、Swift-Hohenberg 模型），采用高效的**半隐式傅里叶谱方法**进行数值求解。模块化架构将核心求解器、分析工具、可视化和输入输出分离为独立的 mixin 类，支持灵活的功能组合和易于扩展的开发模式。

### 核心能力

| 类别 | 功能 |
|------|------|
| **核心模拟** | 谱方法半隐式PFC求解器；多种二维/三维晶格类型；纯材料与二元合金模型 |
| **分析工具** | 缺陷检测（5/7重配位）、Voronoi剖分、ψ₆取向序、晶界识别、结构因子分析 |
| **弹性计算** | 能量-应变曲线、应力-应变关系、弹性常数拟合 |
| **可视化** | 实时场渲染、MP4/GIF视频生成、三维等值面渲染（PyVista） |

---

## 功能特性

### 核心模拟
- **谱方法半隐式PFC求解器** — 傅里叶空间高效数值积分，线性项无条件稳定
- **多种晶格类型** — 六角、正方、三角（二维）；BCC、FCC、SC（三维）
- **纯材料与二元合金** — 单组分模型与双相场耦合，含Cahn-Hilliard浓度动力学
- **三维PFC模拟** — 完整的三维BCC晶格支持，体可视化与弹性常数计算
- **Swift-Hohenberg模型** — 经典图案形成偏微分方程，用于条纹/六角图案研究

### 分析工具
- **缺陷密度分析** — 通过局部峰值分析检测5重/7重配位缺陷
- **Voronoi剖分** — 周期边界条件下的晶粒结构拓扑分析
- **ψ₆取向序参量** — 六角晶格的键取向有序度量化
- **晶界检测** — 基于D参数的晶界原子识别
- **结构因子分析** — 二维与三维倒易空间衍射图案
- **弹性能量计算** — 施加应变下的自由能计算，含参考态管理
- **应力-应变曲线** — 通过数值微分提取力学性质

### 可视化
- **实时场可视化** — 使用Matplotlib渲染密度场和浓度场
- **视频录制** — 自动生成演化动力学MP4（ffmpeg）或GIF（Pillow）
- **三维等值面渲染** — 使用PyVista进行体可视化（含Matplotlib降级方案）
- **全面后处理** — 全套分析图表（能量、缺陷、晶粒尺寸等）

---

## 项目结构

```
Phase-Field-Crystal/
│
├── README.md              # 英文说明文档
├── README.zh.md           # 中文说明文档
│
├── 核心模块
│   ├── pfc_base.py        # 基类：网格设置、k空间构建、模拟日志
│   ├── pfc_pure.py        # 纯材料PFC求解器（六角/正方/三角晶格）
│   ├── pfc_binary.py      # 二元合金PFC求解器（密度场+浓度场双场耦合）
│   ├── pfc_3d.py          # 三维PFC求解器（BCC/FCC/SC晶格，PyVista可视化）
│   └── sh_model.py        # Swift-Hohenberg图案形成求解器
│
├── 分析模块
│   ├── pfc_analysis.py    # 微观结构分析：原子检测、配位数、ψ6、Voronoi、缺陷
│   ├── pfc_elastic.py     # 弹性性质计算：应变施加、能量曲线、拟合
│   └── pfc_plot.py        # 可视化工具包：场图、结构因子、缺陷、晶界等
│
├── IO模块
│   └── pfc_io.py          # 视频录制：内存帧缓存、ffmpeg MP4合成
│
├── 配置模块
│   └── config.py          # 交互式控制台参数输入菜单，含输入验证
│
└── 运行脚本
    ├── run_pure.py        # 标准纯材料PFC模拟（交互式配置）
    ├── run_binary.py      # 二元合金模拟（4种预设模式：快速/交互/高分辨率/相分离）
    ├── run_elastic.py     # 弹性常数计算（能量-应变二次拟合）
    └── run_3d.py          # 三维PFC模拟（4种模式：纯材料/弹性/合金/参数扫描）
```

---

## 安装依赖

### 依赖项

| 包名 | 版本 | 用途 |
|------|------|------|
| `numpy` | ≥1.20.0 | 核心数值数组与FFT |
| `scipy` | ≥1.7.0 | FFT运算、空间算法 |
| `matplotlib` | ≥3.4.0 | 二维绘图与可视化 |
| `pyvista` | ≥0.32.0 | 三维等值面渲染（可选） |
| `scikit-image` | ≥0.18.0 | 三维峰值检测（可选） |
| `ffmpeg` | 系统包 | 视频生成（可选，系统级安装） |

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/RichardWANG-1010/Phase-Field-Crystal.git
cd Phase-Field-Crystal

# 安装核心依赖
pip install numpy scipy matplotlib

# 可选：三维可视化
pip install pyvista scikit-image

# 可选：视频生成（通过系统包管理器安装）
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: 从 https://ffmpeg.org/download.html 下载
```

---

## 快速开始

### 1. 纯材料PFC模拟

```bash
python run_pure.py
```

启动交互式菜单配置模拟参数，然后运行模拟并生成分析图表。

### 2. 弹性常数计算

```bash
python run_elastic.py
```

通过施加单轴应变并拟合能量-应变二次曲线来计算弹性模量。

### 3. 二元合金模拟

```bash
python run_binary.py
```

模拟二元合金中的旋节线分解和相分离过程。支持4种模式：快速模式、交互模式、高分辨率模式、相分离模式。

### 4. 三维PFC模拟

```bash
python run_3d.py
```

运行支持BCC晶格的完整三维PFC模拟。可选模式：纯材料、弹性常数计算、二元合金、参数扫描。

### 5. Swift-Hohenberg模型

```bash
python sh_model.py
```

直接运行Swift-Hohenberg图案形成求解器。

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

通过半隐式傅里叶谱方法求解，兼顾数值稳定性与计算效率。

### 二元合金模型

扩展的PFC模型，包含两个守恒场：总密度 `φ` 和浓度 `c`。

```
F = F_PFC[φ] + F_CH[c] + F_coupling[φ, c]
```

其中 `F_CH` 为Cahn-Hilliard自由能，`F_coupling` 描述密度场与浓度场的相互作用。

---

## 晶格类型

### 二维晶格

| 晶格 | 色散算子 `L(k)` | 晶体结构 |
|------|------------------|----------|
| **六角** | `L(k) = (1 - k²)² + r` | 三角晶格 |
| **正方** | `L(k) = (1-kx²)²(1-ky²)² + r` | 正方晶格 |
| **三角** | `L(k) = (1 - kx² - kx·ky + ky²)² + r` | 斜晶格 |

### 三维晶格

| 晶格 | 说明 | 备注 |
|------|------|------|
| **BCC** | 体心立方 | 标准PFC（Provatas & Elder 第8章） |
| **FCC** | 面心立方 | 需要额外稳定化项 |
| **SC** | 简单立方 | 基本立方对称性 |

---

## 代码文档

### 核心模块

#### `pfc_base.py` — 数值计算基础设施基类

所有PFC求解器的基础。提供：
- **网格设置**：`N×N`（二维）或 `N³`（三维）空间离散化，物理域尺寸 `L`
- **k空间构建**：预计算傅里叶波矢 `kx, ky, kz` 及模长平方 `k2 = kx² + ky² + kz²`，使用 `numpy.fft.fftfreq`
- **模拟日志**：能量、质量、缺陷密度、晶粒尺寸、结构因子峰值的数组记录
- **分析缓存**：检测到的原子位置、邻居列表、ψ₆值的存储
- **视频配置**：帧目录设置与录制标志

**关键方法**：`_build_kspace()` — 构建倒易空间网格，使用正确的FFT频率排序和矩阵式索引。

---

#### `pfc_pure.py` — 纯材料PFC求解器

通过**多继承**（mixin模式）实现标准单组分PFC模型：

```python
class PurePFCSolver(PFCBase, PFCAnalysis, PFCPlot, PFCIO, PFCElastic, PFCAdvancedAnalysis):
```

**功能**：
- **半隐式谱时间步进**：线性项隐式处理，非线性 `φ³` 显式处理
- **晶格相关色散算子**：实时计算六角/正方/三角晶格的 `L(k)`
- **质量守恒强制**：每步后修正平均密度
- **能量计算**：使用帕塞瓦尔定理在k空间高效积分
- **结构因子**：`S(k) = |φ̃(k)|²`，使用 `fftshift` 居中衍射图案

**关键参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N` | 128 | 网格分辨率 |
| `L` | 64.0 | 物理域尺寸 |
| `r` | -0.25 | 温度参数（负值有利于晶相形成） |
| `M` | 1.0 | 迁移率系数 |
| `phi0` | -0.25 | 平均密度 |
| `lattice_type` | "hexagon" | 晶体对称性 |

---

#### `pfc_binary.py` — 二元合金PFC求解器

双守恒场求解器，耦合密度场 `φ` 和浓度场 `c`：

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

**特性**：
- 傅里叶空间中双场同步半隐式更新
- 浓度物理约束：`0 ≤ c ≤ 1`，通过裁剪实现
- 双场视频捕获（密度场+浓度场并排显示）
- 扩展后处理：浓度演化、叠加场、耦合能量
- 浓度场结构因子分析

**关键参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `c0` | 0.3 | 初始平均浓度 |
| `M_c` | 0.1 | 浓度迁移率 |
| `alpha` | 0.1 | 偏析耦合强度 |
| `beta` | 0.0 | 直接耦合强度 |
| `r_c` | -0.5 | Cahn-Hilliard参数 |

---

#### `pfc_3d.py` — 三维PFC求解器

完整的三维扩展，支持BCC、FCC、SC晶格。由4个mixin类组成：

| 类 | 用途 |
|----|------|
| `PFCBase3D` | 三维网格、`N³` k空间（`kx, ky, kz`）、体积计算 |
| `PFCAnalysis3D` | 三维结构因子、`peak_local_max`原子检测、二维切片 |
| `PFCPlot3D` | 三维等值面（PyVista + Matplotlib降级）、正交切片面板 |
| `PFCElastic3D` | 三维应变张量、含泊松效应的单轴应变、体模量 |

**关键特性**：
- **BCC色散**：`L(k) = (1 - |k|²)² + r`（标准PFC，第8章）
- **FCC稳定化**：额外项 `α·(kx²ky² + ky²kz² + kz²kx²)`
- **三维原子检测**：`skimage.feature.peak_local_max`，`min_distance=5`
- **等值面渲染**：PyVista `UniformGrid` + `contour()`，自动降级至Matplotlib `marching_cubes`
- **弹性常数**：完整3×3应变张量支持，`fit_elastic_constant_3d()`

---

#### `sh_model.py` — Swift-Hohenberg求解器

独立的图案形成求解器，用于经典Swift-Hohenberg方程：

```
∂ψ/∂t = ε·ψ - (q₀² + ∇²)²·ψ - ψ³
```

**设计特点**：
- 继承 `PFCBase` 复用网格/k空间基础设施
- 内嵌视频录制（独立于 `pfc_io.py`）
- 预计算线性算子提高效率：`L = -(ε - (q₀² - k²)²)`
- 自动ffmpeg/GIF降级视频生成
- 功率谱分析，标注不稳定环

**参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epsilon` | 0.3 | 控制参数（图案形成阈值） |
| `q0` | 1.0 | 特征波数 |
| `psi_clip` | 2.0 | 场裁剪值（数值稳定性） |

---

### 分析模块

#### `pfc_analysis.py` — 微观结构与缺陷分析

全面的mixin类，提供定量微观结构表征：

**原子检测**：
- `detect_atoms()` — 局部峰值检测，使用 `skimage.feature.peak_local_max`，`min_distance=7`，`threshold_rel=0.5`
- `build_neighbors()` — 周期边界KD树（`cKDTree` + `boxsize`）构建邻居列表
- `coordination_numbers()` — 统计最近邻数量（完美六角晶格 = 6）

**拓扑分析**：
- `voronoi_analysis()` — `scipy.spatial.Voronoi` 剖分，含边界原子过滤
- `compute_psi6()` — 复键取向序参量：`ψ6 = ⟨exp(6i·θ)⟩`
- `global_psi6()` — 局部平均 `|ψ6|` 与全局 `|⟨ψ6⟩|`（对晶界敏感）
- `grain_boundary_parameter()` — D参数：相邻原子ψ6差值平方均值

**缺陷分析**：
- `analyze_defects()` — 5/7重缺陷密度与近似晶粒尺寸：`D ≈ √(A/N_缺陷)`
- `defect_statistics()` — 控制台输出配位数分布统计

---

#### `pfc_elastic.py` — 弹性性质计算

力学性质提取的mixin类：

**应变施加**：
- `apply_strain(eps)` — 重新缩放域 `L = L0·(1+ε)` 并重建k空间
- `save_reference_state()` — 保存平衡态 `phi_ref`、`L_ref`、`dx_ref`

**能量-应变分析**：
- `elastic_energy_curve(strain_list, relax_steps=2000)` — 对每个应变：重置→施加→弛豫→测量能量
- `fit_elastic_constant(strain, energy)` — 二次拟合 `F(ε) = a·ε² + b·ε + c`；提取 `C = 2a = d²F/dε²` 和残余应变 `ε_r = -b/(2a)`
- `compute_stress(strain, energy)` — 数值微分 `σ = dF/dε`，使用 `np.gradient`

---

#### `pfc_plot.py` — 可视化工具包

包含20+绘图方法的广泛mixin类：

**场图**：
- `plot_field()` — 二维密度云图（`coolwarm`）
- `plot_structure_factor()` — 对数尺度衍射图案（`inferno`）

**微观结构图**：
- `plot_voronoi()` — Voronoi元胞拓扑图
- `plot_defects()` — 密度场上叠加5重（红）/7重（蓝）缺陷
- `plot_detected_atoms()` — 密度背景上的原子黑点
- `plot_grain_boundary_atoms()` — D参数阈值分割（85百分位数）

**序参量图**：
- `plot_psi6()` — 按 `|ψ6|` 模长着色的原子（`viridis`）
- `plot_grain_orientation()` — 按 `arg(ψ6)` 相位着色的原子（`hsv`）
- `plot_psi6_field()` — 插值连续有序度场

**演化图**：
- `plot_energy()`、`plot_defect_density()`、`plot_grain_size()`、`plot_structure_peak()`

**弹性图**：
- `plot_elastic_curve()` — 能量-应变曲线及二次拟合
- `plot_stress_strain()` — 应力-应变曲线

**分析流水线**：
- `postprocess()` — 完整10图分析套件
- `analyze_psi6()` — 完整ψ6分析（4张图 + 指标）

---

### IO模块

#### `pfc_io.py` — 视频录制

模拟视频生成的mixin类：

**特性**：
- **内存缓存**：帧以PNG字节形式存储在RAM中（`frame_cache` 列表），模拟期间零磁盘IO
- **ffmpeg合成**：H.264 MP4，CRF=18（高质量），自动清理临时文件
- **可配置**：帧率、图形尺寸、输出文件名

**关键方法**：
- `initialize_io()` — 设置录制参数
- `capture_frame()` — 将当前 `phi` 场转换为内存图像（子类可重写）
- `frames_to_video()` — 通过ffmpeg子进程合成缓存帧

---

### 配置模块

#### `config.py` — 交互式参数菜单

基于控制台的交互式模拟配置界面：

**特性**：
- **晶格选择**：编号菜单（1=六角、2=正方、3=三角），含输入验证
- **验证输入**：`get_float_input()` 和 `get_int_input()`，支持默认值和类型检查
- **参数打包**：返回结构化字典 `{"solver": {...}, "lattice_type": ...}`
- **确认步骤**：模拟开始前用户审核

**配置参数**：
`N`、`L`、`r`、`M`、`dt`、`T`、`phi0`、`noise_amp`、`lattice_type`

---

### 运行脚本

#### `run_pure.py` — 纯材料模拟运行脚本

标准PFC模拟的入口点：
1. 调用 `input_pfc_parameters()` 进行交互式配置
2. 使用用户参数实例化 `PurePFCSolver`
3. 运行主模拟循环 + 高级分析
4. 执行 `postprocess()`（10图套件）
5. 运行 `analyze_psi6()`（完整取向序分析）
6. 空位检测与扩散动画

#### `run_binary.py` — 二元合金模拟运行脚本

灵活的入口点，支持4种运行模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **快速模式** | `N=512`，预配置标准参数 | 快速预览 |
| **交互模式** | 通过命令行自定义所有参数 | 研究调参 |
| **高分辨率模式** | `N=1024`，`T=5000` | 精细微观结构 |
| **相分离模式** | 针对旋节线分解优化 | 合金动力学 |

**命令行参数**：
```bash
python run_binary.py --mode quick        # 快速模式
python run_binary.py --interactive       # 交互模式
python run_binary.py --mode quick --video # 开启视频录制
```

#### `run_elastic.py` — 弹性常数计算脚本

系统的弹性模量测量流程：
1. 弛豫 `PurePFCSolver` 至平衡态（`T=1500`）
2. 保存参考状态
3. 施加13个应变值：`linspace(-0.03, 0.03, 13)`
4. 每个应变点弛豫2000步
5. 拟合 `F(ε) = a·ε² + b·ε + c` → 提取 `C = 2a`
6. 绘制能量-应变和应力-应变曲线

#### `run_3d.py` — 三维模拟运行脚本

交互式模式选择的三维模拟入口：

| 模式 | 功能 | 参考 |
|------|------|------|
| `pure` | 三维BCC晶体生长 | 第8章 |
| `elastic` | 三维弹性常数计算 | 第8.5节 |
| `alloy` | 三维二元合金（简化演示） | 第9章 |
| `sweep` | 不同 `r` 值的参数扫描 | 稳定性分析 |

**内存说明**：
- `N=128`：约200MB内存，每步较快
- `N=256`：约1.5GB内存，每步慢约8倍（推荐）
- `N=512`：约12GB内存，每步慢约64倍（仅工作站）

---

## 使用示例

### 示例1：晶粒生长模拟

```python
from pfc_pure import PurePFCSolver

# 初始化求解器
solver = PurePFCSolver(
    N=256,              # 网格尺寸
    L=128.0,            # 物理域尺寸
    r=-0.25,            # 温度参数
    dt=0.05,            # 时间步长
    T=2000.0,           # 总模拟时间
    lattice_type="hexagon"
)

# 运行模拟
solver.run(sample_interval=10)

# 后处理
solver.postprocess()
solver.analyze_psi6()
```

### 示例2：弹性常数计算

```python
from pfc_pure import PurePFCSolver
import numpy as np

# 初始化并弛豫
solver = PurePFCSolver(N=256, L=128, r=-0.35, T=1500)
solver.run()
solver.save_reference_state()

# 施加应变并计算能量
strain = np.linspace(-0.03, 0.03, 13)
energy, phi_list = solver.elastic_energy_curve(strain)

# 拟合弹性常数
C, eps_r, coef, _, _ = solver.fit_elastic_constant(strain, energy)
print(f"弹性常数 C = {C:.6e}")
```

### 示例3：二元合金相分离

```python
from pfc_binary import BinaryPFCSolver

solver = BinaryPFCSolver(
    N=256,
    L=128.0,
    r=-0.25,
    c0=0.3,          # 初始浓度
    alpha=0.1,       # 耦合强度
    lattice_type="hexagon"
)

solver.run()
solver.postprocess()
```

---

## 输出说明

### 生成文件

```
result/
├── pfc_simulation.mp4      # 演化视频
├── energy_evolution.png    # 能量曲线
├── density_field.png       # 最终密度场
├── structure_factor.png    # 结构因子（衍射图案）
├── voronoi_analysis.png    # Voronoi剖分
├── defect_analysis.png     # 缺陷可视化（5/7重）
├── defect_density.png      # 缺陷密度演化
├── grain_size.png          # 晶粒尺寸演化
├── psi6_order.png          # 键取向有序度
├── grain_boundary.png      # 晶界原子
└── ...
```

### 关键观测量

| 观测量 | 说明 |
|--------|------|
| **自由能** | 系统自由能演化（应单调下降） |
| **密度场** | 原子密度空间分布 `φ(r)` |
| **结构因子** | 倒易空间衍射图案 `S(k) = |φ̃(k)|²` |
| **缺陷密度** | 5/7重配位缺陷原子比例 |
| **晶粒尺寸** | 由缺陷密度估算的平均晶粒直径 |
| **ψ₆序参量** | 键取向有序度（`|ψ6| = 1` 完美有序，`0` 完全无序） |
| **弹性常数** | 由能量-应变二次拟合得到的杨氏模量 |

---

## 作者与引用

**王金鹏 (Jinpeng Wang)**

- 香港理工大学 材料工程系
- 麦克马斯特大学 Mitacs实习生

如在研究中使用本框架，请适当引用。

### 参考文献

1. Elder, K. R., & Grant, M. (2004). Modeling elastic and plastic deformations in nonequilibrium processing using phase field crystals. *Physical Review E*, 70(5), 051605.
2. Provatas, N., & Elder, K. (2010). *Phase-Field Methods in Materials Science and Engineering*. Wiley-VCH.
3. Berry, J., Elder, K. R., & Grant, M. (2008). Phase-field crystal modeling of eutectic solidification. *Physical Review Letters*, 100(4), 045705.

---

## 更新日志

### v1.1.0
- 新增三维PFC模拟（BCC晶格），支持PyVista等值面渲染
- 新增Swift-Hohenberg模型，含内嵌视频录制
- 新增三维弹性常数计算，含泊松效应
- 改进视频生成，自动GIF降级
- 新增二元合金相分离模式（快速/交互/高分辨率）

### v1.0.0
- 初始版本，包含二维PFC框架
- 支持六角、正方、三角晶格
- 纯材料和二元合金模型
- 完整的分析和可视化工具包
- 缺陷检测、Voronoi分析、ψ₆序参量
- 弹性能量和应力-应变计算

---

## 许可证

本项目用于学术研究目的。如在出版物中使用，请适当引用。

---

*最后更新：2026-06*
