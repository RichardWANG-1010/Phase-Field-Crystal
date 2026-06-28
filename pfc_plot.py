"""
pfc_plot.py - Visualization Tools for PFC Simulations
可视化工具模块 - PFC模拟的可视化工具

This module provides plotting and visualization methods for PFC simulation results,
including field visualization, structure factor, Voronoi analysis, defect analysis,
and elastic property plots.
本模块提供PFC模拟结果的绘图和可视化方法，包括场可视化、结构因子、Voronoi分析、
缺陷分析和弹性性质图。

Designed as a mixin class to be used with PFC solvers.
设计为mixin类，与PFC求解器一起使用。

Author: Jinpeng Wang
Department of Material Engineering
"""

# Matplotlib - plotting library
# Matplotlib - 绘图库
import matplotlib.pyplot as plt

# SciPy interpolation - for scattered data interpolation
# SciPy插值 - 用于散点数据插值
from scipy.interpolate import griddata

# SciPy spatial - for Voronoi tessellation plotting
# SciPy空间 - 用于Voronoi剖分绘图
from scipy.spatial import voronoi_plot_2d

# NumPy - numerical computing
# NumPy - 数值计算
import numpy as np


class PFCPlot:
    """
    Mixin class providing visualization methods for PFC simulations.
    为PFC模拟提供可视化方法的mixin类。
    
    This class provides a comprehensive set of plotting functions for
    analyzing PFC simulation results.
    本类提供一套全面的绘图函数，用于分析PFC模拟结果。
    
    Plot categories / 绘图类别:
        - Field visualization / 场可视化
        - Structure factor / 结构因子
        - Voronoi analysis / Voronoi分析
        - Defect analysis / 缺陷分析
        - Grain analysis / 晶粒分析
        - Elastic properties / 弹性性质
        - Order parameters / 序参量
    
    Notes / 说明:
        - Assumes host class provides: phi, energy_log, defect_log, etc.
          假设宿主类提供：phi, energy_log, defect_log等
        - All plots use origin="lower" for consistent coordinate system
          所有图都使用origin="lower"以保持坐标系一致
    """
    
    def plot_energy(self):
        """
        Plot free energy evolution curve against sample step index.
        绘制自由能随采样步演化曲线。
        
        Shows how the system free energy decreases over time, indicating
        relaxation towards equilibrium.
        显示系统自由能如何随时间降低，表明系统向平衡态弛豫。
        
        Notes / 说明:
            - X-axis: sample index (not physical time)
              X轴：采样索引（不是物理时间）
            - Y-axis: free energy density
              Y轴：自由能密度
            - Energy should decrease monotonically (or with fluctuations)
              能量应该单调下降（或有波动）
        """
        # Create figure
        # 创建图形
        plt.figure(figsize=(6, 4))
        
        # Plot energy time series curve
        # 绘制能量时序曲线
        plt.plot(self.energy_log)
        
        # Axis labels and title
        # 坐标轴标签和标题
        plt.xlabel("Sample / 采样")
        plt.ylabel("Free Energy / 自由能")
        plt.title("Energy Evolution / 能量演化")
        
        # Add grid for readability
        # 添加网格以提高可读性
        plt.grid(True)
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_field(self):
        """
        Plot 2D heatmap of density field φ.
        绘制2D密度场φ云图。
        
        Visualizes the spatial distribution of atomic density, showing
        crystal structure, grain boundaries, and defects.
        可视化原子密度的空间分布，显示晶体结构、晶界和缺陷。
        
        Notes / 说明:
            - origin="lower" matches grid ij indexing
              origin="lower"匹配网格ij索引
            - coolwarm colormap shows positive/negative deviations
              coolwarm色图显示正/负偏差
            - Image bottom-left corresponds to coordinate origin
              图像左下角对应坐标原点
        """
        # Create square figure
        # 创建正方形图形
        plt.figure(figsize=(6, 6))
        
        # Plot density field as heatmap
        # 将密度场绘制成云图
        # origin="lower" matches grid ij indexing
        # origin="lower"匹配网格ij索引
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        
        # Add colorbar for density scale
        # 添加颜色条表示密度刻度
        plt.colorbar(label="φ")
        
        # Plot title
        # 图标题
        plt.title("Density Field / 密度场")
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_structure_factor(self):
        """
        Plot log-scaled static structure factor diffraction pattern.
        绘制对数尺度静态结构因子衍射图案。
        
        The structure factor S(k) shows the diffraction pattern in
        reciprocal space, revealing crystal symmetry and quality.
        结构因子S(k)显示倒易空间中的衍射图案，揭示晶体对称性和质量。
        
        Notes / 说明:
            - log10 compresses dynamic range for better visibility
              log10压缩动态范围以获得更好的可见性
            - +1 avoids log10(0) which is undefined
              +1避免无意义的log10(0)
            - Sharp peaks indicate good crystalline order
              尖锐的峰表明良好的晶体有序度
        """
        # Calculate structure factor
        # 计算结构因子
        S = self.structure_factor()
        
        # Create square figure
        # 创建正方形图形
        plt.figure(figsize=(6, 6))
        
        # Plot log-scaled structure factor
        # 绘制对数尺度的结构因子
        # log10 compresses dynamic range, +1 to avoid log10(0)
        # log10压缩动态范围，+1避免log10(0)
        plt.imshow(np.log10(S + 1.0), origin="lower", cmap="inferno")
        
        # Add colorbar with label
        # 添加带标签的颜色条
        plt.colorbar(label="log₁₀(S+1)")
        
        # Plot title
        # 图标题
        plt.title("Structure Factor / 结构因子")
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_voronoi(self):
        """
        Plot Voronoi tessellation topology of crystal atoms.
        绘制原子Voronoi元胞拓扑图。
        
        Voronoi tessellation partitions space into regions around each
        atom, revealing grain structure and topological defects.
        Voronoi剖分将空间划分为每个原子周围的区域，揭示晶粒结构和拓扑缺陷。
        
        Notes / 说明:
            - Uses scipy.spatial.voronoi_plot_2d
              使用scipy.spatial.voronoi_plot_2d
            - Y-axis inverted to match imshow coordinate system
              Y轴反转以匹配imshow坐标系
            - Each polygon is the Voronoi cell of one atom
              每个多边形是一个原子的Voronoi元胞
        """
        # Generate Voronoi topology and filtered atom coordinates
        # 生成Voronoi拓扑和过滤后的原子坐标
        vor, _ = self.voronoi_analysis()
        
        # Create figure
        # 创建图形
        plt.figure(figsize=(8, 8))
        
        # Plot Voronoi diagram
        # 绘制Voronoi图
        # Hide vertices, set line width and point size
        # 隐藏顶点，设置线宽和点尺寸
        voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_width=1, point_size=2)
        
        # Invert Y axis to align with imshow plotting coordinate system
        # 反转Y轴以与imshow绘图坐标系对齐
        plt.gca().invert_yaxis()
        
        # Plot title
        # 图标题
        plt.title("Voronoi Analysis / Voronoi分析")
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_defects(self):
        """
        Mark 5-fold (red) and 7-fold (blue) coordination defect atoms
        over density heatmap.
        在密度场云图上标记5重（红）和7重（蓝）配位缺陷原子。
        
        In a perfect hexagonal lattice, each atom has 6 neighbors.
        5-fold and 7-fold defects correspond to dislocations.
        在完美的六角晶格中，每个原子有6个邻居。
        5重和7重缺陷对应位错。
        
        Notes / 说明:
            - 5-fold defects: one fewer neighbor (vacancy-like)
              5重缺陷：少一个邻居（类空位）
            - 7-fold defects: one extra neighbor (interstitial-like)
              7重缺陷：多一个邻居（类间隙）
            - 5/7 pairs form dislocation cores
              5/7对形成位错核心
        """
        # Get coordination numbers and valid atom positions
        # 获取配位数和有效原子位置
        coord, points = self.valid_coordination()
        
        # Mask for 5-fold coordination defects
        # 5配位缺陷的掩码
        mask5 = coord == 5
        
        # Mask for 7-fold coordination defects
        # 7配位缺陷的掩码
        mask7 = coord == 7
        
        # Create figure
        # 创建图形
        plt.figure(figsize=(8, 8))
        
        # Draw density field as background
        # 绘制密度场作为背景
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        
        # Red scatter marks 5-fold defects
        # 红色散点标记5重缺陷
        plt.scatter(points[mask5, 0], points[mask5, 1], c="red", s=60, label="5-fold / 5重")
        
        # Blue scatter marks 7-fold defects
        # 蓝色散点标记7重缺陷
        plt.scatter(points[mask7, 0], points[mask7, 1], c="blue", s=60, label="7-fold / 7重")
        
        # Add legend
        # 添加图例
        plt.legend()
        
        # Plot title
        # 图标题
        plt.title("Defect Analysis / 缺陷分析")
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_defect_density(self):
        """
        Plot evolution curve of defect density over sample steps.
        绘制缺陷密度随采样步变化曲线。
        
        Shows how defect concentration evolves during grain growth
        or annealing processes.
        显示在晶粒生长或退火过程中缺陷浓度如何演化。
        
        Notes / 说明:
            - Defect density typically decreases during grain growth
              晶粒生长过程中缺陷密度通常降低
            - Uses circle markers for clarity
              使用圆形标记以提高清晰度
        """
        # Create figure
        # 创建图形
        plt.figure()
        
        # Line plot with circle markers
        # 带圆点标记的折线图
        plt.plot(self.defect_log, "o-")
        
        # Axis labels and title
        # 坐标轴标签和标题
        plt.xlabel("Sample / 采样")
        plt.ylabel("Defect Density / 缺陷密度")
        plt.title("Defect Density Evolution / 缺陷密度演化")
        
        # Add grid
        # 添加网格
        plt.grid()
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_grain_size(self):
        """
        Plot time series curve of estimated average grain size.
        绘制估算平均晶粒尺寸时序曲线。
        
        Shows grain growth kinetics during simulation.
        显示模拟过程中的晶粒生长动力学。
        
        Notes / 说明:
            - Grain size estimated from defect density
              从缺陷密度估算晶粒尺寸
            - Normal grain growth: D ~ t^(1/2)
              正常晶粒生长：D ~ t^(1/2)
        """
        # Create figure
        # 创建图形
        plt.figure()
        
        # Plot grain size evolution
        # 绘制晶粒尺寸演化
        plt.plot(self.grain_size_log, "o-")
        
        # Axis labels and title
        # 坐标轴标签和标题
        plt.xlabel("Sample / 采样")
        plt.ylabel("Grain Size / 晶粒尺寸")
        plt.title("Grain Size Evolution / 晶粒尺寸演化")
        
        # Add grid
        # 添加网格
        plt.grid()
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_structure_peak(self):
        """
        Plot evolution curve of maximum structure factor diffraction peak.
        绘制结构因子最大衍射峰值演化曲线。
        
        The peak height indicates degree of crystalline order - higher
        peak means better crystallinity.
        峰高表示晶体有序度 - 峰越高表示结晶度越好。
        
        Notes / 说明:
            - Peak grows during crystallization
              结晶过程中峰值增长
            - Saturates at equilibrium
              在平衡时饱和
        """
        # Create figure
        # 创建图形
        plt.figure()
        
        # Plot structure factor peak evolution
        # 绘制结构因子峰值演化
        plt.plot(self.structure_peak_log, "o-")
        
        # Axis labels and title
        # 坐标轴标签和标题
        plt.xlabel("Sample / 采样")
        plt.ylabel("Structure Factor Peak / 结构因子峰值")
        plt.title("Structure Factor Peak Evolution / 结构因子峰值演化")
        
        # Add grid
        # 添加网格
        plt.grid()
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_detected_atoms(self):
        """
        Draw black dots of all detected atoms over density field.
        在密度场上绘制所有识别出的原子黑点。
        
        Shows the positions of atoms identified by peak detection
        in the density field.
        显示通过密度场峰值检测识别出的原子位置。
        
        Notes / 说明:
            - atoms[:,1] = x (horizontal axis)
              atoms[:,1] = x（横轴）
            - atoms[:,0] = y (vertical axis)
              atoms[:,0] = y（纵轴）
            - Directly matches image coordinates
              直接匹配图像坐标
        """
        # Get raw [i,j] format atom pixel indices
        # 获取原始[i,j]格式原子像素索引
        atoms = self.detect_atoms()
        
        # Create figure
        # 创建图形
        plt.figure(figsize=(8, 8))
        
        # Plot density field as background
        # 绘制密度场作为背景
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        
        # Plot atoms as black dots
        # 将原子绘制成黑点
        # atoms[:,1] = x horizontal axis, atoms[:,0] = y vertical axis
        # atoms[:,1] = x横轴, atoms[:,0] = y纵轴
        # Directly matches image coordinate system
        # 直接匹配图像坐标系
        plt.scatter(atoms[:, 1], atoms[:, 0], s=10, c="k")
        
        # Plot title with atom count
        # 带原子数的图标题
        plt.title(f"Detected Atoms / 检测到的原子 ({len(atoms)})")
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_psi6(self):
        """
        Atom scatter colored plot, color mapped to magnitude of
        bond-orientational order |ψ₆|.
        原子散点着色图，颜色映射键取向有序度|ψ₆|的大小。
        
        ψ₆ (psi6) measures the six-fold bond orientational order.
        |ψ₆| = 1 means perfect hexagonal order, 0 means disordered.
        ψ₆测量六重键取向有序度。
        |ψ₆| = 1表示完美六角有序，0表示无序。
        
        Notes / 说明:
            - High |ψ₆| inside grains, low at grain boundaries
              晶粒内部|ψ₆|高，晶界处低
            - Viridis colormap for intuitive ordering visualization
              Viridis色图用于直观的有序度可视化
        """
        # Get atom coordinates and ψ6 order parameter
        # 获取原子坐标和ψ6序参量
        points, psi6 = self.compute_psi6()
        
        # Create figure
        # 创建图形
        plt.figure(figsize=(8, 8))
        
        # Plot density field as background
        # 绘制密度场作为背景
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        
        # Scatter color determined by absolute value of ψ6
        # 散点颜色由ψ6的绝对值决定
        plt.scatter(points[:, 0], points[:, 1], c=np.abs(psi6), cmap="viridis", s=40)
        
        # Add colorbar with label
        # 添加带标签的颜色条
        plt.colorbar(label="|ψ₆|")
        
        # Plot title
        # 图标题
        plt.title("Bond Orientational Order / 键取向有序度")
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_psi6_field(self):
        """
        Interpolate discrete atom ψ6 values to full continuous order heatmap.
        将离散原子ψ6值插值生成全域连续有序度云图。
        
        Creates a continuous field representation of bond-orientational
        order by interpolating from discrete atom positions.
        通过从离散原子位置插值，创建键取向有序度的连续场表示。
        
        Notes / 说明:
            - Uses linear interpolation from scipy
              使用scipy的线性插值
            - Empty regions (no atoms) filled with 0
              无原子的空区域填充为0
            - Shows spatial distribution of crystalline order
              显示晶体有序度的空间分布
        """
        # Get atom coordinates and ψ6 values
        # 获取原子坐标和ψ6值
        points, psi6 = self.compute_psi6()
        
        # Extract magnitude value |ψ6| of each atom
        # 提取每个原子的|ψ6|数值
        values = np.abs(psi6)
        
        # Generate full-domain interpolation grid X, Y
        # 生成全场插值网格X, Y
        X, Y = np.meshgrid(np.arange(self.N), np.arange(self.N))
        
        # Linear interpolation, fill empty regions with zero
        # 线性插值，无原子区域填充0
        field = griddata(points, values, (X, Y), method="linear", fill_value=0)
        
        # Create figure
        # 创建图形
        plt.figure(figsize=(8, 8))
        
        # Plot interpolated order field
        # 绘制插值后的有序度场
        plt.imshow(field, origin="lower", cmap="viridis", vmin=0, vmax=1)
        
        # Add colorbar
        # 添加颜色条
        plt.colorbar(label="|ψ₆|")
        
        # Plot title
        # 图标题
        plt.title("Psi6 Order Field / Psi6有序度场")
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_grain_orientation(self):
        """
        Atom colored scatter plot, color mapped to ψ6 phase angle
        (representing grain crystal orientation).
        原子着色散点图，颜色映射ψ6相位角（代表晶粒晶体取向）。
        
        The phase of ψ6 indicates the crystallographic orientation.
        Grains with different orientations have different colors.
        ψ6的相位表示晶体学取向。不同取向的晶粒有不同的颜色。
        
        Notes / 说明:
            - Grayscale density field as background
              灰度密度场作为背景
            - HSV colormap corresponds to 0~2π angles
              HSV色图对应0~2π角度
            - Same color = same grain orientation
              相同颜色 = 相同晶粒取向
        """
        # Get atom coordinates and ψ6 values
        # 获取原子坐标和ψ6值
        points, psi6 = self.compute_psi6()
        
        # Create figure
        # 创建图形
        plt.figure(figsize=(8, 8))
        
        # Grayscale density field as background
        # 灰度密度场作为背景
        plt.imshow(self.phi, cmap="gray", origin="lower")
        
        # Color determined by argument of ψ6
        # 颜色由ψ6的辐角决定
        # HSV colormap corresponds to angle 0~2π
        # HSV色图对应0~2π角度
        plt.scatter(points[:, 0], points[:, 1], c=np.angle(psi6), cmap="hsv", s=40)
        
        # Add colorbar
        # 添加颜色条
        plt.colorbar(label="arg(ψ₆)")
        
        # Plot title
        # 图标题
        plt.title("Grain Orientation / 晶粒取向")
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_grain_boundary_atoms(self):
        """
        Highlight grain boundary atoms with large D value;
        light green = all atoms, red = grain boundary atoms.
        高亮标记高D值晶界原子；浅绿 = 全部原子，红色 = 晶界原子。
        
        Uses the D-parameter (disorder parameter) to identify atoms
        at grain boundaries, which have higher disorder.
        使用D参数（无序度参数）识别晶界处的原子，这些原子具有更高的无序度。
        
        Notes / 说明:
            - 85th percentile of D values as grain boundary threshold
              D值的85百分位数作为晶界判定阈值
            - Atoms with D > threshold are grain boundary atoms
              D大于阈值的原子是晶界原子
            - Semi-transparent green for all atoms
              半透明绿色表示所有原子
        """
        # Get atom positions and D-parameter values
        # 获取原子位置和D参数值
        points, D = self.grain_boundary_parameter()
        
        # Take 85th percentile of D values as grain boundary discrimination threshold
        # 取D值的85百分位数作为晶界判定阈值
        threshold = np.percentile(D, 85)
        
        # Atoms with D larger than threshold are grain boundary atoms
        # D大于阈值的原子是晶界原子
        gb_mask = D > threshold
        
        # Create figure
        # 创建图形
        plt.figure(figsize=(8, 8))
        
        # Plot density field as grayscale background
        # 绘制灰度密度场作为背景
        plt.imshow(self.phi, cmap='gray', origin='lower')
        
        # Semi-transparent light green draw all atoms
        # 半透明浅绿色绘制全部原子
        plt.scatter(points[:, 0], points[:, 1], s=20, c='lime', alpha=0.3, label='all atoms / 全部原子')
        
        # Enlarged red dots draw grain boundary atoms
        # 放大的红点绘制晶界原子
        plt.scatter(points[gb_mask, 0], points[gb_mask, 1], s=80, c='red', label='grain boundary / 晶界')
        
        # Add legend
        # 添加图例
        plt.legend()
        
        # Plot title
        # 图标题
        plt.title("Grain Boundary Atoms / 晶界原子")
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def plot_elastic_curve(self, strain, energy):
        """
        Plot elastic energy-strain curve with quadratic fit.
        绘制弹性能量-应变曲线及二次拟合。
        
        Shows the parabolic energy-strain relationship and extracts
        the elastic constant from the fit.
        显示抛物线形的能量-应变关系，并从拟合中提取弹性常数。
        
        Args / 参数:
            strain (ndarray): Strain values / 应变值
            energy (ndarray): Corresponding free energy values / 对应的自由能值
        
        Returns / 返回值:
            Prints elastic constant and residual strain to console
            向控制台打印弹性常数和残余应变
        """
        # Fit elastic constant from energy-strain data
        # 从能量-应变数据拟合弹性常数
        C, esp_r, coef, strain_fit, energy_fit = self.fit_elastic_constant(strain, energy)
        
        # Create figure
        # 创建图形
        plt.figure(figsize=(6, 5))
        
        # Scatter plot of PFC data points
        # PFC数据点散点图
        plt.scatter(strain, energy, s=60, label="PFC data / PFC数据")
        
        # Plot quadratic fit curve
        # 绘制二次拟合曲线
        plt.plot(strain_fit, energy_fit, lw=2, label="Quadratic Fit / 二次拟合")
        
        # Axis labels and title
        # 坐标轴标签和标题
        plt.xlabel("Strain / 应变 ε")
        plt.ylabel("Free Energy / 自由能 F")
        plt.title("Elastic Energy Curve / 弹性能量曲线")
        
        # Add legend
        # 添加图例
        plt.legend()
        
        # Add grid
        # 添加网格
        plt.grid(True)
        
        # Display the plot
        # 显示图形
        plt.show()
        
        # Print elastic properties to console
        # 向控制台打印弹性性质
        print()
        print(f"Elastic constant C / 弹性常数 = {C:.6e}")
        print(f"Residual strain ε_r / 残余应变 = {esp_r:.6e}")
        print(
            f"F(ε) = "
            f"{coef[0]:.6e} ε² + "
            f"{coef[1]:.6e} ε + "
            f"{coef[2]:.6e}"
        )
        
    def plot_stress_strain(
        self,
        strain,
        energy
    ):
        """
        Plot stress-strain curve.
        绘制应力-应变曲线。
        
        Stress is computed as numerical derivative of energy w.r.t. strain.
        The slope of this curve is the elastic modulus.
        应力计算为能量对应变的数值导数。该曲线的斜率是弹性模量。
        
        Args / 参数:
            strain (ndarray): Strain values / 应变值
            energy (ndarray): Corresponding free energy values / 对应的自由能值
        """
        # Compute stress from energy-strain data
        # 从能量-应变数据计算应力
        stress = np.gradient(
            energy,
            strain
        )
        
        # Create figure
        # 创建图形
        plt.figure(figsize=(6, 5))
        
        # Plot stress-strain curve with markers
        # 绘制带标记的应力-应变曲线
        plt.plot(
            strain,
            stress,
            "o-"
        )
        
        # Axis labels and title
        # 坐标轴标签和标题
        plt.xlabel("Strain / 应变 ε")
        plt.ylabel("Stress / 应力 σ")
        plt.title("Stress-Strain Curve / 应力-应变曲线")
        
        # Add grid
        # 添加网格
        plt.grid(True)
        
        # Display the plot
        # 显示图形
        plt.show()
        
    def analyze_psi6(self):
        """
        Full ψ6 analysis pipeline: print local & global order metrics
        + full suite of orientation plots.
        ψ6完整分析流水线：打印局部/全局有序度数值 + 全套取向绘图。
        
        This method provides a comprehensive analysis of bond-orientational
        order in the system.
        该方法提供系统中键取向有序度的综合分析。
        
        Outputs / 输出:
            - Printed psi6_local and psi6_global values
              打印psi6_local和psi6_global值
            - Four plots: ψ6 magnitude, grain orientation, ψ6 field,
              grain boundary atoms
              四张图：ψ6模长、晶粒取向、ψ6场、晶界原子
        """
        # Calculate local and global averaged modulus of ψ6
        # 计算局部和全局平均ψ6模长
        psi6_local, psi6_global = self.global_psi6()
        
        # Print order metrics to console
        # 向控制台打印有序度数值
        print(f"Psi6_local / 局部平均 = {psi6_local:.4f}")
        print(f"Psi6_global / 全局平均 = {psi6_global:.4f}")
        
        # Plot four ψ6 related figures sequentially
        # 依次绘制四张ψ6相关图
        self.plot_psi6()           # ψ6 magnitude / ψ6模长
        self.plot_grain_orientation()  # Grain orientation / 晶粒取向
        self.plot_psi6_field()     # Continuous ψ6 field / 连续ψ6场
        self.plot_grain_boundary_atoms()  # Grain boundary atoms / 晶界原子
        
    def postprocess(self):
        """
        Full post-processing plotting pipeline, plot execution order
        identical to original code.
        全套后处理绘图流程，绘图顺序与原始代码完全一致。
        
        This method runs a comprehensive set of analysis plots after
        simulation completion.
        该方法在模拟完成后运行一套全面的分析图。
        
        Plots generated / 生成的图:
            1. Energy evolution / 能量演化
            2. Density field / 密度场
            3. Structure factor / 结构因子
            4. Voronoi analysis / Voronoi分析
            5. Defect statistics / 缺陷统计
            6. Defect visualization / 缺陷可视化
            7. Defect density evolution / 缺陷密度演化
            8. Grain size evolution / 晶粒尺寸演化
            9. Structure factor peak evolution / 结构因子峰值演化
            10. Detected atoms / 检测到的原子
        """
        # Plot energy evolution
        # 绘制能量演化
        self.plot_energy()
        
        # Plot final density field
        # 绘制最终密度场
        self.plot_field()
        
        # Plot structure factor
        # 绘制结构因子
        self.plot_structure_factor()
        
        # Plot Voronoi tessellation
        # 绘制Voronoi剖分
        self.plot_voronoi()
        
        # Print defect statistics
        # 打印缺陷统计
        self.defect_statistics()
        
        # Plot defect visualization
        # 绘制缺陷可视化
        self.plot_defects()
        
        # Plot defect density evolution
        # 绘制缺陷密度演化
        self.plot_defect_density()
        
        # Plot grain size evolution
        # 绘制晶粒尺寸演化
        self.plot_grain_size()
        
        # Plot structure factor peak evolution
        # 绘制结构因子峰值演化
        self.plot_structure_peak()
        
        # Plot detected atoms
        # 绘制检测到的原子
        self.plot_detected_atoms()
