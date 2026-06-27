import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import voronoi_plot_2d
import numpy as np

class PFCPlot:
    
    def plot_energy(self):
        """
        绘制自由能随采样步演化曲线
        Plot free energy evolution curve against sample step index
        """
        plt.figure(figsize=(6,4))
        # 绘制能量时序曲线
        # Plot time series curve of free energy
        plt.plot(self.energy_log)
        plt.xlabel("Sample")
        plt.ylabel("Free Energy")
        plt.title("Energy Evolution")
        plt.grid(True)
        plt.show()
        
    def plot_field(self):
        """
        绘制2D密度场φ云图
        Plot 2D heatmap of density field φ
        """
        plt.figure(figsize=(6,6))
        # origin="lower"匹配网格ij索引，图像左下角对应坐标原点
        # origin="lower" match grid ij index, bottom-left of image is coordinate origin
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        plt.colorbar()
        plt.title("Density Field")
        plt.show()
        
    def plot_structure_factor(self):
        """
        绘制对数尺度静态结构因子衍射图案
        Plot log-scaled static structure factor diffraction pattern
        """
        # 计算结构因子
        # Calculate structure factor
        S = self.structure_factor()
        plt.figure(figsize=(6,6))
        # log10压缩动态范围，+1避免log10(0)无意义
        # log10 compress dynamic range, +1 to avoid log10(0) undefined
        plt.imshow(np.log10(S + 1.0), origin="lower", cmap="inferno")
        plt.colorbar(label="log10(S+1)")
        plt.title("Structure Factor")
        plt.show()
        
    def plot_voronoi(self):
        """
        绘制原子Voronoi元胞拓扑图
        Plot Voronoi tessellation topology of crystal atoms
        """
        # 生成Voronoi拓扑与过滤后原子坐标
        # Generate Voronoi topology and filtered atom coordinates
        vor, _ = self.voronoi_analysis()
        plt.figure(figsize=(8,8))
        # 绘制沃罗诺伊图，关闭顶点标记、设置线条与点尺寸
        # Plot voronoi diagram, hide vertices, set line width and point size
        voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_width=1, point_size=2)
        # Y轴反转，匹配imshow绘图坐标系
        # Invert Y axis to align with imshow plotting coordinate system
        plt.gca().invert_yaxis()
        plt.title("Voronoi Analysis")
        plt.show()
        
    def plot_defects(self):
        """
        在密度场云图上标记5重(红)、7重(蓝)配位缺陷原子
        Mark 5-fold(red) /7-fold(blue) coordination defect atoms over density heatmap
        """
        coord, points = self.valid_coordination()
        # 5配位缺陷掩码
        # Mask for 5-fold coordination defects
        mask5 = coord == 5
        # 7配位缺陷掩码
        # Mask for 7-fold coordination defects
        mask7 = coord == 7
        plt.figure(figsize=(8,8))
        # 底层绘制密度场
        # Draw density field as background
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        # 红色散点标记5重缺陷
        # Red scatter mark 5-fold defects
        plt.scatter(points[mask5,0], points[mask5,1], c="red", s=60, label="5-fold")
        # 蓝色散点标记7重缺陷
        # Blue scatter mark 7-fold defects
        plt.scatter(points[mask7,0], points[mask7,1], c="blue", s=60, label="7-fold")
        plt.legend()
        plt.title("Defect Analysis")
        plt.show()
        
    def plot_defect_density(self):
        """
        绘制缺陷密度随采样步变化曲线
        Plot evolution curve of defect density over sample steps
        """
        plt.figure()
        # 带圆点标记的折线图
        # Line plot with circle markers
        plt.plot(self.defect_log, "o-")
        plt.xlabel("sample")
        plt.ylabel("Defect Density")
        plt.title("Defect Density Evolution")
        plt.grid()
        plt.show()
        
    def plot_grain_size(self):
        """
        绘制估算平均晶粒尺寸时序曲线
        Plot time series curve of estimated average grain size
        """
        plt.figure()
        plt.plot(self.grain_size_log, "o-")
        plt.xlabel("sample")
        plt.ylabel("Grain Size")
        plt.title("Grain Size Evolution")
        plt.grid()
        plt.show()
        
    def plot_structure_peak(self):
        """
        绘制结构因子最大衍射峰值演化曲线
        Plot evolution curve of maximum structure factor diffraction peak
        """
        plt.figure()
        plt.plot(self.structure_peak_log, "o-")
        plt.xlabel("sample")
        plt.ylabel("Structure Factor Peak")
        plt.title("Structure Factor Peak Evolution")
        plt.grid()
        plt.show()
        
    def plot_detected_atoms(self):
        """
        在密度场上绘制所有识别出的原子黑点
        Draw black dots of all detected atoms over density field
        """
        # 获取原始[i,j]格式原子像素索引
        # Get raw [i,j] format atom pixel indices
        atoms = self.detect_atoms()
        plt.figure(figsize=(8,8))
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        # atoms[:,1]=x横轴, atoms[:,0]=y纵轴，直接匹配图像坐标
        # atoms[:,1]=x horizontal axis, atoms[:,0]=y vertical axis, directly match image coordinate
        plt.scatter(atoms[:,1], atoms[:,0], s=10, c="k")
        plt.title(f"Detected atoms ({len(atoms)})")
        plt.show()
        
    def plot_psi6(self):
        """
        原子散点着色图，颜色映射|ψ6|键取向有序度大小
        Atom scatter colored plot, color mapped to magnitude of bond-orientational order |ψ6|
        """
        # 获取原子坐标与ψ6序参量
        # Load atom coordinates and ψ6 order parameter
        points, psi6 = self.compute_psi6()
        plt.figure(figsize=(8,8))
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        # 散点颜色由|ψ6|决定
        # Scatter color determined by absolute value of ψ6
        plt.scatter(points[:,0], points[:,1], c=np.abs(psi6), cmap="viridis", s=40)
        plt.colorbar(label="|psi6|")
        plt.title("Bond Orientational Order")
        plt.show()
        
    def plot_psi6_field(self):
        """
        散乱原子ψ6插值生成全域连续有序度云图
        Interpolate discrete atom ψ6 values to full continuous order heatmap
        """
        points, psi6 = self.compute_psi6()
        # 提取每个原子|ψ6|数值
        # Extract magnitude value |ψ6| of each atom
        values = np.abs(psi6)
        # 生成全场插值网格XY
        # Generate full-domain interpolation grid X,Y
        X, Y = np.meshgrid(np.arange(self.N), np.arange(self.N))
        # 线性插值，无原子区域填充0
        # Linear interpolation, fill empty region with zero
        field = griddata(points, values, (X,Y), method="linear", fill_value=0)
        plt.figure(figsize=(8,8))
        plt.imshow(field, origin="lower", cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(label="|psi6|")
        plt.title("Psi6 Order Field")
        plt.show()
        
    def plot_grain_orientation(self):
        """
        原子着色图，颜色映射ψ6相位角（代表晶粒晶体取向）
        Atom colored scatter plot, color mapped to ψ6 phase angle (represent grain crystal orientation)
        """
        points, psi6 = self.compute_psi6()
        plt.figure(figsize=(8,8))
        # 灰度密度场作为底图
        # Grayscale density field as background
        plt.imshow(self.phi, cmap="gray", origin="lower")
        # 颜色由ψ6辐角决定，hsv色环对应0~2π角度
        # Color determined by argument of ψ6, hsv colormap correspond to angle 0~2π
        plt.scatter(points[:,0], points[:,1], c=np.angle(psi6), cmap="hsv", s=40)
        plt.colorbar(label="arg(psi6)")
        plt.title("Grain Orientation")
        plt.show()
        
    def plot_grain_boundary_atoms(self):
        """
        高亮标记高D值晶界原子，浅绿为全部原子，红色为晶界原子
        Highlight grain boundary atoms with large D value; light green = all atoms, red = grain boundary atoms
        """
        points, D = self.grain_boundary_parameter()
        # 取D值85百分位数作为晶界判定阈值
        # Take 85th percentile of D values as grain boundary discrimination threshold
        threshold = np.percentile(D, 85)
        # D大于阈值判定为晶界原子
        # Atoms with D larger than threshold are grain boundary atoms
        gb_mask = D > threshold
        plt.figure(figsize=(8,8))
        plt.imshow(self.phi, cmap='gray', origin='lower')
        # 半透明浅绿色绘制全部原子
        # Semi-transparent light green draw all atoms
        plt.scatter(points[:,0], points[:,1], s=20, c='lime', alpha=0.3, label='all atoms')
        # 红色放大绘制晶界原子
        # Enlarged red dots draw grain boundary atoms
        plt.scatter(points[gb_mask,0], points[gb_mask,1], s=80, c='red', label='grain boundary')
        plt.legend()
        plt.title("Grain Boundary Atoms")
        plt.show()
        
    def plot_elastic_curve(self, strain, energy):
        C, esp_r, coef, strain_fit, energy_fit = (self.fit_elastic_constant(strain, energy))
        plt.figure(figsize=(6,5))
        plt.scatter(strain, energy, s = 60, label = "PFC data")
        plt.plot(strain_fit, energy_fit, lw = 2, label = "Quadratic Fit")
        plt.xlabel("Strain")
        plt.ylabel("Free Energy")
        plt.title("Elastic Energy Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
        print()
        print(f"Elastic constant C = {C:.6e}")
        print(f"Residual strain ε_r = {esp_r:.6e}")
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

        stress = np.gradient(
            energy,
            strain
        )

        plt.figure(figsize=(6,5))

        plt.plot(
            strain,
            stress,
            "o-"
        )

        plt.xlabel("Strain")
        plt.ylabel("Stress")

        plt.title(
            "Stress-Strain Curve"
        )

        plt.grid(True)

        plt.show()
        
        
    def analyze_psi6(self):
        """
        ψ6完整分析流水线：打印局部/全局有序度数值 + 全套取向绘图
        Full ψ6 analysis pipeline: print local & global order metrics + full suite of orientation plots
        """
        # 计算局部、全局平均ψ6模长
        # Calculate local and global averaged modulus of ψ6
        psi6_local, psi6_global = self.global_psi6()
        # 控制台打印有序度数值
        # Print order metrics to console
        print(f"Psi6_local = {psi6_local:.4f}")
        print(f"Psi6_global = {psi6_global:.4f}")
        # 依次绘制四张ψ6相关图
        # Plot four ψ6 related figures sequentially
        self.plot_psi6()
        self.plot_grain_orientation()
        self.plot_psi6_field()
        self.plot_grain_boundary_atoms()
        
    def postprocess(self):
        """
        全套后处理绘图流程，绘图顺序与原始代码完全一致
        Full post-processing plotting pipeline, plot execution order identical to original code
        """
        self.plot_energy()
        self.plot_field()
        self.plot_structure_factor()
        self.plot_voronoi()
        self.defect_statistics()
        self.plot_defects()
        self.plot_defect_density()
        self.plot_grain_size()
        self.plot_structure_peak()
        self.plot_detected_atoms()