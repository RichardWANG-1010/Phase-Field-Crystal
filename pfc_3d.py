"""
pfc_3d.py
三维Phase Field Crystal (PFC) 模拟求解器
3D Phase Field Crystal Simulation Solver
参考: Provatas & Elder, Chapter 8 - Phase Field Crystal Modeling of Pure Materials
      支持BCC晶格 (Section 8.4.1)
Reference: Provatas & Elder, Chapter 8 - Phase Field Crystal Modeling of Pure Materials
           Supports BCC lattice (Section 8.4.1)
作者: Jinpeng Wang
Author: Jinpeng Wang
"""
import numpy as np
import scipy.fft as fft
from pathlib import Path


class PFCBase3D:
    """
    三维PFC基础类：网格、波数空间、IO基础设施
    继承并扩展了原二维PFCBase的核心思想到三维

    3D PFC base class: grid, wavenumber space, IO infrastructure
    Inherits and extends the core ideas of the original 2D PFCBase to 3D
    """
    def __init__(
        self,
        N=64,           # 三维网格尺寸 (N x N x N) / 3D grid size (N x N x N)
        L=64.0,         # 物理域尺寸 / Physical domain size
        dt=0.05,        # 时间步长 / Time step
        T=500.0         # 总模拟时间 / Total simulation time
    ):
        # 网格参数
        # Grid parameters
        self.N = int(N)
        self.L = float(L)
        self.L0 = self.L
        self.dx = self.L / self.N
        self.volume = self.L ** 3
        # 时间参数
        # Time parameters
        self.dt = float(dt)
        self.T = float(T)
        self.steps = int(np.ceil(self.T / self.dt))
        # 日志
        # Logs
        self.energy_log = []
        self.mass_log = []
        # IO
        # IO
        self.record_video = True
        self.frame_count = 0
        self.frame_dir = "frames_3d"
        Path(self.frame_dir).mkdir(exist_ok=True)
        # 构建三维k空间
        # Build 3D k-space
        self._build_kspace_3d()

    def _build_kspace_3d(self):
        """
        预生成三维波数网格 kx, ky, kz
        对应书中Eq. 8.2的频域算子

        Pre-generate 3D wavenumber grids kx, ky, kz
        Corresponds to the frequency-domain operator in Eq. 8.2 of the book
        """
        k = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        self.KX, self.KY, self.KZ = np.meshgrid(k, k, k, indexing="ij")
        # 三维波数平方 |k|² = kx² + ky² + kz²
        # 3D wavenumber squared |k|² = kx² + ky² + kz²
        self.k2 = self.KX**2 + self.KY**2 + self.KZ**2
        # 保存各分量供可能的各向异性晶格使用
        # Save individual components for possible anisotropic lattices
        self.kx = self.KX
        self.ky = self.KY
        self.kz = self.KZ


class PFCAnalysis3D:
    """
    三维PFC分析工具
    包含结构因子、原子检测等三维扩展版本

    3D PFC analysis tools
    Includes 3D extended versions of structure factor, atom detection, etc.
    """
    def sample_observables(self, step):
        """
        采样可观测量
        Sample observables
        """
        E = self.compute_energy()
        self.energy_log.append(E)
        self.mass_log.append(np.mean(self.phi))
        return E

    def structure_factor_3d(self):
        """
        三维静态结构因子 S(k)
        对应书中Section 8.4.1的衍射分析
        BCC晶格应在k空间显示(110), (200), (211)等峰

        3D static structure factor S(k)
        Corresponds to diffraction analysis in Section 8.4.1 of the book
        BCC lattice should show peaks at (110), (200), (211), etc. in k-space
        """
        phi_fluct = self.phi - np.mean(self.phi)
        phi_hat = np.fft.fftshift(np.fft.fftn(phi_fluct))
        S = np.abs(phi_hat)**2
        return S

    def detect_atoms_3d(self):
        """
        三维局部峰值检测
        使用skimage的peak_local_max三维版本

        3D local peak detection
        Uses the 3D version of skimage's peak_local_max
        """
        from skimage.feature import peak_local_max
        atoms = peak_local_max(
            self.phi,
            min_distance=5,
            threshold_rel=0.3,
            exclude_border=False
        )
        # 返回 [n_atoms, 3] 数组，每行 [i, j, k]
        # Returns [n_atoms, 3] array, each row [i, j, k]
        return atoms

    def get_slice(self, axis=0, index=None):
        """
        获取三维场的二维切片用于可视化
        Get 2D slice of 3D field for visualization
        axis: 0=x, 1=y, 2=z
        index: 切片位置，None则取中间 / slice position, None for middle
        """
        if index is None:
            index = self.N // 2
        if axis == 0:
            return self.phi[index, :, :]
        elif axis == 1:
            return self.phi[:, index, :]
        else:
            return self.phi[:, :, index]


class PFCPlot3D:
    """
    三维PFC可视化工具
    包含三维体绘制、切片、等值面等

    3D PFC visualization tools
    Includes 3D volume rendering, slicing, isosurfaces, etc.
    """
    def plot_field_slice(self, axis=0, index=None, cmap="coolwarm", save_path=None):
        """
        绘制二维切片云图
        Plot 2D slice heatmap
        """
        import matplotlib.pyplot as plt
        slice_2d = self.get_slice(axis, index)
        plt.figure(figsize=(7, 6))
        plt.imshow(slice_2d, cmap=cmap, origin="lower")
        plt.colorbar(label=r"$\phi$")
        axis_names = ["x", "y", "z"]
        plt.title(f"Density Field Slice (axis={axis_names[axis]})")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_structure_factor_slice(self, axis=0, index=None, save_path=None):
        """
        绘制结构因子的对数尺度切片
        可观察BCC衍射斑点

        Plot log-scale slice of structure factor
        BCC diffraction spots can be observed
        """
        import matplotlib.pyplot as plt
        S = self.structure_factor_3d()
        if index is None:
            index = self.N // 2
        if axis == 0:
            S_slice = S[index, :, :]
        elif axis == 1:
            S_slice = S[:, index, :]
        else:
            S_slice = S[:, :, index]
        plt.figure(figsize=(7, 6))
        plt.imshow(np.log10(S_slice + 1), origin="lower", cmap="inferno")
        plt.colorbar(label="log10(S+1)")
        plt.title("Structure Factor Slice (log scale)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_isosurface(self, level=None, save_path=None):
        """
        绘制三维等值面
        需要安装pyvista: pip install pyvista

        Plot 3D isosurface
        Requires pyvista: pip install pyvista
        """
        try:
            import pyvista as pv
        except ImportError:
            print("pyvista not installed. Falling back to matplotlib isosurface.")
            self._plot_isosurface_mpl(level, save_path)
            return
        if level is None:
            level = np.mean(self.phi)
        # 创建均匀网格
        # Create uniform grid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(self.phi.shape) + 1
        grid.spacing = (self.dx, self.dx, self.dx)
        grid.origin = (0, 0, 0)
        grid.cell_data["phi"] = self.phi.flatten(order="F")
        # 提取等值面
        # Extract isosurface
        contour = grid.contour([level])
        # 绘制
        # Plot
        plotter = pv.Plotter(off_screen=save_path is not None)
        plotter.add_mesh(contour, cmap="coolwarm", opacity=0.8)
        plotter.add_axes()
        plotter.add_title(f"Isosurface φ = {level:.3f}")
        if save_path:
            plotter.screenshot(save_path)
        else:
            plotter.show()

    def _plot_isosurface_mpl(self, level=None, save_path=None):
        """
        使用matplotlib的3D等值面（备用方案）
        3D isosurface using matplotlib (fallback)
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from skimage.measure import marching_cubes
        if level is None:
            level = np.mean(self.phi)
        verts, faces, _, _ = marching_cubes(self.phi, level=level)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # 绘制三角面片
        # Plot triangular faces
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        cmap="coolwarm", alpha=0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Isosurface φ = {level:.3f}")
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_atoms_3d(self, save_path=None):
        """
        三维散点图显示检测到的原子位置
        3D scatter plot showing detected atom positions
        """
        import matplotlib.pyplot as plt
        atoms = self.detect_atoms_3d()
        if len(atoms) == 0:
            print("No atoms detected.")
            return
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # 原子坐标
        # Atom coordinates
        x, y, z = atoms[:, 2], atoms[:, 1], atoms[:, 0]
        # 根据局部密度着色
        # Color by local density
        colors = self.phi[atoms[:, 0], atoms[:, 1], atoms[:, 2]]
        scatter = ax.scatter(x, y, z, c=colors, cmap="coolwarm", s=20, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label=r"$\phi$")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Detected Atoms (n={len(atoms)})")
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_energy(self):
        """
        绘制自由能演化
        Plot free energy evolution
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(self.energy_log)
        plt.xlabel("Sample")
        plt.ylabel("Free Energy")
        plt.title("Energy Evolution (3D)")
        plt.grid(True)
        plt.show()

    def capture_frame_3d(self, step):
        """
        保存三维可视化帧用于后续生成视频
        Save 3D visualization frames for subsequent video generation
        """
        import matplotlib.pyplot as plt
        if not self.record_video:
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # 三个正交切片
        # Three orthogonal slices
        for idx, (ax, axis) in enumerate(zip(axes.flat[:3], [0, 1, 2])):
            slice_2d = self.get_slice(axis)
            im = ax.imshow(slice_2d, cmap="coolwarm", origin="lower")
            ax.set_title(f"Slice axis={['x','y','z'][axis]}")
            plt.colorbar(im, ax=ax, fraction=0.046)
        # 结构因子切片
        # Structure factor slice
        S = self.structure_factor_3d()
        im = axes[1, 1].imshow(np.log10(S[self.N//2, :, :] + 1),
                                origin="lower", cmap="inferno")
        axes[1, 1].set_title("Structure Factor (k-space)")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        plt.suptitle(f"Step {step}, E={self.compute_energy():.4e}")
        plt.tight_layout()
        frame_path = f"{self.frame_dir}/frame_{self.frame_count:06d}.png"
        plt.savefig(frame_path, dpi=100)
        plt.close()
        self.frame_count += 1

    def frames_to_video_3d(self, fps=10, output="pfc_3d.mp4"):
        """
        将帧合成为视频
        Synthesize frames into video
        """
        import subprocess
        import glob
        import os
        frames = sorted(glob.glob(f"{self.frame_dir}/frame_*.png"))
        if len(frames) == 0:
            print("No frames to convert.")
            return
        print(f"Found {len(frames)} frames in '{self.frame_dir}'")
        print(f"First frame: {frames[0]}")
        print(f"Last frame:  {frames[-1]}")
        # 使用绝对路径避免 Windows 路径问题
        # Use absolute paths to avoid Windows path issues
        frame_pattern = os.path.abspath(os.path.join(self.frame_dir, "frame_%06d.png"))
        output_path = os.path.abspath(output)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", frame_pattern,
            "-pix_fmt", "yuv420p", "-crf", "18",
            output_path
        ]
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Video saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed with return code {e.returncode}")
            print(f"STDERR: {e.stderr}")
            print(f"Frames remain in {self.frame_dir}")
        except FileNotFoundError:
            print("ffmpeg command not found. Please ensure ffmpeg is in your PATH.")
            print(f"Frames remain in {self.frame_dir}")


class PFCElastic3D:
    """
    三维PFC弹性计算
    参考书中Section 8.5: Elastic Constants of PFC Model

    3D PFC elastic calculations
    Reference: Section 8.5 of the book: Elastic Constants of PFC Model
    """
    def apply_strain_3d(self, strain_tensor):
        """
        应用三维应变张量
        Apply 3D strain tensor
        strain_tensor: 3x3 对称应变张量 [e_xx, e_xy, e_xz; e_xy, e_yy, e_yz; e_xz, e_yz, e_zz]
                       3x3 symmetric strain tensor
        对于简单单轴应变，可传入对角张量 diag(eps, -nu*eps, -nu*eps)
        For simple uniaxial strain, you can pass diagonal tensor diag(eps, -nu*eps, -nu*eps)
        其中nu为泊松比
        where nu is Poisson's ratio
        """
        # 计算变形后的晶格向量
        # Calculate deformed lattice vectors
        F = np.eye(3) + strain_tensor  # 变形梯度 / deformation gradient
        # 更新物理域尺寸
        # Update physical domain size
        self.L = self.L0 * np.diag(F)
        self.dx = self.L / self.N
        # 重新构建k空间
        # Rebuild k-space
        self._build_kspace_3d()
        return F

    def apply_uniaxial_strain(self, eps, direction=0, nu=0.33):
        """
        应用单轴应变（考虑泊松效应）
        Apply uniaxial strain (considering Poisson effect)
        direction: 0=x, 1=y, 2=z
        """
        strain = np.zeros((3, 3))
        # 主应变
        # Principal strain
        strain[direction, direction] = eps
        # 横向收缩（泊松效应）
        # Transverse contraction (Poisson effect)
        for i in range(3):
            if i != direction:
                strain[i, i] = -nu * eps
        return self.apply_strain_3d(strain)

    def save_reference_state_3d(self):
        """
        保存参考状态
        Save reference state
        """
        self.phi_ref = self.phi.copy()
        self.L_ref = self.L.copy()
        self.dx_ref = self.dx.copy()

    def elastic_energy_curve_3d(self, strain_list, direction=0, nu=0.33, relax_steps=2000):
        """
        计算单轴应变下的弹性能量曲线
        Calculate elastic energy curve under uniaxial strain
        返回: (energy_array, phi_list)
        Returns: (energy_array, phi_list)
        """
        phi_ref = self.phi.copy()
        L_ref = self.L.copy()
        dx_ref = self.dx.copy()
        energy = []
        phi_list = []
        for eps in strain_list:
            # 恢复参考状态
            # Restore reference state
            self.phi = phi_ref.copy()
            self.L = L_ref.copy()
            self.dx = dx_ref.copy()
            self._build_kspace_3d()
            # 应用应变
            # Apply strain
            self.apply_uniaxial_strain(eps, direction, nu)
            # 弛豫
            # Relax
            for _ in range(relax_steps):
                self.step()
            energy.append(self.compute_energy())
            phi_list.append(self.phi.copy())
        return np.array(energy), phi_list

    def fit_elastic_constant_3d(self, strain, energy):
        """
        拟合三维弹性常数
        Fit 3D elastic constant
        对于单轴应变，E = d²F/de²
        For uniaxial strain, E = d²F/de²
        """
        coef = np.polyfit(strain, energy, 2)
        strain_fit = np.linspace(strain.min(), strain.max(), 200)
        energy_fit = np.polyval(coef, strain_fit)
        # 对于单轴应变，C = d²F/de²
        # For uniaxial strain, C = d²F/de²
        C = 2 * coef[0]
        eps_r = -coef[1] / (2 * coef[0])
        return C, eps_r, coef, strain_fit, energy_fit

    def compute_stress_3d(self, strain, energy):
        """
        计算应力-应变关系
        Compute stress-strain relationship
        """
        stress = np.gradient(energy, strain)
        return stress

    def compute_bulk_modulus(self, strain_vol, energy):
        """
        计算体模量 K = V d²F/dV²
        需要体积应变数据

        Compute bulk modulus K = V d²F/dV²
        Requires volumetric strain data
        """
        # 体积应变 ev = exx + eyy + ezz
        # Volumetric strain ev = exx + eyy + ezz
        # 对于均匀膨胀: ev = 3*eps
        # For uniform expansion: ev = 3*eps
        coef = np.polyfit(strain_vol, energy, 2)
        # 简化的体模量估计
        # Simplified bulk modulus estimate
        K = 2 * coef[0]
        return K


class PurePFC3DSolver(PFCBase3D, PFCAnalysis3D, PFCPlot3D, PFCElastic3D):
    """
    三维纯材料PFC求解器
    3D pure material PFC solver
    支持晶格类型:
    Supported lattice types:
    - "bcc": 体心立方 (Body-Centered Cubic) - 第8章标准PFC / Chapter 8 standard PFC
    - "fcc": 面心立方 (Face-Centered Cubic) - 可通过修改色散实现 / can be implemented by modifying dispersion
    - "sc": 简单立方 (Simple Cubic)
    动力学方程 (书中Eq. 8.3):
    Dynamic equation (Eq. 8.3 in book):
        ∂φ/∂t = ∇² · δF/δφ
    其中自由能泛函 (Eq. 8.1):
    where free energy functional (Eq. 8.1):
        F = ∫ [φ/2 · (r + (1 + ∇²)²) φ + φ⁴/4] dr
    半隐式傅里叶迭代:
    Semi-implicit Fourier iteration:
        φ̂^{n+1} = [φ̂^n - dt · M · k² · (φ³)̂] / [1 + dt · M · k² · L(k)]
        L(k) = (1 - k²)² + r
    """
    def __init__(
        self,
        N=64,
        L=64.0,
        r=-0.25,        # 温度参数 (书中为ε，但代码沿用r约定) / temperature parameter (ε in book, but code uses r convention)
        M=1.0,          # 迁移率 / mobility
        dt=0.05,
        T=500.0,
        phi0=-0.25,     # 平均密度 / average density
        noise_amp=0.01,
        lattice_type="bcc"
    ):
        # 初始化基础类
        # Initialize base classes
        PFCBase3D.__init__(self, N=N, L=L, dt=dt, T=T)
        self.r = float(r)
        self.M = float(M)
        self.phi0 = float(phi0)
        self.noise_amp = float(noise_amp)
        self.lattice_type = lattice_type
        # 初始化密度场
        # Initialize density field
        self._initialize_field_3d()

    def _initialize_field_3d(self):
        """
        生成带高斯噪声的三维初始密度场
        修正均值保证初始质量守恒

        Generate 3D initial density field with Gaussian noise
        Correct mean to ensure initial mass conservation
        """
        noise = self.noise_amp * np.random.randn(self.N, self.N, self.N)
        self.phi = self.phi0 + noise
        # 强制均值等于phi0
        # Force mean to equal phi0
        current_mean = np.mean(self.phi)
        self.phi -= (current_mean - self.phi0)
        print(f"Initial mean density = {np.mean(self.phi):.6f}")
        print(f"Grid: {self.N}³, Volume: {self.volume:.1f}")

    def step(self):
        """
        单步三维傅里叶半隐式欧拉迭代
        Single step 3D Fourier semi-implicit Euler iteration
        标准BCC晶格色散 (书中Eq. 8.4):
        Standard BCC lattice dispersion (Eq. 8.4 in book):
            L(k) = (1 - |k|²)² + r
        对于其他晶格可扩展:
        Can be extended for other lattices:
        - FCC: L(k) = (1 - k²)² + r + α·(k_x²k_y² + k_y²k_z² + k_z²k_x²)
        """
        phi = self.phi
        phi_hat = fft.fftn(phi)
        # 非线性项 φ³
        # Nonlinear term φ³
        nonlinear = phi**3
        nonlinear_hat = fft.fftn(nonlinear)
        # 色散算子 L(k)
        # Dispersion operator L(k)
        if self.lattice_type == "bcc":
            # 标准PFC - 支持BCC (书中第8章)
            # Standard PFC - supports BCC (Chapter 8 in book)
            l_k = (1.0 - self.k2)**2 + self.r
        elif self.lattice_type == "fcc":
            # FCC修正 (需要额外项来稳定fcc)
            # FCC correction (requires additional terms to stabilize fcc)
            kx2, ky2, kz2 = self.kx**2, self.ky**2, self.kz**2
            l_k = (1.0 - self.k2)**2 + self.r + 0.5 * (kx2*ky2 + ky2*kz2 + kz2*kx2)
        elif self.lattice_type == "sc":
            # 简单立方
            # Simple cubic
            l_k = (1.0 - self.k2)**2 + self.r
        else:
            raise ValueError(f"Unknown lattice type: {self.lattice_type}")
        # 半隐式更新
        # Semi-implicit update
        numerator = phi_hat - self.dt * self.M * self.k2 * nonlinear_hat
        denominator = 1.0 + self.dt * self.M * self.k2 * l_k
        phi_hat_new = numerator / denominator
        self.phi = np.real(fft.ifftn(phi_hat_new))
        # 质量守恒修正
        # Mass conservation correction
        self.phi -= (np.mean(self.phi) - self.phi0)

    def compute_energy(self):
        """
        计算三维PFC自由能
        Compute 3D PFC free energy
        F = 0.5 ∫ φ · (r + (1+∇²)²) φ dr + 0.25 ∫ φ⁴ dr
        """
        phi = self.phi
        phi_hat = fft.fftn(phi)
        # 线性部分 (频域)
        # Linear part (frequency domain)
        linear_part = np.sum(
            np.real(np.conj(phi_hat) * ((1.0 - self.k2)**2 + self.r) * phi_hat)
        ) / (self.N**3)
        # 非线性部分 (实空间)
        # Nonlinear part (real space)
        nonlinear_part = np.mean(phi**4)
        energy = 0.5 * linear_part + 0.25 * nonlinear_part
        return energy

    def compute_elastic_energy(self):
        """
        计算弹性能量密度 (书中Section 8.5)
        用于弹性常数分析

        Compute elastic energy density (Section 8.5 in book)
        Used for elastic constant analysis
        """
        # 梯度能量部分
        # Gradient energy part
        phi_hat = fft.fftn(self.phi)
        grad2_phi = np.real(fft.ifftn(-self.k2 * phi_hat))
        # 近似弹性能
        # Approximate elastic energy
        elastic = np.mean(self.phi * grad2_phi + 0.5 * grad2_phi**2)
        return elastic

    def print_status(self, step, E):
        """
        打印状态信息
        Print status information
        """
        print(
            f"step={step:6d} "
            f"E={E:.6e} "
            f"mean={np.mean(self.phi):.3e} "
            f"std={np.std(self.phi):.3e} "
            f"min={np.min(self.phi):.3e} "
            f"max={np.max(self.phi):.3e}"
        )

    def run(self, sample_interval=10):
        """
        运行三维PFC模拟主循环
        Run main 3D PFC simulation loop
        Parameters:
        -----------
        sample_interval : int
            采样间隔步数
            Sampling interval in steps
        """
        print(f"\nStarting 3D PFC simulation...")
        print(f"Steps: {self.steps}, dt={self.dt}, lattice={self.lattice_type}")
        print("=" * 60)
        for step in range(self.steps):
            self.step()
            if step % sample_interval == 0:
                E = self.sample_observables(step)
                self.print_status(step, E)
                if self.record_video:
                    self.capture_frame_3d(step)
        print("=" * 60)
        print("Simulation complete!")
        if self.record_video:
            self.frames_to_video_3d()

    def postprocess(self):
        """
        后处理分析
        Post-processing analysis
        """
        print("\nPost-processing...")
        self.plot_energy()
        self.plot_field_slice(axis=0)
        self.plot_field_slice(axis=1)
        self.plot_field_slice(axis=2)
        self.plot_structure_factor_slice(axis=0)
        self.plot_atoms_3d()
