# ============================================================
# pfc_binary.py - Binary Alloy Phase Field Crystal Solver
# 二元合金相场晶体求解器
#
# 双守恒场耦合动力学:
#   - phi (n): 总密度场, 守恒演化 (mass conservation)
#   - c    : 溶质浓度场, Cahn-Hilliard守恒扩散
#
# 自由能泛函:
#   F = F_PFC[phi] + F_CH[c] + F_coupling[phi, c]
#
#   F_PFC      = 0.5*phi * L(k) * phi + 0.25*phi^4
#   F_CH       = 0.5*c*(r_c - nabla^2)*c + 0.25*u_c*c^4
#              = 0.5*r_c*c^2 + 0.5*|nabla c|^2 + 0.25*u_c*c^4
#   F_coupling = alpha * c * phi^2 + beta * c * phi
#
# 变分导数:
#   delta F / delta phi = L(k)*phi + phi^3 + 2*alpha*c*phi + beta*c
#   delta F / delta c   = (r_c - nabla^2)*c + u_c*c^3 + alpha*phi^2 + beta*phi
#
# 演化方程 (守恒动力学):
#   d phi / dt = M_phi * nabla^2 * (delta F / delta phi)
#   d c   / dt = M_c   * nabla^2 * (delta F / delta c)
#
# 频域半隐式欧拉离散:
#   phi_hat^{n+1} = (phi_hat^n - dt*M_phi*k2*N_phi_hat) / (1 + dt*M_phi*k2*L(k))
#   c_hat^{n+1}   = (c_hat^n   - dt*M_c  *k2*N_c_hat)   / (1 + dt*M_c  *k2*(r_c+k2))
#
#   其中 N_phi = phi^3 + 2*alpha*c*phi + beta*c
#        N_c   = u_c*c^3 + alpha*phi^2 + beta*phi
#
# 晶格类型支持: hexagon (六角) / square (正方) / triangle (三角)
# Lattice support: hexagon / square / triangle
# ============================================================

import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import os

from pfc_base import PFCBase
from pfc_analysis import PFCAnalysis
from pfc_plot import PFCPlot
from pfc_io import PFCIO
from pfc_elastic import PFCElastic


class BinaryPFCSolver(PFCBase, PFCAnalysis, PFCPlot, PFCIO, PFCElastic):
    """
    二元合金PFC求解器 (Binary Alloy Phase Field Crystal Solver)

    继承自PFCBase, PFCAnalysis, PFCPlot, PFCIO, PFCElastic,
    与纯材料求解器PurePFCSolver保持完全一致的API接口。

    新增双场耦合:
    - phi: 总密度场 (对应纯材料的phi)
    - c  : 溶质浓度场 (0 <= c <= 1, 守恒量)
    """

    def __init__(
        self,
        # === 基础网格参数 (Grid parameters) ===
        N=256,
        L=128.0,
        dt=0.05,
        T=2000.0,

        # === PFC密度场参数 (Density field parameters) ===
        r=-0.25,
        M_phi=1.0,
        phi0=-0.25,

        # === 浓度场参数 (Concentration field parameters) ===
        M_c=0.1,
        c0=0.3,
        r_c=-0.5,
        u_c=1.0,

        # === 耦合参数 (Coupling parameters) ===
        alpha=0.1,
        beta=0.0,

        # === 初始条件与晶格 (Initial condition & lattice) ===
        noise_amp=0.01,
        lattice_type="hexagon",
    ):
        super().__init__(
            N=N,
            L=L,
            dt=dt,
            T=T,
        )

        # --- PFC密度场参数 ---
        self.r = float(r)
        self.M_phi = float(M_phi)
        self.phi0 = float(phi0)

        # --- 浓度场参数 ---
        self.M_c = float(M_c)
        self.c0 = float(c0)
        self.r_c = float(r_c)
        self.u_c = float(u_c)

        # --- 耦合参数 ---
        self.alpha = float(alpha)
        self.beta = float(beta)

        # --- 初始条件 ---
        self.noise_amp = float(noise_amp)
        self.lattice_type = lattice_type

        # --- 浓度场日志 (Concentration field logs) ---
        self.c_log = []
        self.c_mass_log = []
        self.c_std_log = []  # 浓度标准差，表征相分离程度

        # --- 输出目录 ---
        self.result_dir = "result"
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"  Output directory: {os.path.abspath(self.result_dir)}")

        # --- IO初始化 ---
        self.initialize_io()

        # --- 初始化双场 ---
        self._initialize_field()

    def _initialize_field(self):
        """
        初始化密度场phi和浓度场c，均叠加高斯噪声并修正均值
        Initialize density field phi and concentration field c,
        both with Gaussian noise and mean correction
        """
        # 标准正态分布噪声场
        noise_phi = self.noise_amp * np.random.randn(self.N, self.N)
        noise_c = self.noise_amp * np.random.randn(self.N, self.N)

        # 基础值叠加噪声
        self.phi = self.phi0 + noise_phi
        self.c = self.c0 + noise_c

        # 修正均值保证守恒
        self.phi -= (np.mean(self.phi) - self.phi0)
        self.c -= (np.mean(self.c) - self.c0)

        # 浓度物理约束: 0 <= c <= 1
        self.c = np.clip(self.c, 0.0, 1.0)

        print(f"  Initial mean phi = {np.mean(self.phi):.6f}")
        print(f"  Initial mean c   = {np.mean(self.c):.6f}")

    # ============================================================
    # 核心演化步 (Core Evolution Step)
    # ============================================================

    def step(self):
        """
        单步傅里叶半隐式欧拉迭代更新双场 (phi, c)
        Single-step Fourier semi-implicit Euler update for dual fields
        """
        phi = self.phi
        c = self.c

        # 傅里叶变换到频域
        phi_hat = fft.fft2(phi)
        c_hat = fft.fft2(c)

        # === 非线性项 (实空间计算) ===
        # N_phi = phi^3 + 2*alpha*c*phi + beta*c
        nonlinear_phi = phi ** 3 + 2.0 * self.alpha * c * phi + self.beta * c
        # N_c = u_c*c^3 + alpha*phi^2 + beta*phi
        nonlinear_c = self.u_c * (c ** 3) + self.alpha * (phi ** 2) + self.beta * phi

        # 非线性项变换到频域
        nonlinear_phi_hat = fft.fft2(nonlinear_phi)
        nonlinear_c_hat = fft.fft2(nonlinear_c)

        # === 按晶格类型计算色散算子 L(k) ===
        if self.lattice_type == "hexagon":
            # 六角晶格: L(k) = (1 - k^2)^2 + r
            l_k = (1.0 - self.k2) ** 2 + self.r
        elif self.lattice_type == "square":
            # 正方晶格: L(k) = (1-kx^2)^2 * (1-ky^2)^2 + r
            term_x = (1.0 - self.kx ** 2) ** 2
            term_y = (1.0 - self.ky ** 2) ** 2
            l_k = term_x * term_y + self.r
        elif self.lattice_type == "triangle":
            # 三角晶格: L(k) = (1 - kx^2 - kx*ky + ky^2)^2 + r
            tri_base = 1.0 - self.kx ** 2 - self.kx * self.ky + self.ky ** 2
            l_k = tri_base ** 2 + self.r

        # === 密度场 phi 半隐式更新 ===
        numerator_phi = phi_hat - self.dt * self.M_phi * self.k2 * nonlinear_phi_hat
        denominator_phi = 1.0 + self.dt * self.M_phi * self.k2 * l_k
        phi_hat_new = numerator_phi / denominator_phi
        self.phi = np.real(fft.ifft2(phi_hat_new))
        # 质量守恒修正
        self.phi -= (np.mean(self.phi) - self.phi0)

        # === 浓度场 c 半隐式更新 (Cahn-Hilliard) ===
        # 分母中的 (r_c + k2) 来自 Cahn-Hilliard 梯度项
        numerator_c = c_hat - self.dt * self.M_c * self.k2 * nonlinear_c_hat
        denominator_c = 1.0 + self.dt * self.M_c * self.k2 * (self.r_c + self.k2)
        c_hat_new = numerator_c / denominator_c
        self.c = np.real(fft.ifft2(c_hat_new))
        # 质量守恒修正 + 物理约束
        self.c -= (np.mean(self.c) - self.c0)
        self.c = np.clip(self.c, 0.0, 1.0)

    # ============================================================
    # 能量计算 (Free Energy Calculation)
    # ============================================================

    def compute_energy(self):
        """
        计算二元合金PFC全场自由能
        F = F_PFC + F_CH + F_coupling

        Returns:
            total_energy (float): 自由能密度 (free energy density)
        """
        phi = self.phi
        c = self.c
        phi_hat = fft.fft2(phi)
        c_hat = fft.fft2(c)

        # --- 按晶格类型计算 L(k) ---
        if self.lattice_type == "hexagon":
            l_k = (1.0 - self.k2) ** 2 + self.r
        elif self.lattice_type == "square":
            term_x = (1.0 - self.kx ** 2) ** 2
            term_y = (1.0 - self.ky ** 2) ** 2
            l_k = term_x * term_y + self.r
        elif self.lattice_type == "triangle":
            tri_base = 1.0 - self.kx ** 2 - self.kx * self.ky + self.ky ** 2
            l_k = tri_base ** 2 + self.r

        # === F_PFC: 标准PFC自由能 ===
        linear_phi = np.sum(np.real(np.conj(phi_hat) * l_k * phi_hat)) / (self.N ** 2)
        nonlinear_phi = np.mean(phi ** 4)
        E_pfc = 0.5 * linear_phi + 0.25 * nonlinear_phi

        # === F_CH: Cahn-Hilliard混合自由能 ===
        # 0.5 * c * (r_c - nabla^2) * c 在频域: 0.5 * |c_hat|^2 * (r_c + k2)
        linear_c = np.sum(np.real(np.conj(c_hat) * (self.r_c + self.k2) * c_hat)) / (self.N ** 2)
        nonlinear_c = np.mean(c ** 4)
        E_ch = 0.5 * linear_c + 0.25 * self.u_c * nonlinear_c

        # === F_coupling: 双场耦合自由能 ===
        E_couple = np.mean(self.alpha * c * phi ** 2 + self.beta * c * phi)

        return E_pfc + E_ch + E_couple

    # ============================================================
    # 结构因子 (Structure Factor)
    # ============================================================

    def structure_factor(self):
        """
        基于密度场phi计算静态结构因子 (与纯材料分析一致)
        Calculate static structure factor from density field phi

        Returns:
            S (ndarray): 静态结构因子 S(k)
        """
        phi_fluct = self.phi - np.mean(self.phi)
        phi_hat = np.fft.fftshift(np.fft.fft2(phi_fluct))
        S = np.abs(phi_hat) ** 2
        return S

    def structure_factor_c(self):
        """
        基于浓度场c计算静态结构因子
        Calculate static structure factor from concentration field c

        Returns:
            S_c (ndarray): 浓度场结构因子
        """
        c_fluct = self.c - np.mean(self.c)
        c_hat = np.fft.fftshift(np.fft.fft2(c_fluct))
        S_c = np.abs(c_hat) ** 2
        return S_c

    # ============================================================
    # 采样与状态打印 (Sampling & Status Print)
    # ============================================================

    def sample_observables(self, step):
        """
        采样观测量: 自由能、phi均值、浓度c均值等
        Sample observables: free energy, phi mean, concentration c mean, etc.
        """
        E = self.compute_energy()
        self.energy_log.append(E)
        self.mass_log.append(np.mean(self.phi))
        self.c_mass_log.append(np.mean(self.c))
        self.c_std_log.append(np.std(self.c))

        # 结构因子峰值
        S = self.structure_factor()
        self.structure_peak_log.append(np.max(S))

        # 缺陷分析 (仅在后期执行)
        if step > 1500:
            defect_density, grain_size, _, _ = self.analyze_defects()
            if not np.isnan(defect_density):
                self.defect_log.append(defect_density)
                self.grain_size_log.append(grain_size)

        return E

    def print_status(self, step, E):
        """
        控制台打印当前演化状态 (包含双场信息)
        Print current evolution status to console (dual field info)
        """
        print(
            f"step={step:6d}  "
            f"E={E:.6e}  "
            f"phi={np.mean(self.phi):.3e}  "
            f"c={np.mean(self.c):.3e}  "
            f"c_std={np.std(self.c):.3e}  "
            f"c_range=[{np.min(self.c):.3e}, {np.max(self.c):.3e}]"
        )

    # ============================================================
    # 主运行循环 (Main Run Loop)
    # ============================================================

    def run(self, sample_interval=10):
        """
        主模拟运行循环

        Args:
            sample_interval (int): 采样间隔步数 (sampling interval steps)
        """
        print("\n" + "=" * 60)
        print("  Binary Alloy PFC Simulation Started")
        print(f"  Grid: {self.N}x{self.N}, L={self.L:.1f}, dt={self.dt:.3f}, steps={self.steps}")
        print(f"  Lattice: {self.lattice_type}")
        print(f"  Coupling: alpha={self.alpha:.3f}, beta={self.beta:.3f}")
        print(f"  M_phi={self.M_phi:.2f}, M_c={self.M_c:.2f}")
        print("=" * 60 + "\n")

        for step in range(self.steps):
            self.step()

            if step % sample_interval == 0:
                E = self.sample_observables(step)
                self.print_status(step, E)
                self.capture_frame()

        # 尝试合成视频，失败时不中断
        try:
            self.frames_to_video()
            # 如果视频生成成功，移动到 result 目录
            if os.path.exists("pfc_simulation.mp4"):
                import shutil
                shutil.move("pfc_simulation.mp4", os.path.join(self.result_dir, "pfc_simulation.mp4"))
                print(f"  Video saved to: {os.path.join(self.result_dir, 'pfc_simulation.mp4')}")
        except Exception as e:
            print(f"\n  [Warning] Video generation failed: {e}")
            print("  This is often a Windows/ffmpeg path issue. Individual PNG frames are still available.")
            print("  Tip: Re-run with --no-video to skip video generation entirely.")
        print("\nSimulation completed!")

    # ============================================================
    # 可视化覆盖 (Visualization Overrides)
    # ============================================================

    def capture_frame(self):
        """
        捕获当前双场(phi, c)画面用于视频合成
        Capture current dual-field snapshot for video synthesis
        """
        if not self.record_video:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # --- 密度场 phi ---
        im0 = axes[0].imshow(
            self.phi,
            cmap="coolwarm",
            origin="lower",
            vmin=-0.8, vmax=0.4,
        )
        axes[0].set_title(
            f"Density Field φ\nstep={self.steps}, frame={len(self.frame_cache)}",
            fontsize=11,
        )
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        # --- 浓度场 c ---
        im1 = axes[1].imshow(
            self.c,
            cmap="RdYlBu_r",
            origin="lower",
            vmin=0.0, vmax=1.0,
        )
        axes[1].set_title(
            f"Concentration Field c\nstep={self.steps}, frame={len(self.frame_cache)}",
            fontsize=11,
        )
        plt.colorbar(im1, ax=axes[1], shrink=0.8, label="c")

        plt.tight_layout()
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        self.frame_cache.append(buf.getvalue())
        plt.close(fig)

    def postprocess(self):
        """
        二元合金全套后处理绘图流程
        Full binary alloy post-processing plotting pipeline
        """
        print("\n" + "=" * 60)
        print("  Binary Alloy Post-Processing")
        print("=" * 60 + "\n")

        # 切换到 result 目录，让继承来的 PFCPlot 方法也把图存到这里
        old_cwd = os.getcwd()
        os.chdir(self.result_dir)

        # --- 1. 能量演化曲线 ---
        self.plot_energy()

        # --- 2. 最终密度场 phi ---
        self.plot_field()

        # --- 3. 最终浓度场 c ---
        self._plot_concentration_field()

        # --- 4. phi + c 叠加场 ---
        self._plot_overlay_field()

        # --- 5. 浓度场结构因子 ---
        self._plot_structure_factor_c()

        # --- 6-7. 结构因子 & Voronoi分析 ---
        self.plot_structure_factor()
        self.plot_voronoi()

        # --- 8-12. 缺陷与微观结构分析 ---
        self.defect_statistics()
        self.plot_defects()
        self.plot_defect_density()
        self.plot_grain_size()
        self.plot_structure_peak()
        self.plot_detected_atoms()

        # --- 13. 浓度演化曲线 ---
        self._plot_concentration_evolution()

        # 切回原来的工作目录
        os.chdir(old_cwd)

    # ============================================================
    # 二元合金特有可视化方法
    # ============================================================

    def _plot_concentration_field(self):
        """
        绘制浓度场c的二维云图
        Plot 2D heatmap of concentration field c
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(
            self.c,
            cmap="RdYlBu_r",
            origin="lower",
            vmin=0.0, vmax=1.0,
        )
        plt.colorbar(label="Concentration c")
        plt.title("Final Concentration Field")
        plt.tight_layout()
        plt.savefig("concentration_field.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: concentration_field.png")

    def _plot_overlay_field(self):
        """
        绘制phi + c叠加场: phi用灰度底图, c用颜色叠加
        Plot phi + c overlay: phi as grayscale background, c as color overlay
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        # phi作为灰度底图
        phi_bg = (self.phi - self.phi.min()) / (self.phi.max() - self.phi.min() + 1e-10)
        ax.imshow(phi_bg, cmap="gray", origin="lower", alpha=0.5)

        # c作为颜色叠加 (仅在c偏离平均值区域显示)
        c_overlay = self.c.copy()
        ax.imshow(c_overlay, cmap="RdYlBu_r", origin="lower", alpha=0.6, vmin=0.0, vmax=1.0)

        ax.set_title("Density φ (grayscale) + Concentration c (color)")
        plt.colorbar(
            ax.imshow(c_overlay, cmap="RdYlBu_r", origin="lower", vmin=0.0, vmax=1.0, visible=False),
            ax=ax, shrink=0.8, label="c",
        )
        plt.tight_layout()
        plt.savefig("overlay_field.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: overlay_field.png")

    def _plot_structure_factor_c(self):
        """
        绘制浓度场的对数尺度结构因子
        Plot log-scaled structure factor of concentration field c
        """
        S_c = self.structure_factor_c()
        plt.figure(figsize=(6, 6))
        plt.imshow(np.log10(S_c + 1.0), origin="lower", cmap="inferno")
        plt.colorbar(label="log10(Sc+1)")
        plt.title("Concentration Structure Factor")
        plt.tight_layout()
        plt.savefig("structure_factor_c.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: structure_factor_c.png")

    def _plot_concentration_evolution(self):
        """
        绘制浓度c的均值和标准差演化曲线
        Plot mean and standard deviation evolution of concentration c
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 左图: 浓度均值演化 (守恒，应为平线)
        axes[0].plot(self.c_mass_log, "r-o", markersize=3, linewidth=1)
        axes[0].axhline(y=self.c0, color="gray", linestyle="--", label=f"target c0={self.c0}")
        axes[0].set_xlabel("Sample")
        axes[0].set_ylabel("Mean Concentration")
        axes[0].set_title("Concentration Mean Evolution (Conserved)")
        axes[0].legend()
        axes[0].grid(True)

        # 右图: 浓度标准差演化 (表征相分离/偏析程度)
        axes[1].plot(self.c_std_log, "b-o", markersize=3, linewidth=1)
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("Concentration Std Dev")
        axes[1].set_title("Concentration Std Dev (Phase Separation Metric)")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("concentration_evolution.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: concentration_evolution.png")

    def plot_coupling_energy(self):
        """
        绘制耦合自由能密度空间分布
        Plot spatial distribution of coupling free energy density
        """
        f_couple = self.alpha * self.c * self.phi ** 2 + self.beta * self.c * self.phi
        plt.figure(figsize=(6, 6))
        plt.imshow(f_couple, cmap="coolwarm", origin="lower")
        plt.colorbar(label="Coupling Energy Density")
        plt.title(f"Coupling Energy (α={self.alpha}, β={self.beta})")
        plt.tight_layout()
        plt.savefig("coupling_energy.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: coupling_energy.png")