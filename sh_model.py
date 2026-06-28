#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sh_model.py — Swift-Hohenberg 求解器
Swift-Hohenberg Solver

复用 PFCBase (网格/k空间) + 内嵌视频录制（不依赖 pfc_io）
Reuses PFCBase (grid/k-space) + built-in video recording (independent of pfc_io)

Swift-Hohenberg equation:
    ∂ψ/∂t = ε·ψ - (q₀² + ∇²)²·ψ - ψ³
    
This is a canonical pattern-forming PDE that produces stripe, hexagonal,
and other periodic patterns above a critical threshold ε.
这是一个典型的图案形成偏微分方程，在临界阈值ε以上产生条纹、六角形和其他周期性图案。

Author: Jinpeng Wang
Department of Material Engineering
"""

# OS module for file and directory operations
# OS模块，用于文件和目录操作
import os

# File operations (copy, move, delete)
# 文件操作（复制、移动、删除）
import shutil

# Temporary file handling
# 临时文件处理
import tempfile

# Subprocess for running external commands (ffmpeg)
# Subprocess用于运行外部命令（ffmpeg）
import subprocess

# Warning handling
# 警告处理
import warnings

# NumPy for numerical operations
# NumPy用于数值运算
import numpy as np

# SciPy FFT - fast Fourier transform
# SciPy FFT - 快速傅里叶变换
import scipy.fft as fft

# Matplotlib for plotting
# Matplotlib用于绘图
import matplotlib.pyplot as plt

# BytesIO for in-memory image storage
# BytesIO用于内存图像存储
from io import BytesIO

# Import PFC base class for grid and k-space infrastructure
# 导入PFC基类以获取网格和k空间基础设施
from pfc_base import PFCBase


class SHSolver(PFCBase):
    """
    Swift-Hohenberg equation solver.
    Swift-Hohenberg方程求解器。
    
    Inherits grid and k-space infrastructure from PFCBase.
    从PFCBase继承网格和k空间基础设施。
    
    The Swift-Hohenberg equation is a pattern-forming PDE:
        ∂ψ/∂t = ε·ψ - (q₀² + ∇²)²·ψ - ψ³
    
    Swift-Hohenberg方程是一个图案形成偏微分方程：
        ∂ψ/∂t = ε·ψ - (q₀² + ∇²)²·ψ - ψ³
    
    Features / 功能:
        - Semi-implicit Fourier spectral method
          半隐式傅里叶谱方法
        - Built-in video recording (ffmpeg or GIF fallback)
          内置视频录制（ffmpeg或GIF降级）
        - Energy calculation and spectrum analysis
          能量计算和谱分析
    
    Parameters / 参数:
        epsilon (float): Control parameter (pattern formation threshold)
                         控制参数（图案形成阈值）
        q0 (float): Characteristic wavenumber
                    特征波数
        psi0 (float): Average field value
                      平均场值
        psi_clip (float): Clipping value for field stability
                          场稳定性的裁剪值
    """
    
    def __init__(
        self,
        N=256,
        L=64.0,
        dt=0.005,
        T=500.0,
        epsilon=0.3,
        q0=1.0,
        noise_amp=0.01,
        psi0=0.0,
        record_video=False,
        video_fps=15,
        psi_clip=2.0,
        output_dir=None,   # ← 新增参数 / new parameter
    ):
        """
        Initialize Swift-Hohenberg solver.
        初始化Swift-Hohenberg求解器。
        
        Args / 参数:
            N (int): Grid resolution / 网格分辨率
            L (float): Domain size / 计算域尺寸
            dt (float): Time step / 时间步长
            T (float): Total simulation time / 总模拟时间
            epsilon (float): Control parameter / 控制参数
            q0 (float): Characteristic wavenumber / 特征波数
            noise_amp (float): Initial noise amplitude / 初始噪声幅值
            psi0 (float): Average field value / 平均场值
            record_video (bool): Whether to record video / 是否录制视频
            video_fps (int): Video frames per second / 视频帧率
            psi_clip (float): Field clipping value / 场裁剪值
            output_dir (str): Output directory / 输出目录
        """
        # Initialize base class
        # 初始化基类
        super().__init__(N=N, L=L, dt=dt, T=T)
        
        # Swift-Hohenberg parameters
        # Swift-Hohenberg参数
        self.epsilon = float(epsilon)  # Control parameter / 控制参数
        self.q0 = float(q0)            # Characteristic wavenumber / 特征波数
        self.q0_sq = q0 ** 2           # Squared wavenumber / 波数平方
        self.noise_amp = float(noise_amp)  # Noise amplitude / 噪声幅值
        self.psi0 = float(psi0)        # Average field value / 平均场值
        self.psi_clip = float(psi_clip)  # Clipping value / 裁剪值
        
        # Video recording settings
        # 视频录制设置
        self.record_video = record_video
        self.video_fps = int(video_fps)
        self.frame_cache = []  # In-memory frame cache / 内存帧缓存
        
        # === 输出目录设置 ===
        # === Output directory setup ===
        if output_dir is None:
            # 默认路径：你的 result 文件夹
            # Default path: your result folder
            self.output_dir = os.path.join(
                "C:", "Users", "35180", "PFC", "Phase-Field-Crystal", "result"
            )
        else:
            self.output_dir = output_dir
        
        # Create output directory
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_output_name = os.path.join(self.output_dir, "sh_evolution.mp4")
        # ====================
        
        # Initialize field
        # 初始化场
        self._init_field()
        
        # Precompute linear operator for efficiency
        # 预计算线性算子以提高效率
        # Linear part: -(ε - (q₀² - k²)²)
        # Note: The SH equation has -(q₀²+∇²)²ψ term
        # In Fourier: -(q₀² - k²)²·ψ̂  (since ∇² → -k²)
        self.linear_operator = -(self.epsilon - (self.q0_sq - self.k2) ** 2)
        
        # Logging arrays
        # 日志数组
        self.time_log = []       # Time values / 时间值
        self.psi_max_log = []    # Maximum field values / 场最大值
        self.psi_min_log = []    # Minimum field values / 场最小值
        
        # Check ffmpeg availability
        # 检查ffmpeg可用性
        self._ffmpeg_available = self._check_ffmpeg()
        
        # Print initialization info
        # 打印初始化信息
        print(f"\n{'='*55}")
        print(f"  Swift-Hohenberg Solver")
        print(f"  Swift-Hohenberg求解器")
        print(f"  Grid / 网格: {N}x{N}, L={L}, dt={dt}, T={T}, steps={self.steps}")
        print(f"  ε={epsilon}, q₀={q0}")
        print(f"  Output directory / 输出目录: {self.output_dir}")
        print(f"{'='*55}\n")
    
    def _init_field(self):
        """
        Initialize field with Gaussian noise.
        用高斯噪声初始化场。
        """
        # Generate Gaussian noise
        # 生成高斯噪声
        noise = self.noise_amp * np.random.randn(self.N, self.N)
        
        # Add to base value
        # 加到基础值上
        self.psi = self.psi0 + noise
    
    def _check_ffmpeg(self):
        """
        检查系统是否安装了 ffmpeg
        Check if ffmpeg is installed on the system.
        
        Returns / 返回值:
            bool: True if ffmpeg is available / 如果ffmpeg可用则为True
        """
        try:
            # Try running ffmpeg -version
            # 尝试运行ffmpeg -version
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            return True
        except FileNotFoundError:
            # ffmpeg command not found
            # 未找到ffmpeg命令
            return False
    
    # ==================== 核心演化 ====================
    # ==================== Core Evolution ====================
    
    def step(self):
        """
        单步频域半隐式欧拉
        Single-step Fourier semi-implicit Euler iteration
        
        Semi-implicit scheme:
            Linear terms: implicit (for stability)
            Nonlinear terms: explicit
        
        半隐式格式：
            线性项：隐式（保证稳定性）
            非线性项：显式
        """
        # Fourier transform
        # 傅里叶变换
        psi_hat = fft.fft2(self.psi)
        
        # 非线性项 -ψ³
        # Nonlinear term -ψ³
        nonlinear = - self.psi ** 3
        nonlinear_hat = fft.fft2(nonlinear)
        
        # Semi-implicit update
        # 半隐式更新
        numerator = psi_hat + self.dt * nonlinear_hat
        denominator = 1.0 + self.dt * self.linear_operator
        
        # Update field
        # 更新场
        self.psi = np.real(fft.ifft2(numerator / denominator))
    
    def compute_energy(self):
        """
        计算Swift-Hohenberg自由能
        Calculate Swift-Hohenberg free energy
        
        F = ∫[-ε/2·ψ² + ½((q₀²+∇²)ψ)² - ¼ψ⁴]
        
        Returns / 返回值:
            float: Total free energy / 总自由能
        """
        psi = self.psi
        psi_hat = fft.fft2(psi)
        
        # Term 1: -ε/2 · ψ²
        # 第一项：-ε/2 · ψ²
        term1 = -0.5 * self.epsilon * psi ** 2
        
        # Term 2: ½ · ((q₀²+∇²)ψ)²
        # In Fourier: (q₀² - k²) · ψ̂  (since ∇² → -k²)
        # 第二项：½ · ((q₀²+∇²)ψ)²
        # 傅里叶空间：(q₀² - k²) · ψ̂ （因为∇² → -k²）
        lap_hat = (self.q0_sq - self.k2) * psi_hat
        lap = np.real(fft.ifft2(lap_hat))
        term2 = 0.5 * lap ** 2
        
        # Term 3: + ¼ · ψ⁴  (note: positive because of -ψ³ in equation)
        # 第三项：+ ¼ · ψ⁴（注意：因为方程中有-ψ³，所以是正的）
        term3 = + 0.25 * psi ** 4
        
        # Total energy density
        # 总能量密度
        energy_density = term1 + term2 + term3
        
        # Integrate over domain
        # 在域上积分
        return np.sum(energy_density) * self.dx ** 2
    
    # ==================== 运行控制 ====================
    # ==================== Run Control ====================
    
    def run(self, log_interval=10, frame_interval=50):
        """
        主循环
        Main simulation loop
        
        Args / 参数:
            log_interval (int): Logging interval steps / 日志间隔步数
            frame_interval (int): Frame capture interval steps / 帧捕获间隔步数
        """
        # Main simulation loop
        # 主模拟循环
        for n in range(self.steps):
            # Perform one time step
            # 执行一个时间步
            self.step()
            
            # Log at regular intervals
            # 定期记录
            if n % log_interval == 0:
                t = n * self.dt
                E = self.compute_energy()
                
                # Log observables
                # 记录观测量
                self.time_log.append(t)
                self.energy_log.append(E)
                self.mass_log.append(np.mean(self.psi))
                self.psi_max_log.append(np.max(self.psi))
                self.psi_min_log.append(np.min(self.psi))
                
                # Print status every 5 log intervals
                # 每5个日志间隔打印状态
                if n % (5 * log_interval) == 0:
                    print(f"  step={n:5d}  t={t:7.1f}  E={E:12.4f}  "
                          f"mean={np.mean(self.psi):.4f}  max={np.max(self.psi):.4f}")
            
            # 捕获帧（复用 PFCIO 逻辑，但内嵌避免依赖）
            # Capture frame (reuses PFCIO logic, but embedded to avoid dependency)
            if self.record_video and n % frame_interval == 0:
                self._capture_frame()
        
        # 视频生成（可选，失败不崩溃）
        # Video generation (optional, don't crash on failure)
        if self.record_video and self.frame_cache:
            self._save_video()
    
    # ==================== 视频录制（内嵌，不依赖 pfc_io） ====================
    # ==================== Video Recording (embedded, independent of pfc_io) ====================
    
    def _capture_frame(self):
        """
        将当前 psi 场缓存为内存图像
        Cache current psi field as in-memory image
        """
        # Create figure
        # 创建图形
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Calculate symmetric color range
        # 计算对称颜色范围
        vmax = np.max(np.abs(self.psi))
        if vmax == 0:
            vmax = 1e-6
        
        # Plot field
        # 绘制场
        im = ax.imshow(self.psi, cmap="RdBu_r", origin="lower",
                       vmin=-vmax, vmax=vmax)
        
        # Add title with time
        # 添加带时间的标题
        ax.set_title(f"t = {len(self.frame_cache) * self.dt * 50:.1f}")
        
        # Add colorbar
        # 添加颜色条
        plt.colorbar(im, ax=ax, label=r"$\psi$")
        
        # Adjust layout
        # 调整布局
        plt.tight_layout()
        
        # Save to memory buffer
        # 保存到内存缓冲区
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        self.frame_cache.append(buf.getvalue())
        
        # Close figure to free memory
        # 关闭图形以释放内存
        plt.close(fig)
    
    def _save_video(self):
        """
        合成视频，支持 ffmpeg 或降级为 GIF
        Synthesize video, supports ffmpeg or falls back to GIF
        """
        # Skip if no frames
        # 如果没有帧则跳过
        if not self.frame_cache:
            return
        
        # Create temporary directory
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 写入临时帧
            # Write temporary frames
            for idx, frame_bytes in enumerate(self.frame_cache):
                frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                with open(frame_path, "wb") as f:
                    f.write(frame_bytes)
            
            # 优先尝试 ffmpeg
            # Try ffmpeg first
            if self._ffmpeg_available:
                try:
                    # Build ffmpeg command
                    # 构建ffmpeg命令
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-framerate", str(self.video_fps),
                        "-i", os.path.join(temp_dir, "frame_%06d.png"),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "18",
                        self.video_output_name,
                    ]
                    
                    # Run ffmpeg
                    # 运行ffmpeg
                    subprocess.run(
                        ffmpeg_cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    print(f"\n  Video saved: {os.path.abspath(self.video_output_name)}")
                    print(f"\n  视频已保存: {os.path.abspath(self.video_output_name)}")
                    return
                    
                except subprocess.CalledProcessError as e:
                    print(f"\n  ffmpeg failed, trying GIF fallback...")
                    print(f"\n  ffmpeg失败，尝试GIF降级...")
            else:
                print(f"\n  ffmpeg not found, using GIF fallback...")
                print(f"\n  未找到ffmpeg，使用GIF降级...")
            
            # 降级：用 matplotlib 保存 GIF
            # Fallback: save GIF using matplotlib
            try:
                from matplotlib import animation
                
                # Create figure
                # 创建图形
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Calculate color range
                # 计算颜色范围
                vmax = np.max([np.max(np.abs(f)) for f in self._decode_frames()])
                
                # Display first frame
                # 显示第一帧
                im = ax.imshow(self._decode_frames()[0], cmap="RdBu_r", origin="lower",
                               vmin=-vmax, vmax=vmax)
                ax.axis("off")
                
                # Animation update function
                # 动画更新函数
                def update(i):
                    im.set_array(self._decode_frames()[i])
                    return [im]
                
                # Create animation
                # 创建动画
                anim = animation.FuncAnimation(fig, update, frames=len(self.frame_cache), blit=True)
                
                # Generate GIF filename
                # 生成GIF文件名
                gif_path = self.video_output_name.replace(".mp4", ".gif")
                
                # Save as GIF
                # 保存为GIF
                anim.save(gif_path, writer="pillow", fps=self.video_fps)
                
                print(f"  GIF saved: {os.path.abspath(gif_path)}")
                print(f"  GIF已保存: {os.path.abspath(gif_path)}")
                
                plt.close(fig)
                
            except Exception as e2:
                print(f"  Video/GIF generation failed: {e2}")
                print(f"  视频/GIF生成失败: {e2}")
        
        finally:
            # Clean up temporary directory
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.frame_cache.clear()
    
    def _decode_frames(self):
        """
        辅助：将缓存的字节还原为数组（用于 GIF 生成）
        Helper: decode cached bytes back to arrays (for GIF generation)
        
        Returns / 返回值:
            list: List of numpy arrays / numpy数组列表
        """
        from PIL import Image
        import io
        
        frames = []
        for b in self.frame_cache:
            img = Image.open(io.BytesIO(b))
            frames.append(np.array(img))
        return frames
    
    # ==================== 可视化 ====================
    # ==================== Visualization ====================
    
    def plot_summary(self, save_path=None):
        """
        绘制总结图：最终状态、能量、质量、极值
        Plot summary: final state, energy, mass, extrema
        
        Args / 参数:
            save_path (str, optional): Save path / 保存路径
        """
        # Default save path
        # 默认保存路径
        if save_path is None:
            save_path = os.path.join(self.output_dir, "sh_summary.png")
        
        # Create 2x2 subplot
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # --- Top left: Final state ---
        # --- 左上：最终状态 ---
        ax = axes[0, 0]
        vmax = np.max(np.abs(self.psi))
        im = ax.imshow(self.psi, cmap='RdBu_r', origin='lower',
                       extent=[0, self.L, 0, self.L], vmin=-vmax, vmax=vmax)
        ax.set_title(f"Final State (t={self.T}, ε={self.epsilon})")
        ax.set_title(f"最终状态 (t={self.T}, ε={self.epsilon})")
        plt.colorbar(im, ax=ax, label=r"$\psi$")
        
        # --- Top right: Free energy ---
        # --- 右上：自由能 ---
        ax = axes[0, 1]
        ax.plot(self.time_log, self.energy_log, 'b-', lw=1)
        ax.set_title("Free Energy / 自由能")
        ax.set_xlabel("t / 时间")
        ax.set_ylabel("F / 自由能")
        ax.grid(True, alpha=0.3)
        
        # --- Bottom left: Mass (Mean) ---
        # --- 左下：质量（均值）---
        ax = axes[1, 0]
        ax.plot(self.time_log, self.mass_log, 'g-', lw=1)
        ax.set_title("Mass (Mean) / 质量（均值）")
        ax.set_xlabel("t / 时间")
        ax.set_ylabel(r"$\langle \psi \rangle$")
        ax.grid(True, alpha=0.3)
        
        # --- Bottom right: Extrema ---
        # --- 右下：极值 ---
        ax = axes[1, 1]
        ax.plot(self.time_log, self.psi_max_log, 'r-', lw=1, label='max / 最大')
        ax.plot(self.time_log, self.psi_min_log, 'b-', lw=1, label='min / 最小')
        ax.set_title("Extrema / 极值")
        ax.set_xlabel("t / 时间")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        # 调整布局
        plt.tight_layout()
        
        # Save figure
        # 保存图形
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Summary saved: {save_path}")
        print(f"  总结图已保存: {save_path}")
        
        plt.show()
    
    def plot_spectrum(self, save_path=None):
        """
        绘制功率谱，标注不稳定环
        Plot power spectrum, mark unstable ring
        
        Args / 参数:
            save_path (str, optional): Save path / 保存路径
        """
        # Default save path
        # 默认保存路径
        if save_path is None:
            save_path = os.path.join(self.output_dir, "sh_spectrum.png")
        
        # Compute power spectrum
        # 计算功率谱
        psi_hat = fft.fftshift(fft.fft2(self.psi))
        power = np.abs(psi_hat) ** 2
        
        # Create figure
        # 创建图形
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # 1D wavenumber array for extent
        # 用于范围的一维波数数组
        k1d = 2.0 * np.pi * fft.fftshift(fft.fftfreq(self.N, d=self.dx))
        extent = [k1d[0], k1d[-1], k1d[0], k1d[-1]]
        
        # Plot power spectrum with log scale
        # 用对数尺度绘制功率谱
        im = ax.imshow(fft.fftshift(power), extent=extent, cmap='hot',
                       origin='lower', norm=plt.matplotlib.colors.LogNorm())
        ax.set_title("Power Spectrum / 功率谱")
        
        # Calculate unstable wavenumber range
        # 计算不稳定波数范围
        # For SH equation: unstable when ε - (q₀² - k²)² > 0
        # → |q₀² - k²| < √ε
        # → q₀² - √ε < k² < q₀² + √ε
        k_out = np.sqrt(self.q0_sq + np.sqrt(self.epsilon))
        k_in = np.sqrt(self.q0_sq - np.sqrt(self.epsilon)) if self.epsilon <= self.q0_sq ** 2 else 0
        
        # Draw unstable ring
        # 绘制不稳定环
        theta = np.linspace(0, 2 * np.pi, 200)
        if k_in > 0:
            ax.plot(k_in * np.cos(theta), k_in * np.sin(theta), 'c--', lw=1.5)
        ax.plot(k_out * np.cos(theta), k_out * np.sin(theta), 'c--', lw=1.5, label='unstable ring / 不稳定环')
        
        # Set limits
        # 设置范围
        ax.set_xlim(-3 * self.q0, 3 * self.q0)
        ax.set_ylim(-3 * self.q0, 3 * self.q0)
        ax.legend()
        
        # Add colorbar
        # 添加颜色条
        plt.colorbar(im, label="Power / 功率")
        
        # Adjust layout
        # 调整布局
        plt.tight_layout()
        
        # Save figure
        # 保存图形
        plt.savefig(save_path, dpi=150)
        print(f"  Spectrum saved: {save_path}")
        print(f"  谱图已保存: {save_path}")
        
        plt.show()


# ==================== 主程序 ====================
# ==================== Main Program ====================

if __name__ == "__main__":
    # 默认不录视频，如需视频改为 record_video=True
    # Default: no video recording, set record_video=True for video
    solver = SHSolver(
        N=256, L=64.0,
        dt=0.01,        # 关键：dt 必须小，否则数值爆炸
                        # Critical: dt must be small, otherwise numerical explosion
        T=500.0,
        epsilon=0.3, q0=1.0,
        noise_amp=0.01,
        record_video=False,   # 默认关闭 / default off
        video_fps=15,
    )
    
    # Run simulation
    # 运行模拟
    solver.run(log_interval=10, frame_interval=50)
    
    # Plot results
    # 绘制结果
    solver.plot_summary()
    solver.plot_spectrum()
