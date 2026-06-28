#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sh_model.py — Swift-Hohenberg 求解器
复用 PFCBase (网格/k空间) + PFCIO (视频录制)
"""

import os
import shutil
import tempfile
import subprocess
import warnings
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from io import BytesIO

from pfc_base import PFCBase


class SHSolver(PFCBase):
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
        output_dir=None,   # ← 新增参数
    ):
        super().__init__(N=N, L=L, dt=dt, T=T)

        self.epsilon = float(epsilon)
        self.q0 = float(q0)
        self.q0_sq = q0 ** 2
        self.noise_amp = float(noise_amp)
        self.psi0 = float(psi0)
        self.psi_clip = float(psi_clip)

        self.record_video = record_video
        self.video_fps = int(video_fps)
        self.frame_cache = []

        # === 输出目录设置 ===
        if output_dir is None:
            # 默认路径：你的 result 文件夹
            self.output_dir = os.path.join(
                "C:", "Users", "35180", "PFC", "Phase-Field-Crystal", "result"
            )
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_output_name = os.path.join(self.output_dir, "sh_evolution.mp4")
        # ====================

        self._init_field()
        self.linear_operator = -(self.epsilon - (self.q0_sq - self.k2) ** 2)
        self.time_log = []
        self.psi_max_log = []
        self.psi_min_log = []
        self._ffmpeg_available = self._check_ffmpeg()

        print(f"\n{'='*55}")
        print(f"  Swift-Hohenberg Solver")
        print(f"  Grid: {N}x{N}, L={L}, dt={dt}, T={T}, steps={self.steps}")
        print(f"  ε={epsilon}, q₀={q0}")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*55}\n")

    def _init_field(self):
        noise = self.noise_amp * np.random.randn(self.N, self.N)
        self.psi = self.psi0 + noise

    def _check_ffmpeg(self):
        """检查系统是否安装了 ffmpeg"""
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            return True
        except FileNotFoundError:
            return False

    # ==================== 核心演化 ====================

    def step(self):
        """单步频域半隐式欧拉"""
        psi_hat = fft.fft2(self.psi)
        nonlinear = - self.psi ** 3
        nonlinear_hat = fft.fft2(nonlinear)

        numerator = psi_hat + self.dt * nonlinear_hat
        denominator = 1.0 + self.dt * self.linear_operator

        self.psi = np.real(fft.ifft2(numerator / denominator))

    def compute_energy(self):
        """
        F = ∫[-ε/2·ψ² + ½((q₀²+∇²)ψ)² - ¼ψ⁴]
        """
        psi = self.psi
        psi_hat = fft.fft2(psi)

        term1 = -0.5 * self.epsilon * psi ** 2
        lap_hat = (self.q0_sq - self.k2) * psi_hat
        lap = np.real(fft.ifft2(lap_hat))
        term2 = 0.5 * lap ** 2
        term3 = + 0.25 * psi ** 4

        energy_density = term1 + term2 + term3
        return np.sum(energy_density) * self.dx ** 2

    # ==================== 运行控制 ====================

    def run(self, log_interval=10, frame_interval=50):
        """主循环"""
        for n in range(self.steps):
            self.step()

            if n % log_interval == 0:
                t = n * self.dt
                E = self.compute_energy()
                self.time_log.append(t)
                self.energy_log.append(E)
                self.mass_log.append(np.mean(self.psi))
                self.psi_max_log.append(np.max(self.psi))
                self.psi_min_log.append(np.min(self.psi))

                if n % (5 * log_interval) == 0:
                    print(f"  step={n:5d}  t={t:7.1f}  E={E:12.4f}  "
                          f"mean={np.mean(self.psi):.4f}  max={np.max(self.psi):.4f}")

            # 捕获帧（复用 PFCIO 逻辑，但内嵌避免依赖）
            if self.record_video and n % frame_interval == 0:
                self._capture_frame()

        # 视频生成（可选，失败不崩溃）
        if self.record_video and self.frame_cache:
            self._save_video()

    # ==================== 视频录制（内嵌，不依赖 pfc_io） ====================

    def _capture_frame(self):
        """将当前 psi 场缓存为内存图像"""
        fig, ax = plt.subplots(figsize=(6, 6))
        vmax = np.max(np.abs(self.psi))
        if vmax == 0:
            vmax = 1e-6
        im = ax.imshow(self.psi, cmap="RdBu_r", origin="lower",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"t = {len(self.frame_cache) * self.dt * 50:.1f}")
        plt.colorbar(im, ax=ax, label=r"$\psi$")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        self.frame_cache.append(buf.getvalue())
        plt.close(fig)

    def _save_video(self):
        """合成视频，支持 ffmpeg 或降级为 GIF"""
        if not self.frame_cache:
            return

        temp_dir = tempfile.mkdtemp()
        try:
            # 写入临时帧
            for idx, frame_bytes in enumerate(self.frame_cache):
                frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                with open(frame_path, "wb") as f:
                    f.write(frame_bytes)

            # 优先尝试 ffmpeg
            if self._ffmpeg_available:
                try:
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-framerate", str(self.video_fps),
                        "-i", os.path.join(temp_dir, "frame_%06d.png"),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "18",
                        self.video_output_name,
                    ]
                    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"\n  Video saved: {os.path.abspath(self.video_output_name)}")
                    return
                except subprocess.CalledProcessError as e:
                    print(f"\n  ffmpeg failed, trying GIF fallback...")
            else:
                print(f"\n  ffmpeg not found, using GIF fallback...")

            # 降级：用 matplotlib 保存 GIF
            try:
                from matplotlib import animation
                fig, ax = plt.subplots(figsize=(6, 6))
                vmax = np.max([np.max(np.abs(f)) for f in self._decode_frames()])
                im = ax.imshow(self._decode_frames()[0], cmap="RdBu_r", origin="lower",
                               vmin=-vmax, vmax=vmax)
                ax.axis("off")

                def update(i):
                    im.set_array(self._decode_frames()[i])
                    return [im]

                anim = animation.FuncAnimation(fig, update, frames=len(self.frame_cache), blit=True)
                gif_path = self.video_output_name.replace(".mp4", ".gif")
                anim.save(gif_path, writer="pillow", fps=self.video_fps)
                print(f"  GIF saved: {os.path.abspath(gif_path)}")
                plt.close(fig)
            except Exception as e2:
                print(f"  Video/GIF generation failed: {e2}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.frame_cache.clear()

    def _decode_frames(self):
        """辅助：将缓存的字节还原为数组（用于 GIF 生成）"""
        from PIL import Image
        import io
        frames = []
        for b in self.frame_cache:
            img = Image.open(io.BytesIO(b))
            frames.append(np.array(img))
        return frames

    # ==================== 可视化 ====================

    def plot_summary(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "sh_summary.png")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        vmax = np.max(np.abs(self.psi))
        im = ax.imshow(self.psi, cmap='RdBu_r', origin='lower',
                       extent=[0, self.L, 0, self.L], vmin=-vmax, vmax=vmax)
        ax.set_title(f"Final State (t={self.T}, ε={self.epsilon})")
        plt.colorbar(im, ax=ax, label=r"$\psi$")

        ax = axes[0, 1]
        ax.plot(self.time_log, self.energy_log, 'b-', lw=1)
        ax.set_title("Free Energy")
        ax.set_xlabel("t")
        ax.set_ylabel("F")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(self.time_log, self.mass_log, 'g-', lw=1)
        ax.set_title("Mass (Mean)")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\langle \psi \rangle$")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(self.time_log, self.psi_max_log, 'r-', lw=1, label='max')
        ax.plot(self.time_log, self.psi_min_log, 'b-', lw=1, label='min')
        ax.set_title("Extrema")
        ax.set_xlabel("t")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Summary saved: {save_path}")
        plt.show()

    def plot_spectrum(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "sh_spectrum.png")
        psi_hat = fft.fftshift(fft.fft2(self.psi))
        power = np.abs(psi_hat) ** 2

        fig, ax = plt.subplots(figsize=(7, 6))
        k1d = 2.0 * np.pi * fft.fftshift(fft.fftfreq(self.N, d=self.dx))
        extent = [k1d[0], k1d[-1], k1d[0], k1d[-1]]
        im = ax.imshow(fft.fftshift(power), extent=extent, cmap='hot',
                       origin='lower', norm=plt.matplotlib.colors.LogNorm())
        ax.set_title("Power Spectrum")

        k_out = np.sqrt(self.q0_sq + np.sqrt(self.epsilon))
        k_in = np.sqrt(self.q0_sq - np.sqrt(self.epsilon)) if self.epsilon <= self.q0_sq ** 2 else 0
        theta = np.linspace(0, 2 * np.pi, 200)
        if k_in > 0:
            ax.plot(k_in * np.cos(theta), k_in * np.sin(theta), 'c--', lw=1.5)
        ax.plot(k_out * np.cos(theta), k_out * np.sin(theta), 'c--', lw=1.5, label='unstable ring')
        ax.set_xlim(-3 * self.q0, 3 * self.q0)
        ax.set_ylim(-3 * self.q0, 3 * self.q0)
        ax.legend()
        plt.colorbar(im, label="Power")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"  Spectrum saved: {save_path}")
        plt.show()


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 默认不录视频，如需视频改为 record_video=True
    solver = SHSolver(
        N=256, L=64.0,
        dt=0.01,        # 关键：dt 必须小，否则数值爆炸
        T=500.0,
        epsilon=0.3, q0=1.0,
        noise_amp=0.01,
        record_video=False,   # 默认关闭
        video_fps=15,
    )

    solver.run(log_interval=10, frame_interval=50)
    solver.plot_summary()
    solver.plot_spectrum()