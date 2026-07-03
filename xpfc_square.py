"""
XPFC Square Lattice Solver - 交互选择版
用户输入选择：单晶(single) / 多晶(poly)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import fft2, ifft2, fftfreq, fftshift
import taichi as ti
import time
import os
import subprocess
import sys
import io

os.environ["TI_THREADS"] = "8"
ti.init(arch=ti.cpu, cpu_max_num_threads=8)

# ==================== 配置 ====================
CONFIG = {
    "N": 256,
    "L": 64.0,
    "sigma": 1.5,
    "eta": 1.0,
    "chi": 1.0,
    "dt": 0.01,
    "n0": -0.25,
    "noise_amp": 0.03,      # 多晶默认噪声
    "steps": 50000,
    "sample": 1000,
    "seed_amp": 0.15,
    "frame_interval": 50,
    "video_fps": 30,
    "output_dir": "xpfc_output",
    "video_name": "xpfc_evolution.mp4",
    "preset": "ultrafast",
    "crf": "23",
    "width": 480,
    "height": 480,
}

# ==================== Taichi Kernels ====================

@ti.kernel
def compute_nl_kernel(n: ti.types.ndarray(ndim=2), nl: ti.types.ndarray(ndim=2), eta: ti.f32, chi: ti.f32):
    for i, j in ti.ndrange(n.shape[0], n.shape[1]):
        nv = n[i, j]
        nl[i, j] = -0.5 * eta * nv * nv + 0.3333333333333 * chi * nv * nv * nv

@ti.kernel
def compute_step_kernel(n_k_r: ti.types.ndarray(ndim=2), n_k_i: ti.types.ndarray(ndim=2),
                        nl_k_r: ti.types.ndarray(ndim=2), nl_k_i: ti.types.ndarray(ndim=2),
                        k2: ti.types.ndarray(ndim=2), linear_op: ti.types.ndarray(ndim=2), dt: ti.f32):
    for i, j in ti.ndrange(n_k_r.shape[0], n_k_r.shape[1]):
        denom = 1.0 + dt * k2[i, j] * linear_op[i, j]
        nr = n_k_r[i, j] - dt * k2[i, j] * nl_k_r[i, j]
        ni = n_k_i[i, j] - dt * k2[i, j] * nl_k_i[i, j]
        n_k_r[i, j] = nr / denom
        n_k_i[i, j] = ni / denom

@ti.kernel
def enforce_density_kernel(n_k_r: ti.types.ndarray(ndim=2), n_k_i: ti.types.ndarray(ndim=2), n0: ti.f32, N: ti.i32):
    n_k_r[0, 0] = n0 * N * N
    n_k_i[0, 0] = 0.0

@ti.kernel
def clip_field_kernel(f: ti.types.ndarray(ndim=2)):
    for i, j in ti.ndrange(f.shape[0], f.shape[1]):
        v = f[i, j]
        if v < -2.0: f[i, j] = -2.0
        elif v > 2.0: f[i, j] = 2.0

# ==================== 核心求解器 ====================

class XPFC_Square_CPU:
    def __init__(self, N=256, L=64.0, peaks=None, sigma=1.5,
                 eta=1.0, chi=1.0, dt=0.01, n0=-0.25, noise_amp=0.03):
        self.N = N
        self.L = L
        self.dx = L / N
        self.dt = dt
        self.n0 = n0
        self.noise_amp = noise_amp
        self.eta = eta
        self.chi = chi
        self.sigma = sigma

        if peaks is None:
            peaks = [
                {"q": 1.0, "alpha": 0.25, "weight": 1.0},
                {"q": np.sqrt(2), "alpha": 0.25, "weight": 0.85},
            ]
        self.peaks = peaks
        self._build_grid()
        self._build_C2()

        self.n = np.zeros((N, N), dtype=np.float32)
        self.n_k = np.zeros((N, N), dtype=np.complex64)
        self.nl = np.zeros((N, N), dtype=np.float32)
        self.nl_k = np.zeros((N, N), dtype=np.complex64)
        self.n_k_r = np.zeros((N, N), dtype=np.float32)
        self.n_k_i = np.zeros((N, N), dtype=np.float32)
        self.nl_k_r = np.zeros((N, N), dtype=np.float32)
        self.nl_k_i = np.zeros((N, N), dtype=np.float32)

    def _build_grid(self):
        x = np.linspace(0, self.L, self.N, endpoint=False)
        self.x, self.y = np.meshgrid(x, x)
        kx = 2 * np.pi * fftfreq(self.N, self.dx)
        ky = 2 * np.pi * fftfreq(self.N, self.dx)
        self.kx, self.ky = np.meshgrid(kx, ky)
        self.k2 = (self.kx**2 + self.ky**2).astype(np.float32)

    def _build_C2(self):
        C2_k = np.zeros((self.N, self.N), dtype=np.float32)
        for peak in self.peaks:
            q, alpha, w = peak["q"], peak["alpha"], peak["weight"]
            G1 = [(q, 0), (-q, 0), (0, q), (0, -q)]
            G2 = [(q, q), (q, -q), (-q, q), (-q, -q)]
            for gx, gy in G1 + G2:
                dk2 = (self.kx - gx)**2 + (self.ky - gy)**2
                C2_k += w * np.exp(-dk2 / (2 * alpha**2))
        C2_k *= self.sigma
        C2_k[0, 0] = 0.3
        self.C2_k = C2_k.astype(np.float32)
        self.linear_op = (1.0 - self.C2_k).astype(np.float32)

    def initialize_single_crystal(self, seed_q=None, seed_amp=0.15):
        """单晶：完美周期性晶格，无噪声"""
        if seed_q is None:
            seed_q = 2 * np.pi / 4.0
        seed = seed_amp * (np.cos(seed_q * self.x) + np.cos(seed_q * self.y))
        self.n[:] = (self.n0 + seed).astype(np.float32)
        self.n_k = fft2(self.n)
        self.n_k_r[:] = self.n_k.real
        self.n_k_i[:] = self.n_k.imag

    def initialize_polycrystal(self, seed=True, seed_q=None, seed_amp=0.15, noise_amp=None):
        """多晶：随机噪声 + 种子，形成多个晶畴"""
        if noise_amp is None:
            noise_amp = self.noise_amp
        
        # 随机噪声（不同种子产生不同晶畴）
        np.random.seed(42)
        noise = np.random.normal(0, noise_amp, (self.N, self.N)).astype(np.float32)
        self.n[:] = self.n0 + noise
        
        # 多个随机位置的种子，帮助成核
        if seed:
            if seed_q is None:
                seed_q = 2 * np.pi / 4.0
            
            # 主种子
            seed_pattern = seed_amp * (np.cos(seed_q * self.x) + np.cos(seed_q * self.y))
            self.n += seed_pattern.astype(np.float32)
            
            # 额外的随机相位种子（促进多晶成核）
            n_seeds = 5
            for _ in range(n_seeds):
                px = np.random.uniform(0, self.L)
                py = np.random.uniform(0, self.L)
                phase = np.random.uniform(0, 2*np.pi)
                amp = seed_amp * 0.3  # 较小的振幅
                local_seed = amp * np.cos(seed_q * (self.x - px) + phase) * np.cos(seed_q * (self.y - py) + phase)
                self.n += local_seed.astype(np.float32)
        
        self.n_k = fft2(self.n)
        self.n_k_r[:] = self.n_k.real
        self.n_k_i[:] = self.n_k.imag

    def step(self):
        self.n[:] = ifft2(self.n_k).real
        compute_nl_kernel(self.n, self.nl, self.eta, self.chi)
        self.nl_k = fft2(self.nl)
        self.nl_k_r[:] = self.nl_k.real
        self.nl_k_i[:] = self.nl_k.imag
        compute_step_kernel(self.n_k_r, self.n_k_i, self.nl_k_r, self.nl_k_i,
                           self.k2, self.linear_op, self.dt)
        enforce_density_kernel(self.n_k_r, self.n_k_i, self.n0, self.N)
        self.n_k = self.n_k_r + 1j * self.n_k_i
        self.n[:] = ifft2(self.n_k).real
        clip_field_kernel(self.n)
        self.n_k = fft2(self.n)
        self.n_k_r[:] = self.n_k.real
        self.n_k_i[:] = self.n_k.imag

# ==================== 视频编码器 ====================

class VideoEncoder:
    def __init__(self, output_path, width, height, fps=30, crf=23, preset="ultrafast"):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-preset", preset,
            "-threads", "0",
            output_path
        ]
        
        print(f"🎬  ffmpeg: {' '.join(self.cmd)}")
        
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    def write_frame(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=fig.dpi)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.LANCZOS)
        self.proc.stdin.write(img.tobytes())
        self.proc.stdin.flush()
        self.frame_count += 1
        buf.close()
    
    def close(self):
        print(f"\n🎬  Closing pipe... ({self.frame_count} frames written)")
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("⚠️  timeout, killing ffmpeg...")
            self.proc.kill()
            self.proc.wait()
        if self.proc.returncode == 0:
            print("✅  Video encoding complete")
        else:
            print(f"❌  ffmpeg exited with code {self.proc.returncode}")
        return self.proc.returncode == 0

def create_frame_figure(n, L, step, width, height, vmin=-2, vmax=2):
    dpi = 100
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.12, 0.12, 0.72, 0.72])
    im = ax.imshow(n, cmap='RdBu_r', origin='lower',
                   extent=[0, L, 0, L], vmin=vmin, vmax=vmax)
    ax.set_title(f'Step {step}', fontsize=10)
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('y', fontsize=9)
    cax = fig.add_axes([0.88, 0.12, 0.04, 0.72])
    plt.colorbar(im, cax=cax)
    return fig

def plot_final_result(sim, save_path, init_type):
    """绘制最终结果，标题显示模式"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    title_prefix = "Single Crystal" if init_type == "single" else "Polycrystal"

    im0 = axes[0].imshow(sim.n, cmap='RdBu_r', origin='lower',
                         extent=[0, sim.L, 0, sim.L], vmin=-2, vmax=2)
    axes[0].set_title(f'{title_prefix} - Density Field n(r)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    S = np.log(np.abs(fftshift(sim.n_k))**2 + 1)
    im1 = axes[1].imshow(S, cmap='inferno', origin='lower')
    axes[1].set_title(f'{title_prefix} - Structure Factor S(k)')
    axes[1].set_xlabel('k_x')
    axes[1].set_ylabel('k_y')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    idx = 0
    axes[2].plot(sim.kx[idx, :], sim.C2_k[idx, :], 'b-', label='C2(kx,0)')
    axes[2].plot(sim.ky[:, idx], sim.C2_k[:, idx], 'r--', label='C2(0,ky)')
    axes[2].set_title(f'{title_prefix} - Correlation Function C2(k)')
    axes[2].set_xlabel('k')
    axes[2].set_ylabel('C2(k)')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_xlim([-3, 3])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"✅  Final plot saved: {save_path}")
    plt.show()
    return fig

# ==================== 用户交互 ====================

def get_user_choice():
    """获取用户选择的模拟模式"""
    print("\n" + "=" * 50)
    print("XPFC Square Lattice Solver")
    print("=" * 50)
    print("请选择模拟模式:")
    print("  [1] 单晶 (Single Crystal) - 完美周期性晶格")
    print("  [2] 多晶 (Polycrystal) - 随机噪声成核，有晶界")
    print("=" * 50)
    
    while True:
        choice = input("输入 1 或 2 (默认 1): ").strip()
        
        if choice == "" or choice == "1":
            return "single"
        elif choice == "2":
            return "poly"
        else:
            print("❌ 无效输入，请重新输入 1 或 2")

def get_custom_params():
    """询问是否自定义参数"""
    print("\n是否使用默认参数?")
    print(f"  n0={CONFIG['n0']}, sigma={CONFIG['sigma']}, dt={CONFIG['dt']}")
    print(f"  steps={CONFIG['steps']}, seed_amp={CONFIG['seed_amp']}")
    
    choice = input("输入 y 自定义，或直接回车使用默认: ").strip().lower()
    
    if choice == 'y':
        try:
            CONFIG['n0'] = float(input(f"n0 (默认 {CONFIG['n0']}): ") or CONFIG['n0'])
            CONFIG['sigma'] = float(input(f"sigma (默认 {CONFIG['sigma']}): ") or CONFIG['sigma'])
            CONFIG['dt'] = float(input(f"dt (默认 {CONFIG['dt']}): ") or CONFIG['dt'])
            CONFIG['steps'] = int(input(f"steps (默认 {CONFIG['steps']}): ") or CONFIG['steps'])
            CONFIG['seed_amp'] = float(input(f"seed_amp (默认 {CONFIG['seed_amp']}): ") or CONFIG['seed_amp'])
            
            if CONFIG['steps'] > 100000:
                print("⚠️  步数过多，建议不超过 100000")
        except ValueError:
            print("⚠️  输入无效，使用默认值")

# ==================== 主程序 ====================

def main():
    # 用户选择模式
    init_type = get_user_choice()
    
    # 是否自定义参数
    get_custom_params()
    
    # 根据模式设置输出文件名
    cfg = CONFIG.copy()
    if init_type == "poly":
        cfg["video_name"] = "xpfc_polycrystal.mp4"
        cfg["noise_amp"] = 0.05  # 多晶用更大噪声
    else:
        cfg["video_name"] = "xpfc_single_crystal.mp4"
    
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌  ffmpeg 未安装")
        sys.exit(1)
    
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, cfg["video_name"])
    
    print("\n" + "=" * 50)
    print(f"模式: {'单晶' if init_type == 'single' else '多晶'}")
    print(f"参数: n0={cfg['n0']}, sigma={cfg['sigma']}, dt={cfg['dt']}, steps={cfg['steps']}")
    print("=" * 50)
    
    # 初始化求解器
    sim = XPFC_Square_CPU(
        N=cfg["N"], L=cfg["L"], sigma=cfg["sigma"],
        eta=cfg["eta"], chi=cfg["chi"], dt=cfg["dt"],
        n0=cfg["n0"], noise_amp=cfg["noise_amp"]
    )
    
    # 根据模式初始化
    if init_type == "single":
        sim.initialize_single_crystal(seed_amp=cfg["seed_amp"])
        print("Initialized: Single Crystal (完美晶格)")
    else:
        sim.initialize_polycrystal(seed=True, seed_amp=cfg["seed_amp"], noise_amp=cfg["noise_amp"])
        print("Initialized: Polycrystal (随机噪声 + 多种子)")
    
    print(f"Initial: min={sim.n.min():.4f}, max={sim.n.max():.4f}, amp={sim.n.max()-sim.n.min():.4f}")
    
    print("Warming up...")
    sim.step()
    print(f"After 1 step: min={sim.n.min():.4f}, max={sim.n.max():.4f}, amp={sim.n.max()-sim.n.min():.4f}")
    
    # 视频编码
    encoder = VideoEncoder(
        output_path=video_path,
        width=cfg["width"],
        height=cfg["height"],
        fps=cfg["video_fps"],
        crf=cfg["crf"],
        preset=cfg["preset"]
    )
    
    fig = create_frame_figure(sim.n, sim.L, 0, cfg["width"], cfg["height"])
    encoder.write_frame(fig)
    plt.close(fig)
    print("Frame 0 written")
    
    # 主循环
    print("\nRunning simulation...")
    t_start = time.time()
    
    for i in range(1, cfg["steps"] + 1):
        sim.step()
        
        if i % cfg["sample"] == 0:
            n_mean = sim.n.mean()
            n_max = sim.n.max()
            n_min = sim.n.min()
            amp = n_max - n_min
            print(f"Step {i:5d}/{cfg['steps']}: <n>={n_mean:.4f}, max={n_max:.4f}, min={n_min:.4f}, amp={amp:.4f}")
            
            if amp < 0.01 and i > 5000:
                print(f"  ⚠️  Warning: amplitude collapsed to {amp:.4f}")
        
        if i % cfg["frame_interval"] == 0:
            fig = create_frame_figure(sim.n, sim.L, i, cfg["width"], cfg["height"])
            encoder.write_frame(fig)
            plt.close(fig)
            print(f"  Frame: step {i}")
    
    elapsed = time.time() - t_start
    print(f"\nSimulation: {elapsed:.2f}s | {cfg['steps']/elapsed:.1f} steps/sec")
    
    encoder.close()
    
    # 最终结果
    final_plot_path = os.path.join(out_dir, f"final_{init_type}.png")
    plot_final_result(sim, final_plot_path, init_type)
    
    print(f"\n✅  Output: {os.path.abspath(out_dir)}")
    print(f"    Video: {video_path}")
    print(f"    Plot:  {final_plot_path}")

if __name__ == "__main__":
    main()