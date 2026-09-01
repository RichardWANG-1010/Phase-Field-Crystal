"""
pfc_core.py — Dual-Amplitude (Two-Mode) PFC Core Engine
双振幅（双模）相场晶体核心引擎

This module implements the two-mode PFC free energy functional,
pseudo-spectral time evolution, and energy computation.
本模块实现双模PFC自由能泛函、伪谱时间演化和能量计算。

Free energy / 自由能:
    F = integral [ 1/2 psi L psi - tau/3 psi^3 + 1/4 psi^4 ] dV
    L = [r + (1+laplacian)^2] [(q^2+laplacian)^2 / q^4]

where r is the reduced temperature (mapped from sigma),
q is the ratio of the two wave-numbers (q = sqrt(3) for triangular second shell).
其中 r 为约化温度（由 sigma 映射），q 为两波数之比。
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq, fftshift


class DualPFCConfig:
    """Configuration parameters for the dual-amplitude PFC model.
    双振幅PFC模型的配置参数。"""

    def __init__(self, Nx=256, Ny=256, Lx=None, Ly=None,
                 sigma=0.0, tau=1.0, q=np.sqrt(3.0),
                 k0=1.0, dt=0.5, n_steps=2000,
                 amplitude=0.3, noise=0.01, mean_density=0.0):
        # Grid parameters / 网格参数
        self.Nx = Nx
        self.Ny = Ny
        # Physical domain size; default gives ~8 grid pts per 2pi period
        # 物理区域大小，默认每个2pi周期约8个网格点
        self.Lx = Lx if Lx is not None else Nx * np.pi / 4.0
        self.Ly = Ly if Ly is not None else Ny * np.pi / 4.0
        self.dx = self.Lx / Nx
        self.dy = self.Ly / Ny

        # PFC model parameters / PFC模型参数
        self.sigma = sigma          # control parameter / 控制参数
        self.tau = tau              # cubic coefficient / 三次项系数
        self.q = q                  # wave-number ratio / 波数比
        self.k0 = k0                # primary wave number / 主波数
        self.r = self._sigma_to_r(sigma)  # reduced temperature / 约化温度

        # Time-stepping / 时间步进
        self.dt = dt
        self.n_steps = n_steps

        # Initial condition / 初始条件
        self.amplitude = amplitude  # crystal amplitude / 晶体振幅
        self.noise = noise          # initial noise / 初始噪声
        self.mean_density = mean_density  # average density offset / 平均密度偏移

    @staticmethod
    def _sigma_to_r(sigma):
        """Map sigma to reduced temperature r.
        将sigma映射为约化温度r。
        Low sigma -> square phase; high sigma -> triangular phase.
        低sigma对应正方形相，高sigma对应三角形相。"""
        # Linear mapping: sigma in [-0.5, 0.5] -> r in [-0.6, -0.2]
        # 线性映射，可根据实际相图调整
        return -0.4 + sigma * 0.4

    def clone(self, **kwargs):
        """Create a copy with overridden parameters.
        创建参数被覆盖的副本。"""
        params = {
            'Nx': self.Nx, 'Ny': self.Ny, 'Lx': self.Lx, 'Ly': self.Ly,
            'sigma': self.sigma, 'tau': self.tau, 'q': self.q,
            'k0': self.k0, 'dt': self.dt, 'n_steps': self.n_steps,
            'amplitude': self.amplitude, 'noise': self.noise,
            'mean_density': self.mean_density,
        }
        params.update(kwargs)
        return DualPFCConfig(**params)


class DualPFCEngine:
    """
    Dual-amplitude PFC simulation engine using pseudo-spectral method.
    使用伪谱法的双振幅PFC模拟引擎。

    The linear operator L(k) is evaluated in Fourier space;
    nonlinear terms are computed in real space.
    线性算子L(k)在傅里叶空间计算，非线性项在实空间计算。
    """

    def __init__(self, config: DualPFCConfig):
        self.cfg = config
        self._build_operators()

    # ------------------------------------------------------------------ #
    # Fourier-space operators / 傅里叶空间算子
    # ------------------------------------------------------------------ #
    def _build_operators(self):
        """Precompute k-space grids and the linear operator.
        预计算k空间网格和线性算子。"""
        c = self.cfg
        # Wave number grids / 波数网格
        kx = 2 * np.pi * fftfreq(c.Nx, d=c.dx)
        ky = 2 * np.pi * fftfreq(c.Ny, d=c.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX ** 2 + self.KY ** 2

        q2 = c.q ** 2
        # Linear operator L(k) = [r + (1-k^2)^2] [(q^2-k^2)^2 / q^4]
        self.L_hat = (c.r + (1.0 - self.K2) ** 2) * (q2 - self.K2) ** 2 / q2 ** 2

        # Denominator for semi-implicit stepping / 半隐式步进分母
        # dpsi/dt = laplacian(dF/dpsi) = -k^2 [L psi - tau psi^2 + psi^3]
        # Implicit part: 1 + dt k^2 L(k)
        self.denom = 1.0 + c.dt * self.K2 * self.L_hat

        # Laplacian in k-space / k空间拉普拉斯
        self.k2 = self.K2

    # ------------------------------------------------------------------ #
    # Crystal initial conditions / 晶体初始条件
    # ------------------------------------------------------------------ #
    def square_crystal(self, x, y, amplitude=None):
        """Generate a square-lattice density field.
        生成正方形晶格密度场。
        psi = mean_density + A[cos(k0 x) + cos(k0 y)]"""
        A = amplitude if amplitude is not None else self.cfg.amplitude
        k0 = self.cfg.k0
        return self.cfg.mean_density + A * (np.cos(k0 * x) + np.cos(k0 * y))

    def triangular_crystal(self, x, y, amplitude=None):
        """Generate a triangular (hexagonal) lattice density field.
        生成三角形（六角）晶格密度场。
        psi = mean_density + A[cos(k0 x) + 2 cos(k0 x/2) cos(sqrt(3) k0 y/2)]"""
        A = amplitude if amplitude is not None else self.cfg.amplitude
        k0 = self.cfg.k0
        return self.cfg.mean_density + A * (np.cos(k0 * x) +
                     2.0 * np.cos(k0 * x / 2.0) * np.cos(np.sqrt(3) * k0 * y / 2.0))

    def liquid(self, shape, amplitude=None):
        """Generate a uniform liquid (mean density + noise).
        生成均匀液相（平均密度+噪声）。"""
        amp = amplitude if amplitude is not None else self.cfg.noise
        return self.cfg.mean_density + amp * np.random.randn(*shape)

    def get_coordinate_grids(self):
        """Return real-space coordinate grids X, Y.
        返回实空间坐标网格X, Y。"""
        c = self.cfg
        x = np.linspace(0, c.Lx, c.Nx, endpoint=False)
        y = np.linspace(0, c.Ly, c.Ny, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    # ------------------------------------------------------------------ #
    # Time evolution / 时间演化
    # ------------------------------------------------------------------ #
    def step(self, psi):
        """Perform one semi-implicit time step.
        执行一次半隐式时间步进。

        psi_k(t+dt) = [psi_k(t) - dt k^2 (nonlinear)_k] / [1 + dt k^2 L(k)]
        nonlinear = -tau psi^2 + psi^3
        """
        c = self.cfg
        psi_k = fft2(psi)

        # Nonlinear terms in real space / 实空间非线性项
        nl = -c.tau * psi ** 2 + psi ** 3
        nl_k = fft2(nl)

        # Semi-implicit update / 半隐式更新
        psi_k_new = (psi_k - c.dt * self.k2 * nl_k) / self.denom
        return np.real(ifft2(psi_k_new))

    def relax(self, psi, n_steps=None, callback=None):
        """Relax the field for n_steps iterations.
        弛豫场n_steps步。

        Parameters:
            psi: initial field / 初始场
            n_steps: number of steps (default from config) / 步数
            callback: function(step, psi, energy) called each 50 steps
                      每50步调用的回调函数
        Returns:
            psi: relaxed field / 弛豫后的场
            energies: list of (step, total_energy) / 能量历史
        """
        n = n_steps if n_steps is not None else self.cfg.n_steps
        energies = []
        for i in range(n):
            psi = self.step(psi)
            if i % 50 == 0 or i == n - 1:
                e = self.total_energy(psi)
                energies.append((i, e))
                if callback is not None:
                    callback(i, psi, e)
        return psi, energies

    # ------------------------------------------------------------------ #
    # Energy computation / 能量计算
    # ------------------------------------------------------------------ #
    def total_energy(self, psi):
        """Compute total free energy F = integral f dV.
        计算总自由能F = integral f dV。
        f = 1/2 psi L psi - tau/3 psi^3 + 1/4 psi^4"""
        c = self.cfg
        psi_k = fft2(psi)
        Lpsi = np.real(ifft2(self.L_hat * psi_k))
        f_density = (0.5 * psi * Lpsi
                     - c.tau / 3.0 * psi ** 3
                     + 0.25 * psi ** 4)
        return np.sum(f_density) * c.dx * c.dy

    def energy_density(self, psi):
        """Compute local free energy density.
        计算局部自由能密度。"""
        c = self.cfg
        psi_k = fft2(psi)
        Lpsi = np.real(ifft2(self.L_hat * psi_k))
        return (0.5 * psi * Lpsi
                - c.tau / 3.0 * psi ** 3
                + 0.25 * psi ** 4)
        
    def chemical_potential(self, psi):
        """
        Local chemical potential μ = δF/δψ
        局域化学势，自由能泛函对密度场ψ的变分导数
        Formula: μ = L[ψ] + τ ψ² - ψ³
        L[ψ] = IFFT( L_hat · FFT(ψ) )
        """
        # 线性算子作用于ψ
        psi_k = fft2(psi)
        Lpsi = np.real(ifft2(self.L_hat * psi_k))
        # PFC化学势完整表达式
        mu = Lpsi + self.cfg.tau * psi**2 - psi**3
        return mu

    def bulk_energy(self, crystal_type='square', n_relax=3000):
        """Compute bulk equilibrium energy for a perfect crystal or liquid.
        计算完美晶体或液相的体平衡能量。

        Parameters:
            crystal_type: 'square', 'triangular', or 'liquid'
        Returns:
            E_bulk: total energy / 总能量
            psi: equilibrium field / 平衡场
        """
        X, Y = self.get_coordinate_grids()
        if crystal_type == 'square':
            psi = self.square_crystal(X, Y)
        elif crystal_type == 'triangular':
            psi = self.triangular_crystal(X, Y)
        else:
            psi = self.liquid(X.shape)

        psi, _ = self.relax(psi, n_steps=n_relax)
        return self.total_energy(psi), psi

    # ------------------------------------------------------------------ #
    # Structure factor / 结构因子
    # ------------------------------------------------------------------ #
    def structure_factor(self, psi):
        """Compute the 2D structure factor S(k) = |psi_k|^2.
        计算二维结构因子S(k) = |psi_k|^2。"""
        psi_k = fft2(psi)
        S = np.abs(psi_k) ** 2
        return fftshift(S)

    def k_grid_centered(self):
        """Return centered k-space grids for plotting.
        返回用于绘图的中心化k空间网格。"""
        return fftshift(self.KX), fftshift(self.KY)
    
    
        

