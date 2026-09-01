"""
part3_dislocation.py — Model 3: Dislocation Energy (E_dis)
第三部分：位错能量计算（E_dis）

Creates an edge dislocation by shifting the right half of the crystal
upward by half a lattice unit, then computes the dislocation energy.
通过将晶体右半部分向上移动半个晶格单位创建刃位错，然后计算位错能。

Shift: for x > Lx/2, psi(x,y) = psi_perfect(x, y - delta_y)
位移：x > Lx/2 时，psi(x,y) = psi_perfect(x, y - delta_y)

Domain: 256 x 256 (as specified) / 区域：256 x 256

Simulations / 模拟（共7组）:
  - sigma_1, sigma_2: square crystal / 正方形晶体
  - sigma_1 through sigma_5: triangular crystal / 三角形晶体

Dislocation energy / 位错能:
    E_dis = E_total(dislocated) - E_perfect(perfect crystal)
    Reported as E_dis (energy per unit length in 2D)
"""

import numpy as np
from scipy.ndimage import shift as ndimage_shift
from pfc_core import DualPFCConfig, DualPFCEngine


class DislocationModel:
    """Edge dislocation energy calculator.
    刃位错能量计算器。"""

    def __init__(self, config: DualPFCConfig, crystal_type='square'):
        self.cfg = config
        self.crystal_type = crystal_type  # 'square' or 'triangular'
        self.engine = DualPFCEngine(config)
        self.psi = None
        self.psi_perfect = None
        self.psi_initial = None
        self.energies = []
        self.E_dis = None
        self.delta_y = self._compute_half_lattice_shift()

    def _compute_half_lattice_shift(self):
        """Compute the half-lattice shift in y-direction (physical units).
        计算y方向的半个晶格位移（物理单位）。

        Square: y-period = 2*pi/k0, half = pi/k0
        Triangular: y-period = 4*pi/(sqrt(3)*k0), half = 2*pi/(sqrt(3)*k0)
        """
        k0 = self.cfg.k0
        if self.crystal_type == 'square':
            return np.pi / k0
        else:  # triangular
            return 2.0 * np.pi / (np.sqrt(3.0) * k0)

    def build_perfect_crystal(self):
        """Build a perfect crystal field.
        构建完美晶体场。"""
        X, Y = self.engine.get_coordinate_grids()
        if self.crystal_type == 'square':
            psi = self.engine.square_crystal(X, Y)
        else:
            psi = self.engine.triangular_crystal(X, Y)
        self.psi_perfect = psi.copy()
        return psi

    def build_dislocation(self, from_relaxed=True):
        """Build dislocation by shifting right half upward by half lattice unit.
        通过将右半部分上移半个晶格单位构建位错。

        For x > Lx/2: shift the field up by delta_y.
        Uses sub-pixel interpolation for non-integer shifts.
        对非整数位移使用亚像素插值。

        Parameters:
            from_relaxed: if True, build from relaxed perfect crystal.
                          若为True，从弛豫后的完美晶体制备。
        """
        c = self.cfg
        if self.psi_perfect is None:
            self.build_perfect_crystal()

        psi = self.psi_perfect.copy()

        # Shift in grid units / 以网格单位表示的位移
        delta_grid = self.delta_y / c.dy

        # Shift the right half upward / 将右半部分上移
        # Shifting up = shift array indices downward = shift by -delta_grid
        right_half = psi[c.Nx // 2:, :].copy()
        shifted_right = ndimage_shift(right_half, shift=(-delta_grid, 0),
                                       mode='wrap')
        psi[c.Nx // 2:, :] = shifted_right

        # Enforce same average density as perfect crystal (mass conservation)
        # 强制与完美晶体具有相同的平均密度（质量守恒）
        mean_perfect = np.mean(self.psi_perfect)
        psi += (mean_perfect - np.mean(psi))

        # Conserve average density: add zero-mean noise / 守恒平均密度：添加零均值噪声
        noise = c.noise * np.random.randn(*psi.shape)
        noise -= np.mean(noise)
        psi += noise

        self.psi_initial = psi.copy()
        self.psi = psi.copy()
        return psi

    def run(self, n_steps=None, callback=None):
        """Relax the dislocation configuration.
        弛豫位错构型。"""
        n = n_steps if n_steps is not None else self.cfg.n_steps
        self.psi, self.energies = self.engine.relax(
            self.psi, n_steps=n, callback=callback)
        return self.psi

    def relax_perfect(self, n_steps=None):
        """Relax the perfect crystal for reference energy.
        弛豫完美晶体以获得参考能量。"""
        n = n_steps if n_steps is not None else self.cfg.n_steps
        if self.psi_perfect is None:
            self.build_perfect_crystal()
        self.psi_perfect, _ = self.engine.relax(
            self.psi_perfect, n_steps=n)
        return self.psi_perfect

    def prepare_and_run(self, n_relax_perfect=None, n_relax_disloc=None,
                        callback=None):
        """Proper workflow: relax perfect crystal, build dislocation from it,
        then relax both for the same additional steps so they have equal
        total relaxation time.
        正确流程：弛豫完美晶体，从中制备位错，然后两者再弛豫相同步数，
        确保总弛豫时间相同。

        Parameters:
            n_relax_perfect: initial relaxation of perfect crystal / 完美晶体初始弛豫
            n_relax_disloc: additional relaxation for both / 两者额外弛豫步数
            callback: progress callback / 进度回调
        """
        n_p = n_relax_perfect if n_relax_perfect is not None else self.cfg.n_steps
        n_d = n_relax_disloc if n_relax_disloc is not None else self.cfg.n_steps

        # Step 1: Build and relax perfect crystal / 第一步：构建并弛豫完美晶体
        self.build_perfect_crystal()
        self.relax_perfect(n_steps=n_p)

        # Step 2: Build dislocation from relaxed perfect crystal
        # 第二步：从弛豫后的完美晶体制备位错
        self.build_dislocation(from_relaxed=True)

        # Step 3: Relax both dislocation AND perfect crystal for n_d more steps
        # This ensures equal total relaxation: n_p + n_d for both
        # 第三步：位错和完美晶体都再弛豫n_d步，确保总弛豫相同
        self.run(n_steps=n_d, callback=callback)
        self.relax_perfect(n_steps=n_d)

        # Step 4: Compute energy / 第四步：计算能量
        self.compute_dislocation_energy()
        return self.psi

    def compute_dislocation_energy(self):
        eng = self.engine
        c = self.cfg
        psi_d = self.psi
        psi_p = self.psi_perfect

        # 计算场
        f_d = eng.energy_density(psi_d)
        mu_d = eng.chemical_potential(psi_d)
        f_p = eng.energy_density(psi_p)
        mu_p = eng.chemical_potential(psi_p)

        # 积分总巨势
        Omega_d = np.sum(f_d - mu_d * psi_d) * c.dx * c.dy
        Omega_p = np.sum(f_p - mu_p * psi_p) * c.dx * c.dy

        # 打印调试信息
        print(f"带位错总巨势 Ω_d = {Omega_d:.6f}")
        print(f"完美晶体总巨势 Ω_p = {Omega_p:.6f}")
        print(f"E_dis = Ω_d - Ω_p = {Omega_d - Omega_p:.6f}")

        self.E_dis = Omega_d - Omega_p
        return self.E_dis

    def get_results(self):
        """Return simulation results as a dictionary.
        以字典形式返回模拟结果。"""
        return {
            'model': 'Dislocation / 位错',
            'crystal_type': self.crystal_type,
            'sigma': self.cfg.sigma,
            'E_dis': self.E_dis,
            'delta_y': self.delta_y,
            'energies': self.energies,
            'psi_final': self.psi,
            'psi_perfect': self.psi_perfect,
            'psi_initial': self.psi_initial,
            'config': self.cfg,
        }


def run_dislocation_sigma_series(sigma_values_tri, sigma_values_sq,
                                 Nx=256, Ny=256, n_steps=2000,
                                 callback=None):
    """Run dislocation energy for all 7 simulations.
    运行全部7组位错能模拟。

    Parameters:
        sigma_values_tri: sigma values for triangular (5 values)
                          三角形晶体的sigma值（5个）
        sigma_values_sq: sigma values for square (2 values: sigma1, sigma2)
                         正方形晶体的sigma值（2个：sigma1, sigma2）
    Returns:
        results: list of result dictionaries / 结果字典列表
    """
    results = []

    # Triangular crystals: all 5 sigma / 三角形晶体：全部5个sigma
    print("Triangular crystal dislocations / 三角形晶体位错:")
    for sigma in sigma_values_tri:
        cfg = DualPFCConfig(Nx=Nx, Ny=Ny, sigma=sigma, n_steps=n_steps)
        model = DislocationModel(cfg, crystal_type='triangular')
        model.prepare_and_run(callback=callback)
        E_dis = model.E_dis
        print(f"  sigma={sigma:.3f}, triangular, E_dis={E_dis:.6f}")
        results.append(model.get_results())

    # Square crystals: sigma1, sigma2 / 正方形晶体：sigma1, sigma2
    print("Square crystal dislocations / 正方形晶体位错:")
    for sigma in sigma_values_sq:
        cfg = DualPFCConfig(Nx=Nx, Ny=Ny, sigma=sigma, n_steps=n_steps)
        model = DislocationModel(cfg, crystal_type='square')
        model.prepare_and_run(callback=callback)
        E_dis = model.E_dis
        print(f"  sigma={sigma:.3f}, square, E_dis={E_dis:.6f}")
        results.append(model.get_results())

    return results
