"""
part2_round_interface.py — Model 2: Circular (Round) Interface Energy
第二部分：圆形界面的界面能计算（核生长模型）

Builds a circular solid nucleus inside liquid and computes gamma_SL.
在液相中构建圆形固相晶核并计算界面能gamma_SL。

Uses nucleus growth model: circular seed of solid in liquid matrix.
使用核生长模型：液相基体中的圆形固相晶核。

For sigma_1, sigma_2: solid nucleus is square crystal.
对于sigma1, sigma2：晶核为正方形晶体。
For sigma_3, sigma_4, sigma_5: solid nucleus is triangular crystal.
对于sigma3, sigma4, sigma5：晶核为三角形晶体。

Interface energy / 界面能:
    gamma = (E_total - E_solid * V_s/V - E_liquid * V_l/V) / (2*pi*R)
    For circular interface: A_interface = 2*pi*R (circumference in 2D)
"""

import numpy as np
from pfc_core import DualPFCConfig, DualPFCEngine


class RoundInterfaceModel:
    """Circular nucleus interface energy calculator.
    圆形晶核界面能计算器。"""

    def __init__(self, config: DualPFCConfig, crystal_type='square',
                 nucleus_radius=None):
        self.cfg = config
        self.crystal_type = crystal_type  # 'square' or 'triangular'
        self.engine = DualPFCEngine(config)
        # Default nucleus radius: 1/4 of domain / 默认晶核半径为区域的1/4
        self.R = nucleus_radius if nucleus_radius is not None else min(config.Lx, config.Ly) / 4.0
        self.psi = None
        self.psi_initial = None
        self.energies = []
        self.gamma = None

    def build_initial_condition(self, interface_width=8.0):
        """Build initial circular nucleus: solid inside, liquid outside.
        构建初始圆形晶核：内部固相，外部液相。

        Uses a smooth tanh transition at the nucleus boundary.
        在晶核边界使用平滑tanh过渡。
        """
        c = self.cfg
        X, Y = self.engine.get_coordinate_grids()

        # Solid and liquid fields / 固相和液相场
        if self.crystal_type == 'square':
            psi_solid = self.engine.square_crystal(X, Y)
        else:
            psi_solid = self.engine.triangular_crystal(X, Y)
        psi_liquid = self.engine.liquid(X.shape, amplitude=c.noise)

        # Circular mask with smooth boundary / 带平滑边界的圆形掩膜
        cx, cy = c.Lx / 2.0, c.Ly / 2.0
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        width = interface_width * c.dx
        mix = 0.5 * (1.0 - np.tanh((dist - self.R) / width))

        psi = mix * psi_solid + (1.0 - mix) * psi_liquid
        self.psi_initial = psi.copy()
        self.psi = psi.copy()
        return psi

    def run(self, n_steps=None, callback=None):
        """Relax the nucleus to equilibrium (may grow or shrink).
        弛豫晶核至平衡态（可能生长或收缩）。"""
        n = n_steps if n_steps is not None else self.cfg.n_steps
        self.psi, self.energies = self.engine.relax(
            self.psi, n_steps=n, callback=callback)
        return self.psi

    def measure_nucleus_radius(self):
        """Measure the effective nucleus radius from the density field.
        从密度场测量有效晶核半径。

        Uses the amplitude envelope (absolute value of density modulation).
        使用密度调制的振幅包络。
        """
        c = self.cfg
        # Use local amplitude as solid fraction proxy / 用局部振幅作为固相分数代理
        psi = self.psi
        # Threshold: solid where |psi - mean| > 0.5 * amplitude
        psi_centered = psi - np.mean(psi)
        threshold = 0.3 * self.cfg.amplitude
        solid_mask = np.abs(psi_centered) > threshold

        # Count solid area and compute equivalent radius / 计算固相面积和等效半径
        solid_area = np.sum(solid_mask) * c.dx * c.dy
        R_eff = np.sqrt(solid_area / np.pi)
        return R_eff

    def compute_interface_energy(self, sample_width=20):
        """
        巨势法计算圆形晶核固液界面能（同part1场内取样逻辑，适配非平衡晶核）
        gamma = Omega_excess / (2*pi*R_eff)
        """
        c = self.cfg
        psi = self.psi
        eng = self.engine

        # 1. 全场自由能密度、化学势场（依赖pfc_core新增的chemical_potential）
        f_density = eng.energy_density(psi)
        mu_field = eng.chemical_potential(psi)

        # 2. 场内取样纯液相（远离中心晶核外圈）、纯固相（中心内部无界面区）
        Nx, Ny = c.Nx, c.Ny
        cx, cy = Nx//2, Ny//2
        # 固相取样：晶核中心小方块（远离圆形界面）
        solid_mask_center = np.zeros_like(psi, dtype=bool)
        solid_mask_center[cx-sample_width:cx+sample_width, cy-sample_width:cy+sample_width] = True
        # 液相取样：图像最外圈边界区域（远离晶核界面）
        liquid_mask_outer = np.zeros_like(psi, dtype=bool)
        liquid_mask_outer[:sample_width, :] = True
        liquid_mask_outer[-sample_width:, :] = True
        liquid_mask_outer[:, :sample_width] = True
        liquid_mask_outer[:, -sample_width:] = True

        # 取体相平均自由能密度、化学势、密度
        solid_f = np.mean(f_density[solid_mask_center])
        solid_mu = np.mean(mu_field[solid_mask_center])
        solid_rho = np.mean(psi[solid_mask_center])

        liquid_f = np.mean(f_density[liquid_mask_outer])
        liquid_mu = np.mean(mu_field[liquid_mask_outer])
        liquid_rho = np.mean(psi[liquid_mask_outer])

        # 3. 体相巨势密度 ω = f - μ·ρ
        omega_s = solid_f - solid_mu * solid_rho
        omega_l = liquid_f - liquid_mu * liquid_rho

        # 4. 统计固相面积分数（原有阈值逻辑保留）
        psi_centered = psi - np.mean(psi)
        threshold = 0.3 * c.amplitude
        solid_mask = np.abs(psi_centered) > threshold
        f_s = np.sum(solid_mask) / (c.Nx * c.Ny)
        f_l = 1.0 - f_s

        # 5. 总巨势、体相混合巨势、过剩巨势
        total_area = c.Lx * c.Ly
        Omega_total = np.sum(f_density - mu_field * psi) * c.dx * c.dy
        Omega_bulk_mix = (omega_s * f_s + omega_l * f_l) * total_area
        Omega_excess = Omega_total - Omega_bulk_mix

        # 6. 实测等效半径+真实界面周长
        R_eff = self.measure_nucleus_radius()
        A_interface = 2.0 * np.pi * R_eff

        if A_interface < 1e-6:
            self.gamma = 0.0
        else:
            self.gamma = Omega_excess / A_interface
        return self.gamma, R_eff

    def get_results(self):
        """Return simulation results as a dictionary.
        以字典形式返回模拟结果。"""
        R_eff = self.measure_nucleus_radius() if self.psi is not None else self.R
        return {
            'model': 'Round Interface / 圆形界面',
            'crystal_type': self.crystal_type,
            'sigma': self.cfg.sigma,
            'gamma_SL': self.gamma,
            'nucleus_radius_initial': self.R,
            'nucleus_radius_final': R_eff,
            'energies': self.energies,
            'psi_final': self.psi,
            'psi_initial': self.psi_initial,
            'config': self.cfg,
        }


def run_round_interface_sigma_series(sigma_values, crystal_map,
                                     Nx=256, Ny=256, n_steps=2000,
                                     nucleus_radius=None, callback=None):
    """Run round interface energy for a series of sigma values.
    对一系列sigma值运行圆形界面能计算。

    Parameters:
        sigma_values: list of sigma values / sigma值列表
        crystal_map: dict mapping sigma -> 'square' or 'triangular'
    Returns:
        results: list of result dictionaries / 结果字典列表
    """
    results = []
    for sigma in sigma_values:
        crystal_type = crystal_map.get(sigma, 'square')
        cfg = DualPFCConfig(Nx=Nx, Ny=Ny, sigma=sigma, n_steps=n_steps)
        model = RoundInterfaceModel(cfg, crystal_type, nucleus_radius)
        model.build_initial_condition()
        model.run(callback=callback)
        gamma, R_eff = model.compute_interface_energy()
        print(f"  sigma={sigma:.3f}, {crystal_type:12s}, "
              f"R_eff={R_eff:.2f}, gamma_SL={gamma:.6f}")
        results.append(model.get_results())
    return results
