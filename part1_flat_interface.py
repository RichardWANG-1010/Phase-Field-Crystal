"""
part1_flat_interface.py — Model 1: Flat (Smooth) Interface Energy
第一部分：平滑界面的界面能计算

Builds a solid-liquid flat interface and computes gamma_SL.
构建固-液平滑界面并计算界面能gamma_SL。

For sigma_1, sigma_2: solid phase is square crystal.
对于sigma1, sigma2：固相为正方形晶体。
For sigma_3, sigma_4, sigma_5: solid phase is triangular crystal.
对于sigma3, sigma4, sigma5：固相为三角形晶体。

Interface energy / 界面能:
    gamma = (E_total - E_solid * V_s/V - E_liquid * V_l/V) * L / A_interface
    For flat interface: A_interface = Ly (interface normal along x)
"""

import numpy as np
from pfc_core import DualPFCConfig, DualPFCEngine


class FlatInterfaceModel:
    """Flat solid-liquid interface energy calculator.
    平滑固液界面能计算器。"""

    def __init__(self, config: DualPFCConfig, crystal_type='square'):
        self.cfg = config
        self.crystal_type = crystal_type  # 'square' or 'triangular'
        self.engine = DualPFCEngine(config)
        self.psi = None
        self.psi_initial = None
        self.energies = []
        self.gamma = None

    def build_initial_condition(self, interface_width=8.0):
        """Build initial flat interface: solid on left, liquid on right.
        构建初始平滑界面：左侧固相，右侧液相。

        Uses a smooth tanh transition across the interface.
        使用平滑的tanh函数过渡界面。
        """
        c = self.cfg
        X, Y = self.engine.get_coordinate_grids()

        # Solid and liquid fields / 固相和液相场
        if self.crystal_type == 'square':
            psi_solid = self.engine.square_crystal(X, Y)
        else:
            psi_solid = self.engine.triangular_crystal(X, Y)
        psi_liquid = self.engine.liquid(X.shape, amplitude=c.noise)

        # Smooth mixing function / 平滑混合函数
        # Interface at x = Lx/2, transition width in grid units
        x_center = c.Lx / 2.0
        width = interface_width * c.dx
        mix = 0.5 * (1.0 - np.tanh((X - x_center) / width))

        psi = mix * psi_solid + (1.0 - mix) * psi_liquid
        self.psi_initial = psi.copy()
        self.psi = psi.copy()
        return psi

    def run(self, n_steps=None, callback=None):
        """Relax the interface to equilibrium.
        弛豫界面至平衡态。"""
        n = n_steps if n_steps is not None else self.cfg.n_steps
        self.psi, self.energies = self.engine.relax(
            self.psi, n_steps=n, callback=callback)
        return self.psi

    def compute_interface_energy(self, sample_width=20):
        """
        巨势（Grand Potential）形式的固液界面能计算
        适配平衡态与非平衡态，贴合热力学标准定义
        sample_width: 两侧体相取样的列数，需远离界面
        """
        c = self.cfg
        psi = self.psi

        # 1. 计算全场自由能密度、化学势场
        f_density = self.engine.energy_density(psi)
        mu_field = self.engine.chemical_potential(psi)  # 化学势 μ = δF/δψ

        # 2. 从体系两侧取样体相区域，计算体相参考值
        # 左侧纯固相区
        solid_f = np.mean(f_density[:sample_width, :])
        solid_mu = np.mean(mu_field[:sample_width, :])
        solid_rho = np.mean(psi[:sample_width, :])
        # 右侧纯液相区
        liquid_f = np.mean(f_density[-sample_width:, :])
        liquid_mu = np.mean(mu_field[-sample_width:, :])
        liquid_rho = np.mean(psi[-sample_width:, :])

        # 3. 计算体相巨势密度 ω = f - μ·ρ
        omega_solid = solid_f - solid_mu * solid_rho
        omega_liquid = liquid_f - liquid_mu * liquid_rho

        # 4. 统计固相面积分数（和之前逻辑一致）
        threshold = 0.5 * (solid_rho + liquid_rho)
        solid_mask = psi > threshold
        f_s = np.sum(solid_mask) / (c.Nx * c.Ny)
        f_l = 1.0 - f_s

        # 5. 计算总巨势、体相混合巨势、过剩巨势
        total_area = c.Nx * c.Ny
        Omega_total = np.sum(f_density - mu_field * psi)
        Omega_bulk_mix = (omega_solid * f_s + omega_liquid * f_l) * total_area
        Omega_excess = Omega_total - Omega_bulk_mix

        # 6. 平界面长度 = Ly，得到单位长度界面能
        interface_length = c.Ny  # 二维y方向贯穿全域
        self.gamma = Omega_excess / interface_length

        return self.gamma

    def get_results(self):
        """Return simulation results as a dictionary.
        以字典形式返回模拟结果。"""
        return {
            'model': 'Flat Interface / 平滑界面',
            'crystal_type': self.crystal_type,
            'sigma': self.cfg.sigma,
            'gamma_SL': self.gamma,
            'energies': self.energies,
            'psi_final': self.psi,
            'psi_initial': self.psi_initial,
            'config': self.cfg,
        }


def run_flat_interface_sigma_series(sigma_values, crystal_map,
                                    Nx=256, Ny=256, n_steps=2000,
                                    callback=None):
    """Run flat interface energy for a series of sigma values.
    对一系列sigma值运行平滑界面能计算。

    Parameters:
        sigma_values: list of sigma values / sigma值列表
        crystal_map: dict mapping sigma -> 'square' or 'triangular'
                     映射sigma到晶体类型
    Returns:
        results: list of result dictionaries / 结果字典列表
    """
    results = []
    for sigma in sigma_values:
        crystal_type = crystal_map.get(sigma, 'square')
        cfg = DualPFCConfig(Nx=Nx, Ny=Ny, sigma=sigma, n_steps=n_steps)
        model = FlatInterfaceModel(cfg, crystal_type)
        model.build_initial_condition()
        model.run(callback=callback)
        gamma = model.compute_interface_energy()
        print(f"  sigma={sigma:.3f}, {crystal_type:12s}, gamma_SL={gamma:.6f}")
        results.append(model.get_results())
    return results
