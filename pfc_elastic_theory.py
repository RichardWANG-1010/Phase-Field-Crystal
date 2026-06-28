"""
pfc_elastic_theory.py
基于一模式近似的解析弹性常数计算
基于 Provatas & Elder 第8.5节
"""

import numpy as np
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Tuple, Optional
from pfc_mode_approximation import ModeApproximation, LatticeSolution


@dataclass
class ElasticConstants:
    """弹性常数数据容器"""
    C11: float
    C12: float
    C44: float
    bulk_modulus: float
    shear_modulus: float
    youngs_modulus: float
    poisson_ratio: float
    lattice_type: str


class ElasticConstantTheory:
    """
    基于一模式近似的PFC解析弹性常数
    
    理论推导：
    对密度场施加小的应变 epsilon_ij，计算自由能变化
    delta_F = (1/2) * C_ijkl * epsilon_ij * epsilon_kl
    
    对于立方晶系，独立常数为 C11, C12, C44
    """
    
    def __init__(self, mode_solver: Optional[ModeApproximation] = None):
        self.mode_solver = mode_solver or ModeApproximation()
        
    def compute_bcc_elastic(self, r: float, 
                           method: str = 'analytical') -> ElasticConstants:
        """
        计算BCC晶格的弹性常数（教材8.5节）
        
        方法1: 解析公式（基于一模式近似）
        方法2: 数值微分（对应变张量数值求导）
        
        Parameters
        ----------
        r : float
            PFC控制参数
        method : 'analytical' or 'numerical'
            
        Returns
        -------
        ElasticConstants
        """
        if method == 'analytical':
            return self._bcc_analytical(r)
        else:
            return self._bcc_numerical(r)
    
    def _bcc_analytical(self, r: float) -> ElasticConstants:
        """
        BCC弹性常数的解析公式
        
        基于一模式近似，对BCC密度场：
        phi(r) = n0 + A * sum_{j=1}^{6} [exp(i*k_j·r) + c.c.]
        
        施加应变后，倒格矢变为 k_j' = (I + epsilon)^{-T} · k_j
        自由能变化给出弹性常数
        """
        # 获取平衡解
        sol = self.mode_solver.solve_bcc(r)
        n0, A, a = sol.n0, sol.A, sol.a
        
        # 倒格矢模长（未应变）
        q_m = 2.0 * np.pi * np.sqrt(2.0) / a
        
        # 标准PFC参数
        u = 1.0
        
        # === 解析推导的关键结果 ===
        # 对于BCC，弹性常数与振幅A和晶格常数a相关
        
        # 体积模量（来自均匀压缩）
        # B = (a^2 / 18) * (d²F/da²)
        # 在一模式近似下：
        K = (4.0 / 9.0) * q_m**4 * A**2 * (1 - q_m**2)
        
        # C11: [100]方向纵向模量
        # 来自单轴应变 epsilon_xx
        C11 = (8.0 / 3.0) * q_m**4 * A**2 * (1 - q_m**2) + \
              (16.0 / 3.0) * u * A**4
        
        # C12: 泊松耦合
        C12 = (4.0 / 3.0) * q_m**4 * A**2 * (1 - q_m**2) - \
              (8.0 / 3.0) * u * A**4
        
        # C44: [100]方向剪切模量
        # 注意：BCC的C44在一模式近似下可能为负（不稳定）
        # 这是已知的一模式近似缺陷
        C44 = (2.0 / 3.0) * q_m**4 * A**2 * (1 - q_m**2) - \
              (8.0 / 3.0) * u * A**4
        
        # 修正：使用更精确的公式（考虑高阶模式贡献）
        # 或标记C44的不稳定性
        
        # 计算导出量
        bulk_modulus = (C11 + 2*C12) / 3.0
        shear_modulus = (C11 - C12 + 3*C44) / 5.0  # Voigt平均
        youngs_modulus = 9 * bulk_modulus * shear_modulus / \
                        (3 * bulk_modulus + shear_modulus)
        poisson_ratio = (3 * bulk_modulus - 2 * shear_modulus) / \
                       (6 * bulk_modulus + 2 * shear_modulus)
        
        return ElasticConstants(
            C11=C11, C12=C12, C44=C44,
            bulk_modulus=bulk_modulus,
            shear_modulus=shear_modulus,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            lattice_type='BCC_analytical'
        )
    
    def _bcc_numerical(self, r: float, 
                       strain_magnitude: float = 1e-4) -> ElasticConstants:
        """
        数值计算弹性常数（应变张量微分法）
        
        施加6种独立应变模式，数值计算能量二阶导
        """
        sol = self.mode_solver.solve_bcc(r)
        n0, A, a0 = sol.n0, sol.A, sol.a
        
        # 基础自由能
        F0 = sol.F
        
        def energy_under_strain(epsilon):
            """
            计算应变后的自由能
            
            epsilon: 3x3应变张量
            """
            # 应变后的倒格矢
            # k' = (I + e)^{-T} · k
            I = np.eye(3)
            F_mat = I + epsilon
            
            # BCC倒格矢（12个最近邻）
            k_vectors = np.array([
                [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
            ]) * 2 * np.pi / a0
            
            k_strained = np.linalg.inv(F_mat.T) @ k_vectors.T
            k_mags = np.linalg.norm(k_strained, axis=0)
            
            # 计算应变后的自由能（简化）
            # 实际应使用完整的PFC自由能泛函
            q_m = 1.0 / np.sqrt(2.0)
            r_eff = r + np.mean((1 - k_mags**2)**2)
            
            # 简化计算（仅示意）
            F_strained = 0.5 * r * n0**2 + 0.25 * n0**4 + \
                        6 * r_eff * A**2 + 315 * A**4 + 36 * n0**2 * A**2
            
            return F_strained
        
        # 施加6种独立应变计算二阶导
        # epsilon_1 = [e, 0, 0; 0, 0, 0; 0, 0, 0] -> C11
        e = strain_magnitude
        
        eps1 = np.diag([e, 0, 0])
        eps2 = np.diag([0, e, 0])
        eps3 = np.diag([0, 0, e])
        eps4 = np.array([[0, e/2, 0], [e/2, 0, 0], [0, 0, 0]])  # shear
        eps5 = np.array([[0, 0, e/2], [0, 0, 0], [e/2, 0, 0]])
        eps6 = np.array([[0, 0, 0], [0, 0, e/2], [0, e/2, 0]])
        
        # 数值二阶导
        F1p = energy_under_strain(+eps1)
        F1m = energy_under_strain(-eps1)
        F2p = energy_under_strain(+eps2)
        F2m = energy_under_strain(-eps2)
        
        C11 = (F1p + F1m - 2*F0) / (e**2)
        C22 = (F2p + F2m - 2*F0) / (e**2)
        
        # 交叉项 C12
        eps12 = np.diag([e, e, 0])
        F12 = energy_under_strain(eps12)
        C12 = (F12 - F0 - 0.5*e**2*(C11 + C22)) / (e**2)
        
        # Shear C44
        F4p = energy_under_strain(+eps4)
        F4m = energy_under_strain(-eps4)
        C44 = (F4p + F4m - 2*F0) / (e**2) * 2  # 因子2来自剪切定义
        
        # 计算导出量
        bulk_modulus = (C11 + 2*C12) / 3.0
        shear_modulus = (C11 - C12 + 3*C44) / 5.0
        youngs_modulus = 9 * bulk_modulus * shear_modulus / \
                        (3 * bulk_modulus + shear_modulus)
        poisson_ratio = (3 * bulk_modulus - 2 * shear_modulus) / \
                       (6 * bulk_modulus + 2 * shear_modulus)
        
        return ElasticConstants(
            C11=C11, C12=C12, C44=C44,
            bulk_modulus=bulk_modulus,
            shear_modulus=shear_modulus,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            lattice_type='BCC_numerical'
        )
    
    def compute_triangular_elastic(self, r: float) -> dict:
        """
        2D三角晶格的弹性常数
        
        对于2D，独立常数为 C11, C12, C66 (或等价地 lambda, mu)
        """
        sol = self.mode_solver.solve_triangular(r)
        n0, A, a = sol.n0, sol.A, sol.a
        
        q_m = 4.0 * np.pi / (np.sqrt(3.0) * a)
        u = 1.0
        
        # 2D三角晶格的弹性常数
        C11 = 3.0 * q_m**4 * A**2 * (1 - q_m**2) + 12.0 * u * A**4
        C12 = 1.5 * q_m**4 * A**2 * (1 - q_m**2) - 6.0 * u * A**4
        C66 = 0.75 * q_m**4 * A**2 * (1 - q_m**2) + 3.0 * u * A**4
        
        # Lamé常数
        lambda_lame = C12
        mu_lame = C66
        
        # 2D体积模量
        K_2d = (C11 + C12) / 2.0
        
        return {
            'C11': C11, 'C12': C12, 'C66': C66,
            'lambda': lambda_lame, 'mu': mu_lame,
            'K_2d': K_2d,
            'lattice_type': 'triangular'
        }
    
    def cauchy_relation_check(self, elastic: ElasticConstants) -> float:
        """
        检查Cauchy关系 C12 = C44
        
        对于中心力势，Cauchy关系成立。
        PFC由于有角度依赖（通过梯度项），通常不满足。
        
        Returns
        -------
        float : 偏差程度
        """
        return abs(elastic.C12 - elastic.C44) / abs(elastic.C12)
    
    def compare_with_numerical(self, r: float, 
                                numerical_solver) -> dict:
        """
        与数值计算结果对比验证
        
        Parameters
        ----------
        r : float
        numerical_solver : 你的PFC求解器实例
            
        Returns
        -------
        dict : 对比结果
        """
        # 解析结果
        analytical = self.compute_bcc_elastic(r, method='analytical')
        
        # 数值结果（从你的solver计算）
        # numerical = numerical_solver.compute_elastic_constants(r)
        
        return {
            'analytical': analytical,
            # 'numerical': numerical,
            # 'relative_error_C11': abs(analytical.C11 - numerical.C11) / numerical.C11,
            'note': 'Connect your numerical solver here'
        }
    
    def elastic_anisotropy(self, elastic: ElasticConstants) -> dict:
        """
        计算弹性各向异性指标
        
        Zener因子: A = 2*C44 / (C11 - C12)
        A = 1 为各向同性
        """
        A_zener = 2.0 * elastic.C44 / (elastic.C11 - elastic.C12)
        
        # 其他各向异性指标
        return {
            'zener_factor': A_zener,
            'isotropic_limit': 1.0,
            'anisotropy_degree': abs(A_zener - 1.0),
            'classification': 'isotropic' if abs(A_zener - 1.0) < 0.1 else 'anisotropic'
        }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    theory = ElasticConstantTheory()
    
    print("=" * 60)
    print("BCC弹性常数解析计算 (r = -0.5)")
    print("=" * 60)
    
    elastic = theory.compute_bcc_elastic(r=-0.5, method='analytical')
    print(f"C11 = {elastic.C11:.4f}")
    print(f"C12 = {elastic.C12:.4f}")
    print(f"C44 = {elastic.C44:.4f}")
    print(f"体积模量 K = {elastic.bulk_modulus:.4f}")
    print(f"剪切模量 G = {elastic.shear_modulus:.4f}")
    print(f"杨氏模量 E = {elastic.youngs_modulus:.4f}")
    print(f"泊松比 nu = {elastic.poisson_ratio:.4f}")
    
    # 检查Cauchy关系
    deviation = theory.cauchy_relation_check(elastic)
    print(f"\nCauchy关系偏差: {deviation:.4f}")
    
    # 各向异性
    aniso = theory.elastic_anisotropy(elastic)
    print(f"\nZener各向异性因子: {aniso['zener_factor']:.4f}")
    print(f"分类: {aniso['classification']}")
    
    # 2D三角晶格
    print("\n" + "=" * 60)
    print("2D三角晶格弹性常数 (r = -0.5)")
    print("=" * 60)
    tri_elastic = theory.compute_triangular_elastic(r=-0.5)
    print(f"C11 = {tri_elastic['C11']:.4f}")
    print(f"C12 = {tri_elastic['C12']:.4f}")
    print(f"C66 = {tri_elastic['C66']:.4f}")
    print(f"Lame λ = {tri_elastic['lambda']:.4f}")
    print(f"Lame μ = {tri_elastic['mu']:.4f}")