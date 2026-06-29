"""
pfc_mode_approximation.py
PFC一模式近似求解器
基于 Provatas & Elder 第8章
"""

import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings


@dataclass
class LatticeSolution:
    """晶格平衡解的数据容器"""
    n0: float           # 平均密度
    A: float            # 振幅
    a: float            # 晶格常数
    k_m: float          # 主倒格矢模长
    F: float            # 自由能密度
    G: float            # 巨势密度
    mu: float           # 化学势
    stable: bool        # 稳定性标志
    lattice_type: str   # 晶格类型


class ModeApproximation:
    """
    PFC一模式近似求解器
    
    将密度场近似为：
    phi(r) = n0 + A * sum_j(exp(i*k_j·r)) + c.c.
    
    其中对于不同晶格，倒格矢集合 {k_j} 不同：
    - 1D: 1对
    - 2D三角: 3对  
    - 3D BCC: 6对 (对应12个最近邻)
    """
    
    def __init__(self):
        self.solutions = {}
        
    # 1D Plane Wave
    # ==================== 1D 平面波 ====================
    
    def solve_1d(self, r: float, n0_guess: float = -0.2,
                 A_guess: float = 0.5) -> LatticeSolution:
        """
        1D平面波解（教材8.4.3节）
        
        phi(x) = n0 + 2A*cos(q*x)
        
        Parameters
        ----------
        r : float
            控制参数 r < 0
        n0_guess, A_guess : float
            初始猜测
            
        Returns
        -------
        LatticeSolution
        """
        def equations(vars):
            n0, A = vars
            if abs(A) < 1e-10:
                return [1e10, 1e10]
            
            # Optimal wave vector q = 1/sqrt(2) for standard PFC operator
            # 最优波矢 q = 1/sqrt(2) 对于标准PFC算符
            q = 1.0 / np.sqrt(2.0)
            
            # Partial derivatives of free energy w.r.t. n0 and A = 0
            # 自由能对n0和A的偏导 = 0
            # F = (r/2)*n0^2 + (u/4)*n0^4 + 3*(r+1)*A^2 + 9*u*A^4 + 6*u*n0^2*A^2
            # Simplified: Standard PFC free energy density
            # 简化：标准PFC自由能密度
            u = 1.0  # 标准值
            
            # dF/dn0 = 0
            eq1 = r * n0 + u * n0**3 + 6 * u * n0 * A**2
            
            # dF/dA = 0 (considering single-mode approximation)
            # dF/dA = 0 (考虑一模式近似)
            # 有效r: r_eff = r + (1-q^2)^2 = r (当q=1时)
            # 但标准PFC中 q_m = 1/sqrt(2) 对应峰值
            r_eff = r + (1 - 2*q**2)**2  # 修正到峰值位置
            
            eq2 = r_eff * A + 3 * u * A**3 + 3 * u * n0**2 * A
            
            return [eq1, eq2]
        
        from scipy.optimize import fsolve
        sol = fsolve(equations, [n0_guess, A_guess], full_output=True)
        
        n0, A = sol[0]
        
        # Compute free energy
        # 计算自由能
        q = 1.0 / np.sqrt(2.0)
        F = self._free_energy_1d(n0, A, r)
        
        # [EN] 化学势
        # 化学势
        mu = self._chemical_potential_1d(n0, A, r)
        
        # Grand potential
        # 巨势
        G = F - mu * n0
        
        # Stability check
        # 稳定性检查
        stable = self._check_stability_1d(n0, A, r)
        
        return LatticeSolution(
            n0=n0, A=A, a=2*np.pi/q, k_m=q,
            F=F, G=G, mu=mu,
            stable=stable, lattice_type='1D_plane'
        )
    
    def _free_energy_1d(self, n0: float, A: float, r: float, u: float = 1.0) -> float:
        """1D自由能密度"""
        q = 1.0 / np.sqrt(2.0)
        r_eff = r + (1 - 2*q**2)**2
        # Free energy = liquid part + solid correction
        # 自由能 = 液体部分 + 固体修正
        F_liquid = 0.5 * r * n0**2 + 0.25 * u * n0**4
        F_solid = 3 * r_eff * A**2 + 9 * u * A**4 + 6 * u * n0**2 * A**2
        return F_liquid + F_solid
    
    def _chemical_potential_1d(self, n0: float, A: float, r: float, u: float = 1.0) -> float:
        """化学势 mu = dF/dn0"""
        return r * n0 + u * n0**3 + 6 * u * n0 * A**2
    
    def _check_stability_1d(self, n0: float, A: float, r: float) -> bool:
        """检查Hessian矩阵正定性"""
        u = 1.0
        q = 1.0 / np.sqrt(2.0)
        r_eff = r + (1 - 2*q**2)**2
        
        # Hessian = [[d²F/dn0², d²F/dn0dA], [d²F/dA dn0, d²F/dA²]]
        H11 = r + 3*u*n0**2 + 6*u*A**2
        H22 = 6*r_eff + 54*u*A**2 + 6*u*n0**2
        H12 = 12*u*n0*A
        
        det = H11 * H22 - H12**2
        return H11 > 0 and det > 0
    
    # 2D Triangular Lattice
    # ==================== 2D 三角晶格 ====================
    
    def solve_triangular(self, r: float, n0_guess: float = -0.2,
                         A_guess: float = 0.3) -> LatticeSolution:
        """
        2D三角晶格解（教材8.4.2节）
        
        phi(r) = n0 + A * sum_{j=1}^{3} [exp(i*k_j·r) + c.c.]
        
        倒格矢: k_j = q_m * [cos(2πj/3), sin(2πj/3)]
        三角晶格常数 a = 4π/(sqrt(3)*q_m)
        
        Parameters
        ----------
        r : float
            控制参数
        """
        # But in standard PFC, q_m = 1/sqrt(2) corresponds to peak
        # 标准PFC中峰值位置 q_m = 1/sqrt(2)
        q_m = 1.0 / np.sqrt(2.0)
        
        def free_energy_triangular(vars):
            """计算三角晶格自由能密度（教材方程8.4.2相关）"""
            n0, A = vars
            
            u = 1.0
            
            # Liquid part
            # 液体部分
            F_liquid = 0.5 * r * n0**2 + 0.25 * u * n0**4
            
            # Solid part (3 pairs of reciprocal lattice vectors)
            # 固体部分（3对倒格矢）
            # 每个模式的能量贡献
            r_eff = r + (1 - q_m**2)**2  # 在q_m处的有效r
            
            # Linear term: 3 * r_eff * A^2 (3 mode pairs)
            # 线性项: 3 * r_eff * A^2 (3对模式)
            F_linear = 3 * r_eff * A**2
            
            # Self-interaction in quartic term
            # 四次项中的自相互作用
            # 来自 (phi^4) 展开，考虑umklapp过程
            # 3对模式产生特定的角向依赖
            F_quartic_self = 45 * u * A**4  # 自相互作用
            
            # Coupling with n0
            # 与n0的耦合
            F_coupling = 18 * u * n0**2 * A**2
            
            # Umklapp term (triangular lattice special)
            # umklapp项（三角晶格特殊）
            # k1 + k2 + k3 = 0 的共振
            F_umklapp = 2 * u * A**3 * n0  # 三模式耦合（如果存在）
            # 实际上标准PFC中主要是4次项
            
            # Correct umklapp: 6A^4 term from phi^4
            # 正确的umklapp: 来自phi^4的6A^4项
            # 对于三角晶格，有特定的几何因子
            F_total = F_liquid + F_linear + F_quartic_self + F_coupling
            
            return F_total
        
        def equations(vars):
            n0, A = vars
            if A < 0:
                A = abs(A)  # 振幅取正
            
            u = 1.0
            q_m = 1.0 / np.sqrt(2.0)
            r_eff = r + (1 - q_m**2)**2
            
            # dF/dn0 = 0
            eq1 = r * n0 + u * n0**3 + 18 * u * n0 * A**2
            
            # dF/dA = 0
            eq2 = 6 * r_eff * A + 180 * u * A**3 + 36 * u * n0**2 * A
            
            return [eq1, eq2]
        
        sol = fsolve(equations, [n0_guess, abs(A_guess)], full_output=True)
        n0, A = sol[0]
        A = abs(A)
        
        # Compute lattice constant
        # 计算晶格常数
        a_tri = 4.0 * np.pi / (np.sqrt(3.0) * q_m)
        
        # [EN] 自由能
        # 自由能
        F = free_energy_triangular([n0, A])
        
        # [EN] 化学势
        # 化学势
        mu = r * n0 + n0**3 + 18 * n0 * A**2
        
        # Grand potential
        # 巨势
        G = F - mu * n0
        
        # [EN] 稳定性
        # 稳定性
        stable = self._check_stability_triangular(n0, A, r)
        
        return LatticeSolution(
            n0=n0, A=A, a=a_tri, k_m=q_m,
            F=F, G=G, mu=mu,
            stable=stable, lattice_type='2D_triangular'
        )
    
    def _check_stability_triangular(self, n0: float, A: float, r: float) -> bool:
        """检查三角晶格稳定性"""
        u = 1.0
        q_m = 1.0 / np.sqrt(2.0)
        r_eff = r + (1 - q_m**2)**2
        
        H11 = r + 3*u*n0**2 + 18*u*A**2
        H22 = 6*r_eff + 540*u*A**2 + 36*u*n0**2
        H12 = 36*u*n0*A
        
        det = H11 * H22 - H12**2
        return H11 > 0 and det > 0
    
    # 3D BCC Lattice
    # ==================== 3D BCC晶格 ====================
    
    def solve_bcc(self, r: float, n0_guess: float = -0.2,
                  A_guess: float = 0.2) -> LatticeSolution:
        """
        3D BCC晶格解（教材8.4.1节）
        
        phi(r) = n0 + A * sum_{j=1}^{6} [exp(i*k_j·r) + c.c.]
        
        倒格矢对应BCC的12个最近邻（6对）：
        k_j = (2π/a) * (±1, ±1, 0) 及其排列
        
        Parameters
        ----------
        r : float
            控制参数
        """
        q_m = 1.0 / np.sqrt(2.0)
        
        def equations(vars):
            n0, A = vars
            if A < 0:
                A = abs(A)
            
            u = 1.0
            r_eff = r + (1 - q_m**2)**2
            
            # dF/dn0 = 0
            # 6 mode pairs, each coupled with n0
            # 6对模式，每对与n0耦合
            eq1 = r * n0 + u * n0**3 + 36 * u * n0 * A**2
            
            # dF/dA = 0
            # 6 mode pairs, complex umklapp terms
            # 6对模式，复杂的umklapp项
            eq2 = 12 * r_eff * A + 1260 * u * A**3 + 72 * u * n0**2 * A
            
            return [eq1, eq2]
        
        sol = fsolve(equations, [n0_guess, abs(A_guess)], full_output=True)
        n0, A = sol[0]
        A = abs(A)
        
        # BCC lattice constant
        # BCC晶格常数
        a_bcc = 2.0 * np.pi * np.sqrt(2.0) / q_m
        
        # [EN] 自由能
        # 自由能
        F = self._free_energy_bcc(n0, A, r)
        
        # [EN] 化学势
        # 化学势
        mu = r * n0 + n0**3 + 36 * n0 * A**2
        
        # Grand potential
        # 巨势
        G = F - mu * n0
        
        # [EN] 稳定性
        # 稳定性
        stable = self._check_stability_bcc(n0, A, r)
        
        return LatticeSolution(
            n0=n0, A=A, a=a_bcc, k_m=q_m,
            F=F, G=G, mu=mu,
            stable=stable, lattice_type='3D_BCC'
        )
    
    def _free_energy_bcc(self, n0: float, A: float, r: float, u: float = 1.0) -> float:
        """BCC自由能密度"""
        q_m = 1.0 / np.sqrt(2.0)
        r_eff = r + (1 - q_m**2)**2
        
        F_liquid = 0.5 * r * n0**2 + 0.25 * u * n0**4
        F_linear = 6 * r_eff * A**2  # 6对模式
        F_quartic = 315 * u * A**4    # BCC几何因子
        F_coupling = 36 * u * n0**2 * A**2
        
        return F_liquid + F_linear + F_quartic + F_coupling
    
    def _check_stability_bcc(self, n0: float, A: float, r: float) -> bool:
        """检查BCC稳定性"""
        u = 1.0
        q_m = 1.0 / np.sqrt(2.0)
        r_eff = r + (1 - q_m**2)**2
        
        H11 = r + 3*u*n0**2 + 36*u*A**2
        H22 = 12*r_eff + 3780*u*A**2 + 72*u*n0**2
        H12 = 72*u*n0*A
        
        det = H11 * H22 - H12**2
        return H11 > 0 and det > 0
    
    # Phase Stability Comparison
    # ==================== 相稳定性比较 ====================
    
    def compare_lattices(self, r_range: Tuple[float, float], 
                         num_points: int = 100) -> dict:
        """
        比较不同晶格结构的稳定性（教材8.4节核心结果）
        
        确定给定r下哪种晶格结构最稳定
        
        Parameters
        ----------
        r_range : (r_min, r_max)
        num_points : int
            
        Returns
        -------
        dict : 各r值下的最稳定晶格
        """
        r_values = np.linspace(r_range[0], r_range[1], num_points)
        
        results = {
            'r': r_values,
            'stable_lattice': [],
            'G_liquid': [],
            'G_1d': [],
            'G_triangular': [],
            'G_bcc': []
        }
        
        for r in r_values:
            # Liquid reference state
            # 液体参考态
            G_liquid = 0.0  # 参考点
            
            # Solutions for each lattice
            # 各晶格解
            try:
                sol_1d = self.solve_1d(r)
                G_1d = sol_1d.G if sol_1d.stable else np.inf
            except:
                G_1d = np.inf
            
            try:
                sol_tri = self.solve_triangular(r)
                G_tri = sol_tri.G if sol_tri.stable else np.inf
            except:
                G_tri = np.inf
            
            try:
                sol_bcc = self.solve_bcc(r)
                G_bcc = sol_bcc.G if sol_bcc.stable else np.inf
            except:
                G_bcc = np.inf
            
            results['G_liquid'].append(G_liquid)
            results['G_1d'].append(G_1d)
            results['G_triangular'].append(G_tri)
            results['G_bcc'].append(G_bcc)
            
            # Determine most stable
            # 确定最稳定的
            G_values = {'liquid': G_liquid, '1D': G_1d, 
                       'triangular': G_tri, 'BCC': G_bcc}
            best = min(G_values, key=G_values.get)
            results['stable_lattice'].append(best)
        
        return results
    
    def find_phase_boundaries(self, r_range: Tuple[float, float],
                              tolerance: float = 1e-3) -> List[dict]:
        """
        寻找相边界（液-固、固-固转变点）
        
        Returns
        -------
        list of dict : 每个相变点的信息
        """
        comparison = self.compare_lattices(r_range, num_points=500)
        
        boundaries = []
        r_vals = comparison['r']
        lattices = comparison['stable_lattice']
        
        for i in range(len(r_vals)-1):
            if lattices[i] != lattices[i+1]:
                # Find phase transition point
                # 找到相变点
                r_cross = (r_vals[i] + r_vals[i+1]) / 2
                boundaries.append({
                    'r': r_cross,
                    'phase_left': lattices[i],
                    'phase_right': lattices[i+1]
                })
        
        return boundaries


# Usage Example
# ==================== 使用示例 ====================

if __name__ == "__main__":
    solver = ModeApproximation()
    
    # Solve BCC lattice
    # 求解BCC晶格
    print("=" * 50)
    print("BCC晶格解 (r = -0.5)")
    print("=" * 50)
    bcc = solver.solve_bcc(r=-0.5)
    print(f"平均密度 n0 = {bcc.n0:.4f}")
    print(f"振幅 A = {bcc.A:.4f}")
    print(f"晶格常数 a = {bcc.a:.4f}")
    print(f"自由能密度 F = {bcc.F:.6f}")
    print(f"巨势 G = {bcc.G:.6f}")
    print(f"稳定性: {bcc.stable}")
    
    # Solve triangular lattice
    # 求解三角晶格
    print("\n" + "=" * 50)
    print("三角晶格解 (r = -0.5)")
    print("=" * 50)
    tri = solver.solve_triangular(r=-0.5)
    print(f"平均密度 n0 = {tri.n0:.4f}")
    print(f"振幅 A = {tri.A:.4f}")
    print(f"晶格常数 a = {tri.a:.4f}")
    print(f"自由能密度 F = {tri.F:.6f}")
    print(f"巨势 G = {tri.G:.6f}")
    print(f"稳定性: {tri.stable}")
    
    # Compare different lattices
    # 比较不同晶格
    print("\n" + "=" * 50)
    print("相稳定性比较 (r ∈ [-1, 0])")
    print("=" * 50)
    comparison = solver.compare_lattices((-1.0, 0.0), num_points=50)
    
    # Statistics of regions where each phase appears
    # 统计各相出现的区域
    from collections import Counter
    phase_counts = Counter(comparison['stable_lattice'])
    print("各相稳定区域占比:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count/len(comparison['r'])*100:.1f}%")
    
    # Find phase boundaries (liquid-solid, solid-solid transition points)
    # 寻找相边界
    boundaries = solver.find_phase_boundaries((-1.0, 0.0))
    print(f"\n发现 {len(boundaries)} 个相边界:")
    for b in boundaries:
        print(f"  r = {b['r']:.4f}: {b['phase_left']} -> {b['phase_right']}")