"""
pfc_vacancy.py
PFC空位扩散分析模块
基于 Provatas & Elder 第8.5.2节

PFC的独特优势：能自洽描述晶体缺陷（空位、位错等）
而不需要额外引入缺陷自由度
"""

import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable
import warnings


@dataclass
class VacancyState:
    """空位状态数据"""
    position: Tuple[int, ...]      # 空位位置（网格坐标）
    formation_energy: float        # 形成能
    migration_energy: float        # 迁移能
    lifetime: float                # 寿命
    trajectory: List[Tuple]        # 运动轨迹


class VacancyDiffusion:
    """
    PFC空位扩散分析
    
    核心思想：在PFC中，空位不是人为引入的，而是密度场phi(r)的自然结果。
    空位对应于局部密度最小值（phi < 平均密度）。
    
    教材8.5.2节描述的方法：
    1. 识别密度场中的局部最小值作为空位位置
    2. 计算空位形成能 = 有空位的系统能量 - 完整晶体能量
    3. 追踪空位位置随时间的演化得到扩散系数
    """
    
    def __init__(self, dx: float = 1.0, dt: float = 1.0):
        """
        Parameters
        ----------
        dx : float
            空间步长
        dt : float
            时间步长
        """
        self.dx = dx
        self.dt = dt
        self.vacancy_history = []
        
    # ==================== 空位识别 ====================
    
    def find_vacancies(self, phi: np.ndarray, 
                       threshold: Optional[float] = None,
                       min_distance: int = 3) -> List[Tuple[int, ...]]:
        """
        从密度场中识别空位位置
        
        空位是局部密度最小值，满足：
        1. phi < threshold（低于周围密度）
        2. 是局部极小值点
        
        Parameters
        ----------
        phi : ndarray
            PFC密度场
        threshold : float or None
            密度阈值，None则使用自动估计
        min_distance : int
            两个空位之间的最小距离（像素）
            
        Returns
        -------
        list of tuple : 空位位置坐标
        """
        if threshold is None:
            # 自动阈值：使用平均密度减去标准差
            threshold = np.mean(phi) - 0.5 * np.std(phi)
        
        # 找到低于阈值的区域
        vacancy_mask = phi < threshold
        
        # 使用形态学操作找到局部最小值
        from scipy.ndimage import minimum_filter
        
        # 局部最小值滤波
        local_min = minimum_filter(phi, size=3) == phi
        
        # 空位是低于阈值且是局部最小值的点
        vacancy_points = vacancy_mask & local_min
        
        # 标记连通区域
        labeled, num_features = label(vacancy_points)
        
        # 计算每个区域的中心
        vacancies = []
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) > 0:
                # 使用质心作为空位位置
                com = center_of_mass(phi, labeled, i)
                # 取最近的整数坐标
                pos = tuple(int(round(c)) for c in com)
                vacancies.append(pos)
        
        # 过滤距离太近的空位（可能是同一个空位的多个标记）
        filtered = self._filter_by_distance(vacancies, min_distance)
        
        return filtered
    
    def _filter_by_distance(self, positions: List[Tuple], 
                           min_dist: int) -> List[Tuple]:
        """按距离过滤过近的点"""
        if not positions:
            return []
        
        filtered = [positions[0]]
        for pos in positions[1:]:
            too_close = False
            for existing in filtered:
                dist = np.sqrt(sum((a - b)**2 for a, b in zip(pos, existing)))
                if dist < min_dist:
                    too_close = True
                    break
            if not too_close:
                filtered.append(pos)
        
        return filtered
    
    # ==================== 空位创建与消除 ====================
    
    def create_vacancy(self, phi: np.ndarray, 
                       position: Tuple[int, ...],
                       radius: int = 2) -> np.ndarray:
        """
        在指定位置创建空位
        
        方法：局部降低密度，模拟原子缺失
        
        Parameters
        ----------
        phi : ndarray
            原始密度场
        position : tuple
            空位中心位置
        radius : int
            空位影响半径
            
        Returns
        -------
        ndarray : 修改后的密度场
        """
        phi_new = phi.copy()
        dims = phi.ndim
        
        # 创建球形/圆形空位区域
        slices = []
        for i, p in enumerate(position):
            start = max(0, p - radius)
            end = min(phi.shape[i], p + radius + 1)
            slices.append(slice(start, end))
        
        # 提取局部区域
        local = phi_new[tuple(slices)]
        
        # 创建距离矩阵
        grids = np.ogrid[[slice(0, s) for s in local.shape]]
        center = tuple(radius for _ in range(dims))
        dist_sq = sum((g - c)**2 for g, c in zip(grids, center))
        
        # 高斯型空位分布
        vacancy_profile = -np.exp(-dist_sq / (2 * (radius/2)**2))
        
        # 叠加到密度场
        local += vacancy_profile * 0.5  # 幅度可调
        
        phi_new[tuple(slices)] = local
        
        return phi_new
    
    def remove_vacancy(self, phi: np.ndarray,
                       position: Tuple[int, ...],
                       perfect_crystal: Callable) -> np.ndarray:
        """
        消除空位（用完美晶体填充）
        
        Parameters
        ----------
        phi : ndarray
            含空位的密度场
        position : tuple
            空位位置
        perfect_crystal : callable
            生成完美晶体的函数 phi(r)
            
        Returns
        -------
        ndarray : 修复后的密度场
        """
        # 在空位位置用完美晶体替换
        phi_new = phi.copy()
        # 实际实现需要知道完美晶体的相位
        # 这里简化处理
        return phi_new
    
    # ==================== 空位能量计算 ====================
    
    def compute_formation_energy(self, phi_with_vacancy: np.ndarray,
                                  phi_perfect: np.ndarray,
                                  free_energy_func: Callable,
                                  n0: float) -> float:
        """
        计算空位形成能
        
        E_form = F[phi_with_vacancy] - F[phi_perfect]
        
        教材8.5.2节：PFC中形成能自然包含熵贡献
        
        Parameters
        ----------
        phi_with_vacancy : ndarray
            含空位的密度场
        phi_perfect : ndarray
            完美晶体密度场
        free_energy_func : callable
            自由能泛函 F[phi]
        n0 : float
            平均密度（用于巨势计算）
            
        Returns
        -------
        float : 形成能
        """
        F_vac = free_energy_func(phi_with_vacancy)
        F_perf = free_energy_func(phi_perfect)
        
        # 巨势差（考虑粒子数变化）
        # Omega = F - mu*N
        # 形成能用巨势差更合适
        E_form = F_vac - F_perf
        
        return E_form
    
    def compute_migration_energy(self, phi: np.ndarray,
                                  vacancy_pos: Tuple[int, ...],
                                  neighbor_directions: List[Tuple],
                                  free_energy_func: Callable) -> float:
        """
        计算空位迁移能
        
        方法：将空位沿不同方向移动，找到能量最低路径
        
        Parameters
        ----------
        phi : ndarray
            密度场
        vacancy_pos : tuple
            当前空位位置
        neighbor_directions : list of tuple
            可能的迁移方向（最近邻位置）
        free_energy_func : callable
            
        Returns
        -------
        float : 迁移能垒
        """
        # 当前能量
        E0 = free_energy_func(phi)
        
        energies = [E0]
        
        for direction in neighbor_directions:
            # 尝试将空位移到邻居位置
            new_pos = tuple(v + d for v, d in zip(vacancy_pos, direction))
            
            # 检查边界
            if all(0 <= p < s for p, s in zip(new_pos, phi.shape)):
                # 创建迁移后的构型
                phi_trial = self._move_vacancy(phi, vacancy_pos, new_pos)
                E_trial = free_energy_func(phi_trial)
                energies.append(E_trial)
        
        # 迁移能 = 鞍点能量 - 初始能量
        E_saddle = max(energies)
        E_migration = E_saddle - E0
        
        return E_migration
    
    def _move_vacancy(self, phi: np.ndarray,
                      from_pos: Tuple[int, ...],
                      to_pos: Tuple[int, ...]) -> np.ndarray:
        """将空位从一个位置移动到另一个位置"""
        phi_new = phi.copy()
        
        # 简化的移动：交换密度值
        # 实际应该用更物理的方法
        temp = phi_new[from_pos]
        phi_new[from_pos] = phi_new[to_pos]
        phi_new[to_pos] = temp
        
        return phi_new
    
    # ==================== 扩散系数计算 ====================
    
    def compute_diffusion_coefficient(self, 
                                      phi_history: List[np.ndarray],
                                      time_interval: Optional[float] = None) -> dict:
        """
        从密度场时间序列计算空位扩散系数
        
        D = <(r(t) - r(0))^2> / (2*d*t)
        
        Parameters
        ----------
        phi_history : list of ndarray
            时间序列密度场
        time_interval : float or None
            时间步长，None则使用self.dt
            
        Returns
        -------
        dict : 扩散系数及相关统计量
        """
        if time_interval is None:
            time_interval = self.dt
        
        # 追踪每个时间步的空位位置
        trajectories = []
        current_pos = None
        
        for i, phi in enumerate(phi_history):
            vacancies = self.find_vacancies(phi)
            
            if not vacancies:
                continue
            
            if current_pos is None:
                current_pos = vacancies[0]
                trajectories = [current_pos]
            else:
                # 找到最近的空位（假设空位连续移动）
                distances = [np.linalg.norm(np.array(v) - np.array(current_pos)) 
                           for v in vacancies]
                nearest_idx = np.argmin(distances)
                current_pos = vacancies[nearest_idx]
                trajectories.append(current_pos)
        
        if len(trajectories) < 2:
            return {'D': None, 'error': 'Insufficient data'}
        
        # 计算均方位移
        trajectory_array = np.array(trajectories)
        displacements = np.diff(trajectory_array, axis=0)
        msd = np.cumsum(np.sum(displacements**2, axis=1))
        
        # 线性拟合求D
        times = np.arange(1, len(msd) + 1) * time_interval
        dims = phi_history[0].ndim
        
        # MSD = 2 * d * D * t
        slope = np.polyfit(times, msd, 1)[0]
        D = slope / (2.0 * dims)
        
        # 计算误差
        residuals = msd - (2 * dims * D * times)
        std_error = np.std(residuals) / np.sqrt(len(times))
        
        return {
            'D': D,
            'D_units': f'{self.dx**2/self.dt} (dx^2/dt)',
            'msd_data': msd,
            'times': times,
            'trajectory': trajectories,
            'std_error': std_error,
            'num_steps': len(trajectories)
        }
    
    def compute_correlation_function(self, phi_history: List[np.ndarray],
                                     max_lag: int = 100) -> np.ndarray:
        """
        计算空位位置的自相关函数
        
        用于分析扩散机制（随机行走 vs 关联跳跃）
        """
        # 提取空位轨迹
        trajectory = []
        for phi in phi_history[:max_lag]:
            vacancies = self.find_vacancies(phi)
            if vacancies:
                trajectory.append(vacancies[0])
        
        if len(trajectory) < 2:
            return np.array([])
        
        # 计算自相关
        trajectory = np.array(trajectory)
        mean_pos = np.mean(trajectory, axis=0)
        
        # 位置涨落
        fluctuations = trajectory - mean_pos
        
        # 自相关
        autocorr = np.correlate(fluctuations[:, 0], fluctuations[:, 0], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        return autocorr
    
    # ==================== 可视化工具 ====================
    
    def visualize_vacancy(self, phi: np.ndarray, 
                          vacancy_positions: List[Tuple[int, ...]],
                          save_path: Optional[str] = None):
        """
        可视化空位位置
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 密度场
        if phi.ndim == 2:
            im1 = axes[0].imshow(phi, cmap='RdBu_r', origin='lower')
            axes[0].set_title('Density Field φ(r)')
            
            # 标记空位
            for pos in vacancy_positions:
                axes[0].plot(pos[1], pos[0], 'ko', markersize=10, 
                           markerfacecolor='none', markeredgewidth=2)
            
            plt.colorbar(im1, ax=axes[0])
            
            # 密度直方图
            axes[1].hist(phi.flatten(), bins=50, alpha=0.7, color='blue')
            axes[1].axvline(np.mean(phi), color='r', linestyle='--', label='Mean')
            for pos in vacancy_positions:
                axes[1].axvline(phi[pos], color='g', linestyle=':', alpha=0.5)
            axes[1].set_xlabel('Density')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Density Distribution')
            axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def animate_diffusion(self, phi_history: List[np.ndarray],
                          interval: int = 100,
                          save_path: Optional[str] = None):
        """
        创建空位扩散动画
        """
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 初始帧
        im = ax.imshow(phi_history[0], cmap='RdBu_r', origin='lower')
        scatter, = ax.plot([], [], 'ko', markersize=15, 
                         markerfacecolor='none', markeredgewidth=3)
        
        def init():
            scatter.set_data([], [])
            return [im, scatter]
        
        def update(frame):
            phi = phi_history[frame]
            im.set_array(phi)
            
            vacancies = self.find_vacancies(phi)
            if vacancies:
                y, x = zip(*vacancies)
                scatter.set_data(x, y)
            
            ax.set_title(f'Time step {frame}')
            return [im, scatter]
        
        anim = FuncAnimation(fig, update, frames=len(phi_history),
                           init_func=init, interval=interval, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        plt.show()
        return anim


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建测试用的PFC密度场（模拟含空位的BCC晶体）
    from pfc_mode_approximation import ModeApproximation
    
    mode_solver = ModeApproximation()
    sol = mode_solver.solve_bcc(r=-0.5)
    
    # 生成简单测试场
    L = 64
    x = np.linspace(0, 4*np.pi, L)
    y = np.linspace(0, 4*np.pi, L)
    X, Y = np.meshgrid(x, y)
    
    # 简化的2D密度场（三角晶格近似）
    phi_test = sol.n0 + 2*sol.A * (np.cos(X) + np.cos(0.5*X + np.sqrt(3)/2*Y) 
                                  + np.cos(0.5*X - np.sqrt(3)/2*Y))
    
    # 添加空位（局部密度降低）
    vacancy_center = (L//2, L//2)
    Y_idx, X_idx = np.ogrid[:L, :L]
    dist_sq = (Y_idx - vacancy_center[0])**2 + (X_idx - vacancy_center[1])**2
    phi_test[dist_sq < 9] *= 0.5  # 局部降低密度
    
    # 初始化分析器
    analyzer = VacancyDiffusion(dx=4*np.pi/L, dt=0.1)
    
    print("=" * 50)
    print("空位分析测试")
    print("=" * 50)
    
    # 识别空位
    vacancies = analyzer.find_vacancies(phi_test, threshold=sol.n0)
    print(f"识别到 {len(vacancies)} 个空位:")
    for v in vacancies:
        print(f"  位置: {v}, 密度值: {phi_test[v]:.4f}")
    
    # 可视化
    analyzer.visualize_vacancy(phi_test, vacancies)