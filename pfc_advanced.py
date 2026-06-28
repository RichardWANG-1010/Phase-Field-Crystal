"""
pfc_advanced.py - 高级分析模块集成
将 pfc_mode_approximation, pfc_elastic_theory, pfc_vacancy 与 PFC 求解器连接
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

# 导入三个分析模块（需放在同一目录）
from pfc_mode_approximation import ModeApproximation, LatticeSolution
from pfc_elastic_theory import ElasticConstantTheory, ElasticConstants
from pfc_vacancy import VacancyDiffusion, VacancyState


class PFCAdvancedAnalysis:
    """
    Mixin 类：集成一模式近似、弹性理论、空位分析三大模块
    
    使用方式：
        class PurePFCSolver(PFCBase, ..., PFCAdvancedAnalysis):
            pass
    """
    
    def __init__(self, *args, **kwargs):
        # 关键：调用 MRO 中下一个父类的 __init__，支持多继承
        super().__init__(*args, **kwargs)
        
        # 分析器实例（惰性初始化，节省内存）
        self._mode_solver = None
        self._elastic_theory = None
        self._vacancy_analyzer = None
        
        # 空位追踪历史（循环缓冲区）
        self._phi_history = []
        self._history_interval = 10      # 每10步记录一次
        self._max_history = 500          # 最多保存500帧
        
        # 分析日志
        self.vacancy_log = []            # [(step, num_vacancies), ...]
        self.elastic_log = []            # [(step, elastic_constants), ...]
        self.mode_compare_log = []       # [(step, comparison_dict), ...]
        
    # ========== 属性：惰性初始化 ==========
    
    @property
    def mode_solver(self):
        if self._mode_solver is None:
            self._mode_solver = ModeApproximation()
        return self._mode_solver
    
    @property
    def elastic_theory(self):
        if self._elastic_theory is None:
            self._elastic_theory = ElasticConstantTheory(self.mode_solver)
        return self._elastic_theory
    
    @property
    def vacancy_analyzer(self):
        if self._vacancy_analyzer is None:
            self._vacancy_analyzer = VacancyDiffusion(dx=self.dx, dt=self.dt)
        return self._vacancy_analyzer
    
    # ========== 一模式近似集成 ==========
    
    def compare_mode_approximation(self, r: Optional[float] = None) -> Dict:
        """
        对比当前数值解与一模式近似解析解。
        从密度场统计估算 n0 和 A，与解析解对比。
        """
        if r is None:
            r = getattr(self, 'r', -0.25)
            
        n0_num = float(np.mean(self.phi))
        # 估算振幅：FFT 最大峰值 / N^2（近似）
        phi_fluct = self.phi - n0_num
        phi_hat = np.fft.fft2(phi_fluct)
        A_est = float(np.max(np.abs(phi_hat))) / (self.N ** 2)
        
        result = {
            'r': r,
            'n0_numerical': n0_num,
            'A_numerical': A_est,
            'lattice_type': getattr(self, 'lattice_type', 'hexagon'),
        }
        
        try:
            if self.lattice_type in ['hexagon', 'triangle']:
                sol = self.mode_solver.solve_triangular(r)
            elif self.lattice_type == 'square':
                # 模式近似未实现 square，回退到 triangular 做参考
                sol = self.mode_solver.solve_triangular(r)
            else:
                sol = self.mode_solver.solve_bcc(r)
            
            result['n0_mode'] = sol.n0
            result['A_mode'] = sol.A
            result['a_mode'] = sol.a
            result['F_mode'] = sol.F
            result['stable'] = sol.stable
            result['mode_solution'] = sol
            result['n0_error'] = abs(n0_num - sol.n0) / (abs(sol.n0) + 1e-10)
            result['A_error'] = abs(A_est - sol.A) / (abs(sol.A) + 1e-10)
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    # ========== 弹性理论集成 ==========
    
    def compute_elastic_constants_theory(self, r: Optional[float] = None,
                                          method: str = 'analytical'):
        """
        基于一模式近似计算理论弹性常数。
        2D三角/六角晶格返回 dict，3D BCC 返回 ElasticConstants。
        """
        if r is None:
            r = getattr(self, 'r', -0.25)
            
        try:
            if self.lattice_type in ['hexagon', 'triangle']:
                elastic = self.elastic_theory.compute_triangular_elastic(r)
                print(f"[Elastic Theory] 2D Triangular @ r={r:.3f}: "
                      f"C11={elastic['C11']:.4f}, C12={elastic['C12']:.4f}, C66={elastic['C66']:.4f}")
                return elastic
            else:
                elastic = self.elastic_theory.compute_bcc_elastic(r, method=method)
                print(f"[Elastic Theory] BCC @ r={r:.3f}: "
                      f"C11={elastic.C11:.4f}, C12={elastic.C12:.4f}, C44={elastic.C44:.4f}")
                return elastic
        except Exception as e:
            print(f"[Elastic Theory] 计算失败: {e}")
            return None
    
    # ========== 空位分析集成 ==========
    
    def analyze_current_vacancies(self, threshold: Optional[float] = None,
                                    min_distance: int = 3) -> List[Tuple[int, ...]]:
        """
        识别当前密度场中的空位位置。
        """
        return self.vacancy_analyzer.find_vacancies(
            self.phi, threshold=threshold, min_distance=min_distance
        )
    
    def record_phi_snapshot(self, step: int):
        """按间隔记录密度场快照，用于后续空位扩散分析。"""
        if step % self._history_interval == 0:
            self._phi_history.append(self.phi.copy())
            if len(self._phi_history) > self._max_history:
                self._phi_history.pop(0)
    
    def compute_vacancy_diffusion(self) -> Dict:
        """
        基于已记录的历史密度场计算空位扩散系数。
        """
        if len(self._phi_history) < 10:
            print("[Vacancy] 历史数据不足（需≥10个快照），跳过扩散分析")
            return {'D': None, 'error': 'Insufficient history'}
        
        return self.vacancy_analyzer.compute_diffusion_coefficient(
            self._phi_history, time_interval=self.dt * self._history_interval
        )
    
    # ========== 增强运行循环 ==========
    
    def run_with_advanced_analysis(self, analysis_interval: int = 200,
                                    vacancy_interval: int = 50,
                                    elastic_interval: int = 500):
        """
        运行模拟并定期执行高级分析。
        
        Parameters
        ----------
        analysis_interval : int
            综合模式近似对比的步长间隔
        vacancy_interval : int
            空位识别与记录的步长间隔
        elastic_interval : int
            弹性常数计算的最小步数（晶格需先稳定）
        """
        print("=" * 60)
        print("PFC Simulation with Advanced Analysis")
        print(f"Grid: {self.N}x{self.N}, L={self.L}, r={self.r}, lattice={self.lattice_type}")
        print("=" * 60)
        
        for step in range(self.steps):
            # 1. 标准 PFC 时间步
            self.step()
            
            # 2. 记录历史（用于空位扩散）
            self.record_phi_snapshot(step)
            
            # 3. 标准采样输出（每10步）
            if step % 10 == 0:
                if hasattr(self, 'sample_observables'):
                    E = self.sample_observables(step)
                else:
                    E = self.compute_energy() if hasattr(self, 'compute_energy') else 0.0
                    
                if hasattr(self, 'print_status'):
                    self.print_status(step, E)
                if hasattr(self, 'capture_frame'):
                    self.capture_frame()
            
            # 4. 空位分析（较频繁）
            if step % vacancy_interval == 0 and step > 0:
                try:
                    vacancies = self.analyze_current_vacancies()
                    self.vacancy_log.append((step, len(vacancies)))
                    if vacancies:
                        print(f"  [Vacancy] Step {step}: {len(vacancies)} vacancies detected")
                except Exception as e:
                    pass  # 空位分析失败不中断模拟
            
            # 5. 综合分析（模式近似 + 弹性，较稀疏）
            if step % analysis_interval == 0 and step > 0:
                print(f"\n--- Advanced Analysis @ step {step} ---")
                
                # 模式近似对比
                comp = self.compare_mode_approximation()
                self.mode_compare_log.append((step, comp))
                if 'n0_mode' in comp:
                    print(f"  [Mode] n0_mode={comp['n0_mode']:.4f} vs num={comp['n0_numerical']:.4f} "
                          f"(err={comp.get('n0_error', 0)*100:.1f}%)")
                
                # 弹性常数（晶格稳定后才计算）
                if step >= elastic_interval:
                    elastic = self.compute_elastic_constants_theory()
                    if elastic is not None:
                        self.elastic_log.append((step, elastic))
                
                print("-" * 40)
        
        # 6. 后处理：空位扩散
        if len(self._phi_history) > 10:
            print("\n" + "=" * 60)
            print("Post-processing: Vacancy Diffusion")
            print("=" * 60)
            diff = self.compute_vacancy_diffusion()
            if diff.get('D') is not None:
                print(f"  D = {diff['D']:.6e} (dx²/dt)")
                print(f"  Steps tracked: {diff.get('num_steps', 0)}")
        
        # 7. 收尾
        if hasattr(self, 'frames_to_video'):
            self.frames_to_video()
        print("\nSimulation completed.")