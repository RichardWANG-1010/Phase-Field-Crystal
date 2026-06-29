"""
pfc_advanced.py - 高级分析模块集成（Mixin）
将 pfc_mode_approximation, pfc_elastic_theory, pfc_vacancy 与 PFC 求解器连接

修复说明：本版本不依赖 __init__，所有属性首次使用时自动初始化，
避免多继承链中 PFCBase 未调用 super() 导致的问题。
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

from pfc_mode_approximation import ModeApproximation, LatticeSolution
from pfc_elastic_theory import ElasticConstantTheory, ElasticConstants
from pfc_vacancy import VacancyDiffusion, VacancyState


class PFCAdvancedAnalysis:
    """
    Mixin：集成一模式近似、弹性理论、空位分析。
    不定义 __init__，完全惰性初始化，安全混入任何多继承结构。
    """

    # Lazy Properties (Analyzer Instances)
    # ==================== 惰性属性（分析器实例） ====================

    @property
    def mode_solver(self):
        if not hasattr(self, '_mode_solver'):
            self._mode_solver = ModeApproximation()
        return self._mode_solver

    @property
    def elastic_theory(self):
        if not hasattr(self, '_elastic_theory'):
            self._elastic_theory = ElasticConstantTheory(self.mode_solver)
        return self._elastic_theory

    @property
    def vacancy_analyzer(self):
        if not hasattr(self, '_vacancy_analyzer'):
            dx = getattr(self, 'dx', 1.0)
            dt = getattr(self, 'dt', 1.0)
            self._vacancy_analyzer = VacancyDiffusion(dx=dx, dt=dt)
        return self._vacancy_analyzer

    # Internal Helper: Ensure History Buffer Exists
    # ==================== 内部辅助：确保历史缓冲区存在 ====================

    def _ensure_history_attrs(self):
        """首次使用时初始化空位追踪相关的实例属性"""
        if not hasattr(self, '_phi_history'):
            self._phi_history = []
            self._history_interval = 10
            self._max_history = 500
            self.vacancy_log = []
            self.elastic_log = []
            self.mode_compare_log = []

    # Single-Mode Approximation Integration
    # ==================== 一模式近似集成 ====================

    def compare_mode_approximation(self, r: Optional[float] = None) -> Dict:
        if r is None:
            r = getattr(self, 'r', -0.25)

        n0_num = float(np.mean(self.phi))
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
                sol = self.mode_solver.solve_triangular(r)
            else:
                sol = self.mode_solver.solve_bcc(r)

            result['n0_mode'] = sol.n0
            result['A_mode'] = sol.A
            result['a_mode'] = sol.a
            result['F_mode'] = sol.F
            result['stable'] = sol.stable
            result['n0_error'] = abs(n0_num - sol.n0) / (abs(sol.n0) + 1e-10)
            result['A_error'] = abs(A_est - sol.A) / (abs(sol.A) + 1e-10)
        except Exception as e:
            result['error'] = str(e)

        return result

    # Elastic Theory Integration
    # ==================== 弹性理论集成 ====================

    def compute_elastic_constants_theory(self, r: Optional[float] = None,
                                          method: str = 'analytical'):
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

    # Vacancy Analysis Integration
    # ==================== 空位分析集成 ====================

    def analyze_current_vacancies(self, threshold: Optional[float] = None,
                                    min_distance: int = 3) -> List[Tuple[int, ...]]:
        return self.vacancy_analyzer.find_vacancies(
            self.phi, threshold=threshold, min_distance=min_distance
        )

    # Enhanced Run Loop (Fixed Version)
    # ==================== 增强运行循环（修复版） ====================

    def run_with_advanced_analysis(self,
                                    analysis_interval: int = 200,
                                    vacancy_interval: int = 50,
                                    elastic_interval: int = 300):
        """
        运行模拟并定期执行高级分析。
        """
        print("=" * 60)
        print("PFC Simulation with Advanced Analysis")
        print(f"Grid: {self.N}x{self.N}, L={self.L}, r={self.r}, lattice={self.lattice_type}")
        print("=" * 60)

        # Ensure history buffer exists (critical fix)
        # 确保历史缓冲区存在（关键修复）
        self._ensure_history_attrs()

        for step in range(self.steps):
            # Standard PFC time step
            # 1. 标准 PFC 时间步
            self.step()

            # Record history (for vacancy diffusion) - inline, no external method calls
            # 2. 记录历史（用于空位扩散）—— 直接内联，不再调用外部方法
            if step % self._history_interval == 0:
                self._phi_history.append(self.phi.copy())
                if len(self._phi_history) > self._max_history:
                    self._phi_history.pop(0)

            # Standard sampling output (every 10 steps)
            # 3. 标准采样输出（每10步）
            if step % 10 == 0:
                E = self.compute_energy() if hasattr(self, 'compute_energy') else 0.0
                self.print_status(step, E)
                if hasattr(self, 'capture_frame'):
                    self.capture_frame()

            # Vacancy analysis
            # 4. 空位分析
            if step % vacancy_interval == 0 and step > 0:
                try:
                    vacancies = self.analyze_current_vacancies()
                    self.vacancy_log.append((step, len(vacancies)))
                    if vacancies:
                        print(f"  [Vacancy] Step {step}: {len(vacancies)} vacancies detected")
                except Exception:
                    pass

            # Comprehensive analysis (mode approximation + elasticity)
            # 5. 综合分析（模式近似 + 弹性）
            if step % analysis_interval == 0 and step > 0:
                print(f"\n--- Advanced Analysis @ step {step} ---")

                comp = self.compare_mode_approximation()
                self.mode_compare_log.append((step, comp))
                if 'n0_mode' in comp:
                    print(f"  [Mode] n0_mode={comp['n0_mode']:.4f} vs num={comp['n0_numerical']:.4f} "
                          f"(err={comp.get('n0_error', 0)*100:.1f}%)")

                if step >= elastic_interval:
                    elastic = self.compute_elastic_constants_theory()
                    if elastic is not None:
                        self.elastic_log.append((step, elastic))

                print("-" * 40)

        # Post-processing: Vacancy diffusion
        # 6. 后处理：空位扩散
        if len(self._phi_history) > 10:
            print("\n" + "=" * 60)
            print("Post-processing: Vacancy Diffusion")
            print("=" * 60)
            diff = self.vacancy_analyzer.compute_diffusion_coefficient(
                self._phi_history, time_interval=self.dt * self._history_interval
            )
            if diff.get('D') is not None:
                print(f"  D = {diff['D']:.6e} (dx²/dt)")
                print(f"  Steps tracked: {diff.get('num_steps', 0)}")

        # [EN] 7. 收尾
        # 7. 收尾
        if hasattr(self, 'frames_to_video'):
            self.frames_to_video()
        print("\nSimulation completed.")