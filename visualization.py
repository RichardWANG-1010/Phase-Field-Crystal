"""
visualization.py — Interactive Visualization Module
可视化模块（交互式）

Provides real-time and post-simulation visualization for all three models:
提供三种模型的实时和后处理可视化：
  - Density field / 密度场
  - Energy density / 能量密度
  - Structure factor (k-space) / 结构因子（k空间）
  - Energy convergence / 能量收敛曲线
  - Interactive controls: play/pause, step, slider
    交互控件：播放/暂停、单步、滑块
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend for interactivity / 使用Tk后端实现交互
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.gridspec import GridSpec

# Configure Chinese font support / 配置中文字体支持
def _setup_chinese_font():
    """Try to set a font that supports Chinese characters.
    尝试设置支持中文字符的字体。"""
    import matplotlib.font_manager as fm
    # Common Chinese fonts on Windows / Windows上常见的中文字体
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi',
                     'FangSong', 'Arial Unicode MS', 'PingFang SC',
                     'Heiti SC', 'STHeiti']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return font
    # Fallback: use default, Chinese may show as boxes
    # 回退：使用默认字体，中文可能显示为方框
    plt.rcParams['axes.unicode_minus'] = False
    return None

_setup_chinese_font()


class PFCVisualizer:
    """Interactive visualizer for PFC simulation results.
    PFC模拟结果的交互式可视化器。"""

    def __init__(self, engine=None):
        self.engine = engine
        self.fig = None
        self.axes = {}
        self.artists = {}
        self._running = False
        self._step_callback = None

    # ------------------------------------------------------------------ #
    # Single-field snapshot / 单场快照
    # ------------------------------------------------------------------ #
    def plot_field(self, psi, title='Density Field / 密度场',
                   cmap='RdBu_r', show=True):
        """Plot a 2D field with colorbar.
        绘制二维场并显示色标。"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(psi.T, origin='lower', cmap=cmap, aspect='auto')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='psi')
        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def plot_energy_density(self, psi, engine=None, title='Energy Density / 能量密度',
                            cmap='hot', show=True):
        """Plot local free energy density.
        绘制局部自由能密度。"""
        eng = engine if engine is not None else self.engine
        if eng is None:
            raise ValueError("No engine provided / 未提供引擎")
        f_density = eng.energy_density(psi)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(f_density.T, origin='lower', cmap=cmap, aspect='auto')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='f(x,y)')
        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def plot_structure_factor(self, psi, engine=None,
                              title='Structure Factor / 结构因子',
                              cmap='viridis', show=True):
        """Plot the 2D structure factor S(k) = |psi_k|^2.
        绘制二维结构因子S(k) = |psi_k|^2。"""
        eng = engine if engine is not None else self.engine
        if eng is None:
            raise ValueError("No engine provided / 未提供引擎")
        S = eng.structure_factor(psi)
        KX, KY = eng.k_grid_centered()
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        im = ax.pcolormesh(KX, KY, np.log1p(S), cmap=cmap, shading='auto')
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_title(title + ' (log scale)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='log(1+S)')
        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------ #
    # Multi-panel result view / 多面板结果视图
    # ------------------------------------------------------------------ #
    def plot_result_comprehensive(self, result, show=True):
        """Comprehensive 4-panel view of a simulation result.
        模拟结果的四面板综合视图。

        Panels: density field, energy density, structure factor, energy convergence
        面板：密度场、能量密度、结构因子、能量收敛曲线
        """
        psi = result['psi_final']
        energies = result.get('energies', [])
        crystal = result.get('crystal_type', 'unknown')
        sigma = result.get('sigma', 0.0)
        model_name = result.get('model', 'Unknown')

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Panel 1: Density field / 面板1：密度场
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(psi.T, origin='lower', cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'Density Field / 密度场\n{model_name}, {crystal}, sigma={sigma:.3f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, label='psi')

        # Panel 2: Energy density / 面板2：能量密度
        ax2 = fig.add_subplot(gs[0, 1])
        if self.engine is not None:
            f_dens = self.engine.energy_density(psi)
        else:
            f_dens = np.zeros_like(psi)
        im2 = ax2.imshow(f_dens.T, origin='lower', cmap='hot', aspect='auto')
        ax2.set_title('Energy Density / 能量密度')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, label='f(x,y)')

        # Panel 3: Structure factor / 面板3：结构因子
        ax3 = fig.add_subplot(gs[1, 0])
        if self.engine is not None:
            S = self.engine.structure_factor(psi)
            KX, KY = self.engine.k_grid_centered()
            im3 = ax3.pcolormesh(KX, KY, np.log1p(S), cmap='viridis', shading='auto')
            ax3.set_aspect('equal')
        else:
            im3 = ax3.imshow(np.zeros((10, 10)))
        ax3.set_title('Structure Factor / 结构因子 (log)')
        ax3.set_xlabel('kx')
        ax3.set_ylabel('ky')
        plt.colorbar(im3, ax=ax3, label='log(1+S)')

        # Panel 4: Energy convergence / 面板4：能量收敛曲线
        ax4 = fig.add_subplot(gs[1, 1])
        if energies:
            steps = [e[0] for e in energies]
            vals = [e[1] for e in energies]
            ax4.plot(steps, vals, 'b-', linewidth=1.5)
            ax4.set_xlabel('Step / 步数')
            ax4.set_ylabel('Total Energy / 总能量')
            ax4.set_title('Energy Convergence / 能量收敛')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No energy data / 无能量数据',
                     ha='center', va='center', transform=ax4.transAxes)

        # Print key result / 输出关键结果
        if 'gamma_SL' in result and result['gamma_SL'] is not None:
            fig.suptitle(f'gamma_SL = {result["gamma_SL"]:.6f}',
                         fontsize=14, fontweight='bold', y=0.98)
        elif 'E_dis' in result and result['E_dis'] is not None:
            fig.suptitle(f'E_dis = {result["E_dis"]:.6f}',
                         fontsize=14, fontweight='bold', y=0.98)

        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------ #
    # Sigma series comparison / sigma系列对比
    # ------------------------------------------------------------------ #
    def plot_sigma_series(self, results, quantity='gamma_SL',
                          title=None, show=True):
        """Plot a quantity vs sigma for a series of simulations.
        绘制某物理量随sigma的变化曲线。

        Parameters:
            results: list of result dictionaries / 结果字典列表
            quantity: 'gamma_SL' or 'E_dis' / 物理量名称
        """
        sigmas = [r['sigma'] for r in results]
        values = [r.get(quantity, 0) for r in results]
        crystals = [r.get('crystal_type', '') for r in results]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Color by crystal type / 按晶体类型着色
        colors = {'square': 'blue', 'triangular': 'red'}
        for i, (s, v, cr) in enumerate(zip(sigmas, values, crystals)):
            c = colors.get(cr, 'gray')
            ax.scatter(s, v, color=c, s=100, zorder=5,
                       label=cr if i == 0 or crystals.index(cr) == i else '')
            ax.annotate(f'{v:.4f}', (s, v), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=9)

        ax.plot(sigmas, values, 'k--', alpha=0.4, zorder=1)
        ax.set_xlabel('sigma / 控制参数', fontsize=12)
        ylabel = 'gamma_SL (Interface Energy / 界面能)' if quantity == 'gamma_SL' else 'E_dis (Dislocation Energy / 位错能)'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title if title else f'{quantity} vs sigma', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()
        return fig

    # ------------------------------------------------------------------ #
    # Real-time interactive simulation viewer / 实时交互模拟查看器
    # ------------------------------------------------------------------ #
    def create_live_viewer(self, initial_psi, engine, step_fn,
                           title='Live PFC Simulation / 实时PFC模拟'):
        """Create an interactive window with play/pause/step/slider controls.
        创建带有播放/暂停/单步/滑块控件的交互窗口。

        Parameters:
            initial_psi: initial field / 初始场
            engine: DualPFCEngine instance / PFC引擎
            step_fn: function(psi) -> psi_next (one time step)
                     单步演化函数
        Returns:
            fig, ax_dict: figure and axes dictionary
        """
        self.engine = engine
        self._step_fn = step_fn
        self._current_psi = initial_psi.copy()
        self._step_count = 0
        self._running = False

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                      height_ratios=[3, 1])

        # Density field / 密度场
        ax_dens = fig.add_subplot(gs[0, 0])
        im_dens = ax_dens.imshow(initial_psi.T, origin='lower',
                                 cmap='RdBu_r', aspect='auto')
        ax_dens.set_title('Density Field / 密度场')
        ax_dens.set_xlabel('x')
        ax_dens.set_ylabel('y')
        plt.colorbar(im_dens, ax=ax_dens, label='psi')

        # Energy density / 能量密度
        ax_eng = fig.add_subplot(gs[0, 1])
        f_dens = engine.energy_density(initial_psi)
        im_eng = ax_eng.imshow(f_dens.T, origin='lower',
                               cmap='hot', aspect='auto')
        ax_eng.set_title('Energy Density / 能量密度')
        ax_eng.set_xlabel('x')
        ax_eng.set_ylabel('y')
        plt.colorbar(im_eng, ax=ax_eng, label='f')

        # Energy convergence / 能量收敛
        ax_conv = fig.add_subplot(gs[1, :])
        (line_conv,) = ax_conv.plot([], [], 'b-', linewidth=1.5)
        ax_conv.set_xlabel('Step / 步数')
        ax_conv.set_ylabel('Total Energy / 总能量')
        ax_conv.set_title('Energy Convergence / 能量收敛')
        ax_conv.grid(True, alpha=0.3)

        # Control buttons / 控制按钮
        ax_play = plt.axes([0.35, 0.02, 0.08, 0.03])
        ax_pause = plt.axes([0.44, 0.02, 0.08, 0.03])
        ax_step = plt.axes([0.53, 0.02, 0.08, 0.03])
        ax_reset = plt.axes([0.62, 0.02, 0.08, 0.03])

        btn_play = Button(ax_play, 'Play / 播放')
        btn_pause = Button(ax_pause, 'Pause / 暂停')
        btn_step = Button(ax_step, 'Step / 单步')
        btn_reset = Button(ax_reset, 'Reset / 重置')

        # Store references / 存储引用
        self.fig = fig
        self.axes = {'density': ax_dens, 'energy': ax_eng, 'convergence': ax_conv}
        self.artists = {'im_dens': im_dens, 'im_eng': im_eng, 'line_conv': line_conv}
        self._energy_history = []
        self._initial_psi = initial_psi.copy()

        # Callbacks / 回调函数
        def _update_display():
            psi = self._current_psi
            self.artists['im_dens'].set_data(psi.T)
            f_dens = engine.energy_density(psi)
            self.artists['im_eng'].set_data(f_dens.T)
            self.artists['im_eng'].set_clim(vmin=np.min(f_dens), vmax=np.max(f_dens))
            # Update convergence / 更新收敛曲线
            if self._energy_history:
                steps = [e[0] for e in self._energy_history]
                vals = [e[1] for e in self._energy_history]
                self.artists['line_conv'].set_data(steps, vals)
                ax_conv.relim()
                ax_conv.autoscale_view()
            fig.canvas.draw_idle()

        def _do_step(n=1):
            for _ in range(n):
                self._current_psi = self._step_fn(self._current_psi)
                self._step_count += 1
            e = engine.total_energy(self._current_psi)
            self._energy_history.append((self._step_count, e))
            _update_display()

        def on_play(event):
            self._running = True
            while self._running:
                _do_step(10)
                plt.pause(0.01)

        def on_pause(event):
            self._running = False

        def on_step(event):
            self._running = False
            _do_step(1)

        def on_reset(event):
            self._running = False
            self._current_psi = self._initial_psi.copy()
            self._step_count = 0
            self._energy_history = []
            _update_display()

        btn_play.on_clicked(on_play)
        btn_pause.on_clicked(on_pause)
        btn_step.on_clicked(on_step)
        btn_reset.on_clicked(on_reset)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.ion()
        plt.show()

        return fig, self.axes

    # ------------------------------------------------------------------ #
    # Side-by-side comparison / 并排对比
    # ------------------------------------------------------------------ #
    def plot_compare_fields(self, psi1, psi2, label1='Initial / 初始',
                            label2='Final / 最终', cmap='RdBu_r', show=True):
        """Plot two fields side by side for comparison.
        并排绘制两个场进行对比。"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        im1 = ax1.imshow(psi1.T, origin='lower', cmap=cmap, aspect='auto')
        ax1.set_title(label1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, label='psi')

        im2 = ax2.imshow(psi2.T, origin='lower', cmap=cmap, aspect='auto')
        ax2.set_title(label2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, label='psi')

        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_dislocation_comparison(self, result, show=True):
        """Special comparison for dislocation: perfect vs dislocated.
        位错的特殊对比：完美晶体 vs 位错晶体。"""
        psi_perfect = result.get('psi_perfect')
        psi_dislocated = result.get('psi_final')
        if psi_perfect is None or psi_dislocated is None:
            raise ValueError("Missing field data / 缺少场数据")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Perfect crystal / 完美晶体
        im0 = axes[0].imshow(psi_perfect.T, origin='lower', cmap='RdBu_r', aspect='auto')
        axes[0].set_title('Perfect Crystal / 完美晶体')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0], label='psi')

        # Dislocated / 位错
        im1 = axes[1].imshow(psi_dislocated.T, origin='lower', cmap='RdBu_r', aspect='auto')
        axes[1].set_title('Dislocated / 位错')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1], label='psi')

        # Difference / 差值
        diff = psi_dislocated - psi_perfect
        im2 = axes[2].imshow(diff.T, origin='lower', cmap='seismic', aspect='auto')
        axes[2].set_title('Difference / 差值 (dislocated - perfect)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[2], label='Delta psi')

        E_dis = result.get('E_dis', 0)
        fig.suptitle(f'E_dis = {E_dis:.6f}  |  {result.get("crystal_type","")}  |  '
                     f'sigma={result.get("sigma",0):.3f}',
                     fontsize=13, fontweight='bold')

        if show:
            plt.tight_layout()
            plt.show()
        return fig
    
        # ------------------------------------------------------------------ #
    # 新增：初始场对比与扩展可视化函数
    # New: Initial field comparison & extended visualization functions
    # ------------------------------------------------------------------ #

    def plot_initial_vs_final(self, psi_initial, psi_final,
                              label1='Initial / 初始构型',
                              label2='Final / 弛豫终态',
                              title='Initial vs Final Comparison / 初始与终态对比',
                              cmap='RdBu_r', show=True):
        """并排对比初始场与终态场。
        Side-by-side comparison of initial and final density fields.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # 初始场
        im1 = ax1.imshow(psi_initial.T, origin='lower', cmap=cmap, aspect='auto')
        ax1.set_title(label1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, label='psi')
        # 终态场
        im2 = ax2.imshow(psi_final.T, origin='lower', cmap=cmap, aspect='auto')
        ax2.set_title(label2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, label='psi')

        fig.suptitle(title, fontsize=13, fontweight='bold')
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_result_comprehensive_with_initial(self, result, show=True):
        """带初始场的综合视图（5面板）：初始密度场、终态密度场、能量密度、结构因子、能量收敛。
        Comprehensive view with initial field (5 panels):
        initial density, final density, energy density, structure factor, energy convergence.
        """
        psi_init = result.get('psi_initial')
        psi_final = result['psi_final']
        energies = result.get('energies', [])
        crystal = result.get('crystal_type', 'unknown')
        sigma = result.get('sigma', 0.0)
        model_name = result.get('model', 'Unknown')

        if psi_init is None:
            raise ValueError("No initial field data in result / 结果中无初始场数据")

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 面板1：初始密度场
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(psi_init.T, origin='lower', cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'Initial Density / 初始密度场\n{model_name}, {crystal}, sigma={sigma:.3f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, label='psi')

        # 面板2：终态密度场
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(psi_final.T, origin='lower', cmap='RdBu_r', aspect='auto')
        ax2.set_title('Final Density / 终态密度场')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, label='psi')

        # 面板3：能量密度
        ax3 = fig.add_subplot(gs[0, 2])
        if self.engine is not None:
            f_dens = self.engine.energy_density(psi_final)
        else:
            f_dens = np.zeros_like(psi_final)
        im3 = ax3.imshow(f_dens.T, origin='lower', cmap='hot', aspect='auto')
        ax3.set_title('Energy Density / 能量密度')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(im3, ax=ax3, label='f(x,y)')

        # 面板4：结构因子
        ax4 = fig.add_subplot(gs[1, 0])
        if self.engine is not None:
            S = self.engine.structure_factor(psi_final)
            KX, KY = self.engine.k_grid_centered()
            im4 = ax4.pcolormesh(KX, KY, np.log1p(S), cmap='viridis', shading='auto')
            ax4.set_aspect('equal')
        else:
            im4 = ax4.imshow(np.zeros((10, 10)))
        ax4.set_title('Structure Factor / 结构因子 (log)')
        ax4.set_xlabel('kx')
        ax4.set_ylabel('ky')
        plt.colorbar(im4, ax=ax4, label='log(1+S)')

        # 面板5：能量收敛曲线（跨两列）
        ax5 = fig.add_subplot(gs[1, 1:])
        if energies:
            steps = [e[0] for e in energies]
            vals = [e[1] for e in energies]
            ax5.plot(steps, vals, 'b-', linewidth=1.5)
            ax5.set_xlabel('Step / 步数')
            ax5.set_ylabel('Total Energy / 总能量')
            ax5.set_title('Energy Convergence / 能量收敛')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No energy data / 无能量数据',
                     ha='center', va='center', transform=ax5.transAxes)

        # 顶部标题
        if 'gamma_SL' in result and result['gamma_SL'] is not None:
            fig.suptitle(f'gamma_SL = {result["gamma_SL"]:.6f}',
                         fontsize=14, fontweight='bold', y=0.98)
        elif 'E_dis' in result and result['E_dis'] is not None:
            fig.suptitle(f'E_dis = {result["E_dis"]:.6f}',
                         fontsize=14, fontweight='bold', y=0.98)

        if show:
            plt.show()
        return fig

    def plot_dislocation_comparison_with_initial(self, result, show=True):
        """位错全流程对比：完美晶体、初始位错、弛豫位错、差值。
        Full dislocation comparison: perfect crystal, initial dislocation, relaxed dislocation, difference.
        """
        psi_perfect = result.get('psi_perfect')
        psi_init = result.get('psi_initial')
        psi_final = result.get('psi_final')
        if psi_perfect is None or psi_init is None or psi_final is None:
            raise ValueError("Missing field data / 缺少场数据")

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        # 1. 完美晶体
        im0 = axes[0].imshow(psi_perfect.T, origin='lower', cmap='RdBu_r', aspect='auto')
        axes[0].set_title('Perfect Crystal / 完美晶体')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0], label='psi')

        # 2. 初始位错构型
        im1 = axes[1].imshow(psi_init.T, origin='lower', cmap='RdBu_r', aspect='auto')
        axes[1].set_title('Initial Dislocation / 初始位错构型')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1], label='psi')

        # 3. 弛豫后位错
        im2 = axes[2].imshow(psi_final.T, origin='lower', cmap='RdBu_r', aspect='auto')
        axes[2].set_title('Relaxed Dislocation / 弛豫后位错')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[2], label='psi')

        # 4. 差值（弛豫位错 - 完美晶体）
        diff = psi_final - psi_perfect
        im3 = axes[3].imshow(diff.T, origin='lower', cmap='seismic', aspect='auto')
        axes[3].set_title('Difference / 差值 (dislocated - perfect)')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        plt.colorbar(im3, ax=axes[3], label='Delta psi')

        E_dis = result.get('E_dis', 0)
        fig.suptitle(f'E_dis = {E_dis:.6f}  |  {result.get("crystal_type","")}  |  '
                     f'sigma={result.get("sigma",0):.3f}',
                     fontsize=13, fontweight='bold')
        if show:
            plt.tight_layout()
            plt.show()
        return fig
