"""
main.py — Interactive Main Interface for Dual-Amplitude PFC Simulator
双振幅PFC模拟器交互式主界面

Usage / 使用方法:
    python main.py

This provides a GUI to select and run three models:
提供图形界面选择并运行三种模型：
  1. Flat Interface Energy / 平滑界面能
  2. Round Interface Energy / 圆形界面能（核生长）
  3. Dislocation Energy / 位错能

Author: Auto-generated for PFC research / 为PFC研究自动生成
"""

import sys
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading

# Add current directory to path / 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pfc_core import DualPFCConfig, DualPFCEngine
from part1_flat_interface import FlatInterfaceModel
from part2_round_interface import RoundInterfaceModel
from part3_dislocation import DislocationModel
from visualization import PFCVisualizer


# Default sigma values / 默认sigma值
# sigma1-2: square phase; sigma3-5: triangular phase
# sigma1-2：正方形相；sigma3-5：三角形相
DEFAULT_SIGMAS = [-0.4, -0.2, 0.0, 0.2, 0.4]
SIGMA_LABELS = ['sigma1', 'sigma2', 'sigma3', 'sigma4', 'sigma5']


class PFCMainApp:
    """Main application window for dual-amplitude PFC simulator.
    双振幅PFC模拟器主应用窗口。"""

    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Amplitude PFC Simulator / 双振幅PFC模拟器")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        self.current_result = None
        self.current_engine = None
        self.visualizer = PFCVisualizer()
        self._sim_thread = None
        self._running = False

        self._build_ui()

    def _build_ui(self):
        """Build the user interface.
        构建用户界面。"""
        # Main container / 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title / 标题
        title_label = ttk.Label(
            main_frame,
            text="Dual-Amplitude (Two-Mode) PFC Simulator\n双振幅（双模）相场晶体模拟器",
            font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 15))

        # Model selection / 模型选择
        model_frame = ttk.LabelFrame(main_frame, text="Select Model / 选择模型",
                                     padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        self.model_var = tk.IntVar(value=1)
        models = [
            ("Model 1: Flat Interface Energy / 平滑界面能 (sigma1,2 square; sigma3-5 triangular)", 1),
            ("Model 2: Round Interface Energy / 圆形界面能 (Nucleus Growth / 核生长)", 2),
            ("Model 3: Dislocation Energy / 位错能 E_dis (7 simulations / 7组模拟)", 3),
        ]
        for text, val in models:
            ttk.Radiobutton(model_frame, text=text, variable=self.model_var,
                            value=val).pack(anchor=tk.W, pady=2)

        # Parameters / 参数设置
        param_frame = ttk.LabelFrame(main_frame, text="Parameters / 参数设置",
                                     padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))

        # Grid layout for parameters / 参数网格布局
        params = [
            ("Nx (grid x / x方向网格数):", "256"),
            ("Ny (grid y / y方向网格数):", "256"),
            ("dt (time step / 时间步):", "0.5"),
            ("n_steps (iterations / 迭代步数):", "2000"),
            ("tau (cubic coeff / 三次项系数):", "1.0"),
            ("q (wave ratio / 波数比):", "1.732"),
            ("amplitude (crystal amp / 晶体振幅):", "0.3"),
            ("mean_density (avg density / 平均密度):", "0.0"),
        ]
        self.param_entries = {}
        for i, (label, default) in enumerate(params):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(param_frame, text=label).grid(row=row, column=col,
                                                     sticky=tk.W, padx=5, pady=3)
            entry = ttk.Entry(param_frame, width=12)
            entry.insert(0, default)
            entry.grid(row=row, column=col + 1, padx=5, pady=3)
            key = label.split()[0].split('(')[0]
            self.param_entries[key] = entry

        # Sigma selection / sigma选择
        sigma_frame = ttk.Frame(param_frame)
        sigma_frame.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=5)
        ttk.Label(sigma_frame, text="Sigma values / sigma值 (comma separated / 逗号分隔):").pack(side=tk.LEFT)
        self.sigma_entry = ttk.Entry(sigma_frame, width=40)
        self.sigma_entry.insert(0, "-0.4, -0.2, 0.0, 0.2, 0.4")
        self.sigma_entry.pack(side=tk.LEFT, padx=5)

        # Run mode / 运行模式
        mode_frame = ttk.Frame(param_frame)
        mode_frame.grid(row=5, column=0, columnspan=4, sticky=tk.W, pady=5)
        self.live_view_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mode_frame, text="Live visualization / 实时可视化",
                        variable=self.live_view_var).pack(side=tk.LEFT, padx=5)
        self.series_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_frame, text="Run full sigma series / 运行完整sigma系列",
                        variable=self.series_var).pack(side=tk.LEFT, padx=5)

        # Control buttons / 控制按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.run_btn = ttk.Button(btn_frame, text="Run Simulation / 运行模拟",
                                  command=self._run_simulation)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop / 停止",
                                   command=self._stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="View Results / 查看结果",
                   command=self._view_results).pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="Plot Sigma Series / 绘制sigma系列",
                   command=self._plot_sigma_series).pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="Export Data / 导出数据",
                   command=self._export_data).pack(side=tk.LEFT, padx=5)

        # Output log / 输出日志
        log_frame = ttk.LabelFrame(main_frame, text="Output Log / 输出日志",
                                   padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=12,
                                                   font=('Consolas', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self._log("Welcome to Dual-Amplitude PFC Simulator / 欢迎使用双振幅PFC模拟器")
        self._log("Select a model and parameters, then click Run / 选择模型和参数后点击运行")
        self._log("=" * 60)

        # Status bar / 状态栏
        self.status_var = tk.StringVar(value="Ready / 就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------ #
    # Logging / 日志
    # ------------------------------------------------------------------ #
    def _log(self, message):
        """Append message to log.
        向日志追加消息。"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    # ------------------------------------------------------------------ #
    # Parameter parsing / 参数解析
    # ------------------------------------------------------------------ #
    def _get_config(self, sigma=0.0):
        """Parse parameters and create a DualPFCConfig.
        解析参数并创建配置对象。"""
        try:
            Nx = int(self.param_entries['Nx'].get())
            Ny = int(self.param_entries['Ny'].get())
            dt = float(self.param_entries['dt'].get())
            n_steps = int(self.param_entries['n_steps'].get())
            tau = float(self.param_entries['tau'].get())
            q = float(self.param_entries['q'].get())
            amplitude = float(self.param_entries['amplitude'].get())
            mean_density = float(self.param_entries['mean_density'].get())
        except ValueError as e:
            messagebox.showerror("Parameter Error / 参数错误",
                                 f"Invalid parameter / 无效参数: {e}")
            return None

        return DualPFCConfig(
            Nx=Nx, Ny=Ny, sigma=sigma, tau=tau, q=q,
            dt=dt, n_steps=n_steps, amplitude=amplitude,
            mean_density=mean_density
        )

    def _get_sigma_list(self):
        """Parse sigma values from entry.
        从输入框解析sigma值。"""
        try:
            sigmas = [float(s.strip()) for s in self.sigma_entry.get().split(',')]
            return sigmas
        except ValueError:
            messagebox.showerror("Sigma Error / sigma错误",
                                 "Invalid sigma values / 无效的sigma值")
            return None

    @staticmethod
    def _crystal_type_for_sigma(sigma, sigmas_list):
        """Determine crystal type based on sigma index.
        根据sigma索引确定晶体类型。
        sigma1, sigma2 -> square; sigma3,4,5 -> triangular.
        """
        if sigma in sigmas_list:
            idx = sigmas_list.index(sigma)
            return 'square' if idx < 2 else 'triangular'
        return 'square' if sigma < 0 else 'triangular'

    # ------------------------------------------------------------------ #
    # Simulation execution / 模拟执行
    # ------------------------------------------------------------------ #
    def _run_simulation(self):
        """Run the selected simulation in a background thread.
        在后台线程中运行选定的模拟。"""
        if self._running:
            return

        model_type = self.model_var.get()
        sigmas = self._get_sigma_list()
        if sigmas is None:
            return

        self._running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Running / 运行中...")

        # Run in thread to keep GUI responsive / 在线程中运行以保持GUI响应
        self._sim_thread = threading.Thread(
            target=self._run_simulation_thread,
            args=(model_type, sigmas),
            daemon=True
        )
        self._sim_thread.start()

    def _run_simulation_thread(self, model_type, sigmas):
        """Background thread for simulation.
        模拟后台线程。"""
        try:
            results = []
            run_series = self.series_var.get()
            sigma_to_run = sigmas if run_series else [sigmas[0]]

            if model_type == 1:
                self._log("\n--- Model 1: Flat Interface Energy / 平滑界面能 ---")
                for sigma in sigma_to_run:
                    if not self._running:
                        break
                    crystal = self._crystal_type_for_sigma(sigma, sigmas)
                    cfg = self._get_config(sigma)
                    if cfg is None:
                        return
                    self._log(f"Running sigma={sigma:.3f}, crystal={crystal}...")
                    model = FlatInterfaceModel(cfg, crystal)
                    model.build_initial_condition()
                    model.run(callback=self._live_callback if self.live_view_var.get() else None)
                    gamma = model.compute_interface_energy()
                    self._log(f"  Result: gamma_SL = {gamma:.6f}")
                    results.append(model.get_results())
                    self.current_engine = model.engine

            elif model_type == 2:
                self._log("\n--- Model 2: Round Interface Energy / 圆形界面能 ---")
                for sigma in sigma_to_run:
                    if not self._running:
                        break
                    crystal = self._crystal_type_for_sigma(sigma, sigmas)
                    cfg = self._get_config(sigma)
                    if cfg is None:
                        return
                    self._log(f"Running sigma={sigma:.3f}, crystal={crystal}...")
                    model = RoundInterfaceModel(cfg, crystal)
                    model.build_initial_condition()
                    model.run(callback=self._live_callback if self.live_view_var.get() else None)
                    gamma, R_eff = model.compute_interface_energy()
                    self._log(f"  Result: gamma_SL = {gamma:.6f}, R_eff = {R_eff:.2f}")
                    results.append(model.get_results())
                    self.current_engine = model.engine

            elif model_type == 3:
                self._log("\n--- Model 3: Dislocation Energy / 位错能 ---")
                self._log("Domain: 256x256 / 区域：256x256")
                # Triangular: all sigmas; Square: sigma1, sigma2
                # 三角形：全部sigma；正方形：sigma1, sigma2
                tri_sigmas = sigmas
                sq_sigmas = sigmas[:2] if len(sigmas) >= 2 else sigmas

                if not run_series:
                    tri_sigmas = [sigmas[0]]
                    sq_sigmas = []

                # Triangular dislocations / 三角形位错
                self._log("Triangular crystal / 三角形晶体:")
                for sigma in tri_sigmas:
                    if not self._running:
                        break
                    cfg = self._get_config(sigma)
                    if cfg is None:
                        return
                    self._log(f"  Running sigma={sigma:.3f}, triangular...")
                    model = DislocationModel(cfg, 'triangular')
                    model.prepare_and_run(
                        callback=self._live_callback if self.live_view_var.get() else None)
                    self._log(f"    E_dis = {model.E_dis:.6f}")
                    results.append(model.get_results())
                    self.current_engine = model.engine

                # Square dislocations / 正方形位错
                self._log("Square crystal / 正方形晶体:")
                for sigma in sq_sigmas:
                    if not self._running:
                        break
                    cfg = self._get_config(sigma)
                    if cfg is None:
                        return
                    self._log(f"  Running sigma={sigma:.3f}, square...")
                    model = DislocationModel(cfg, 'square')
                    model.prepare_and_run(
                        callback=self._live_callback if self.live_view_var.get() else None)
                    self._log(f"    E_dis = {model.E_dis:.6f}")
                    results.append(model.get_results())
                    self.current_engine = model.engine

            self.current_result = results
            self._log("\n=== Simulation Complete / 模拟完成 ===")
            self.status_var.set("Complete / 完成")

        except Exception as e:
            self._log(f"\nERROR / 错误: {e}")
            import traceback
            self._log(traceback.format_exc())
            self.status_var.set("Error / 错误")
        finally:
            self._running = False
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def _live_callback(self, step, psi, energy):
        """Callback for live visualization (called from simulation thread).
        实时可视化回调（从模拟线程调用）。"""
        if step % 200 == 0:
            self._log(f"  Step {step}, energy = {energy:.4f}")

    def _stop_simulation(self):
        """Stop the running simulation.
        停止正在运行的模拟。"""
        self._running = False
        self._log("Stopping simulation... / 正在停止模拟...")
        self.status_var.set("Stopped / 已停止")

    # ------------------------------------------------------------------ #
    # Result viewing / 结果查看
    # ------------------------------------------------------------------ #
    def _view_results(self):
        """Open visualization for the latest result.
        打开最新结果的可视化。"""
        if self.current_result is None or len(self.current_result) == 0:
            messagebox.showinfo("No Results / 无结果",
                                "Please run a simulation first / 请先运行模拟")
            return
        self.visualizer.engine = self.current_engine
        # Show the first result in comprehensive view / 在综合视图中显示第一个结果
        result = self.current_result[0]
        model_type = self.model_var.get()
        if model_type == 3:
            # 调用带初始位错的新版对比图
            self.visualizer.plot_dislocation_comparison_with_initial(result)
        else:
            # 调用带初始场的新版综合图
            self.visualizer.plot_result_comprehensive_with_initial(result)

    def _plot_sigma_series(self):
        """Plot sigma series comparison.
        绘制sigma系列对比图。"""
        if self.current_result is None or len(self.current_result) < 2:
            messagebox.showinfo("Insufficient Data / 数据不足",
                                "Run with 'full sigma series' checked / 请勾选'运行完整sigma系列'")
            return

        model_type = self.model_var.get()
        quantity = 'E_dis' if model_type == 3 else 'gamma_SL'
        title = 'Dislocation Energy vs sigma / 位错能随sigma变化' if model_type == 3 \
            else 'Interface Energy vs sigma / 界面能随sigma变化'

        self.visualizer.plot_sigma_series(self.current_result, quantity, title)

    def _export_data(self):
        """Export simulation results to a text file.
        导出模拟结果到文本文件。"""
        if self.current_result is None:
            messagebox.showinfo("No Results / 无结果",
                                "Please run a simulation first / 请先运行模拟")
            return

        try:
            filepath = os.path.join(os.path.dirname(__file__), 'simulation_results.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("Dual-Amplitude PFC Simulation Results\n")
                f.write("双振幅PFC模拟结果\n")
                f.write("=" * 50 + "\n\n")
                for i, r in enumerate(self.current_result):
                    f.write(f"--- Simulation {i+1} ---\n")
                    f.write(f"Model: {r.get('model', 'N/A')}\n")
                    f.write(f"Crystal: {r.get('crystal_type', 'N/A')}\n")
                    f.write(f"Sigma: {r.get('sigma', 'N/A')}\n")
                    if r.get('gamma_SL') is not None:
                        f.write(f"gamma_SL: {r['gamma_SL']:.8f}\n")
                    if r.get('E_dis') is not None:
                        f.write(f"E_dis: {r['E_dis']:.8f}\n")
                    if r.get('nucleus_radius_final') is not None:
                        f.write(f"R_eff: {r['nucleus_radius_final']:.4f}\n")
                    f.write("\n")

            # Also save numpy data / 同时保存numpy数据
            for i, r in enumerate(self.current_result):
                if r.get('psi_final') is not None:
                    np.save(os.path.join(os.path.dirname(__file__),
                                         f'result_{i+1}_psi.npy'),
                            r['psi_final'])

            self._log(f"Data exported to / 数据已导出: {filepath}")
            messagebox.showinfo("Export Complete / 导出完成",
                                f"Results saved to / 结果已保存至:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error / 导出错误", str(e))


def main():
    """Main entry point.
    主入口函数。"""
    root = tk.Tk()
    # Set theme / 设置主题
    try:
        style = ttk.Style()
        style.theme_use('clam')
    except Exception:
        pass
    app = PFCMainApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
