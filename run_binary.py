#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# run_binary.py - Binary Alloy PFC Simulation Runner
# 二元合金PFC模拟运行脚本
#
# 使用示例:
#   python run_binary.py
#
# Usage examples:
#   python run_binary.py
#
# 支持两种模式:
#   1. 快速模式: 直接运行预配置参数 (默认)
#   2. 交互模式: 通过命令行交互式输入参数
#
# Supported modes:
#   1. Quick mode: run with pre-configured parameters (default)
#   2. Interactive mode: input parameters interactively via command line
# ============================================================
import sys
import argparse
from pfc_binary import BinaryPFCSolver


def get_param_with_default(prompt, default_value, param_type=float):
    """
    交互式参数输入，支持默认值回车跳过
    Interactive parameter input with default value support
    """
    while True:
        val = input(f"  {prompt} (default={default_value}): ").strip()
        if val == "":
            return param_type(default_value)
        try:
            return param_type(val)
        except ValueError:
            print(f"    Invalid input, please enter a {param_type.__name__}")


def select_mode_menu():
    """
    启动时的模式选择菜单
    Startup mode selection menu
    """
    print("\n" + "=" * 60)
    print("  Binary Alloy PFC Simulation - Mode Selection")
    print("=" * 60 + "\n")
    print("Please select a simulation mode:")
    print("  1 - Quick Mode        (快速模式: 标准参数，适合快速预览)")
    print("  2 - Interactive Mode  (交互模式: 自定义所有参数)")
    print("  3 - High-Resolution   (高分辨率: 精细微观结构分析)")
    print("  4 - Phase Separation  (相分离模式: 观察spinodal分解)")
    print("\n  Tip: Use command line args to skip this menu:")
    print("       --mode <quick|high-res|phase-separation> | --interactive")
    print("=" * 60)
    while True:
        choice = input("\n  Enter your choice (1/2/3/4): ").strip()
        if choice == "1":
            print("\n  [Quick Mode] Using pre-configured standard parameters")
            return quick_mode()
        elif choice == "2":
            print("\n  [Interactive Mode] Starting parameter configuration...")
            return interactive_mode()
        elif choice == "3":
            print("\n  [High-Resolution Mode] Using fine-grid parameters")
            return high_resolution_mode()
        elif choice == "4":
            print("\n  [Phase Separation Mode] Using spinodal decomposition parameters")
            return phase_separation_mode()
        else:
            print("    Invalid choice, please enter 1, 2, 3, or 4")


def interactive_mode():
    """
    交互式参数配置模式
    Interactive parameter configuration mode
    """
    print("\n" + "=" * 60)
    print("  Binary Alloy PFC - Interactive Parameter Setup")
    print("=" * 60 + "\n")
    # --- 晶格类型选择 ---
    # --- Lattice type selection ---
    print("Lattice Type:")
    print("  1 - hexagon   (六角晶格, Standard PFC)")
    print("  2 - square    (正方晶格)")
    print("  3 - triangle  (三角晶格)")
    while True:
        choice = input("  Select lattice type (1/2/3) [default=1]: ").strip()
        if choice == "" or choice == "1":
            lattice = "hexagon"
            break
        elif choice == "2":
            lattice = "square"
            break
        elif choice == "3":
            lattice = "triangle"
            break
        else:
            print("    Invalid choice, please enter 1, 2, or 3")
    # --- 基础参数 ---
    # --- Basic parameters ---
    print("\n--- Grid & Time Parameters ---")
    N = get_param_with_default("Grid size N", 512, int)
    L = get_param_with_default("Domain size L", 128.0, float)
    dt = get_param_with_default("Time step dt", 0.05, float)
    T = get_param_with_default("Total simulation time T", 1000.0, float)
    # --- 密度场参数 ---
    # --- Density field parameters ---
    print("\n--- Density Field Parameters ---")
    r = get_param_with_default("PFC parameter r", -0.25, float)
    M_phi = get_param_with_default("Mobility M_phi", 1.0, float)
    phi0 = get_param_with_default("Average density phi0", -0.25, float)
    # --- 浓度场参数 ---
    # --- Concentration field parameters ---
    print("\n--- Concentration Field Parameters ---")
    M_c = get_param_with_default("Mobility M_c", 0.1, float)
    c0 = get_param_with_default("Average concentration c0", 0.3, float)
    r_c = get_param_with_default("CH parameter r_c", -0.5, float)
    u_c = get_param_with_default("CH parameter u_c", 1.0, float)
    # --- 耦合参数 ---
    # --- Coupling parameters ---
    print("\n--- Coupling Parameters ---")
    print("  (alpha: segregation strength, beta: direct coupling)")
    alpha = get_param_with_default("Coupling alpha", 0.1, float)
    beta = get_param_with_default("Coupling beta", 0.0, float)
    # --- 初始条件 ---
    # --- Initial conditions ---
    print("\n--- Initial Condition ---")
    noise_amp = get_param_with_default("Noise amplitude", 0.01, float)
    params = {
        "N": N,
        "L": L,
        "dt": dt,
        "T": T,
        "r": r,
        "M_phi": M_phi,
        "phi0": phi0,
        "M_c": M_c,
        "c0": c0,
        "r_c": r_c,
        "u_c": u_c,
        "alpha": alpha,
        "beta": beta,
        "noise_amp": noise_amp,
        "lattice_type": lattice,
    }
    # 参数确认
    # Parameter confirmation
    print("\n" + "-" * 60)
    print("  Parameter Summary:")
    for k, v in params.items():
        print(f"    {k:15s} = {v}")
    print("-" * 60)
    confirm = input("\n  Confirm and start simulation? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Simulation cancelled.")
        sys.exit(0)
    return params


def quick_mode():
    """
    快速模式: 使用预配置的标准参数，优化运行速度
    Quick mode: pre-configured parameters optimized for speed
    N=512, L=128: 每个像素代表 0.25 物理单位，晶格周期约 6.28，
    对应约 25 个像素/周期，画面细腻无锯齿感。

    N=512, L=128: each pixel represents 0.25 physical units, lattice period ~ 6.28,
    corresponding to ~25 pixels/period, smooth image without aliasing.
    """
    params = {
        # 网格参数 (512x512，像素感大幅降低)
        # Grid parameters (512x512, pixel feel greatly reduced)
        "N": 512,
        "L": 128.0,
        # 空间分辨率提高后，dt适当减小保证稳定性
        # After spatial resolution increases, dt is appropriately reduced to ensure stability
        "dt": 0.05,
        # 总时间稍延长，让高分辨率结构充分演化
        # Total time slightly extended to allow high-resolution structures to fully evolve
        "T": 1000.0,
        # 密度场参数 (r=-0.25产生多晶结构)
        # Density field parameters (r=-0.25 produces polycrystalline structure)
        "r": -0.25,
        "M_phi": 1.0,
        "phi0": -0.25,
        # 浓度场参数 (M_c < M_phi使浓度扩散慢于密度场)
        # Concentration field parameters (M_c < M_phi makes concentration diffusion slower than density field)
        "M_c": 0.05,
        "c0": 0.3,
        "r_c": -0.3,
        "u_c": 1.0,
        # 耦合参数 (alpha主导溶质在密度峰/谷的偏析)
        # Coupling parameters (alpha dominates solute segregation at density peaks/valleys)
        "alpha": -0.15,
        "beta": 0.0,
        # 初始条件
        # Initial conditions
        "noise_amp": 0.01,
        "lattice_type": "hexagon",
    }
    return params


def high_resolution_mode():
    """
    高分辨率模式: 用于精细的微观结构分析
    High-resolution mode for fine microstructure analysis
    """
    params = {
        "N": 1024,
        "L": 256.0,
        "dt": 0.02,
        "T": 5000.0,
        "r": -0.25,
        "M_phi": 1.0,
        "phi0": -0.25,
        "M_c": 0.05,
        "c0": 0.3,
        "r_c": -0.3,
        "u_c": 1.0,
        "alpha": -0.15,
        "beta": 0.0,
        "noise_amp": 0.01,
        "lattice_type": "hexagon",
    }
    return params


def phase_separation_mode():
    """
    相分离模式: 用于观察spinodal分解
    Phase separation mode for spinodal decomposition observation
    """
    params = {
        "N": 512,
        "L": 128.0,
        "dt": 0.05,
        "T": 4000.0,
        "r": -0.3,
        "M_phi": 1.0,
        "phi0": -0.3,
        "M_c": 0.5,
        "c0": 0.5,
        "r_c": -1.0,
        "u_c": 1.0,
        "alpha": 0.3,
        "beta": 0.0,
        "noise_amp": 0.02,
        "lattice_type": "hexagon",
    }
    return params


def main():
    """
    主入口函数
    Main entry function
    """
    parser = argparse.ArgumentParser(
        description="Binary Alloy PFC Simulation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 启动模式选择菜单 (默认，无参数时)
  python run_binary.py
  # 交互式参数配置
  python run_binary.py --interactive
  # 快速模式 (默认关闭录像，N=512)
  python run_binary.py --mode quick
  # 快速模式 + 录像 (更慢)
  python run_binary.py --mode quick --video
  # 高分辨率模式
  python run_binary.py --mode high-res
  # 相分离模式
  python run_binary.py --mode phase-separation
        """,
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive parameter configuration mode",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["quick", "high-res", "phase-separation"],
        default="quick",
        help="Preset simulation mode (use this to skip the selection menu)",
    )
    parser.add_argument(
        "--video", "-v",
        action="store_true",
        help="Enable video recording (slow, off by default)",
    )
    args = parser.parse_args()
    # 检查用户是否通过命令行显式指定了 --mode 参数
    # Check if user explicitly specified --mode parameter via command line
    mode_explicitly_set = any(arg in sys.argv for arg in ["--mode", "-m"])
    # --- 选择参数模式 ---
    # --- Select parameter mode ---
    if args.interactive:
        params = interactive_mode()
    elif mode_explicitly_set:
        # 用户显式指定了模式，直接执行对应模式
        # User explicitly specified mode, execute corresponding mode directly
        if args.mode == "quick":
            print("\n  [Quick Mode] N=512, L=128 (fine grid)")
            params = quick_mode()
        elif args.mode == "high-res":
            print("\n  [High-Resolution Mode] Using fine-grid parameters")
            params = high_resolution_mode()
        elif args.mode == "phase-separation":
            print("\n  [Phase Separation Mode] Using spinodal decomposition parameters")
            params = phase_separation_mode()
    else:
        # 没有显式指定任何模式，显示交互式选择菜单
        # No mode explicitly specified, show interactive selection menu
        params = select_mode_menu()
    # --- 创建求解器 ---
    # --- Create solver ---
    print("\n  Creating BinaryPFCSolver...")
    solver = BinaryPFCSolver(**params)
    # 录像设置：默认关闭，需用 --video 手动开启
    # Video setting: off by default, need to manually enable with --video
    solver.record_video = args.video
    if args.video:
        print("  [Video ON] Recording will be enabled (slower)")
    else:
        print("  [Video OFF] Use --video to enable recording")
    # --- 运行模拟 ---
    # --- Run simulation ---
    # 快速模式使用更大的采样间隔以减少绘图开销
    # Quick mode uses larger sampling interval to reduce plotting overhead
    if args.mode == "quick" and not args.interactive:
        sample_interval = 50
        print(f"  [Quick Mode] sample_interval={sample_interval} (reduced for speed)")
    else:
        sample_interval = 10
    solver.run(sample_interval=sample_interval)
    # --- 后处理 ---
    # --- Post-processing ---
    solver.postprocess()
    solver.analyze_psi6()
    # --- 额外分析 ---
    # --- Additional analysis ---
    print("\n" + "=" * 60)
    print("  Additional Binary Alloy Analysis")
    print("=" * 60)
    solver.plot_coupling_energy()
    print("\n  All done!")


if __name__ == "__main__":
    main()
