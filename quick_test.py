"""
quick_test.py — Quick verification test for the PFC engine
快速验证测试：检查PFC引擎是否正常运行

Run with: python quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pfc_core import DualPFCConfig, DualPFCEngine
from part1_flat_interface import FlatInterfaceModel
from part2_round_interface import RoundInterfaceModel
from part3_dislocation import DislocationModel


def test_core_engine():
    """Test core PFC engine / 测试核心PFC引擎"""
    print("=" * 50)
    print("Test 1: Core Engine / 核心引擎测试")
    print("=" * 50)

    cfg = DualPFCConfig(Nx=64, Ny=64, sigma=0.0, n_steps=100, dt=0.5)
    engine = DualPFCEngine(cfg)

    print(f"  Grid: {cfg.Nx}x{cfg.Ny}")
    print(f"  Lx={cfg.Lx:.2f}, Ly={cfg.Ly:.2f}")
    print(f"  dx={cfg.dx:.4f}, dy={cfg.dy:.4f}")
    print(f"  r (reduced temp) = {cfg.r:.4f}")
    print(f"  L_hat shape: {engine.L_hat.shape}")
    print(f"  L_hat min={engine.L_hat.min():.4f}, max={engine.L_hat.max():.4f}")

    X, Y = engine.get_coordinate_grids()
    psi_sq = engine.square_crystal(X, Y)
    psi_tri = engine.triangular_crystal(X, Y)

    print(f"  Square crystal: mean={psi_sq.mean():.4f}, std={psi_sq.std():.4f}")
    print(f"  Triangular crystal: mean={psi_tri.mean():.4f}, std={psi_tri.std():.4f}")

    # Test one step / 测试单步演化
    psi_new = engine.step(psi_sq)
    e_initial = engine.total_energy(psi_sq)
    e_after = engine.total_energy(psi_new)
    print(f"  Energy before step: {e_initial:.4f}")
    print(f"  Energy after step: {e_after:.4f}")
    print(f"  Energy change: {e_after - e_initial:.6f}")

    # Test relaxation / 测试弛豫
    psi_relaxed, energies = engine.relax(psi_sq, n_steps=200)
    print(f"  After 200 steps: energy={energies[-1][1]:.4f}")
    print(f"  Energy converged: {abs(energies[-1][1] - energies[-2][1]) < 0.01}")
    print("  PASSED\n")
    return True


def test_flat_interface():
    """Test flat interface model / 测试平滑界面模型"""
    print("=" * 50)
    print("Test 2: Flat Interface Model / 平滑界面模型")
    print("=" * 50)

    cfg = DualPFCConfig(Nx=64, Ny=64, sigma=-0.3, n_steps=200, dt=0.5)
    model = FlatInterfaceModel(cfg, crystal_type='square')
    psi = model.build_initial_condition()
    print(f"  Initial field shape: {psi.shape}")
    print(f"  Initial energy: {model.engine.total_energy(psi):.4f}")

    model.run(n_steps=200)
    gamma = model.compute_interface_energy(n_bulk_relax=200)
    print(f"  gamma_SL = {gamma:.6f}")
    print("  PASSED\n")
    return True


def test_round_interface():
    """Test round interface model / 测试圆形界面模型"""
    print("=" * 50)
    print("Test 3: Round Interface Model / 圆形界面模型")
    print("=" * 50)

    cfg = DualPFCConfig(Nx=64, Ny=64, sigma=0.2, n_steps=200, dt=0.5)
    model = RoundInterfaceModel(cfg, crystal_type='triangular')
    psi = model.build_initial_condition()
    print(f"  Initial field shape: {psi.shape}")
    print(f"  Nucleus radius: {model.R:.2f}")

    model.run(n_steps=200)
    gamma, R_eff = model.compute_interface_energy(n_bulk_relax=200)
    print(f"  gamma_SL = {gamma:.6f}, R_eff = {R_eff:.2f}")
    print("  PASSED\n")
    return True


def test_dislocation():
    """Test dislocation model / 测试位错模型"""
    print("=" * 50)
    print("Test 4: Dislocation Model / 位错模型")
    print("=" * 50)

    cfg = DualPFCConfig(Nx=64, Ny=64, sigma=0.0, n_steps=200, dt=0.5)

    # Test square / 测试正方形
    model_sq = DislocationModel(cfg, crystal_type='square')
    model_sq.prepare_and_run(n_relax_perfect=200, n_relax_disloc=200)
    print(f"  Square dislocation delta_y = {model_sq.delta_y:.4f}")
    print(f"  Field shape: {model_sq.psi.shape}")
    E_dis_sq = model_sq.E_dis
    print(f"  Square E_dis = {E_dis_sq:.6f}")

    # Test triangular / 测试三角形
    model_tri = DislocationModel(cfg, crystal_type='triangular')
    model_tri.prepare_and_run(n_relax_perfect=200, n_relax_disloc=200)
    print(f"  Triangular dislocation delta_y = {model_tri.delta_y:.4f}")
    E_dis_tri = model_tri.E_dis
    print(f"  Triangular E_dis = {E_dis_tri:.6f}")
    print("  PASSED\n")
    return True


def test_visualization():
    """Test visualization module (non-interactive) / 测试可视化模块"""
    print("=" * 50)
    print("Test 5: Visualization Module / 可视化模块")
    print("=" * 50)

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend / 非交互后端
    from visualization import PFCVisualizer

    cfg = DualPFCConfig(Nx=64, Ny=64, sigma=0.0, n_steps=100)
    engine = DualPFCEngine(cfg)
    X, Y = engine.get_coordinate_grids()
    psi = engine.triangular_crystal(X, Y)
    psi, _ = engine.relax(psi, n_steps=100)

    viz = PFCVisualizer(engine)

    # Test plots (save to file instead of showing) / 测试绘图（保存到文件）
    fig1, _ = viz.plot_field(psi, show=False)
    fig1.savefig(os.path.join(os.path.dirname(__file__), 'test_field.png'), dpi=72)
    plt = matplotlib.pyplot
    plt.close(fig1)

    fig2, _ = viz.plot_structure_factor(psi, show=False)
    fig2.savefig(os.path.join(os.path.dirname(__file__), 'test_sf.png'), dpi=72)
    plt.close(fig2)

    print("  Field plot saved / 密度场图已保存")
    print("  Structure factor plot saved / 结构因子图已保存")
    print("  PASSED\n")
    return True


if __name__ == '__main__':
    print("\nDual-Amplitude PFC Simulator — Quick Test Suite")
    print("双振幅PFC模拟器 — 快速测试套件\n")

    all_passed = True
    try:
        all_passed &= test_core_engine()
    except Exception as e:
        print(f"  FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_flat_interface()
    except Exception as e:
        print(f"  FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_round_interface()
    except Exception as e:
        print(f"  FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_dislocation()
    except Exception as e:
        print(f"  FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_visualization()
    except Exception as e:
        print(f"  FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("=" * 50)
    if all_passed:
        print("ALL TESTS PASSED / 全部测试通过!")
    else:
        print("SOME TESTS FAILED / 部分测试失败")
    print("=" * 50)
