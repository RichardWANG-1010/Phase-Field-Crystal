#!/usr/bin/env python3
"""
run_3d.py
三维PFC模拟运行脚本
3D PFC Simulation Runner
对应书中Chapter 8 (纯材料PFC) 和 Chapter 9 (二元合金PFC)
使用方法:
    python run_3d.py           # 交互式选择模式
    python run_3d.py pure      # 运行三维纯材料PFC (BCC晶格)
    python run_3d.py elastic   # 运行三维弹性模拟
    python run_3d.py alloy     # 运行三维二元合金PFC (Chapter 9)
分辨率说明:
    默认 N=256, L=128.0 (dx=0.5)
    如需更高分辨率可改为 N=512, L=256.0，但计算时间和内存会显著增加:
    - N=128: 约 200MB 内存, 每步较快
    - N=256: 约 1.5GB 内存, 每步慢约 8 倍
    - N=512: 约 12GB 内存, 每步慢约 64 倍 (仅推荐工作站使用)
作者: Jinpeng Wang
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# 导入三维求解器
from pfc_3d import PurePFC3DSolver


def run_pure_3d():
    """
    运行三维纯材料PFC模拟
    参考: Provatas & Elder, Chapter 8
    参数设置对应书中BCC晶格稳定区域:
    - r < 0 时系统形成晶体结构
    - r ≈ -0.5 为典型BCC稳定点
    - 平均密度 φ0 控制液相线
    """
    print("=" * 70)
    print("3D Pure Material PFC Simulation (Chapter 8)")
    print("=" * 70)
    # 创建求解器
    # N=256, L=128.0: dx=0.5，切片图细腻无像素感
    solver = PurePFC3DSolver(
        N=256,               # 256³ 网格 (画质细腻)
        L=128.0,             # 物理域尺寸保持不变
        r=-0.25,            # 温度参数 (ε in book)
        M=1.0,              # 迁移率
        dt=0.05,            # 时间步长
        T=200.0,            # 总模拟时间
        phi0=-0.25,         # 平均密度
        noise_amp=0.01,     # 初始噪声幅度
        lattice_type="bcc"  # BCC晶格
    )
    # 运行模拟
    solver.run(sample_interval=10)
    # 后处理
    solver.postprocess()
    # 保存最终状态
    np.save("phi_3d_final.npy", solver.phi)
    print("\nFinal state saved to phi_3d_final.npy")
    return solver


def run_elastic_3d():
    """
    运行三维弹性模拟
    参考: Provatas & Elder, Section 8.5
    计算弹性常数的流程:
    1. 弛豫到平衡态
    2. 保存参考状态
    3. 施加一系列单轴应变
    4. 每个应变下弛豫并记录能量
    5. 拟合 E(ε) 曲线得到弹性模量
    """
    print("=" * 70)
    print("3D Elastic Simulation (Section 8.5)")
    print("=" * 70)
    # 创建求解器
    solver = PurePFC3DSolver(
        N=256,
        L=128.0,
        r=-0.25,
        M=1.0,
        dt=0.05,
        T=500.0,
        phi0=-0.25,
        noise_amp=0.01,
        lattice_type="bcc"
    )
    # Step 1: 弛豫到平衡态
    print("\n[Phase 1] Relaxing to equilibrium...")
    solver.run(sample_interval=20)
    # 保存参考状态
    solver.save_reference_state_3d()
    print("\nReference state saved.")
    # Step 2: 施加应变并计算弹性能量
    print("\n[Phase 2] Computing elastic energy curve...")
    strain_list = np.linspace(-0.02, 0.02, 11)
    print(f"Strain range: [{strain_list.min():.3f}, {strain_list.max():.3f}]")
    # 计算能量曲线
    energy, phi_list = solver.elastic_energy_curve_3d(
        strain_list=strain_list,
        direction=0,        # x方向单轴应变
        nu=0.33,            # 泊松比 (近似值)
        relax_steps=1000    # 每个应变点弛豫步数
    )
    # Step 3: 拟合弹性常数
    print("\n[Phase 3] Fitting elastic constants...")
    C, eps_r, coef, strain_fit, energy_fit = solver.fit_elastic_constant_3d(
        strain_list, energy
    )
    print(f"\nElastic constant C = {C:.4e}")
    print(f"Reference strain ε_r = {eps_r:.6f}")
    # 绘制能量-应变曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 能量-应变
    ax = axes[0]
    ax.plot(strain_list, energy, 'bo', label="Simulation")
    ax.plot(strain_fit, energy_fit, 'r-', label="Parabolic fit")
    ax.axvline(x=eps_r, color='g', linestyle='--', label=f"ε_r = {eps_r:.4f}")
    ax.set_xlabel("Strain ε")
    ax.set_ylabel("Free Energy")
    ax.set_title("Elastic Energy vs Strain")
    ax.legend()
    ax.grid(True)
    # 应力-应变
    ax = axes[1]
    stress = solver.compute_stress_3d(strain_list, energy)
    ax.plot(strain_list, stress, 'gs-', label="Stress")
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel("Strain ε")
    ax.set_ylabel("Stress σ")
    ax.set_title("Stress-Strain Curve")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("elastic_3d_analysis.png", dpi=150)
    plt.show()
    # 保存数据
    np.savez("elastic_3d_data.npz",
             strain=strain_list,
             energy=energy,
             C=C,
             eps_r=eps_r)
    print("\nData saved to elastic_3d_data.npz")
    return solver, energy, strain_list


def run_alloy_3d():
    """
    运行三维二元合金PFC模拟
    参考: Provatas & Elder, Chapter 9
    二元合金PFC模型引入第二组分浓度场 n(r)
    自由能泛函包含密度场 φ 和浓度场 n 的耦合
    """
    print("=" * 70)
    print("3D Binary Alloy PFC Simulation (Chapter 9)")
    print("=" * 70)
    print("\nNote: 3D alloy PFC requires additional concentration field.")
    print("This is a simplified demonstration.")
    # 创建纯材料求解器作为基础
    solver = PurePFC3DSolver(
        N=256,
        L=128.0,
        r=-0.25,
        M=1.0,
        dt=0.05,
        T=200.0,
        phi0=-0.25,
        noise_amp=0.01,
        lattice_type="bcc"
    )
    # 初始化浓度场 (二元合金的第二组分)
    # 书中Section 9.1: n(r) 表示浓度偏差
    c0 = 0.1  # 平均浓度
    solver.concentration = c0 + 0.01 * np.random.randn(solver.N, solver.N, solver.N)
    print(f"Initial concentration mean = {np.mean(solver.concentration):.4f}")
    # 运行耦合模拟 (简化版本)
    # 实际实现需要修改step()方法包含浓度演化
    solver.run(sample_interval=10)
    # 可视化浓度场
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (ax, axis) in enumerate(zip(axes, [0, 1, 2])):
        if axis == 0:
            slice_c = solver.concentration[solver.N//2, :, :]
        elif axis == 1:
            slice_c = solver.concentration[:, solver.N//2, :]
        else:
            slice_c = solver.concentration[:, :, solver.N//2]
        im = ax.imshow(slice_c, cmap="viridis", origin="lower")
        ax.set_title(f"Concentration Slice (axis={['x','y','z'][axis]})")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("alloy_3d_concentration.png", dpi=150)
    plt.show()
    return solver


def run_parameter_sweep_3d():
    """
    三维参数扫描
    扫描不同r值下的晶格稳定性
    """
    print("=" * 70)
    print("3D Parameter Sweep")
    print("=" * 70)
    r_values = [-0.5, -0.4, -0.3, -0.25, -0.2, -0.1]
    results = []
    for r in r_values:
        print(f"\nRunning with r = {r}...")
        # 扫描模式用较小网格以节省时间
        solver = PurePFC3DSolver(
            N=128,           # 扫描用128³，速度优先
            L=128.0,
            r=r,
            M=1.0,
            dt=0.05,
            T=100.0,
            phi0=-0.25,
            noise_amp=0.01,
            lattice_type="bcc"
        )
        solver.run(sample_interval=20)
        # 记录最终能量和标准差
        final_E = solver.compute_energy()
        final_std = np.std(solver.phi)
        results.append({
            'r': r,
            'E': final_E,
            'std': final_std
        })
        print(f"  Final E = {final_E:.6e}, std = {final_std:.4f}")
    # 绘制结果
    r_arr = [r['r'] for r in results]
    E_arr = [r['E'] for r in results]
    std_arr = [r['std'] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(r_arr, E_arr, 'bo-')
    axes[0].set_xlabel("r (temperature parameter)")
    axes[0].set_ylabel("Free Energy")
    axes[0].set_title("Energy vs r")
    axes[0].grid(True)
    axes[1].plot(r_arr, std_arr, 'rs-')
    axes[1].set_xlabel("r (temperature parameter)")
    axes[1].set_ylabel("Density Std Dev")
    axes[1].set_title("Order Parameter vs r")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("parameter_sweep_3d.png", dpi=150)
    plt.show()
    return results


def main():
    """主函数 - 交互式选择运行模式"""
    print(__doc__)
    print("=" * 70)
    print("请选择要运行的模式:")
    print("  [1] pure    - 三维纯材料PFC模拟 (BCC晶格)")
    print("  [2] elastic - 三维弹性常数计算")
    print("  [3] alloy   - 三维二元合金PFC模拟")
    print("  [4] sweep   - 参数扫描 (不同r值)")
    print("=" * 70)
    mode = input("请输入模式编号或名称 (默认: pure): ").strip().lower()
    # 处理空输入，默认使用 pure
    if mode == "" or mode == "1":
        mode = "pure"
    elif mode == "2":
        mode = "elastic"
    elif mode == "3":
        mode = "alloy"
    elif mode == "4":
        mode = "sweep"
    if mode == "pure":
        solver = run_pure_3d()
    elif mode == "elastic":
        solver, energy, strain = run_elastic_3d()
    elif mode == "alloy":
        solver = run_alloy_3d()
    elif mode == "sweep":
        results = run_parameter_sweep_3d()
    else:
        print(
            f"错误: 未知模式 '{mode}'")
        print("可选模式: pure | elastic | alloy | sweep")
        print("请重新运行脚本并输入有效模式。")
        sys.exit(1)


if __name__ == "__main__":
    main()
