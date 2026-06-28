"""
run_elastic.py - Elastic Constant Calculation Script for PFC Model
PFC模型弹性常数计算脚本

This script calculates elastic constants of the PFC model by applying
uniform strain and measuring the resulting free energy change.
本脚本通过施加均匀应变并测量由此产生的自由能变化来计算PFC模型的弹性常数。

Method / 方法:
    1. Run PFC simulation to obtain equilibrium crystal structure
       运行PFC模拟获得平衡晶体结构
    2. Save reference (unstrained) state
       保存参考（无应变）状态
    3. Apply series of strains (compression and tension)
       施加一系列应变（压缩和拉伸）
    4. Measure free energy at each strain
       测量每个应变下的自由能
    5. Fit quadratic curve: F(ε) = a·ε² + b·ε + c
       拟合二次曲线：F(ε) = a·ε² + b·ε + c
    6. Extract elastic constant: C = 2·a = d²F/dε²
       提取弹性常数：C = 2·a = d²F/dε²

Author: Jinpeng Wang
Department of Material Engineering
"""

# Import pure material PFC solver
# 导入纯材料PFC求解器
from pfc_pure import PurePFCSolver

# NumPy for numerical operations
# NumPy用于数值运算
import numpy as np


def main():
    """
    Main function for elastic constant calculation.
    弹性常数计算的主函数。
    
    This function performs a complete elastic constant calculation:
    该函数执行完整的弹性常数计算：
    1. Create and run PFC simulation to equilibrium
       创建并运行PFC模拟到平衡态
    2. Save reference state
       保存参考状态
    3. Generate strain sequence
       生成应变序列
    4. Calculate elastic energy curve
       计算弹性能量曲线
    5. Plot energy-strain and stress-strain curves
       绘制能量-应变和应力-应变曲线
    6. Fit and print elastic constant
       拟合并打印弹性常数
    """
    # ============================================================
    # Step 1: Create and run PFC solver
    # 步骤1：创建并运行PFC求解器
    # ============================================================
    
    # Create PurePFCSolver with specified parameters
    # 使用指定参数创建PurePFCSolver
    # N=256: 256x256 grid resolution
    # L=128: Physical domain size of 128 units
    # r=-0.35: PFC control parameter (undercooling)
    # M=1.0: Mobility coefficient
    # dt=0.05: Time step size
    # T=1500: Total simulation time
    # phi0=-0.25: Average density
    # noise_amp=0.01: Initial noise amplitude
    solver = PurePFCSolver(
        N=256,
        L=128,
        r=-0.35,
        M=1.0,
        dt=0.05,
        T=1500,
        phi0=-0.25,
        noise_amp=0.01
    )
    
    # Run the simulation to reach equilibrium
    # 运行模拟以达到平衡态
    solver.run()
    
    # ============================================================
    # Step 2: Save reference state
    # 步骤2：保存参考状态
    # ============================================================
    
    # Save the current equilibrium state as reference (zero strain)
    # 将当前平衡状态保存为参考（零应变）
    # This will be used as the starting point for each strain calculation
    # 这将用作每个应变计算的起点
    solver.save_reference_state()
    
    # ============================================================
    # Step 3: Generate strain sequence
    # 步骤3：生成应变序列
    # ============================================================
    
    # Generate 13 strain values from -0.03 to +0.03
    # 生成从-0.03到+0.03的13个应变值
    # -0.03 = 3% compression (压缩)
    # +0.03 = 3% tension (拉伸)
    # 13 points ensures good sampling for quadratic fit
    # 13个点确保二次拟合有良好的采样
    strain = np.linspace(
        -0.03,
        0.03,
        13
    )
    
    # ============================================================
    # Step 4: Calculate elastic energy curve
    # 步骤4：计算弹性能量曲线
    # ============================================================
    
    # Calculate free energy at each strain value
    # 计算每个应变值下的自由能
    # For each strain: reset to reference, apply strain, relax, measure energy
    # 对于每个应变：重置到参考状态，施加应变，弛豫，测量能量
    energy, phi_list = (
        solver.elastic_energy_curve(
            strain
        )
    )
    
    # ============================================================
    # Step 5: Plot energy-strain curve (F-ε)
    # 步骤5：绘制能量-应变曲线（F-ε）
    # ============================================================
    
    # Plot elastic energy vs strain with quadratic fit
    # 绘制弹性能量随应变的变化及二次拟合
    # This shows the parabolic energy-strain relationship
    # 这显示了抛物线形的能量-应变关系
    solver.plot_elastic_curve(
        strain,
        energy
    )
    
    # ============================================================
    # Step 6: Calculate and plot stress-strain curve (σ-ε)
    # 步骤6：计算并绘制应力-应变曲线（σ-ε）
    # ============================================================
    
    # Compute stress as numerical derivative of energy w.r.t. strain
    # 计算应力作为能量对应变的数值导数
    # σ = dF/dε
    stress = solver.compute_stress(
        strain,
        energy
    )
    
    # Plot stress-strain curve
    # 绘制应力-应变曲线
    # The slope of this curve is the elastic modulus
    # 该曲线的斜率是弹性模量
    solver.plot_stress_strain(
        strain,
        energy
    )
    
    # ============================================================
    # Step 7: Fit elastic constant and print results
    # 步骤7：拟合弹性常数并打印结果
    # ============================================================
    
    # Fit quadratic curve and extract elastic constant
    # 拟合二次曲线并提取弹性常数
    # C = 2·a = d²F/dε² (Young's modulus for 2D)
    # C = 2·a = d²F/dε²（二维杨氏模量）
    # eps_r = residual strain (strain at energy minimum)
    # eps_r = 残余应变（能量最小时的应变）
    C, eps_r, _, _, _ = (
        solver.fit_elastic_constant(
            strain,
            energy
        )
    )
    
    # ============================================================
    # Print results to console
    # 向控制台打印结果
    # ============================================================
    
    print()
    print(f"Elastic constant / 弹性常数 C: {C:.6e}")
    print(f"Residual strain / 残余应变 ε_r: {eps_r:.6e}")


# ============================================================
# Script entry point
# 脚本入口点
# ============================================================

if __name__ == "__main__":
    # Execute main function when script is run directly
    # 当脚本直接运行时执行主函数
    main()
