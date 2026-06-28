"""
run_pure.py - Interactive Pure Material PFC Simulation Runner
纯材料PFC模拟交互式运行脚本

This script provides an interactive interface for running pure material
PFC simulations. It uses the config module for parameter input and the
PurePFCSolver for the actual simulation.
本脚本提供运行纯材料PFC模拟的交互式界面。使用config模块进行参数输入，
使用PurePFCSolver进行实际模拟。

Usage / 使用方法:
    python run_pure.py

Workflow / 工作流程:
    1. Interactive parameter input / 交互式参数输入
    2. Create solver instance / 创建求解器实例
    3. Run simulation / 运行模拟
    4. Post-processing analysis / 后处理分析
    5. Psi6 order analysis / Psi6有序度分析

Author: Jinpeng Wang
Department of Material Engineering
"""

# Import interactive configuration function
# 导入交互式配置函数
from config import input_pfc_parameters

# Import pure material PFC solver
# 导入纯材料PFC求解器
from pfc_pure import PurePFCSolver


def main():
    """
    Main function for pure material PFC simulation.
    纯材料PFC模拟的主函数。
    
    This function:
    1. Gets simulation parameters interactively from user
    2. Creates a PurePFCSolver instance
    3. Runs the simulation
    4. Performs post-processing analysis
    5. Analyzes psi6 orientational order
    
    该函数：
    1. 从用户交互式获取模拟参数
    2. 创建PurePFCSolver实例
    3. 运行模拟
    4. 执行后处理分析
    5. 分析psi6取向有序度
    """
    # ============================================================
    # Step 1: Get simulation parameters interactively
    # 步骤1：交互式获取模拟参数
    # ============================================================
    
    # Call interactive parameter input function
    # 调用交互式参数输入函数
    # Returns dictionary with solver parameters and lattice type
    # 返回包含求解器参数和晶格类型的字典
    params = input_pfc_parameters()
    
    # ============================================================
    # Step 2: Create solver instance
    # 步骤2：创建求解器实例
    # ============================================================
    
    # Create PurePFCSolver with user-specified parameters
    # 使用用户指定的参数创建PurePFCSolver
    # **params["solver"] unpacks all solver keyword arguments
    # **params["solver"]解包所有求解器关键字参数
    solver = PurePFCSolver(
        **params["solver"],
        lattice_type=params["lattice_type"]
    )
    
    # ============================================================
    # Step 3: Run simulation
    # 步骤3：运行模拟
    # ============================================================
    
    # Run the main simulation loop
    # 运行主模拟循环
    solver.run()
    
    # ============================================================
    # Step 4: Post-processing analysis
    # 步骤4：后处理分析
    # ============================================================
    
    # Run full post-processing pipeline
    # 运行全套后处理流程
    # Includes energy, field, structure factor, Voronoi, defects, etc.
    # 包括能量、场、结构因子、Voronoi、缺陷等
    solver.postprocess()
    
    # ============================================================
    # Step 5: Psi6 orientational order analysis
    # 步骤5：Psi6取向有序度分析
    # ============================================================
    
    # Perform comprehensive psi6 analysis
    # 进行全面的psi6分析
    # Includes local/global order metrics and orientation plots
    # 包括局部/全局有序度指标和取向图
    solver.analyze_psi6()


# ============================================================
# Script entry point
# 脚本入口点
# ============================================================

if __name__ == "__main__":
    # Execute main function when script is run directly
    # 当脚本直接运行时执行主函数
    main()
