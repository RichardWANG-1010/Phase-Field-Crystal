"""
config.py - Interactive Configuration Menu for PFC Simulations
配置模块 - PFC模拟的交互式参数输入菜单

This module provides a console-based interactive interface for users to configure
PFC simulation parameters, including lattice type selection and numerical settings.
本模块提供基于控制台的交互式界面，供用户配置PFC模拟参数，包括晶格类型选择和数值设置。

Author: Jinpeng Wang
Department of Material Engineering
"""

# System module for exit functionality
# 系统模块，用于程序退出功能
import sys


def input_pfc_parameters():
    """
    Interactive console input function for PFC simulation parameters.
    PFC模拟参数的交互式控制台输入函数。
    
    This function guides the user through selecting a lattice type and inputting
    all simulation parameters with default values and input validation.
    该函数引导用户选择晶格类型并输入所有模拟参数，提供默认值和输入验证。
    
    Returns / 返回值:
        dict: Parameter dictionary with structure / 参数字典，结构如下：
            {
                "solver": {
                    "N": int,           # Grid size / 网格尺寸
                    "L": float,         # Domain size / 计算域尺寸
                    "r": float,         # PFC control parameter / PFC控制参数
                    "M": float,         # Mobility / 迁移率
                    "dt": float,        # Time step / 时间步长
                    "T": float,         # Total time / 总时间
                    "phi0": float,      # Average density / 平均密度
                    "noise_amp": float  # Noise amplitude / 噪声幅值
                },
                "lattice_type": str     # Lattice type / 晶格类型
            }
    
    Raises / 异常:
        SystemExit: If user cancels the simulation / 如果用户取消模拟
    """
    # Print welcome banner
    # 打印欢迎横幅
    print("=" * 60)
    print("        2D PFC Multi-Lattice Simulation Parameter Configurator")
    print("        二维PFC多晶格仿真交互参数输入程序")
    print("=" * 60)
    
    # Display lattice type options
    # 显示晶格类型选项
    print("Available lattice types / 可选晶格类型：")
    print("  1 - hexagon  Hexagonal lattice (standard PFC) / 六角晶格（标准PFC）")
    print("  2 - square   Square lattice / 正方形晶格")
    print("  3 - triangle Triangular lattice / 三角形晶格")
    print("-" * 60)

    # ============================================================
    # Step 1: Lattice type selection with input validation
    # 步骤1：晶格选择，带输入验证
    # ============================================================
    while True:
        # Prompt user for lattice choice
        # 提示用户输入晶格选择
        lattice_choice = input("Please enter lattice number (1/2/3) / 请输入晶格编号(1/2/3)：").strip()
        
        if lattice_choice == "1":
            # Hexagonal lattice - most common for PFC
            # 六角晶格 - PFC最常用的晶格类型
            lattice = "hexagon"
            break
        elif lattice_choice == "2":
            # Square lattice - for square crystal structures
            # 正方晶格 - 用于正方晶体结构
            lattice = "square"
            break
        elif lattice_choice == "3":
            # Triangular lattice - oblique lattice structure
            # 三角晶格 - 斜晶格结构
            lattice = "triangle"
            break
        else:
            # Invalid input - prompt user to try again
            # 无效输入 - 提示用户重新输入
            print("Invalid input! Please enter 1, 2, or 3.\n")
            print("输入错误！只能输入 1 / 2 / 3，请重新输入\n")

    # ============================================================
    # Helper functions for validated input
    # 辅助函数：带验证的输入函数
    # ============================================================
    
    def get_float_input(prompt, default):
        """
        Get a float input from user with validation and default value.
        获取用户输入的浮点数值，带验证和默认值。
        
        Args / 参数:
            prompt (str): Prompt message / 提示信息
            default (float): Default value if user presses Enter / 用户回车时的默认值
        
        Returns / 返回值:
            float: Validated float value / 验证后的浮点数值
        """
        while True:
            # Display prompt with default value
            # 显示带默认值的提示
            val = input(f"{prompt} (default={default}) / (默认值={default})：").strip()
            
            # If user presses Enter, use default value
            # 如果用户直接回车，使用默认值
            if val == "":
                return float(default)
            
            # Try to convert to float
            # 尝试转换为浮点数
            try:
                return float(val)
            except ValueError:
                # Conversion failed - show error and retry
                # 转换失败 - 显示错误并重试
                print("Input must be a number. Please try again.\n")
                print("输入必须为数字，请重新输入\n")

    def get_int_input(prompt, default):
        """
        Get an integer input from user with validation and default value.
        获取用户输入的整数值，带验证和默认值。
        
        Args / 参数:
            prompt (str): Prompt message / 提示信息
            default (int): Default value if user presses Enter / 用户回车时的默认值
        
        Returns / 返回值:
            int: Validated integer value / 验证后的整数值
        """
        while True:
            # Display prompt with default value
            # 显示带默认值的提示
            val = input(f"{prompt} (default={default}) / (默认值={default})：").strip()
            
            # If user presses Enter, use default value
            # 如果用户直接回车，使用默认值
            if val == "":
                return int(default)
            
            # Try to convert to int
            # 尝试转换为整数
            try:
                return int(val)
            except ValueError:
                # Conversion failed - show error and retry
                # 转换失败 - 显示错误并重试
                print("Input must be an integer. Please try again.\n")
                print("输入必须为整数，请重新输入\n")

    # ============================================================
    # Step 2: Input general simulation parameters
    # 步骤2：输入通用仿真参数
    # ============================================================
    print("\n--- Simulation Parameters (press Enter for defaults) ---")
    print("--- 仿真基础参数（直接回车使用默认值）---")
    
    # Grid resolution - number of grid points in each dimension
    # 网格分辨率 - 每个维度的网格点数
    N = get_int_input("Grid size N (N×N square grid) / 网格尺寸N（方阵N×N）", 256)
    
    # Physical domain size - length of the simulation box
    # 物理域尺寸 - 模拟盒子的边长
    L = get_float_input("Domain size L / 计算域边长L", 128.0)
    
    # PFC control parameter - related to undercooling/temperature
    # PFC控制参数 - 与过冷度/温度相关
    r = get_float_input("PFC control parameter r / PFC线性控制参数r", -0.35)
    
    # Mobility coefficient - controls diffusion speed
    # 迁移率系数 - 控制扩散速度
    M = get_float_input("Mobility M / 动力学迁移系数M", 1.0)
    
    # Time step - numerical integration step size
    # 时间步长 - 数值积分步长
    dt = get_float_input("Time step dt / 仿真时间步长dt", 0.05)
    
    # Total simulation time - physical duration
    # 总模拟时间 - 物理时长
    T = get_float_input("Total simulation time T / 总仿真物理时长T", 1500.0)
    
    # Average density - mean value of the density field
    # 平均密度 - 密度场的平均值
    phi0 = get_float_input("Average density phi0 / 全局平均密度phi0", -0.25)
    
    # Initial noise amplitude - random perturbation magnitude
    # 初始噪声幅值 - 随机扰动的大小
    noise_amp = get_float_input("Initial noise amplitude / 初始高斯噪声幅值", 0.01)

    # ============================================================
    # Step 3: Package parameters into dictionary
    # 步骤3：将参数打包成字典
    # ============================================================
    param_dict = {
        # Solver parameters - passed directly to PFC solver constructor
        # 求解器参数 - 直接传递给PFC求解器构造函数
        "solver": {
            "N": N,              # Grid resolution / 网格分辨率
            "L": L,              # Domain size / 计算域尺寸
            "r": r,              # Control parameter / 控制参数
            "M": M,              # Mobility / 迁移率
            "dt": dt,            # Time step / 时间步长
            "T": T,              # Total time / 总时间
            "phi0": phi0,        # Average density / 平均密度
            "noise_amp": noise_amp  # Noise amplitude / 噪声幅值
        },
        # Lattice type - determines dispersion operator
        # 晶格类型 - 决定色散算子形式
        "lattice_type": lattice
    }

    # ============================================================
    # Step 4: Confirm configuration with user
    # 步骤4：与用户确认配置
    # ============================================================
    print("\n==== Current Simulation Configuration / 已确认当前仿真配置 ====")
    for k, v in param_dict.items():
        print(f"{k} : {v}")
    
    # Ask user to confirm before starting
    # 开始前询问用户确认
    confirm = input("Start simulation? (y/n) / 确认开始仿真？(y/n)：").strip().lower()
    
    if confirm != "y":
        # User cancelled - exit gracefully
        # 用户取消 - 优雅退出
        print("Simulation cancelled. Exiting program.")
        print("已取消仿真，程序退出")
        sys.exit(0)

    # Return validated parameter dictionary
    # 返回验证后的参数字典
    return param_dict
