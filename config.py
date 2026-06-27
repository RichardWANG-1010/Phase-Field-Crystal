import sys

def input_pfc_parameters():
    """
    交互式控制台输入：选择晶格、输入全部仿真参数，返回参数字典
    Interactive console input: select lattice & input all simulation parameters
    """
    print("="*60)
    print("        2D PFC多晶格仿真交互参数输入程序")
    print("="*60)
    print("可选晶格类型：")
    print("  1 - hexagon  六角晶格（标准PFC）")
    print("  2 - square   正方形晶格")
    print("  3 - triangle 三角形晶格")
    print("-"*60)

    # 1. 晶格选择交互
    while True:
        lattice_choice = input("请输入晶格编号(1/2/3)：").strip()
        if lattice_choice == "1":
            lattice = "hexagon"
            break
        elif lattice_choice == "2":
            lattice = "square"
            break
        elif lattice_choice == "3":
            lattice = "triangle"
            break
        else:
            print("输入错误！只能输入 1 / 2 / 3，请重新输入\n")

    # 2. 通用参数输入函数（带异常捕获，防止非数字崩溃）
    def get_float_input(prompt, default):
        while True:
            val = input(f"{prompt} (默认值={default})：").strip()
            if val == "":
                return float(default)
            try:
                return float(val)
            except ValueError:
                print("输入必须为数字，请重新输入\n")

    def get_int_input(prompt, default):
        while True:
            val = input(f"{prompt} (默认值={default})：").strip()
            if val == "":
                return int(default)
            try:
                return int(val)
            except ValueError:
                print("输入必须为整数，请重新输入\n")

    print("\n--- 仿真基础参数（直接回车使用默认值）---")
    N = get_int_input("网格尺寸N（方阵N×N）", 256)
    L = get_float_input("计算域边长L", 128.0)
    r = get_float_input("PFC线性控制参数r", -0.35)
    M = get_float_input("动力学迁移系数M", 1.0)
    dt = get_float_input("仿真时间步长dt", 0.05)
    T = get_float_input("总仿真物理时长T", 1500.0)
    phi0 = get_float_input("全局平均密度phi0", -0.25)
    noise_amp = get_float_input("初始高斯噪声幅值", 0.01)

    # 3. 返回打包参数
    param_dict = {
        
        "solver": {
            "N": N,
            "L": L,
            "r": r,
            "M": M,
            "dt": dt,
            "T": T,
            "phi0": phi0,
            "noise_amp": noise_amp
        },
        "lattice_type": lattice
    }
    print("\n==== 已确认当前仿真配置 ====")
    for k, v in param_dict.items():
        print(f"{k} : {v}")
    confirm = input("确认开始仿真？(y/n)：").strip().lower()
    if confirm != "y":
        print("已取消仿真，程序退出")
        sys.exit(0)
    return param_dict