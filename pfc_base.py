"""
pfc_base.py - Core Numerical Infrastructure for PFC Simulations
数值计算核心库 - PFC模拟的核心数值基础设施

This module provides the base class for all PFC solvers, including grid setup,
k-space construction, and common simulation logging infrastructure.
本模块提供所有PFC求解器的基类，包括网格设置、k空间构建和通用模拟日志基础设施。

The PFC (Phase Field Crystal) model describes crystalline structures at atomic
length scales while operating on diffusive time scales.
PFC（相场晶体）模型在原子长度尺度上描述晶体结构，同时在扩散时间尺度上运行。

Author: Jinpeng Wang
Department of Material Engineering
"""

# NumPy - fundamental package for numerical computing
# NumPy - 数值计算的基础包
import numpy as np


class PFCBase:
    """
    Base class for 2D Phase Field Crystal simulations.
    二维相场晶体模拟的基类。
    
    This class provides the fundamental numerical infrastructure including:
    本类提供基础数值基础设施，包括：
    - Grid parameters and spatial discretization / 网格参数和空间离散化
    - Fourier space (k-space) construction / 傅里叶空间（k空间）构建
    - Simulation data logging / 模拟数据记录
    - Video recording configuration / 视频录制配置
    
    This is an abstract base class - actual solvers (PurePFCSolver, BinaryPFCSolver, etc.)
    inherit from this class and implement the specific evolution equations.
    这是一个抽象基类 - 实际的求解器（PurePFCSolver、BinaryPFCSolver等）继承此类并实现特定的演化方程。
    
    Attributes / 属性:
        N (int): Grid resolution (number of points per dimension) / 网格分辨率（每维点数）
        L (float): Physical domain size / 物理域尺寸
        L0 (float): Reference domain size (for strain calculations) / 参考域尺寸（用于应变计算）
        dx (float): Grid spacing / 网格间距
        dt (float): Time step / 时间步长
        T (float): Total simulation time / 总模拟时间
        steps (int): Total number of time steps / 总时间步数
        kx (ndarray): x-component of wave vector / 波矢的x分量
        ky (ndarray): y-component of wave vector / 波矢的y分量
        k2 (ndarray): Squared magnitude of wave vector (kx² + ky²) / 波矢模长平方
        energy_log (list): Free energy history / 自由能历史
        mass_log (list): Mass (mean density) history / 质量（平均密度）历史
        defect_log (list): Defect density history / 缺陷密度历史
        grain_size_log (list): Grain size history / 晶粒尺寸历史
        structure_peak_log (list): Structure factor peak history / 结构因子峰值历史
    """
    
    def __init__(
        self,
        N=128,
        L=64.0,
        dt=0.05,
        T=500.0
    ):
        """
        Initialize the PFC base class with grid and time parameters.
        使用网格和时间参数初始化PFC基类。
        
        Args / 参数:
            N (int, optional): Grid resolution (N×N). Defaults to 128.
                              网格分辨率（N×N），默认128。
            L (float, optional): Physical domain size. Defaults to 64.0.
                                物理域尺寸，默认64.0。
            dt (float, optional): Time step size. Defaults to 0.05.
                                 时间步长，默认0.05。
            T (float, optional): Total simulation time. Defaults to 500.0.
                                总模拟时间，默认500.0。
        """
        # ============================================================
        # Grid parameters / 网格参数
        # ============================================================
        
        # Grid resolution - number of grid points in each dimension
        # 网格分辨率 - 每个维度的网格点数
        self.N = int(N)
        
        # Physical domain size - length of the simulation box
        # 物理域尺寸 - 模拟盒子的边长
        self.L = float(L)
        
        # Reference domain size - saved for strain calculations
        # 参考域尺寸 - 保存用于应变计算
        self.L0 = self.L
        
        # Grid spacing - physical distance between adjacent grid points
        # 网格间距 - 相邻网格点之间的物理距离
        self.dx = self.L / self.N
        
        # ============================================================
        # Time parameters / 时间参数
        # ============================================================
        
        # Time step - numerical integration step size
        # 时间步长 - 数值积分步长
        self.dt = float(dt)
        
        # Total simulation time - physical duration of the simulation
        # 总模拟时间 - 模拟的物理时长
        self.T = float(T)
        
        # Total number of time steps - calculated as ceil(T/dt)
        # 总时间步数 - 计算为 ceil(T/dt)
        self.steps = int(np.ceil(self.T / self.dt))
        
        # ============================================================
        # Simulation logs / 模拟日志
        # ============================================================
        
        # Free energy history - sampled at regular intervals
        # 自由能历史 - 定期采样
        self.energy_log = []
        
        # Mass conservation history - mean density over time
        # 质量守恒历史 - 随时间变化的平均密度
        self.mass_log = []
        
        # Defect density history - concentration of 5/7-fold defects
        # 缺陷密度历史 - 5/7重缺陷的浓度
        self.defect_log = []
        
        # Grain size history - estimated average grain diameter
        # 晶粒尺寸历史 - 估算的平均晶粒直径
        self.grain_size_log = []
        
        # Structure factor peak history - maximum diffraction intensity
        # 结构因子峰值历史 - 最大衍射强度
        self.structure_peak_log = []
        
        # ============================================================
        # Analysis cache / 分析缓存
        # ============================================================
        
        # Detected atom positions (pixel indices)
        # 检测到的原子位置（像素索引）
        self.atoms = None
        
        # Atom coordinates in physical space
        # 物理空间中的原子坐标
        self.points = None
        
        # Neighbor list for each atom
        # 每个原子的邻居列表
        self.neighbors = None
        
        # Psi6 orientational order parameter for each atom
        # 每个原子的Psi6取向序参量
        self.psi6 = None
        
        # ============================================================
        # Video recording configuration / 视频录制配置
        # ============================================================
        
        # Flag to enable/disable video recording
        # 启用/禁用视频录制的标志
        self.record_video = True
        
        # Counter for video frames
        # 视频帧计数器
        self.frame_count = 0
        
        # Directory to store frame images
        # 存储帧图像的目录
        self.frame_dir = "frames"
        
        # ============================================================
        # Build k-space / 构建k空间
        # ============================================================
        
        # Pre-compute Fourier space wave vectors
        # 预计算傅里叶空间波矢
        self._build_kspace()
        
    def _build_kspace(self):
        """
        Pre-generate 2D wave number grid (kx, ky) for spectral methods.
        预生成二维波数网格（kx, ky），用于谱方法计算。
        
        This method constructs the reciprocal space grid used for Fourier
        transform operations. The k2 array stores kx² + ky², which is
        used for Laplacian operations in Fourier space.
        该方法构建用于傅里叶变换操作的倒易空间网格。k2数组存储kx² + ky²，
        用于傅里叶空间中的拉普拉斯运算。
        
        The dispersion operator L(k) is NOT pre-computed here because it
        depends on the lattice type and is calculated dynamically in the
        step() method of each solver.
        色散算子L(k)不在这里预计算，因为它取决于晶格类型，在每个求解器的
        step()方法中动态计算。
        
        Key variables / 关键变量:
            kx (ndarray): x-component of wave vector / 波矢的x分量
            ky (ndarray): y-component of wave vector / 波矢的y分量
            k2 (ndarray): Squared wave number = kx² + ky² / 波数平方
            
        Notes / 说明:
            - Uses numpy.fft.fftfreq for correct FFT frequency ordering
              使用numpy.fft.fftfreq以获得正确的FFT频率排序
            - indexing="ij" for matrix-style indexing (row, column)
              使用indexing="ij"进行矩阵式索引（行、列）
            - k2 is shared by all lattice types (hexagon, square, triangle)
              k2被所有晶格类型共享（六角、正方、三角）
        """
        # ============================================================
        # 1D wave vector array / 一维波矢序列
        # ============================================================
        
        # Generate 1D wave vector array using FFT frequency convention
        # 使用FFT频率约定生成一维波矢数组
        # fftfreq returns frequencies in cycles per unit, multiply by 2π for angular frequency
        # fftfreq返回每单位的周期数，乘以2π得到角频率
        k = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        
        # ============================================================
        # 2D wave vector meshgrid / 二维波矢网格
        # ============================================================
        
        # Create 2D meshgrid of wave vectors
        # 创建波矢的二维网格
        # indexing="ij" uses matrix-style (row, column) indexing
        # indexing="ij"使用矩阵式（行、列）索引
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        
        # Save kx, ky as instance variables for use in square/triangle lattice operators
        # 将kx、ky保存为实例变量，供正方/三角晶格算子使用
        # These are needed for anisotropic dispersion relations
        # 这些对于各向异性色散关系是必需的
        self.kx = self.KX
        self.ky = self.KY
        
        # ============================================================
        # Squared wave number / 波数平方
        # ============================================================
        
        # Global unified: self.k2 only equals kx² + ky², shared by all lattice types
        # 全局统一：self.k2仅等于kx² + ky²，三种晶格共用
        # This is used for Laplacian (∇²) operations in Fourier space: ∇² → -k²
        # 这用于傅里叶空间中的拉普拉斯（∇²）运算：∇² → -k²
        self.k2 = self.kx ** 2 + self.ky ** 2
