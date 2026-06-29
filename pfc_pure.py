"""
pfc_pure.py - Pure Material Phase Field Crystal Solver
纯材料相场晶体求解器

This module implements the standard Phase Field Crystal (PFC) model for pure
materials, supporting hexagonal, square, and triangular lattices.
本模块实现标准的纯材料相场晶体（PFC）模型，支持六角、正方和三角晶格。

The PFC model free energy functional:
PFC模型自由能泛函：
    F = ∫ [φ/2 · (r + (1 + ∇²)²) φ + φ⁴/4] dr

The dynamic equation (conserved dynamics):
动力学方程（守恒动力学）：
    ∂φ/∂t = ∇² · δF/δφ

Solved using semi-implicit Fourier spectral method for numerical stability.
使用半隐式傅里叶谱方法求解，保证数值稳定性。

Author: Jinpeng Wang
Department of Material Engineering
"""

# NumPy - numerical computing
# NumPy - 数值计算
import numpy as np

# SciPy FFT - fast Fourier transform
# SciPy FFT - 快速傅里叶变换
import scipy.fft as fft

# Import base classes using multiple inheritance (mixin pattern)
# 使用多继承导入基类（mixin模式）
from pfc_base import PFCBase          # Core numerical infrastructure / 数值计算核心
from pfc_analysis import PFCAnalysis  # Microstructure analysis / 微观结构分析
from pfc_plot import PFCPlot          # Visualization tools / 可视化工具
from pfc_io import PFCIO              # Video recording / 视频录制
from pfc_elastic import PFCElastic    # Elasticity calculations / 弹性计算
from pfc_advanced import PFCAdvancedAnalysis  # Advanced features / 高级功能


class PurePFCSolver(
    PFCBase,
    PFCAnalysis,
    PFCPlot,
    PFCIO,
    PFCElastic,
    PFCAdvancedAnalysis
):
    """
    Pure material Phase Field Crystal solver.
    纯材料相场晶体求解器。
    
    This class implements the standard PFC model for single-component systems,
    using a semi-implicit Fourier spectral method for numerical integration.
    本类实现单组分系统的标准PFC模型，使用半隐式傅里叶谱方法进行数值积分。
    
    Multiple inheritance is used to compose functionality from mixin classes:
    使用多继承从mixin类组合功能：
    - PFCBase: Grid and k-space setup / 网格和k空间设置
    - PFCAnalysis: Defect and microstructure analysis / 缺陷和微观结构分析
    - PFCPlot: Visualization methods / 可视化方法
    - PFCIO: Video recording and output / 视频录制和输出
    - PFCElastic: Elastic property calculations / 弹性性质计算
    - PFCAdvancedAnalysis: Mode approximation, elastic theory, vacancy / 高级分析：模式近似、弹性理论、空位扩散
    
    Supported lattice types / 支持的晶格类型:
        - "hexagon": Hexagonal (triangular) lattice / 六角（三角）晶格
        - "square": Square lattice / 正方晶格
        - "triangle": Triangular (oblique) lattice / 三角（斜）晶格
    
    Attributes / 属性:
        r (float): Reduced temperature parameter / 约化温度参数
        M (float): Mobility coefficient / 迁移率系数
        phi0 (float): Average density / 平均密度
        noise_amp (float): Initial noise amplitude / 初始噪声幅值
        lattice_type (str): Type of crystal lattice / 晶体晶格类型
        phi (ndarray): Density field / 密度场
    """
    
    def __init__(
        self,
        N=128,
        L=64.0,
        r=-0.25,
        M=1.0,
        dt=0.05,
        T=500.0,
        phi0=-0.25,
        noise_amp=0.01,
        lattice_type="hexagon"
    ):
        """
        Initialize the pure PFC solver.
        初始化纯材料PFC求解器。
        
        Args / 参数:
            N (int, optional): Grid resolution. Defaults to 128.
                              网格分辨率，默认128。
            L (float, optional): Domain size. Defaults to 64.0.
                                计算域尺寸，默认64.0。
            r (float, optional): PFC control parameter (reduced temperature).
                                PFC控制参数（约化温度），默认-0.25。
            M (float, optional): Mobility coefficient. Defaults to 1.0.
                                迁移率系数，默认1.0。
            dt (float, optional): Time step. Defaults to 0.05.
                                 时间步长，默认0.05。
            T (float, optional): Total simulation time. Defaults to 500.0.
                                总模拟时间，默认500.0。
            phi0 (float, optional): Average density. Defaults to -0.25.
                                   平均密度，默认-0.25。
            noise_amp (float, optional): Initial noise amplitude. Defaults to 0.01.
                                        初始噪声幅值，默认0.01。
            lattice_type (str, optional): Lattice type ("hexagon", "square", "triangle").
                                         晶格类型，默认"hexagon"。
        """
        # Initialize base class with grid and time parameters
        # 使用网格和时间参数初始化基类
        super().__init__(
            N=N,
            L=L,
            dt=dt,
            T=T
        )
        
        # ============================================================
        # PFC model parameters / PFC模型参数
        # ============================================================
        
        # Reduced temperature parameter - controls phase behavior
        # Negative values favor crystalline phase
        # 约化温度参数 - 控制相行为
        # 负值有利于晶相形成
        self.r = float(r)
        
        # Mobility coefficient - controls diffusion speed
        # Higher M = faster evolution
        # 迁移率系数 - 控制扩散速度
        # M越大 = 演化越快
        self.M = float(M)
        
        # Average density - mean value of the density field
        # Determines which phase is thermodynamically favored
        # 平均密度 - 密度场的平均值
        # 决定哪个相在热力学上更有利
        self.phi0 = float(phi0)
        
        # Initial noise amplitude - magnitude of random perturbation
        # Triggers spontaneous crystallization
        # 初始噪声幅值 - 随机扰动的大小
        # 触发自发结晶
        self.noise_amp = float(noise_amp)
        
        # Lattice type - determines dispersion operator form
        # 晶格类型 - 决定色散算子形式
        self.lattice_type = lattice_type
        
        # ============================================================
        # Initialize IO and field / 初始化IO和场
        # ============================================================
        
        # Initialize video recording infrastructure
        # 初始化视频录制基础设施
        self.initialize_io()
        
        # Initialize density field with noise
        # 用噪声初始化密度场
        self._initialize_field()
        
    def _initialize_field(self):
        """
        Generate initial density field with Gaussian noise, adjust mean to
        satisfy initial mass conservation.
        生成带高斯噪声的初始密度场，修正均值以保证初始质量守恒。
        
        The initial condition is a uniform mean density plus small random
        perturbations. The mean is adjusted to exactly match phi0, ensuring
        mass conservation from t=0.
        初始条件是均匀平均密度加上小的随机扰动。调整均值以精确匹配phi0，
        确保从t=0开始质量守恒。
        
        Notes / 说明:
            - Uses np.random.randn for standard normal distribution
              使用np.random.randn生成标准正态分布
            - Mean correction ensures exact mass conservation
              均值修正确保精确的质量守恒
            - Small noise triggers spontaneous symmetry breaking
              小噪声触发自发对称性破缺
        """
        # ============================================================
        # Generate noise field / 生成噪声场
        # ============================================================
        
        # Generate standard normal distributed noise field
        # 生成标准正态分布噪声场
        # randn returns samples from standard normal distribution (mean=0, std=1)
        # randn返回标准正态分布的样本（均值=0，标准差=1）
        noise = self.noise_amp * np.random.randn(self.N, self.N)
        
        # ============================================================
        # Initialize density field / 初始化密度场
        # ============================================================
        
        # Superpose noise onto base average density
        # 将噪声叠加到基础平均密度上
        self.phi = self.phi0 + noise
        
        # ============================================================
        # Mass conservation correction / 质量守恒修正
        # ============================================================
        
        # Calculate current global field average
        # 计算当前全场均值
        current_mean = np.mean(self.phi)
        
        # Offset whole field value to force average equal to target phi0
        # 偏移全场数值，强制均值等于目标phi0
        # This ensures exact mass conservation at t=0
        # 这确保了t=0时精确的质量守恒
        self.phi -= (current_mean - self.phi0)
        
        # ============================================================
        # Print initialization status / 打印初始化状态
        # ============================================================
        
        # Print real average density after initialization for verification
        # 打印初始化后的实际平均密度用于验证
        print(f"Initial mean density = {np.mean(self.phi):.6f}")
        
    def step(self):
        """
        Single step Fourier semi-implicit Euler update for density field φ.
        单步傅里叶半隐式欧拉迭代更新密度场φ。
        
        The dispersion operator L(k) is calculated in real-time based on the
        selected lattice type.
        根据所选晶格类型实时计算色散算子L(k)。
        
        Semi-implicit scheme / 半隐式格式:
            φ̂^{n+1} = [φ̂^n - dt · M · k² · (φ³)̂] / [1 + dt · M · k² · L(k)]
        
        Where L(k) is the dispersion operator, different for each lattice:
        其中L(k)是色散算子，每种晶格不同：
            - Hexagon: L(k) = (1 - k²)² + r
            - Square:  L(k) = (1-kx²)²(1-ky²)² + r
            - Triangle: L(k) = (1 - kx² - kx·ky + ky²)² + r
        
        Notes / 说明:
            - Linear terms are treated implicitly for stability
              线性项隐式处理以保证稳定性
            - Nonlinear term φ³ is treated explicitly
              非线性项φ³显式处理
            - Mass conservation is enforced after each step
              每步后强制执行质量守恒
        """
        # ============================================================
        # Cache current density field / 缓存当前密度场
        # ============================================================
        
        # Cache copy of current density field for this time step
        # 缓存当前时间步的密度场副本
        phi = self.phi
        
        # ============================================================
        # Fourier transform to k-space / 傅里叶变换到k空间
        # ============================================================
        
        # 2D Fourier transform of density field to k-space
        # 密度场的二维傅里叶变换到频域
        phi_hat = fft.fft2(phi)
        
        # ============================================================
        # Nonlinear term calculation / 非线性项计算
        # ============================================================
        
        # Nonlinear cubic term φ³ (calculated in real space)
        # 非线性立方项φ³（在实空间计算）
        nonlinear = phi ** 3
        
        # Fourier transform of cubic nonlinear term to k-space
        # 立方非线性项的傅里叶变换到k空间
        nonlinear_hat = fft.fft2(nonlinear)
        
        # ============================================================
        # Dispersion operator L(k) by lattice type
        # 按晶格类型计算色散算子L(k)
        # ============================================================
        
        if self.lattice_type == "hexagon":
            # Hexagonal lattice standard PFC dispersion
            # 六角晶格标准PFC色散
            # L(k) = (1 - k²)² + r
            # This is the standard PFC dispersion relation
            # 这是标准的PFC色散关系
            l_k = (1.0 - self.k2) ** 2 + self.r
            
        elif self.lattice_type == "square":
            # Square lattice dispersion
            # 正方晶格色散
            # L(k) = (1-kx²)²(1-ky²)² + r
            # Anisotropic dispersion for square lattice symmetry
            # 正方晶格对称性的各向异性色散
            term_x = (1.0 - self.kx ** 2) ** 2
            term_y = (1.0 - self.ky ** 2) ** 2
            l_k = term_x * term_y + self.r
            
        elif self.lattice_type == "triangle":
            # Triangular lattice dispersion
            # 三角晶格色散
            # L(k) = (1 - kx² - kx·ky + ky²)² + r
            # Oblique lattice with mixed x-y coupling
            # 带有x-y耦合的斜晶格
            tri_base = 1.0 - self.kx ** 2 - self.kx * self.ky + self.ky ** 2
            l_k = tri_base ** 2 + self.r
        
        # ============================================================
        # Semi-implicit update / 半隐式更新
        # ============================================================
        
        # Standard semi-implicit iteration formula
        # 标准半隐式迭代公式
        # Numerator: explicit nonlinear part + old field
        # 分子：显式非线性部分 + 旧场
        numerator = phi_hat - self.dt * self.M * self.k2 * nonlinear_hat
        
        # Denominator: implicit linear part for stability
        # 分母：隐式线性部分以保证稳定性
        denominator = 1.0 + self.dt * self.M * self.k2 * l_k
        
        # Update density field in k-space
        # 在k空间更新密度场
        phi_hat_new = numerator / denominator
        
        # ============================================================
        # Inverse Fourier transform / 逆傅里叶变换
        # ============================================================
        
        # Inverse Fourier transform back to real space
        # Discard tiny imaginary part from numerical errors
        # 逆傅里叶变换回实空间
        # 丢弃数值误差导致的微小虚部
        self.phi = np.real(fft.ifft2(phi_hat_new))
        
        # ============================================================
        # Mass conservation correction / 质量守恒修正
        # ============================================================
        
        # Re-adjust average after iteration to maintain global mass conservation
        # 迭代后重新调整均值以维持全局质量守恒
        # Small numerical errors can accumulate, so we enforce exact conservation
        # 小的数值误差可能累积，因此我们强制执行精确守恒
        self.phi -= (np.mean(self.phi) - self.phi0)
        
    def compute_energy(self):
        """
        Calculate global PFC free energy, contains linear gradient term
        and 4th-order nonlinear term.
        计算PFC全场自由能，包含线性梯度项和四阶非线性项。
        
        Free energy formula / 自由能公式:
            F = 0.5 · ∫ φ · (r + (1 + ∇²)²) φ dr + 0.25 · ∫ φ⁴ dr
        
        The linear term is computed in Fourier space for efficiency:
        线性项在傅里叶空间计算以提高效率：
            ∫ φ · (1 + ∇²)² φ dr = Σ |φ̂|² · (1 - k²)² / N²
        
        Returns / 返回值:
            float: Free energy density / 自由能密度
        """
        # ============================================================
        # Cache current density field / 缓存当前密度场
        # ============================================================
        
        # Cache current density field
        # 缓存当前密度场
        phi = self.phi
        
        # ============================================================
        # Fourier transform / 傅里叶变换
        # ============================================================
        
        # Fourier transform of density field
        # 密度场的傅里叶变换
        phi_hat = fft.fft2(phi)
        
        # ============================================================
        # Linear free energy part / 线性自由能部分
        # ============================================================
        
        # Integrate k-space to calculate linear free energy component
        # 在k空间积分计算线性自由能分量
        # Using Parseval's theorem: ∫ |f|² dr = (1/N²) · Σ |f̂|²
        # 使用帕塞瓦尔定理
        # The (1 - k²)² + r term comes from the (1 + ∇²)² + r operator
        # (1 - k²)² + r项来自(1 + ∇²)² + r算子
        # Note: ∇² in Fourier space is -k², so (1 + ∇²) → (1 - k²)
        # 注意：傅里叶空间中的∇²是-k²，所以(1 + ∇²) → (1 - k²)
        linear_part = np.sum(np.real(np.conj(phi_hat) * ((1.0 - self.k2) ** 2 + self.r) * phi_hat)) / (self.N ** 2)
        
        # ============================================================
        # Nonlinear free energy part / 非线性自由能部分
        # ============================================================
        
        # Real-space average for 4th-order nonlinear free energy component
        # 实空间平均计算四阶非线性自由能分量
        # F_nonlinear = (1/4) · <φ⁴>
        nonlinear_part = np.mean(phi ** 4)
        
        # ============================================================
        # Total free energy / 总自由能
        # ============================================================
        
        # Standard PFC free energy formula F = 0.5*linear + 0.25*quartic
        # 标准PFC自由能公式 F = 0.5*线性项 + 0.25*四阶项
        energy = 0.5 * linear_part + 0.25 * nonlinear_part
        
        return energy
    
    def structure_factor(self):
        """
        Calculate static structure factor S(k), characterizes crystal
        diffraction pattern.
        计算静态结构因子S(k)，表征晶体衍射图案。
        
        The structure factor is the squared modulus of the Fourier transform
        of the density fluctuations:
        结构因子是密度涨落傅里叶变换的模平方：
            S(k) = |φ̃(k)|²
        
        Returns / 返回值:
            ndarray: Static structure factor S(k) / 静态结构因子
        """
        # ============================================================
        # Density fluctuation field / 密度涨落场
        # ============================================================
        
        # Subtract global average to get density fluctuation field
        # 减去全场均值得到密度涨落场
        # This removes the DC component (k=0 peak)
        # 这移除了直流分量（k=0峰）
        phi_fluct = self.phi - np.mean(self.phi)
        
        # ============================================================
        # Fourier transform with shift / 带平移的傅里叶变换
        # ============================================================
        
        # Fourier transform + fftshift to shift zero wave vector to image center
        # 傅里叶变换 + fftshift将零波矢移到图像中心
        # fftshift makes the plot more intuitive (low k in center)
        # fftshift使图像更直观（低k在中心）
        phi_hat = np.fft.fftshift(np.fft.fft2(phi_fluct))
        
        # ============================================================
        # Structure factor calculation / 结构因子计算
        # ============================================================
        
        # Structure factor equals squared modulus of Fourier amplitude
        # 结构因子等于傅里叶振幅的模平方
        # S(k) = |φ̃(k)|²
        S = np.abs(phi_hat) ** 2
        
        return S
    
    def print_status(self, step, E):
        """
        Print current simulation status to console.
        向控制台打印当前模拟状态。
        
        Args / 参数:
            step (int): Current step number / 当前步数
            E (float): Current free energy / 当前自由能
        """
        print(
            f"step={step:6d} "
            f"E={E:.6e} "
            f"mean={np.mean(self.phi):.3e} "
            f"std={np.std(self.phi):.3e} "
            f"min={np.min(self.phi):.3e} "
            f"max={np.max(self.phi):.3e}"
        )
        
    def run(self):
        """
        Main simulation loop.
        主模拟循环。
        
        Runs the simulation for the specified number of steps, sampling
        observables and capturing frames at regular intervals.
        运行指定步数的模拟，定期采样观测量并捕获帧。
        """
        # Iterate over all time steps
        # 遍历所有时间步
        for step in range(self.steps):
            # Perform one time step
            # 执行一个时间步
            self.step()
            
            # Sample and output every 10 steps
            # 每10步采样和输出一次
            if step % 10 == 0:
                # Sample observables (energy, mass, etc.)
                # 采样观测量（能量、质量等）
                E = self.sample_observables(step)
                
                # Print status to console
                # 向控制台打印状态
                self.print_status(step, E)
                
                # Capture video frame
                # 捕获视频帧
                self.capture_frame()
        
        # Convert frames to video after simulation completes
        # 模拟完成后将帧转换为视频
        self.frames_to_video()
