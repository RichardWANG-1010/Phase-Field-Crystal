import numpy as np
import scipy.fft as fft

from pfc_base import PFCBase
from pfc_analysis import PFCAnalysis
from pfc_plot import PFCPlot
from pfc_io import PFCIO
from pfc_elastic import PFCElastic


class PurePFCSolver(
    PFCBase,
    PFCAnalysis,
    PFCPlot,
    PFCIO,
    PFCElastic
):

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

        super().__init__(
            N=N,
            L=L,
            dt=dt,
            T=T
        )

        self.r = float(r)
        self.M = float(M)
        self.phi0 = float(phi0)
        self.noise_amp = float(noise_amp)
        self.lattice_type = lattice_type
        
        self.initialize_io()
        self._initialize_field()
        
    def _initialize_field(self):
        """
        生成带高斯噪声的初始密度场，修正均值保证初始质量守恒
        Generate initial density field with gaussian noise, adjust mean to satisfy initial mass conservation
        """
        # 生成标准正态分布噪声场
        # Generate standard normal distributed noise field
        noise = (self.noise_amp * np.random.randn(self.N, self.N))
        # 基础密度叠加噪声
        # Superpose noise onto base average density
        self.phi = self.phi0 + noise
        # 计算当前全场均值
        # Calculate current global field average
        current_mean = np.mean(self.phi)
        
        # 偏移全场数值，强制均值等于phi0
        # Offset whole field value to force average equal to target phi0
        self.phi -= (current_mean - self.phi0) 
        
        # 打印初始化完成后的实际平均密度
        # Print real average density after initialization
        print( 
            f"Initial mean density = "
            f"{np.mean(self.phi):.6f}" 
        )
        
    def step(self):
        """
        单步傅里叶半隐式欧拉迭代更新密度场φ，按晶格类型实时计算对应标准色散算子L(k)
        Single step Fourier semi-implicit Euler update for density field φ, calculate L(k) by lattice type in real-time
        """
        # 缓存当前密度场副本
        # Cache copy of current density field
        phi = self.phi
        # 密度场二维傅里叶变换到频域
        # 2D Fourier transform of density field to k-space
        phi_hat = fft.fft2(phi)
        # 非线性立方项 φ³
        # Nonlinear cubic term φ³
        nonlinear = phi**3
        # 立方项变换到频域
        # Fourier transform of cubic nonlinear term
        nonlinear_hat = fft.fft2(nonlinear)

        # ===================== 核心修复：分晶格实时计算标准L(k)色散算子 =====================
        # Core fix: calculate standard dispersion L(k) separately for each lattice
        if self.lattice_type == "hexagon":
            # 六角晶格标准PFC色散 L(k) = (1 - k²)² + r
            # Hexagon standard PFC dispersion L(k) = (1 - k²)² + r
            l_k = (1.0 - self.k2) ** 2 + self.r
        elif self.lattice_type == "square":
            # 正方晶格色散 L(k) = (1-kx²)²(1-ky²)² + r
            # Square lattice dispersion L(k) = (1-kx²)²(1-ky²)² + r
            term_x = (1.0 - self.kx ** 2) ** 2
            term_y = (1.0 - self.ky ** 2) ** 2
            l_k = term_x * term_y + self.r
        elif self.lattice_type == "triangle":
            # 三角晶格色散 L(k) = (1 - kx² - kx*ky + ky²)² + r
            # Triangle lattice dispersion L(k) = (1 - kx² - kx*ky + ky²)² + r
            tri_base = 1.0 - self.kx**2 - self.kx * self.ky + self.ky**2
            l_k = tri_base ** 2 + self.r

        # 标准半隐式迭代公式（与最初单晶格原版数学完全对齐）
        # Standard semi-implicit iteration formula, math identical to original single-lattice code
        numerator = phi_hat - self.dt * self.M * self.k2 * nonlinear_hat
        denominator = 1.0 + self.dt * self.M * self.k2 * l_k
        # 频域更新密度场
        # Update density field in k-space
        phi_hat_new = numerator / denominator
        # 逆傅里叶变换回实空间，丢弃微小虚部
        # Inverse Fourier transform back to real space, discard tiny imaginary part
        self.phi = np.real(fft.ifft2(phi_hat_new))
        # 迭代后再次修正均值，维持全局质量守恒
        # Re-adjust average after iteration to maintain global mass conservation
        self.phi -= (np.mean(self.phi)- self.phi0)
        
    def compute_energy(self):
        """
        计算PFC全场自由能，包含线性梯度项与四阶非线性项
        Calculate global PFC free energy, contains linear gradient term and 4th-order nonlinear term
        """
        # 缓存当前密度场
        # Cache current density field
        phi = self.phi
        # 密度场傅里叶变换
        # Fourier transform of density field
        phi_hat = fft.fft2(phi)

        # 频域积分计算线性自由能部分
        # Integrate k-space to calculate linear free energy component
        linear_part = np.sum(np.real(np.conj(phi_hat) * ((1.0 - self.k2)**2 + self.r) * phi_hat)) / (self.N**2)
        # 实空间平均计算四阶非线性自由能部分
        # Real-space average for 4th-order nonlinear free energy component
        nonlinear_part = np.mean(phi**4)
        # 标准PFC自由能公式 F = 0.5*线性项 + 0.25*四阶项
        # Standard PFC free energy formula F = 0.5*linear + 0.25*quartic
        energy = (0.5 * linear_part + 0.25 * nonlinear_part)

        return energy
    
    def structure_factor(self):
        """
        计算静态结构因子S(k)，表征晶体衍射图案
        Calculate static structure factor S(k), characterize crystal diffraction pattern
        """
        # 减去全场均值，得到密度涨落场
        # Subtract global average to get density fluctuation field
        phi_fluct = (self.phi - np.mean(self.phi))
        # 傅里叶变换+fftshift将零波矢移到图像中心
        # Fourier transform + fftshift to shift zero wave vector to image center
        phi_hat = np.fft.fftshift(np.fft.fft2(phi_fluct))
        # 结构因子为傅里叶模的平方 |φ̃(k)|²
        # Structure factor equals squared modulus of Fourier amplitude |φ̃(k)|²
        S = np.abs(phi_hat)**2

        return S
    
    def print_status(self, step, E):
    
        print(
            f"step={step:6d} "
            f"E={E:.6e} "
            f"mean={np.mean(self.phi):.3e} "
            f"std={np.std(self.phi):.3e} "
            f"min={np.min(self.phi):.3e} "
            f"max={np.max(self.phi):.3e}"
        )
        
    def run(self):
    
        for step in range(self.steps):

            self.step()

            if step % 10 == 0:

                E = self.sample_observables(
                    step
                )

                self.print_status(
                    step,
                    E
                )

                self.capture_frame()

        self.frames_to_video()