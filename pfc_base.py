# 数值计算核心库
# Core numerical computation library
import numpy as np


class PFCBase:
    
    def __init__(
        self,
        N = 128,
        L = 64.0,
        dt = 0.05,
        T = 500.0
    ):
        # Grid parameters
        self.N = int(N)
        self.L = float(L)
        self.L0 = self.L
        self.dx = self.L / self.N
        
        # Time parameters
        self.dt = float(dt)
        self.T = float(T)
        self.steps = int(np.ceil(self.T / self.dt))
        
        # Logs for simulation
        self.energy_log = []
        self.mass_log = []
        self.defect_log = []
        self.grain_size_log = []
        self.structure_peak_log = []
        
        # Analysis Cache
        self.atoms = None
        self.points = None
        self.neighbors = None
        self.psi6 = None
        
        self.record_video = True
        self.frame_count = 0
        self.frame_dir = "frames"
        
        # k-space
        self._build_kspace()
        
    def _build_kspace(self):
        """
        预生成kx、ky二维波数网格，self.k2统一存储 kx²+ky²，色散算子L(k)移至step内实时计算
        Pre-generate 2D kx,ky wave grid, self.k2 only stores kx²+ky², dispersion L(k) calculated inside step()
        """
        # 一维波矢序列
        # 1D wave vector array
        k = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        # 保存kx、ky到实例变量，供正方/三角晶格算子使用
        # Save kx, ky to instance variable for square/triangle lattice operator
        self.kx = self.KX
        self.ky = self.KY
        # 全局统一：self.k2 仅等于 kx² + ky²，三种晶格共用
        # Global unified: self.k2 only equals kx² + ky², shared by all lattices
        self.k2 = self.kx ** 2 + self.ky ** 2