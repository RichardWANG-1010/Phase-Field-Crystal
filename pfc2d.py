# 数值计算核心库
# Core numerical computation library
import numpy as np
# 傅里叶变换模块
# Fourier transform module
import scipy.fft as fft

# ===================== 科学拓扑分析工具 =====================
# Scientific topology analysis tools
from scipy.spatial import (
    # 沃罗诺伊元胞划分
    Voronoi,
    # 沃罗诺伊绘图工具
    voronoi_plot_2d,
    # 周期边界KD近邻搜索树
    cKDTree
)
# 散乱点插值工具，用于ψ6有序场全域插值
# Scattered point interpolation for full-domain ψ6 order field
from scipy.interpolate import griddata
# 新增交互输入工具
import sys
import os

# ===================== 可视化绘图库 =====================
# Visualization plotting library
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# 子进程调用ffmpeg合成视频
import subprocess
# 临时存储帧图片的路径管理
import tempfile
import shutil
# 图像局部峰值检测，识别晶体原子位置
# Local peak detection to locate crystal atom positions
from skimage.feature import peak_local_max


class PFCSolver:
    """
    2D傅里叶半隐式相场晶体(PFC)求解器，守恒Model-B动力学
    2D Fourier semi-implicit Phase Field Crystal(PFC) solver, conserved Model-B dynamics
    支持原子识别、配位数缺陷统计、ψ6键取向序、Voronoi晶粒拓扑分析
    Support atom detection, coordination defect statistics, ψ6 bond-orientational order, Voronoi grain topology analysis
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
        # 新增晶格参数：hexagon六角 / square正方 / triangle三角
        lattice_type="hexagon"
    ):
        # 网格一维离散点数
        # 1D grid discretization count
        self.N = int(N)
        # 计算域边长
        # Simulation domain side length
        self.L = float(L)
        # 晶格类型存储
        self.lattice_type = lattice_type.strip().lower()
        # 合法晶格校验
        valid_lattice = ["hexagon", "square", "triangle"]
        if self.lattice_type not in valid_lattice:
            raise ValueError(f"晶格仅支持{valid_lattice}，输入错误：{lattice_type}")

        # PFC模型线性控制参数r
        self.r = float(r)
        # 动力学迁移系数M
        self.M = float(M)
        # 仿真时间步长
        self.dt = float(dt)
        # 总仿真物理时长
        self.T = float(T)
        # 全场平均密度目标均值
        self.phi0 = float(phi0)
        # 总迭代步数
        self.steps = int(np.ceil(T / dt))
        # 空间步长dx
        self.dx = self.L / self.N
        # 初始噪声幅值
        self.noise_amp = noise_amp

        # 时序日志容器
        self.energy_log = []
        self.mass_log = []
        self.defect_log = []
        self.grain_size_log = []
        self.structure_peak_log = []

        # 预计算波数网格（适配多晶格）
        self._build_kspace()
        # 初始化密度场
        self._initialize_field()
        
        # 视频录制配置（原有不变）
        self.record_video = True
        self.video_figsize = (6, 6)
        self.video_fps = 10
        self.video_output_name = f"{self.lattice_type}_pfc_simulation.mp4"
        self.frame_cache = []
        
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
    
    def detect_atoms(self):
        """
        局部峰值检测，提取所有晶体原子像素坐标(i,j)
        Local peak detection, extract pixel coordinate (i,j) of all crystal atoms
        """
        atoms = peak_local_max(
            self.phi,
            # 原子间最小像素间距，避免重复识别同一原子
            # Minimum pixel distance between atoms, avoid duplicate detection
            min_distance=7,
            # 峰值相对阈值，过滤微弱噪声伪峰
            # Relative peak threshold, filter weak noise false peaks
            threshold_rel=0.5,
            # 允许识别靠近计算域边界的原子
            # Allow detection of atoms near simulation boundary
            exclude_border=False
        )
        # 返回数组每行格式 [纵向像素i, 横向像素j]
        # Return array each row format: [vertical pixel i, horizontal pixel j]
        return atoms
    
    def voronoi_analysis(self):
        """
        基于原子坐标生成周期域Voronoi元胞，用于晶粒拓扑划分
        Generate Voronoi tessellation from atom coordinates for grain topology partition
        """
        # 获取所有原子像素索引
        # Get all atom pixel indices
        atoms = self.detect_atoms()
        # 坐标翻转：[i,j] → [x,y]，匹配绘图横轴纵轴
        # Coordinate flip: [i,j] → [x,y], match plot horizontal & vertical axis
        points = atoms[:, ::-1]
        # 边界过滤余量
        # Boundary filter margin
        margin = 0
        # 过滤紧贴边界的原子，避免Voronoi元胞畸形
        # Filter atoms attached to boundary to avoid distorted Voronoi cells
        mask = (
            (points[:,0] > margin)
            &
            (points[:,0] < self.N - margin)
            &
            (points[:,1] > margin)
            &
            (points[:,1] < self.N - margin)
        )
        points = points[mask]
        # 构建Voronoi拓扑对象
        # Build Voronoi topology object
        vor = Voronoi(points)
        # 返回拓扑对象与过滤后原子xy坐标
        # Return topology object & filtered atom xy coordinates
        return vor, points
    
    def build_neighbors(self):
        """
        构建周期边界KD树，计算每个原子的近邻原子列表
        Build periodic boundary KD-Tree, calculate neighbor list for every atom
        """
        # 获取原子像素索引
        # Get atom pixel indices
        atoms = self.detect_atoms()
        # 转换为绘图匹配的xy坐标
        # Convert to plot-aligned xy coordinates
        points = atoms[:, ::-1]
        # 带周期盒子尺寸的KD近邻树
        # KD-Tree with periodic box boundary size
        tree = cKDTree(points, boxsize=self.N)
        # 预查询每个原子前6近邻，用于估算晶格常数a0
        # Pre-query top 6 nearest neighbors for each atom to estimate lattice constant a0
        dists, _ = tree.query(points, k=6)
        # 第二近邻距离中位数作为晶格常数a0
        # Median distance of second nearest neighbor as lattice constant a0
        a0 = np.median(dists[:,1])
        # 近邻搜索截断半径：1.35倍晶格常数
        # Neighbor search cutoff radius: 1.35 times lattice constant
        r_cut = 1.35 * a0
        neighbors = []
        # 遍历全部原子，查询半径内所有近邻
        # Iterate all atoms, query all neighbors within cutoff radius
        for p in points:
            neigh = tree.query_ball_point(p, r_cut)
            neighbors.append(neigh)
        # 返回原子xy坐标、对应近邻索引列表
        # Return atom xy coordinates & matched neighbor index list
        return points, neighbors
    
    def coordination_numbers(self):
        """
        计算每个原子配位数（近邻原子数量，排除自身）
        Calculate coordination number of each atom (neighbor count, exclude self)
        """
        # 获取原子坐标与近邻列表
        # Get atom coordinates and neighbor list
        points, neighbors = self.build_neighbors()
        # 每个原子近邻列表长度-1 = 配位数
        # Neighbor list length minus one equals coordination number
        coord = np.array([len(n)-1 for n in neighbors])
        # 返回配位数数组、原子xy坐标
        # Return coordination number array & atom xy coordinates
        return coord, points
    
    def compute_psi6(self):
        """
        计算ψ6复键取向序参量，表征六边形晶体局部有序度
        Compute complex ψ6 bond-orientational order parameter, characterize local hexagonal crystal order
        """
        # 读取原子坐标与近邻列表
        # Load atom coordinates and neighbor list
        points, neighbors = self.build_neighbors()
        # 初始化复数ψ6数组
        # Initialize complex ψ6 array
        psi6 = np.zeros(len(points), dtype=complex)
        # 遍历每一个原子
        # Iterate every single atom
        for i,p in enumerate(points):
            # 筛选排除自身的近邻索引
            # Filter neighbor indices, exclude self index
            neigh = [j for j in neighbors[i] if j != i]
            # 无近邻则跳过，保持ψ6=0
            # Skip atom with zero neighbors, keep ψ6=0
            if len(neigh)==0:
                continue
            angles = []
            # 遍历所有有效近邻，计算键向量夹角
            # Iterate all valid neighbors, calculate bond vector angle
            for j in neigh:
                dx = points[j,0]-p[0]
                dy = points[j,1]-p[1]
                # 周期边界向量修正，处理跨盒边界原子
                # Periodic vector correction for atoms crossing simulation box
                dx -= self.N*np.round(dx/self.N)
                dy -= self.N*np.round(dy/self.N)
                # 计算键向量与x轴夹角
                # Calculate angle between bond vector and x-axis
                angles.append(np.arctan2(dy,dx))
            # 六重对称相位平均得到ψ6
            # Average 6-fold symmetric phase to get ψ6 value
            psi6[i] = np.mean(np.exp(6j*np.array(angles)))
        # 返回原子坐标、对应ψ6复数值
        # Return atom coordinates & matched complex ψ6 values
        return points, psi6
    
    def global_psi6(self):
        """
        求解局部平均|ψ6|、全局平均复ψ6模长，两个有序度评价指标
        Calculate two order metrics: local average |ψ6|, modulus of global complex averaged ψ6
        """
        # 读取全部原子ψ6值
        # Load ψ6 value of all atoms
        _, psi6 = self.compute_psi6()
        # 局部有序度：所有原子|ψ6|算术平均
        # Local order: arithmetic average of |ψ6| over all atoms
        psi6_local = np.mean(np.abs(psi6))
        # 全局有序度：全部原子复ψ6求和后取模
        # Global order: modulus of summed complex ψ6 of all atoms
        psi6_global = np.abs(np.mean(psi6))
        return (psi6_local, psi6_global)
        
    def grain_boundary_parameter(self):
        """
        晶界判别参数D：相邻原子ψ6差值平方均值，D越大越靠近晶界
        Grain boundary discriminant D: mean squared ψ6 difference between adjacent atoms, larger D means grain boundary
        """
        # 获取原子坐标、近邻列表
        # Get atom coordinates and neighbor list
        points, neighbors = self.build_neighbors()
        # 获取每个原子ψ6序参量
        # Get ψ6 order parameter of each atom
        _, psi6 = self.compute_psi6()
        D = np.zeros(len(points))
        # 遍历全部原子
        # Iterate all atoms
        for i in range(len(points)):
            # 剔除自身索引
            # Remove self index from neighbor list
            neigh = neighbors[i]
            neigh = [j for j in neigh if j != i]
            # 无近邻跳过
            # Skip atom without neighbors
            if len(neigh) == 0:
                continue
            # 计算与所有近邻|ψ6|差值平方的平均值
            # Calculate average squared difference of |ψ6| with all neighbors
            D[i] = np.mean(np.abs(psi6[i] - psi6[neigh]) ** 2)
        # 返回原子坐标、对应晶界参数D
        # Return atom coordinates & matched grain boundary parameter D
        return points, D
    
    def analyze_defects(self):
        """
        缺陷统计：5/7配位原子缺陷密度、估算平均晶粒尺寸
        Defect statistics: concentration of 5/7-fold coordination atoms, estimated average grain size
        """
        # 读取配位数与原子坐标
        # Load coordination numbers and atom coordinates
        coord, points = self.coordination_numbers()
        # 无原子直接返回空值
        # Return empty value if zero atoms detected
        if len(coord) == 0:
            return np.nan, np.nan, np.array([]), np.empty((0, 2))
        # 过滤有效数值
        # Filter valid numeric entries
        valid = ~np.isnan(coord)
        coord = coord[valid]
        points = points[valid]
        # 总识别原子数量
        # Total number of detected atoms
        total_atoms = len(coord)
        # 非6配位原子判定为缺陷
        # Atoms with coordination not equal to 6 are marked as defects
        defect_mask = (coord != 6)
        # 缺陷原子总数
        # Total count of defect atoms
        defect_number = np.sum(defect_mask)
        # 缺陷密度 = 缺陷数 / 总原子数
        # Defect density = defect count / total atom count
        defect_density = defect_number / total_atoms
        # 晶粒尺寸近似公式：总面积 / 缺陷数开根号
        # Approximate grain size formula: sqrt(total area / defect count)
        grain_size = np.sqrt(self.L * self.L / max(defect_number, 1))
        # 返回缺陷密度、晶粒尺寸、配位数数组、原子坐标
        # Return defect density, grain size, coordination array, atom coordinates
        return (defect_density, grain_size, coord, points)
    
    def run(self):
        """
        仿真主循环，固定间隔采样能量、密度、缺陷、结构因子数据
        Main simulation loop, sample energy, density, defect, structure factor data at fixed interval
        """
        # 遍历全部迭代步
        # Iterate all simulation steps
        for step in range(self.steps):
            # 执行单步密度场更新
            # Execute single step density field update
            self.step()
            # 每10步采样一次数据
            # Sample data once every 10 simulation steps
            if step % 10 == 0:
                # 计算当前步自由能
                # Calculate free energy of current step
                E = self.compute_energy()
                # 存入自由能日志
                # Append free energy to log list
                self.energy_log.append(E)
                # 记录当前全场平均密度
                # Record current global average density
                self.mass_log.append(np.mean(self.phi))
                # 仅仿真步数超过1500步后才统计缺陷数据
                # Only start defect statistics after simulation step exceeds 1500
                if step > 1500:
                    # 计算缺陷密度、晶粒尺寸
                    # Calculate defect density and grain size
                    defect_density, grain_size, _, _ = self.analyze_defects()
                    # 无NaN数值则存入日志
                    # Append to log only if value is not NaN
                    if not np.isnan(defect_density):
                        self.defect_log.append(defect_density)
                        self.grain_size_log.append(grain_size)
                # 计算当前结构因子
                # Calculate current structure factor
                S = self.structure_factor()
                # 记录结构因子全局峰值
                # Record global maximum peak of structure factor
                self.structure_peak_log.append(np.max(S))
                # 控制台打印当前步关键数值
                # Print key numerical data of current step to console
                print(
                    f"step={step:6d} "
                    f"E={E:.6e} "
                    f"mean={np.mean(self.phi):.3e}",
                    f"std={np.std(self.phi):.3e}",
                    f"min={np.min(self.phi):.3e}",
                    f"max={np.max(self.phi):.3e}"
                )
                # ========== 新增：每次采样捕获一帧画面 ==========
                # New: capture one frame every sample step
                self.capture_frame()
        # ===================== 补充缺失的步骤5代码 =====================
        # 仿真全部迭代完成后，自动合成mp4视频
        # After all simulation iterations finished, auto generate mp4 video
        self.frames_to_video()
                
    def capture_frame(self):
        """
        捕获当前密度场phi画面，保存内存图像帧用于后期合成视频
        Capture current density field phi snapshot, store frame in memory for video synthesis
        """
        if not self.record_video:
            return
        # 创建临时画布，不弹出可视化窗口
        fig, ax = plt.subplots(figsize=self.video_figsize)
        # 接收imshow返回的图像对象im
        im = ax.imshow(self.phi, cmap="coolwarm", origin="lower")
        ax.set_title(f"Simulation step total: {self.steps}, current sampled frame: {len(self.frame_cache)}")
        # 传入im，指定绑定的坐标轴
        plt.colorbar(im, ax=ax)
        # 将画布转为内存二进制图像，不生成本地临时图片文件
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        self.frame_cache.append(buf.getvalue())
        plt.close(fig)
        
    def frames_to_video(self):
        """
        读取内存缓存的所有帧，调用系统ffmpeg合成mp4视频，自动清理临时文件
        Read all cached frames, call system ffmpeg to synthesize mp4, auto clean temp files
        """
        if not self.record_video or len(self.frame_cache) == 0:
            print("未开启录制/无仿真帧，跳过视频生成")
            print("Video recording disabled or zero frames captured, skip video generation")
            return
        
        # 创建临时文件夹存放帧图片
        # Create temp directory to store frame images
        temp_dir = tempfile.mkdtemp()
        try:
            # 逐帧写入临时png图片
            # Write each cached frame to temporary png file
            for idx, frame_bytes in enumerate(self.frame_cache):
                frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                with open(frame_path, "wb") as f:
                    f.write(frame_bytes)
            
            # ffmpeg合成命令：无损压缩、固定帧率、覆盖旧文件
            # ffmpeg command: lossless compression, fixed fps, overwrite old file
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # 自动覆盖已有视频文件
                "-framerate", str(self.video_fps),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",  # 画质参数，数值越小画质越高(0~51)
                self.video_output_name
            ]
            # 执行ffmpeg命令
            # Execute ffmpeg subprocess
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"仿真视频已生成：{os.path.abspath(self.video_output_name)}")
            print(f"Simulation video saved at: {os.path.abspath(self.video_output_name)}")
        finally:
            # 仿真结束自动删除临时帧文件夹
            # Auto delete temp frame folder after video generate
            shutil.rmtree(temp_dir)
            # 清空内存帧缓存释放内存
            # Clear frame cache to free memory
            self.frame_cache.clear()
                
    def plot_energy(self):
        """
        绘制自由能随采样步演化曲线
        Plot free energy evolution curve against sample step index
        """
        plt.figure(figsize=(6,4))
        # 绘制能量时序曲线
        # Plot time series curve of free energy
        plt.plot(self.energy_log)
        plt.xlabel("Sample")
        plt.ylabel("Free Energy")
        plt.title("Energy Evolution")
        plt.grid(True)
        plt.show()
        
    def plot_field(self):
        """
        绘制2D密度场φ云图
        Plot 2D heatmap of density field φ
        """
        plt.figure(figsize=(6,6))
        # origin="lower"匹配网格ij索引，图像左下角对应坐标原点
        # origin="lower" match grid ij index, bottom-left of image is coordinate origin
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        plt.colorbar()
        plt.title("Density Field")
        plt.show()
        
    def plot_structure_factor(self):
        """
        绘制对数尺度静态结构因子衍射图案
        Plot log-scaled static structure factor diffraction pattern
        """
        # 计算结构因子
        # Calculate structure factor
        S = self.structure_factor()
        plt.figure(figsize=(6,6))
        # log10压缩动态范围，+1避免log10(0)无意义
        # log10 compress dynamic range, +1 to avoid log10(0) undefined
        plt.imshow(np.log10(S + 1.0), origin="lower", cmap="inferno")
        plt.colorbar(label="log10(S+1)")
        plt.title("Structure Factor")
        plt.show()
        
    def plot_voronoi(self):
        """
        绘制原子Voronoi元胞拓扑图
        Plot Voronoi tessellation topology of crystal atoms
        """
        # 生成Voronoi拓扑与过滤后原子坐标
        # Generate Voronoi topology and filtered atom coordinates
        vor, points = self.voronoi_analysis()
        plt.figure(figsize=(8,8))
        # 绘制沃罗诺伊图，关闭顶点标记、设置线条与点尺寸
        # Plot voronoi diagram, hide vertices, set line width and point size
        voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_width=1, point_size=2)
        # Y轴反转，匹配imshow绘图坐标系
        # Invert Y axis to align with imshow plotting coordinate system
        plt.gca().invert_yaxis()
        plt.title("Voronoi Analysis")
        plt.show()
        
    def defect_statistics(self):
        """
        控制台打印不同配位数原子数量统计
        Print count statistics of atoms with different coordination numbers to console
        """
        # 获取配位数与原子坐标
        # Load coordination numbers and atom coordinates
        coord, points = self.coordination_numbers()
        # 过滤有效数值
        # Filter valid numeric entries
        valid = ~np.isnan(coord)
        coord = coord[valid]
        points = points[valid]
        # 统计每种配位数原子数量
        # Count atom quantity for each coordination value
        unique, counts = np.unique(coord.astype(int), return_counts=True)
        print()
        print("Number of atoms =", len(points))
        # 逐行打印配位数与对应原子数量
        # Print coordination value and matched atom count line by line
        for u, c in zip(unique, counts):
            print(f"{u}-fold: {c}")
        # 返回配位数数组与原子坐标
        # Return coordination array and atom coordinates
        return coord, points
    
    def plot_defects(self):
        """
        在密度场云图上标记5重(红)、7重(蓝)配位缺陷原子
        Mark 5-fold(red) /7-fold(blue) coordination defect atoms over density heatmap
        """
        # 获取配位数与原子坐标
        # Load coordination numbers and atom coordinates
        coord, points = self.coordination_numbers()
        valid = ~np.isnan(coord)
        coord = coord[valid]
        points = points[valid]
        # 5配位缺陷掩码
        # Mask for 5-fold coordination defects
        mask5 = coord == 5
        # 7配位缺陷掩码
        # Mask for 7-fold coordination defects
        mask7 = coord == 7
        plt.figure(figsize=(8,8))
        # 底层绘制密度场
        # Draw density field as background
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        # 红色散点标记5重缺陷
        # Red scatter mark 5-fold defects
        plt.scatter(points[mask5,0], points[mask5,1], c="red", s=60, label="5-fold")
        # 蓝色散点标记7重缺陷
        # Blue scatter mark 7-fold defects
        plt.scatter(points[mask7,0], points[mask7,1], c="blue", s=60, label="7-fold")
        plt.legend()
        plt.title("Defect Analysis")
        plt.show()
        
    def plot_defect_density(self):
        """
        绘制缺陷密度随采样步变化曲线
        Plot evolution curve of defect density over sample steps
        """
        plt.figure()
        # 带圆点标记的折线图
        # Line plot with circle markers
        plt.plot(self.defect_log, "o-")
        plt.xlabel("sample")
        plt.ylabel("Defect Density")
        plt.title("Defect Density Evolution")
        plt.grid()
        plt.show()
        
    def plot_grain_size(self):
        """
        绘制估算平均晶粒尺寸时序曲线
        Plot time series curve of estimated average grain size
        """
        plt.figure()
        plt.plot(self.grain_size_log, "o-")
        plt.xlabel("sample")
        plt.ylabel("Grain Size")
        plt.title("Grain Size Evolution")
        plt.grid()
        plt.show()
        
    def plot_structure_peak(self):
        """
        绘制结构因子最大衍射峰值演化曲线
        Plot evolution curve of maximum structure factor diffraction peak
        """
        plt.figure()
        plt.plot(self.structure_peak_log, "o-")
        plt.xlabel("sample")
        plt.ylabel("Structure Factor Peak")
        plt.title("Structure Factor Peak Evolution")
        plt.grid()
        plt.show()
        
    def plot_detected_atoms(self):
        """
        在密度场上绘制所有识别出的原子黑点
        Draw black dots of all detected atoms over density field
        """
        # 获取原始[i,j]格式原子像素索引
        # Get raw [i,j] format atom pixel indices
        atoms = self.detect_atoms()
        plt.figure(figsize=(8,8))
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        # atoms[:,1]=x横轴, atoms[:,0]=y纵轴，直接匹配图像坐标
        # atoms[:,1]=x horizontal axis, atoms[:,0]=y vertical axis, directly match image coordinate
        plt.scatter(atoms[:,1], atoms[:,0], s=10, c="k")
        plt.title(f"Detected atoms ({len(atoms)})")
        plt.show()
        
    def plot_psi6(self):
        """
        原子散点着色图，颜色映射|ψ6|键取向有序度大小
        Atom scatter colored plot, color mapped to magnitude of bond-orientational order |ψ6|
        """
        # 获取原子坐标与ψ6序参量
        # Load atom coordinates and ψ6 order parameter
        points, psi6 = self.compute_psi6()
        plt.figure(figsize=(8,8))
        plt.imshow(self.phi, cmap="coolwarm", origin="lower")
        # 散点颜色由|ψ6|决定
        # Scatter color determined by absolute value of ψ6
        plt.scatter(points[:,0], points[:,1], c=np.abs(psi6), cmap="viridis", s=40)
        plt.colorbar(label="|psi6|")
        plt.title("Bond Orientational Order")
        plt.show()
        
    def plot_psi6_field(self):
        """
        散乱原子ψ6插值生成全域连续有序度云图
        Interpolate discrete atom ψ6 values to full continuous order heatmap
        """
        points, psi6 = self.compute_psi6()
        # 提取每个原子|ψ6|数值
        # Extract magnitude value |ψ6| of each atom
        values = np.abs(psi6)
        # 生成全场插值网格XY
        # Generate full-domain interpolation grid X,Y
        X, Y = np.meshgrid(np.arange(self.N), np.arange(self.N))
        # 线性插值，无原子区域填充0
        # Linear interpolation, fill empty region with zero
        field = griddata(points, values, (X,Y), method="linear", fill_value=0)
        plt.figure(figsize=(8,8))
        plt.imshow(field, origin="lower", cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(label="|psi6|")
        plt.title("Psi6 Order Field")
        plt.show()
        
    def plot_grain_orientation(self):
        """
        原子着色图，颜色映射ψ6相位角（代表晶粒晶体取向）
        Atom colored scatter plot, color mapped to ψ6 phase angle (represent grain crystal orientation)
        """
        points, psi6 = self.compute_psi6()
        plt.figure(figsize=(8,8))
        # 灰度密度场作为底图
        # Grayscale density field as background
        plt.imshow(self.phi, cmap="gray", origin="lower")
        # 颜色由ψ6辐角决定，hsv色环对应0~2π角度
        # Color determined by argument of ψ6, hsv colormap correspond to angle 0~2π
        plt.scatter(points[:,0], points[:,1], c=np.angle(psi6), cmap="hsv", s=40)
        plt.colorbar(label="arg(psi6)")
        plt.title("Grain Orientation")
        plt.show()
        
    def plot_grain_boundary_atoms(self):
        """
        高亮标记高D值晶界原子，浅绿为全部原子，红色为晶界原子
        Highlight grain boundary atoms with large D value; light green = all atoms, red = grain boundary atoms
        """
        points, D = self.grain_boundary_parameter()
        # 取D值85百分位数作为晶界判定阈值
        # Take 85th percentile of D values as grain boundary discrimination threshold
        threshold = np.percentile(D, 85)
        # D大于阈值判定为晶界原子
        # Atoms with D larger than threshold are grain boundary atoms
        gb_mask = D > threshold
        plt.figure(figsize=(8,8))
        plt.imshow(self.phi, cmap='gray', origin='lower')
        # 半透明浅绿色绘制全部原子
        # Semi-transparent light green draw all atoms
        plt.scatter(points[:,0], points[:,1], s=20, c='lime', alpha=0.3, label='all atoms')
        # 红色放大绘制晶界原子
        # Enlarged red dots draw grain boundary atoms
        plt.scatter(points[gb_mask,0], points[gb_mask,1], s=80, c='red', label='grain boundary')
        plt.legend()
        plt.title("Grain Boundary Atoms")
        plt.show()
            
    def analyze_psi6(self):
        """
        ψ6完整分析流水线：打印局部/全局有序度数值 + 全套取向绘图
        Full ψ6 analysis pipeline: print local & global order metrics + full suite of orientation plots
        """
        # 计算局部、全局平均ψ6模长
        # Calculate local and global averaged modulus of ψ6
        psi6_local, psi6_global = self.global_psi6()
        # 控制台打印有序度数值
        # Print order metrics to console
        print(f"Psi6_local = {psi6_local:.4f}")
        print(f"Psi6_global = {psi6_global:.4f}")
        # 依次绘制四张ψ6相关图
        # Plot four ψ6 related figures sequentially
        self.plot_psi6()
        self.plot_grain_orientation()
        self.plot_psi6_field()
        self.plot_grain_boundary_atoms()
        
    def postprocess(self):
        """
        全套后处理绘图流程，绘图顺序与原始代码完全一致
        Full post-processing plotting pipeline, plot execution order identical to original code
        """
        self.plot_energy()
        self.plot_field()
        self.plot_structure_factor()
        self.plot_voronoi()
        self.defect_statistics()
        self.plot_defects()
        self.plot_defect_density()
        self.plot_grain_size()
        self.plot_structure_peak()
        self.plot_detected_atoms()

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
        "N": N,
        "L": L,
        "r": r,
        "M": M,
        "dt": dt,
        "T": T,
        "phi0": phi0,
        "noise_amp": noise_amp,
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


if __name__ == "__main__":
    """
    交互入口：读取用户输入参数 → 初始化求解器 → 运行仿真 → 绘图分析
    Interactive entry: read user params → init solver → run simulation → plot analysis
    """
    # 调用交互输入函数获取全部参数
    sim_params = input_pfc_parameters()
    # 实例化多晶格PFC求解器
    solver = PFCSolver(**sim_params)
    # 启动仿真循环（自动捕获帧、仿真结束自动ffmpeg生成视频）
    solver.run()
    # 执行全套静态后处理绘图
    solver.postprocess()
    # 执行ψ6键取向序专项分析绘图
    solver.analyze_psi6()
    print("\n==== 全部仿真、视频、绘图任务执行完成 ====")