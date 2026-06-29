"""
pfc_analysis.py - Microstructure and Defect Analysis for PFC Simulations
微观结构分析模块 - PFC模拟的微观结构与缺陷分析

This module provides analysis tools for PFC simulation results, including
atom detection, coordination analysis, Voronoi tessellation, ψ6 orientational
order, defect analysis, and grain boundary detection.
本模块提供PFC模拟结果的分析工具，包括原子检测、配位分析、Voronoi剖分、
ψ6取向序、缺陷分析和晶界检测。

Designed as a mixin class for PFC solvers.
设计为PFC求解器的mixin类。

Author: Jinpeng Wang
Department of Material Engineering
"""

# NumPy - numerical computing
# NumPy - 数值计算
import numpy as np

# SciPy spatial - KDTree for neighbor search
# SciPy空间 - 用于近邻搜索的KDTree
from scipy.spatial import cKDTree

# scikit-image - peak detection for atom finding
# scikit-image - 用于原子查找的峰值检测
from skimage.feature import peak_local_max

# SciPy spatial - Voronoi tessellation
# SciPy空间 - Voronoi剖分
from scipy.spatial import Voronoi


class PFCAnalysis:
    """
    Mixin class providing microstructure analysis methods for PFC simulations.
    为PFC模拟提供微观结构分析方法的mixin类。
    
    This class implements a comprehensive set of analysis tools for
    characterizing the microstructure of PFC systems.
    本类实现了一套全面的分析工具，用于表征PFC系统的微观结构。
    
    Analysis categories / 分析类别:
        - Atom detection / 原子检测
        - Coordination analysis / 配位分析
        - Voronoi tessellation / Voronoi剖分
        - ψ6 orientational order / ψ6取向序
        - Defect analysis / 缺陷分析
        - Grain boundary detection / 晶界检测
    
    Notes / 说明:
        - Assumes host class provides: phi, N, L, energy_log, etc.
          假设宿主类提供：phi, N, L, energy_log等
        - Uses periodic boundary conditions for neighbor search
          近邻搜索使用周期边界条件
        - All coordinates are in pixel units (grid indices)
          所有坐标都以像素为单位（网格索引）
    """
    
    def sample_observables(self, step):
        """
        Sample key observables during simulation.
        在模拟过程中采样关键观测量。
        
        Samples free energy, mass, defect density, grain size, and
        structure factor peak at regular intervals.
        定期采样自由能、质量、缺陷密度、晶粒尺寸和结构因子峰值。
        
        Args / 参数:
            step (int): Current simulation step / 当前模拟步数
        
        Returns / 返回值:
            float: Current free energy / 当前自由能
        
        Notes / 说明:
            - Defect analysis only after step 1500 (saves computation)
              仅在1500步后进行缺陷分析（节省计算）
            - All results appended to corresponding log lists
              所有结果追加到对应的日志列表中
        """
        # Compute current free energy
        # 计算当前自由能
        E = self.compute_energy()
        
        # Append energy to log
        # 将能量追加到日志
        self.energy_log.append(E)
        
        # Append mean density (mass conservation check)
        # 追加平均密度（质量守恒检查）
        self.mass_log.append(np.mean(self.phi))
        
        # Defect analysis only after step 1500 (saves computation time)
        # 仅在1500步后进行缺陷分析（节省计算时间）
        if step > 1500:
            # Analyze defects and get defect density and grain size
            # 分析缺陷并获取缺陷密度和晶粒尺寸
            defect_density, grain_size, _, _ = self.analyze_defects()
            
            # Only log if valid (not NaN)
            # 仅在有效时记录（非NaN）
            if not np.isnan(defect_density):
                self.defect_log.append(defect_density)
                self.grain_size_log.append(grain_size)
        
        # Compute structure factor and log peak value
        # 计算结构因子并记录峰值
        S = self.structure_factor()
        self.structure_peak_log.append(np.max(S))
        
        return E
    
    def detect_atoms(self):
        """
        Local peak detection, extract pixel coordinates (i,j) of all crystal atoms.
        局部峰值检测，提取所有晶体原子的像素坐标(i,j)。
        
        Identifies atom positions by finding local maxima in the density field.
        通过在密度场中寻找局部极大值来识别原子位置。
        
        Returns / 返回值:
            ndarray: Array of atom positions with shape (n_atoms, 2),
                    each row is [vertical_pixel_i, horizontal_pixel_j]
                    原子位置数组，形状为(n_atoms, 2)，
                    每行是[纵向像素i, 横向像素j]
        
        Parameters / 参数:
            - min_distance=7: Minimum pixel distance between atoms (prevents duplicates)
              原子间最小像素间距（避免重复识别）
            - threshold_rel=0.5: Relative peak threshold (filters noise)
              相对峰值阈值（过滤噪声）
            - exclude_border=False: Allow atoms near domain boundary
              允许靠近计算域边界的原子
        """
        # Find local maxima in density field
        # 在密度场中寻找局部极大值
        atoms = peak_local_max(
            self.phi,
            # Minimum pixel distance between atoms, avoid duplicate detection
            # 原子间最小像素间距，避免重复识别同一原子
            min_distance=7,
            # Relative peak threshold, filter weak noise false peaks
            # 峰值相对阈值，过滤微弱噪声伪峰
            threshold_rel=0.5,
            # Allow detection of atoms near simulation boundary
            # 允许识别靠近计算域边界的原子
            exclude_border=False
        )
        
        # Return array with format [vertical pixel i, horizontal pixel j]
        # 返回数组，每行格式为[纵向像素i, 横向像素j]
        return atoms
    
    def build_neighbors(self):
        """
        Build periodic boundary KD-Tree, calculate neighbor list for every atom.
        构建周期边界KD树，计算每个原子的近邻原子列表。
        
        Uses a KD-tree with periodic boundary conditions for efficient
        neighbor search in the periodic domain.
        使用带周期边界条件的KD树在周期域中进行高效近邻搜索。
        
        Returns / 返回值:
            tuple: (points, neighbors)
                - points (ndarray): Atom xy coordinates (n_atoms, 2)
                  原子xy坐标
                - neighbors (list): List of neighbor indices for each atom
                  每个原子的近邻索引列表
        
        Notes / 说明:
            - Lattice constant a0 estimated from median 2nd neighbor distance
              晶格常数a0从第二近邻距离的中位数估算
            - Cutoff radius = 1.35 * a0 (includes first coordination shell)
              截断半径 = 1.35 * a0（包含第一配位壳层）
            - Periodic boundary conditions via boxsize parameter
              通过boxsize参数实现周期边界条件
        """
        # Get atom pixel indices
        # 获取原子像素索引
        atoms = self.detect_atoms()
        
        # Convert to plot-aligned xy coordinates [i,j] -> [x,y]
        # 转换为绘图匹配的xy坐标 [i,j] -> [x,y]
        points = atoms[:, ::-1]
        
        # Build KD-Tree with periodic box boundary
        # 构建带周期盒子边界的KD近邻树
        tree = cKDTree(points, boxsize=self.N)
        
        # Pre-query top 6 nearest neighbors for each atom to estimate lattice constant a0
        # 预查询每个原子前6近邻，用于估算晶格常数a0
        dists, _ = tree.query(points, k=6)
        
        # Median distance of second nearest neighbor as lattice constant a0
        # 第二近邻距离的中位数作为晶格常数a0
        # Using 2nd neighbor because 1st might have variations
        # 使用第二近邻是因为第一近邻可能有变化
        a0 = np.median(dists[:, 1])
        
        # Neighbor search cutoff radius: 1.35 times lattice constant
        # 近邻搜索截断半径：1.35倍晶格常数
        # This captures all first-shell neighbors in hexagonal lattice
        # 这捕获了六角晶格中所有第一壳层近邻
        r_cut = 1.35 * a0
        
        # Initialize neighbor list
        # 初始化近邻列表
        neighbors = []
        
        # Iterate all atoms, query all neighbors within cutoff radius
        # 遍历全部原子，查询截断半径内的所有近邻
        for p in points:
            # Find all points within cutoff radius
            # 查找截断半径内的所有点
            neigh = tree.query_ball_point(p, r_cut)
            neighbors.append(neigh)
        
        # Return atom xy coordinates & matched neighbor index list
        # 返回原子xy坐标和对应的近邻索引列表
        return points, neighbors
    
    def coordination_numbers(self):
        """
        Calculate coordination number of each atom (neighbor count, exclude self).
        计算每个原子的配位数（近邻原子数量，排除自身）。
        
        Coordination number = number of nearest neighbors.
        In perfect hexagonal lattice, coordination number = 6.
        配位数 = 最近邻的数量。
        在完美六角晶格中，配位数 = 6。
        
        Returns / 返回值:
            tuple: (coord, points)
                - coord (ndarray): Coordination number for each atom
                  每个原子的配位数
                - points (ndarray): Atom xy coordinates
                  原子xy坐标
        """
        # Get atom coordinates and neighbor list
        # 获取原子坐标和近邻列表
        points, neighbors = self.build_neighbors()
        
        # Neighbor list length minus one equals coordination number
        # 每个原子近邻列表长度-1 = 配位数
        # Minus 1 because the atom itself is included in the neighbor list
        # 减1是因为原子自身也包含在近邻列表中
        coord = np.array([len(n) - 1 for n in neighbors])
        
        # Return coordination number array & atom xy coordinates
        # 返回配位数数组和原子xy坐标
        return coord, points
    
    def valid_coordination(self):
        """
        Filter valid coordination atoms, exclude isolated and boundary atoms.
        过滤有效配位数原子，排除孤立原子和边界原子。
        
        Removes atoms with invalid (NaN) coordination numbers, which
        typically correspond to isolated atoms or boundary artifacts.
        移除具有无效（NaN）配位数的原子，这些通常对应于孤立原子或边界伪影。
        
        Returns / 返回值:
            tuple: (coord, points)
                - coord (ndarray): Valid coordination numbers
                  有效配位数
                - points (ndarray): Corresponding atom coordinates
                  对应的原子坐标
        """
        # Get coordination numbers and atom coordinates
        # 获取配位数和原子坐标
        coord, points = self.coordination_numbers()
        
        # Return empty values if zero atoms detected
        # 如果没有检测到原子，直接返回空值
        if len(coord) == 0:
            return np.nan, np.nan, np.array([]), np.empty((0, 2))
        
        # Filter valid numeric entries, exclude isolated and boundary atoms
        # 过滤有效数值，排除孤立原子和边界原子
        valid = ~np.isnan(coord)
        coord = coord[valid]
        points = points[valid]
        
        # Return valid coordination number array & matched atom coordinates
        # 返回有效配位数数组和对应的原子坐标
        return coord, points
    
    def voronoi_analysis(self):
        """
        Generate Voronoi tessellation from atom coordinates for grain topology partition.
        基于原子坐标生成周期域Voronoi元胞，用于晶粒拓扑划分。
        
        Voronoi tessellation partitions space into convex polygons (cells)
        around each atom, where each cell contains points closer to its
        atom than any other.
        Voronoi剖分将空间划分为每个原子周围的凸多边形（元胞），
        每个元胞包含距离其原子比其他原子更近的点。
        
        Returns / 返回值:
            tuple: (vor, points)
                - vor (Voronoi): Voronoi topology object
                  Voronoi拓扑对象
                - points (ndarray): Filtered atom xy coordinates
                  过滤后的原子xy坐标
        
        Notes / 说明:
            - Boundary atoms filtered to avoid distorted Voronoi cells
              过滤边界原子以避免Voronoi元胞畸形
            - Margin of 0 pixels (can be increased for cleaner results)
              边界余量为0像素（可以增加以获得更干净的结果）
        """
        # Get all atom pixel indices
        # 获取所有原子像素索引
        atoms = self.detect_atoms()
        
        # Coordinate flip: [i,j] → [x,y], match plot horizontal & vertical axis
        # 坐标翻转：[i,j] → [x,y]，匹配绘图横轴纵轴
        points = atoms[:, ::-1]
        
        # Boundary filter margin
        # 边界过滤余量
        margin = 0
        
        # Filter atoms attached to boundary to avoid distorted Voronoi cells
        # 过滤紧贴边界的原子，避免Voronoi元胞畸形
        mask = (
            (points[:, 0] > margin)
            &
            (points[:, 0] < self.N - margin)
            &
            (points[:, 1] > margin)
            &
            (points[:, 1] < self.N - margin)
        )
        points = points[mask]
        
        # Build Voronoi topology object
        # 构建Voronoi拓扑对象
        vor = Voronoi(points)
        
        # Return topology object & filtered atom xy coordinates
        # 返回拓扑对象和过滤后的原子xy坐标
        return vor, points
    
    def compute_psi6(self):
        """
        Compute complex ψ6 bond-orientational order parameter, characterize
        local hexagonal crystal order.
        计算ψ6复键取向序参量，表征六角形晶体局部有序度。
        
        ψ6 is a complex number that measures six-fold bond orientational order.
        For each atom, it's the average of exp(6i·θ) over all neighbor bonds,
        where θ is the bond angle.
        ψ6是一个复数，测量六重键取向有序度。
        对于每个原子，它是所有近邻键的exp(6i·θ)的平均值，其中θ是键角。
        
        |ψ6| = 1: perfect hexagonal order / 完美六角有序
        |ψ6| = 0: completely disordered / 完全无序
        
        Returns / 返回值:
            tuple: (points, psi6)
                - points (ndarray): Atom xy coordinates
                  原子xy坐标
                - psi6 (ndarray): Complex ψ6 values for each atom
                  每个原子的复ψ6值
        
        Notes / 说明:
            - Periodic boundary conditions applied to bond vectors
              对键向量应用周期边界条件
            - Atoms with zero neighbors get psi6 = 0
              零近邻的原子psi6 = 0
        """
        # Load atom coordinates and neighbor list
        # 读取原子坐标和近邻列表
        points, neighbors = self.build_neighbors()
        
        # Initialize complex ψ6 array
        # 初始化复数ψ6数组
        psi6 = np.zeros(len(points), dtype=complex)
        
        # Iterate every single atom
        # 遍历每一个原子
        for i, p in enumerate(points):
            # Filter neighbor indices, exclude self index
            # 筛选排除自身的近邻索引
            neigh = [j for j in neighbors[i] if j != i]
            
            # Skip atom with zero neighbors, keep ψ6 = 0
            # 无近邻则跳过，保持ψ6=0
            if len(neigh) == 0:
                continue
            
            # List to store bond angles
            # 存储键角的列表
            angles = []
            
            # Iterate all valid neighbors, calculate bond vector angle
            # 遍历所有有效近邻，计算键向量夹角
            for j in neigh:
                # Compute bond vector components
                # 计算键向量分量
                dx = points[j, 0] - p[0]
                dy = points[j, 1] - p[1]
                
                # Periodic vector correction for atoms crossing simulation box
                # 周期边界向量修正，处理跨盒边界原子
                # Minimum image convention
                # 最小镜像约定
                dx -= self.N * np.round(dx / self.N)
                dy -= self.N * np.round(dy / self.N)
                
                # Calculate angle between bond vector and x-axis
                # 计算键向量与x轴的夹角
                angles.append(np.arctan2(dy, dx))
            
            # Average 6-fold symmetric phase to get ψ6 value
            # 六重对称相位平均得到ψ6
            # ψ6 = (1/N) · Σ exp(6i·θ_j)
            psi6[i] = np.mean(np.exp(6j * np.array(angles)))
        
        # Return atom coordinates & matched complex ψ6 values
        # 返回原子坐标和对应的复ψ6值
        return points, psi6
    
    def global_psi6(self):
        """
        Calculate two order metrics: local average |ψ6|, modulus of
        global complex averaged ψ6.
        求解局部平均|ψ6|、全局平均复ψ6模长，两个有序度评价指标。
        
        Two different measures of bond-orientational order:
        两种不同的键取向有序度度量：
        
        1. ψ6_local: Average of |ψ6| over all atoms
           - Measures average local order
           - 测量平均局部有序度
           
        2. ψ6_global: |<ψ6>| = |mean(ψ6)| (modulus of complex average)
           - Measures global/long-range order
           - 测量全局/长程有序度
           - Sensitive to grain boundaries and misorientation
             对晶界和取向差敏感
        
        Returns / 返回值:
            tuple: (psi6_local, psi6_global)
                - psi6_local (float): Local average |ψ6|
                  局部平均|ψ6|
                - psi6_global (float): Modulus of global averaged ψ6
                  全局平均ψ6的模长
        """
        # Load ψ6 value of all atoms
        # 读取全部原子的ψ6值
        _, psi6 = self.compute_psi6()
        
        # Local order: arithmetic average of |ψ6| over all atoms
        # 局部有序度：所有原子|ψ6|的算术平均
        # This measures average local crystalline order
        # 这测量平均局部晶体有序度
        psi6_local = np.mean(np.abs(psi6))
        
        # Global order: modulus of summed complex ψ6 of all atoms
        # 全局有序度：全部原子复ψ6求和后取模
        # This measures long-range orientational order
        # Lower value means more grain boundaries / misorientation
        # 这测量长程取向有序度
        # 值越低意味着晶界/取向差越多
        psi6_global = np.abs(np.mean(psi6))
        
        return (psi6_local, psi6_global)
    
    def grain_boundary_parameter(self):
        """
        Grain boundary discriminant D: mean squared ψ6 difference between
        adjacent atoms, larger D means grain boundary.
        晶界判别参数D：相邻原子ψ6差值平方均值，D越大越靠近晶界。
        
        D parameter measures the local variation in ψ6, which is large
        at grain boundaries where orientation changes abruptly.
        D参数测量ψ6的局部变化，在取向突变的晶界处较大。
        
        Returns / 返回值:
            tuple: (points, D)
                - points (ndarray): Atom xy coordinates
                  原子xy坐标
                - D (ndarray): Grain boundary parameter for each atom
                  每个原子的晶界参数
        
        Notes / 说明:
            - D = mean of |ψ6_i - ψ6_j|² over all neighbors j
              D = 所有近邻j的|ψ6_i - ψ6_j|²的平均值
            - High D = grain boundary atom
              高D = 晶界原子
            - Low D = grain interior atom
              低D = 晶粒内部原子
        """
        # Get atom coordinates and neighbor list
        # 获取原子坐标和近邻列表
        points, neighbors = self.build_neighbors()
        
        # Get ψ6 order parameter of each atom
        # 获取每个原子的ψ6序参量
        _, psi6 = self.compute_psi6()
        
        # Initialize D parameter array
        # 初始化D参数数组
        D = np.zeros(len(points))
        
        # Iterate all atoms
        # 遍历全部原子
        for i in range(len(points)):
            # Remove self index from neighbor list
            # 从近邻列表中剔除自身索引
            neigh = neighbors[i]
            neigh = [j for j in neigh if j != i]
            
            # Skip atom without neighbors
            # 无近邻则跳过
            if len(neigh) == 0:
                continue
            
            # Calculate average squared difference of |ψ6| with all neighbors
            # 计算与所有近邻|ψ6|差值平方的平均值
            D[i] = np.mean(np.abs(psi6[i] - psi6[neigh]) ** 2)
        
        # Return atom coordinates & matched grain boundary parameter D
        # 返回原子坐标和对应的晶界参数D
        return points, D
    
    def analyze_defects(self):
        """
        Defect statistics: concentration of 5/7-fold coordination atoms,
        estimated average grain size.
        缺陷统计：5/7配位原子缺陷密度、估算平均晶粒尺寸。
        
        In a perfect hexagonal lattice, each atom has 6 neighbors.
        Atoms with coordination ≠ 6 are considered defects:
        在完美的六角晶格中，每个原子有6个邻居。
        配位数≠6的原子被视为缺陷：
        
        - 5-fold: vacancy-like (missing neighbor)
          5重：类空位（少一个邻居）
        - 7-fold: interstitial-like (extra neighbor)
          7重：类间隙（多一个邻居）
        - 5/7 pairs form dislocation cores
          5/7对形成位错核心
        
        Returns / 返回值:
            tuple: (defect_density, grain_size, coord, points)
                - defect_density (float): Fraction of defective atoms
                  缺陷原子的比例
                - grain_size (float): Estimated average grain size
                  估算的平均晶粒尺寸
                - coord (ndarray): Coordination numbers
                  配位数
                - points (ndarray): Atom coordinates
                  原子坐标
        
        Notes / 说明:
            - Grain size estimated as sqrt(area / defect_count)
              晶粒尺寸估算为sqrt(面积 / 缺陷数)
            - This is an approximation, not exact grain size
              这是一个近似值，不是精确的晶粒尺寸
        """
        # Load valid coordination numbers and atom coordinates
        # 获取有效配位数和原子坐标
        coord, points = self.valid_coordination()
        
        # Total number of detected atoms
        # 总识别原子数量
        total_atoms = len(coord)
        
        # Atoms with coordination not equal to 6 are marked as defects
        # 非6配位原子判定为缺陷
        defect_mask = (coord != 6)
        
        # Total count of defect atoms
        # 缺陷原子总数
        defect_number = np.sum(defect_mask)
        
        # Defect density = defect count / total atom count
        # 缺陷密度 = 缺陷数 / 总原子数
        defect_density = defect_number / total_atoms
        
        # Approximate grain size formula: sqrt(total area / defect count)
        # 近似晶粒尺寸公式：sqrt(总面积 / 缺陷数)
        # This assumes defects are distributed at grain boundaries
        # 这假设缺陷分布在晶界处
        # max(defect_number, 1) avoids division by zero
        # max(defect_number, 1)避免除以零
        grain_size = np.sqrt(self.L * self.L / max(defect_number, 1))
        
        # Return defect density, grain size, coordination array, atom coordinates
        # 返回缺陷密度、晶粒尺寸、配位数数组、原子坐标
        return (defect_density, grain_size, coord, points)
    
    def defect_statistics(self):
        """
        Print count statistics of atoms with different coordination numbers to console.
        控制台打印不同配位数原子数量统计。
        
        Prints the number of atoms for each coordination number value,
        which shows the distribution of defects.
        打印每种配数值的原子数量，显示缺陷的分布。
        
        Returns / 返回值:
            tuple: (coord, points)
                - coord (ndarray): Coordination number array
                  配位数数组
                - points (ndarray): Atom coordinates
                  原子坐标
        """
        # Load valid coordination numbers and atom coordinates
        # 获取有效配位数和原子坐标
        coord, points = self.valid_coordination()
        
        # Count atom quantity for each coordination value
        # 统计每种配位数原子数量
        unique, counts = np.unique(coord.astype(int), return_counts=True)
        
        # Print statistics header
        # 打印统计标题
        print()
        print("Number of atoms / 原子总数 =", len(points))
        
        # Print coordination value and matched atom count line by line
        # 逐行打印配位数和对应的原子数量
        for u, c in zip(unique, counts):
            print(f"{u}-fold / {u}重配位: {c} atoms")
        
        # Return coordination array and atom coordinates
        # 返回配位数数组和原子坐标
        return coord, points
