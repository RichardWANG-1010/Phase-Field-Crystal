import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from scipy.spatial import Voronoi


class PFCAnalysis:
    
    def sample_observables(self, step):
        
        E = self.compute_energy()

        self.energy_log.append(E)

        self.mass_log.append(
            np.mean(self.phi)
        )

        if step > 1500:

            defect_density, grain_size, _, _ = (
                self.analyze_defects()
            )

            if not np.isnan(defect_density):

                self.defect_log.append(
                    defect_density
                )

                self.grain_size_log.append(
                    grain_size
                )

        S = self.structure_factor()

        self.structure_peak_log.append(
            np.max(S)
        )
        
        return E
    
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
    
    def valid_coordination(self):
        """
        过滤有效配位数原子，排除孤立原子和边界原子
        Filter valid coordination atoms, exclude isolated and boundary atoms
        """
        # 获取配位数与原子坐标
        # Get coordination numbers and atom coordinates
        coord, points = self.coordination_numbers()
        # 无原子直接返回空值
        # Return empty value if zero atoms detected
        if len(coord) == 0:
            return np.nan, np.nan, np.array([]), np.empty((0, 2))
        # 过滤有效数值，排除孤立原子和边界原子
        # Filter valid numeric entries, exclude isolated and boundary atoms
        valid = ~np.isnan(coord)
        coord = coord[valid]
        points = points[valid]
        # 返回有效配位数数组、对应原子坐标
        # Return valid coordination number array & matched atom coordinates
        return coord, points
    
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
        # 获取有效配位数与原子坐标
        # Load valid coordination numbers and atom coordinates
        coord, points = self.valid_coordination()
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
    
    def defect_statistics(self):
        """
        控制台打印不同配位数原子数量统计
        Print count statistics of atoms with different coordination numbers to console
        """
        # 获取有效配位数与原子坐标
        # Load valid coordination numbers and atom coordinates
        coord, points = self.valid_coordination()
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