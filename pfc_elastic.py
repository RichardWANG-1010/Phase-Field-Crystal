"""
pfc_elastic.py - Elastic Property Calculations for PFC Model
弹性计算模块 - PFC模型的弹性性质计算

This module provides methods for calculating elastic properties of PFC systems,
including strain application, elastic energy curves, and elastic constant fitting.
本模块提供计算PFC系统弹性性质的方法，包括应变施加、弹性能量曲线和弹性常数拟合。

The elastic constant C is obtained by fitting the energy-strain curve:
弹性常数C通过拟合能量-应变曲线获得：
    F(ε) = a·ε² + b·ε + c
    C = 2·a  (second derivative of energy with respect to strain)

Author: Jinpeng Wang
Department of Material Engineering
"""

# NumPy - numerical computing
# NumPy - 数值计算
import numpy as np


class PFCElastic:
    """
    Mixin class for elastic property calculations in PFC simulations.
    PFC模拟中弹性性质计算的mixin类。
    
    This class provides methods to apply strain, compute elastic energy curves,
    and fit elastic constants from energy-strain data.
    本类提供施加应变、计算弹性能量曲线和从能量-应变数据拟合弹性常数的方法。
    
    Designed to be used as a mixin with PFCBase and solver classes.
    设计为与PFCBase和求解器类一起使用的mixin。
    
    Key concepts / 关键概念:
        - Strain (ε): Deformation relative to reference state
          应变（ε）：相对于参考状态的变形
        - Stress (σ): Force per unit area, dF/dε
          应力（σ）：单位面积的力，dF/dε
        - Elastic constant (C): Second derivative of energy w.r.t. strain
          弹性常数（C）：能量对应变的二阶导数
    
    Notes / 说明:
        - Assumes the host class has: phi, L, L0, dx, _build_kspace(), step(), compute_energy()
          假设宿主类具有：phi, L, L0, dx, _build_kspace(), step(), compute_energy()
        - Strain is applied by rescaling the domain size and rebuilding k-space
          通过重新缩放域大小和重建k空间来施加应变
    """
    
    def apply_strain(self, eps):
        """
        Apply uniform strain to the system by rescaling the domain.
        通过重新缩放计算域来对系统施加均匀应变。
        
        This method rescales the physical domain size and rebuilds the
        k-space grid to account for the change in length scales.
        该方法重新缩放物理域尺寸并重建k空间网格以考虑长度尺度的变化。
        
        Args / 参数:
            eps (float): Strain value (ε = ΔL/L0) / 应变值
            
        Notes / 说明:
            - Positive eps = expansion / 正eps = 膨胀
            - Negative eps = compression / 负eps = 压缩
            - Rebuilds k-space because wave vectors scale with domain size
              重建k空间，因为波矢随域大小缩放
        """
        # Rescale domain size: L = L0 · (1 + ε)
        # 重新缩放域大小
        self.L = self.L0 * (1 + eps)
        
        # Update grid spacing accordingly
        # 相应地更新网格间距
        self.dx = self.L / self.N
        
        # Rebuild k-space with new domain size
        # 使用新的域大小重建k空间
        # Wave vectors scale inversely with domain size
        # 波矢与域大小成反比
        self._build_kspace()
        
    def save_reference_state(self):
        """
        Save the current state as the reference (unstrained) state.
        将当前状态保存为参考（无应变）状态。
        
        This saves the density field and domain parameters for use as
        the starting point for strain calculations.
        这将保存密度场和域参数，用作应变计算的起点。
        
        Saved quantities / 保存的量:
            - phi_ref: Reference density field / 参考密度场
            - L_ref: Reference domain size / 参考域尺寸
            - dx_ref: Reference grid spacing / 参考网格间距
        """
        # Save reference density field
        # 保存参考密度场
        self.phi_ref = self.phi.copy()
        
        # Save reference domain size
        # 保存参考域尺寸
        self.L_ref = self.L
        
        # Save reference grid spacing
        # 保存参考网格间距
        self.dx_ref = self.dx
        
    def elastic_energy_curve(self, strain_list, relax_steps=2000):
        """
        Calculate elastic energy as a function of strain.
        计算作为应变函数的弹性能量。
        
        For each strain value, the system is reset to the reference state,
        strained, and then relaxed for a number of steps before measuring
        the free energy.
        对于每个应变值，系统重置到参考状态，施加应变，然后弛豫若干步，
        再测量自由能。
        
        Args / 参数:
            strain_list (array-like): List of strain values to test
                                     要测试的应变值列表
            relax_steps (int, optional): Number of relaxation steps after strain.
                                        Defaults to 2000.
                                        应变后的弛豫步数，默认2000。
        
        Returns / 返回值:
            tuple: (energy_array, phi_list)
                - energy_array (ndarray): Free energy at each strain
                  每个应变下的自由能
                - phi_list (list): Density field at each strain (for further analysis)
                  每个应变下的密度场（用于进一步分析）
        
        Notes / 说明:
            - Always starts from the reference state for each strain
              每个应变都从参考状态开始
            - Relaxation allows the system to reach mechanical equilibrium
              弛豫使系统达到力学平衡
            - Strain values should be small (typically ±0.05) for linear elasticity
              对于线弹性，应变值应该很小（通常±0.05）
        """
        # Save reference state (in case not already saved)
        # 保存参考状态（以防尚未保存）
        phi_ref = self.phi.copy()
        L_ref = self.L
        dx_ref = self.dx
        
        # Initialize output lists
        # 初始化输出列表
        energy = []
        phi_list = []
        
        # Iterate over each strain value
        # 遍历每个应变值
        for eps in strain_list:
            # ============================================================
            # Reset to reference state / 重置到参考状态
            # ============================================================
            
            # Restore density field from reference
            # 从参考状态恢复密度场
            self.phi = phi_ref.copy()
            
            # Restore domain size
            # 恢复域大小
            self.L = L_ref
            self.dx = dx_ref
            
            # Rebuild k-space with reference domain size
            # 使用参考域大小重建k空间
            self._build_kspace()
            
            # ============================================================
            # Apply strain / 施加应变
            # ============================================================
            
            # Apply the current strain value
            # 施加当前应变值
            self.apply_strain(eps)
            
            # ============================================================
            # Relax to equilibrium / 弛豫到平衡
            # ============================================================
            
            # Relax for specified number of steps
            # 弛豫指定的步数
            # This allows atomic positions to adjust to the new strain
            # 这允许原子位置调整到新的应变状态
            for _ in range(relax_steps):
                self.step()
            
            # ============================================================
            # Measure energy / 测量能量
            # ============================================================
            
            # Compute and store the free energy
            # 计算并存储自由能
            energy.append(self.compute_energy())
            
            # Store the relaxed density field
            # 存储弛豫后的密度场
            phi_list.append(self.phi.copy())
        
        # ============================================================
        # Return results / 返回结果
        # ============================================================
        
        # Convert energy list to numpy array and return
        # 将能量列表转换为numpy数组并返回
        return (
            np.array(energy),
            phi_list
        )
    
    def fit_elastic_constant(self, strain, energy):
        """
        Fit elastic constant from energy-strain data using quadratic fit.
        使用二次拟合从能量-应变数据拟合弹性常数。
        
        Fits F(ε) = a·ε² + b·ε + c and extracts:
        拟合F(ε) = a·ε² + b·ε + c并提取：
            - Elastic constant C = 2·a (Young's modulus)
              弹性常数C = 2·a（杨氏模量）
            - Residual strain ε_r = -b/(2a) (strain at minimum energy)
              残余应变ε_r = -b/(2a)（能量最小时的应变）
        
        Args / 参数:
            strain (ndarray): Strain values / 应变值
            energy (ndarray): Corresponding free energy values / 对应的自由能值
        
        Returns / 返回值:
            tuple: (C, eps_r, coef, strain_fit, energy_fit)
                - C (float): Elastic constant / 弹性常数
                - eps_r (float): Residual strain (energy minimum) / 残余应变
                - coef (ndarray): Polynomial coefficients [a, b, c] / 多项式系数
                - strain_fit (ndarray): Fine strain values for plotting / 用于绘图的精细应变值
                - energy_fit (ndarray): Fitted energy values / 拟合的能量值
        
        Notes / 说明:
            - Assumes quadratic (parabolic) energy-strain relation
              假设二次（抛物线）能量-应变关系
            - Valid for small strains (linear elasticity regime)
              适用于小应变（线弹性区域）
            - C = d²F/dε² evaluated at the minimum
              C = d²F/dε² 在最小值处计算
        """
        # ============================================================
        # Quadratic polynomial fit / 二次多项式拟合
        # ============================================================
        
        # Fit 2nd-order polynomial to energy-strain data
        # 对能量-应变数据拟合二阶多项式
        # coef = [a, b, c] where F(ε) = a·ε² + b·ε + c
        coef = np.polyfit(strain, energy, 2)
        
        # ============================================================
        # Generate fine curve for plotting / 生成精细曲线用于绘图
        # ============================================================
        
        # Generate fine strain values for smooth curve
        # 生成精细应变值以获得平滑曲线
        strain_fit = np.linspace(strain.min(), strain.max(), 200)
        
        # Evaluate polynomial at fine strain values
        # 在精细应变值处计算多项式
        energy_fit = np.polyval(coef, strain_fit)
        
        # ============================================================
        # Extract elastic properties / 提取弹性性质
        # ============================================================
        
        # Elastic constant C = 2·a = d²F/dε²
        # Second derivative of energy with respect to strain
        # 弹性常数C = 2·a = d²F/dε²
        # 能量对应变的二阶导数
        C = 2 * coef[0]
        
        # Residual strain ε_r = -b/(2a)
        # Strain at which energy is minimum
        # 残余应变ε_r = -b/(2a)
        # 能量最小时的应变
        eps_r = (-coef[1]) / (2 * coef[0])
        
        # ============================================================
        # Return results / 返回结果
        # ============================================================
        
        return C, eps_r, coef, strain_fit, energy_fit
    
    def compute_stress(
        self,
        strain,
        energy
    ):
        """
        Compute stress from energy-strain data using numerical differentiation.
        使用数值微分从能量-应变数据计算应力。
        
        Stress is the first derivative of energy with respect to strain:
        应力是能量对应变的一阶导数：
            σ = dF/dε
        
        Args / 参数:
            strain (ndarray): Strain values / 应变值
            energy (ndarray): Corresponding free energy values / 对应的自由能值
        
        Returns / 返回值:
            ndarray: Stress values / 应力值
        
        Notes / 说明:
            - Uses np.gradient for numerical differentiation
              使用np.gradient进行数值微分
            - Second-order accurate for uniform spacing
              对于均匀间距是二阶精度
            - Stress-strain curve slope = elastic modulus
              应力-应变曲线的斜率 = 弹性模量
        """
        # Compute stress as numerical derivative of energy w.r.t. strain
        # 计算应力作为能量对应变的数值导数
        # σ = dF/dε
        stress = np.gradient(
            energy,
            strain
        )
        
        return stress
