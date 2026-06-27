import numpy as np

class PFCElastic:
    
    def apply_strain(self, eps):

        self.L = self.L0 * (1 + eps)

        self.dx = self.L / self.N

        self._build_kspace()
        
    def save_reference_state(self):
        self.phi_ref = self.phi.copy()
        self.L_ref = self.L
        self.dx_ref = self.dx


    def elastic_energy_curve(self, strain_list, relax_steps=2000):
        
        phi_ref = self.phi.copy()
        L_ref = self.L
        dx_ref = self.dx
        
        energy = []
        phi_list = []

        for eps in strain_list:

            self.phi = phi_ref.copy()
            self.L = L_ref
            self.dx = dx_ref
            
            self._build_kspace()

            self.apply_strain(eps)

            for _ in range(relax_steps):
                self.step()

            energy.append(
                self.compute_energy()
            )

            phi_list.append(
                self.phi.copy()
            )

        return (
            np.array(energy),
            phi_list
        )
    
    def fit_elastic_constant(self, strain, energy):
        
        coef = np.polyfit(strain, energy, 2)
        strain_fit = np.linspace(strain.min(), strain.max(), 200)
        energy_fit = np.polyval(coef, strain_fit)
        C = 2 * coef[0]
        eps_r = (-coef[1]) / (2 * coef[0])
        return C, eps_r, coef, strain_fit, energy_fit
    
    def compute_stress(
        self,
        strain,
        energy
    ):

        stress = np.gradient(
            energy,
            strain
        )

        return stress
    
    