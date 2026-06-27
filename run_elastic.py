from pfc_pure import PurePFCSolver
import numpy as np

def main():
    
    solver = PurePFCSolver(
        N=256,
        L=128,
        r=-0.35,
        M=1.0,
        dt=0.05,
        T=1500,
        phi0=-0.25,
        noise_amp=0.01
    )

    solver.run()

    solver.save_reference_state()

    strain = np.linspace(
        -0.03,
        0.03,
        13
    )

    energy, phi_list = (
        solver.elastic_energy_curve(
            strain
        )
    )

    # F-ε
    solver.plot_elastic_curve(
        strain,
        energy
    )

    # σ-ε
    stress = solver.compute_stress(
        strain,
        energy
    )

    solver.plot_stress_strain(
        strain,
        energy
    )
    
    C, eps_r, _, _, _ = (
        solver.fit_elastic_constant(
            strain,
            energy
        )
    )
    
    print()
    print(f"Elastic constant: {C:.6e}")
    print(f"Residual strain: {eps_r:.6e}")
    
if __name__ == "__main__":
    main()