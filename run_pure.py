from config import input_pfc_parameters
from pfc_pure import PurePFCSolver


def main():
    params = input_pfc_parameters()
    solver = PurePFCSolver(
        **params["solver"],
        lattice_type=params["lattice_type"]
    )

    solver.run()
    solver.postprocess()
    solver.analyze_psi6()


if __name__ == "__main__":

    main()