from .wos_solver import ZombieSolver
from . import zombie_bindings as zombie_bindings
import yaml
import torch
import numpy as np


class WOSPoisson3DSolver(ZombieSolver):
    def __init__(self, config):
        super().__init__(config)

    def scene_setup(self, absr, diff, dirichlet, general, geometry):
        """
        Handle specific coefficients for the Poisson 2D problem.
        Instantiate the scene based on these coefficients.
        """

        self.scene = zombie_bindings.Scene3DVar(
            self.scene_config,
            list(absr),
            list(diff),
            list(dirichlet),
            list(general),
            list(geometry),
        )
        # Additional logic to set up the scene

    def run_solver(self, samples):
        """
        Iterate over the dataset and solve the PDE for each sample using the WOS solver.
        Update the solution tensor.
        """
        # print(type(self.solver_config))
        # print(type(self.output_config))
        # print(wost_data['solver'])
        # print(self.solver_config)
        pts, solution, gradient = zombie_bindings.wost_3dvar(
            self.scene, self.solver_config, self.output_config, samples
        )

        return solution, gradient

    def solve(self, points, coefs, geometry):
        absr = [1, 2]
        diff = [2, 3]
        dirichlet = [1, 3]
        general = [4, 6]
        self.scene_setup(absr, diff, dirichlet, general, geometry)
        p_arr, grad_arr = self.run_solver(points)

        return p_arr, grad_arr

    def define_source(self):
        """
        Define the source term for the Poisson 2D problem.
        """
        # Implement the specific source function
        pass

    def define_boundary(self):
        """
        Define the boundary conditions for the Poisson 2D problem.
        """
        # Implement the specific boundary function
        pass


if __name__ == "__main__":
    # Load the YAML file
    with open("../../../configs/zombie_poisson_3d.yaml", "r") as file:
        data = yaml.safe_load(file)
    solver = WOSPoisson3DSolver(data)
    sample = solver.dataset[0]
    samples, grad_arr = solver.solve(
        sample["points"], coefs=None, geometry=sample["geometry"]
    )
    loss = torch.nn.MSELoss()
    print(loss(torch.Tensor(samples), torch.Tensor(sample["solution"])))
