from .wos_solver import ZombieSolver
from . import zombie_bindings as zombie_bindings
import yaml
import torch
import numpy as np


class WOSPoisson2DSolverBVC(ZombieSolver):
    def __init__(self, config):
        super().__init__(config)

    def scene_setup(self, mu1, mu2, beta, r, geometry):
        """
        Handle specific coefficients for the Poisson 2D problem.
        Instantiate the scene based on these coefficients.
        """
        self.scene = zombie_bindings.Scene(
            self.scene_config, list(mu1), list(mu2), list(beta), list(r), list(geometry)
        )
        # Additional logic to set up the scene

    def run_solver(
        self, samples, boundarycache, domaincache, normcache, usecache=False
    ):
        """
        Iterate over the dataset and solve the PDE for each sample using the WOS solver.
        Update the solution tensor.
        """
        # print(type(self.solver_config))
        # print(type(self.output_config))
        # print(wost_data['solver'])
        # print(self.solver_config)

        pts, solution, gradient, bcache, dcache, ncache = zombie_bindings.bvc(
            self.scene,
            self.solver_config,
            self.output_config,
            samples,
            boundarycache,
            domaincache,
            normcache,
            usecache,
        )

        return pts, solution, gradient, bcache, dcache, ncache

    def solve(self, points, coefs, geometry, bcache, dcache, ncache, usecache=False):
        mu1 = coefs["mu_1"].detach().cpu().numpy().tolist()
        mu2 = coefs["mu_2"].detach().cpu().numpy().tolist()
        beta = coefs["beta"].detach().cpu().numpy().tolist()
        r = coefs["b"].detach().cpu().numpy().tolist()
        self.scene_setup(mu1, mu2, beta, r, geometry)
        samples, p_arr, grad_arr, bcache, dcache, ncache = self.run_solver(
            points, bcache, dcache, ncache, usecache
        )

        return samples, p_arr, grad_arr, bcache, dcache, ncache

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
    with open("../../../configs/zombie_poisson_2d.yaml", "r") as file:
        data = yaml.safe_load(file)
    solver = WOSPoisson2DSolver(data)
    sample = solver.dataset[0]
    samples, p_arr, grad_arr = solver.solve(
        sample["train_points_domain"], sample["coefs"], sample["geometry"]
    )
    loss = torch.nn.MSELoss()
    print(loss(torch.Tensor(samples), torch.Tensor(sample["train_values_domain"])))
