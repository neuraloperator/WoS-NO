from abc import ABC, abstractmethod
import numpy as np


class PDE(ABC):
    def __init__(self, domain_bounds):
        """
        Initialize PDE with domain boundaries.
        :param domain_bounds: Tuple specifying the domain range (e.g., [(xmin, xmax), (ymin, ymax)] for 2D).
        """
        self.domain_bounds = domain_bounds

    @abstractmethod
    def source(self, x):
        """Defines the source term f(x) in the PDE."""
        pass

    @abstractmethod
    def dirichlet_boundary(self, x):
        """Defines the Dirichlet boundary condition u(x) on the boundary."""
        pass

    @abstractmethod
    def neumann_boundary(self, x):
        """Defines the Neumann boundary condition u(x) on the boundary."""
        pass

    def sample_domain_points(self):
        """Samples points within the domain."""
        raise NotImplementedError("Must be implemented in subclass")

    def sample_boundary_points(self):
        """Samples points on the boundary of the domain."""
        raise NotImplementedError("Must be implemented in subclass")

    def plot_solution(self, solution_func, resolution=50):
        """Plots the solution for visualization."""
        raise NotImplementedError("Must be implemented in subclass")
