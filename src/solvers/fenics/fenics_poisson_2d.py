from .fenics_solver import FenicsSolver
import numpy as np
import jax
import scipy

from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

try:
    from pde import Poisson2D
except ImportError:
    from src.pde import Poisson2D

pde_2d = Poisson2D(domain_bounds=[[-1, 1], [-1, 1]])


class Poisson2DSolver(FenicsSolver):
    def __init__(self, resolution=32, boundary_points=96):
        super().__init__(resolution, boundary_points)
        self.poisson_pde = pde_2d

    def sample_points(self, params, n_points=1024):
        k1, k2 = jax.random.split(self.key)
        points_on_boundary = self.poisson_pde.sample_boundary_points(
            k1, n_points, params
        )
        points_in_domain = self.poisson_pde.sample_domain_points(k2, n_points, params)
        return points_in_domain, points_on_boundary

    def sample_params(self, seed=0):
        self.key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(self.key, 3)

        source_params = jax.random.normal(
            k1,
            shape=(
                2,
                3,
            ),
        )
        bc_params = jax.random.uniform(k2, minval=-1.0, maxval=1.0, shape=(5,))
        geo_params = jax.random.uniform(k3, minval=-0.2, maxval=0.2, shape=(2,))

        return source_params, bc_params, geo_params

    def point_theta(self, theta, c1, c2):
        r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)
        x = r0 * np.cos(theta)
        y = r0 * np.sin(theta)
        import fenics as fa

        return fa.Point(np.array([x, y]))

    def make_domain(self, c1, c2, n_points):
        thetas = np.linspace(0.0, 1.0, n_points, endpoint=False) * 2 * np.pi
        points = [self.point_theta(t, c1, c2) for t in thetas]
        import mshr

        return mshr.Polygon(points)

    def solve_poisson(self, seed=0, boundary_points=96):
        source_params, bc_params, geo_params = self.sample_params(seed)
        self.params = (source_params, bc_params, geo_params)
        self.poisson_pde.set_params(self.params)
        c1, c2 = geo_params
        domain = self.make_domain(c1, c2, self.boundary_points)
        self.generate_mesh(domain)
        import fenics as fa

        class Boundary(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        class BCExpression(fa.UserExpression):
            def eval(self, value, x):
                value[0] = np.array(pde_2d.dirichlet_boundary(bc_params, x))

            def value_shape(self):
                return ()

        class SourceExpression(fa.UserExpression):
            def eval(self, value, x):
                value[0] = np.array(pde_2d.source(source_params, x))

            def value_shape(self):
                return ()

        source_V = fa.interpolate(SourceExpression(), self.V)
        bc_V = fa.interpolate(BCExpression(), self.V)
        bc = fa.DirichletBC(self.V, bc_V, Boundary())
        F = (
            fa.inner(fa.grad(self.u), fa.grad(self.v)) * fa.dx
            + source_V * self.v * fa.dx
        )
        solution = self.solve_fenics(a=F, L=0, boundary_condition=bc)

        return solution

    def get_pointwise_solution(self, solution):
        points_domain, points_boundary = self.sample_points(self.params, n_points=1024)
        solution.set_allow_extrapolation(True)

        values_boundary = np.array([solution(x) for x in points_boundary])
        values_domain = np.array([solution(x) for x in points_domain])

        return values_boundary, values_domain, points_boundary, points_domain


if __name__ == "__main__":
    poisson = Poisson2DSolver(resolution=16)
    solution = poisson.solve_poisson(seed=15)
    bound_val, dom_val, bound_pt, dom_pt = poisson.get_pointwise_solution(solution)
    poisson.poisson_pde.plot_solution(dom_val, bound_val, dom_pt, bound_pt)
    print(poisson.sample_params())
