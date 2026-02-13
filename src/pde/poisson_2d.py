import numpy as np
import matplotlib.pyplot as plt
import scipy
import jax
from .pde_base import PDE


class Poisson2D(PDE):
    """
    A class representing the 2D Poisson equation with Dirichlet boundary conditions.

    Attributes:
        domain_bounds (list of tuple): The spatial domain boundaries for the problem.
    """

    def __init__(self, domain_bounds=[(-1, 1), (-1, 1)]):
        """
        Initializes the Poisson2D equation with the given domain bounds.

        Args:
            domain_bounds (list of tuple, optional): The boundaries of the domain. Defaults to [(-1, 1), (-1, 1)].
        """
        super().__init__(domain_bounds)
        self.params = None

    def set_params(self, params):
        self.params = params

    def source(self, r, x):
        """
        Defines the source term f(x) in the Poisson equation.

        Args:
            r (ndarray): Coefficients for the source term.
            x (ndarray): Spatial coordinates.

        Returns:
            float: The computed source term value at x.
        """
        x = x.reshape([1, -1]) * np.ones([r.shape[0], x.shape[0]])
        results = r[:, 2] * np.exp(
            -((x[:, 0] - r[:, 0]) ** 2 + (x[:, 1] - r[:, 1]) ** 2)
        )
        return results.sum()

    def dirichlet_boundary(self, r, x):
        """
        Defines the Dirichlet boundary condition u(x).

        Args:
            r (ndarray): Boundary coefficients.
            x (ndarray): Spatial coordinates.

        Returns:
            float: Boundary value at x.
        """
        theta = np.arctan2(x[1], x[0])

        return (
            r[0]
            + r[1] / 4 * np.cos(theta)
            + r[2] / 4 * np.sin(theta)
            + r[3] / 4 * np.cos(2 * theta)
            + r[4] / 4 * np.sin(2 * theta)
        ).sum()

    def neumann_boundary(self, x):
        return

    def sample_boundary_points(self, key, n_points, params):
        """
        Samples points on the boundary of the domain.

        Args:
            key (jax.random.PRNGKey): Random key for reproducibility.
            n_points (int): Number of boundary points to sample.
            params (tuple): Contains problem parameters, including geometric parameters.

        Returns:
            ndarray: Array of sampled boundary points.
        """
        _, _, geo_params = params
        c1, c2 = geo_params
        theta = np.linspace(0.0, 2 * np.pi, n_points)
        theta += jax.random.uniform(
            key, minval=0.0, maxval=(2 * np.pi / n_points), shape=(n_points,)
        )
        r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)
        x = r0 * np.cos(theta)
        y = r0 * np.sin(theta)

        shuffled_indices = jax.random.permutation(key, n_points)
        points = np.stack([x, y], axis=1)
        points = points[shuffled_indices]

        return points

    def is_in_hole(self, xy, geo_params, tol=1e-7):
        """
        Checks whether a given point lies within a hole in the domain.

        Args:
            xy (ndarray): Spatial coordinates to check.
            geo_params (tuple): Geometric parameters defining the hole.
            tol (float, optional): Tolerance for boundary inclusion. Defaults to 1e-7.

        Returns:
            bool: True if the point is inside the hole, False otherwise.
        """
        c1, c2 = geo_params
        theta = np.arctan2(*xy[:2])
        length = np.linalg.norm(xy[:2])
        r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)
        return r0 < length + tol

    def sample_domain_points(self, key, n_points, params):
        """
        Samples points inside the domain while avoiding holes.

        Args:
            key (jax.random.PRNGKey): Random key for reproducibility.
            n_points (int): Number of points to sample.
            params (tuple): Contains problem parameters, including geometric parameters.

        Returns:
            ndarray: Array of sampled domain points.
        """
        _, _, geo_params = params
        k1, k2, k3 = jax.random.split(key, 3)

        # Generate a large pool of candidate points
        n_x, n_y = 10 * n_points, 10 * n_points
        xs = jax.random.uniform(k1, (n_x,), minval=-1.0, maxval=1.0)
        ys = jax.random.uniform(k2, (n_y,), minval=-1.0, maxval=1.0)
        xy = np.stack((xs.flatten(), ys.flatten()), axis=1)

        # Filter points that are inside the hole
        in_hole = self.is_in_hole(xy, geo_params)

        # Randomly select valid points
        idxs = jax.random.choice(k3, xy.shape[0], replace=False, shape=(n_points,))
        return xy[idxs]

    def plot_solution(
        self,
        values_domain,
        values_boundary,
        points_domain,
        points_boundary,
        save_path="./solution.png",
    ):
        """
        Plots and saves the solution of the Poisson equation.

        Args:
            values_domain (ndarray): Computed values at domain points.
            values_boundary (ndarray): Computed values at boundary points.
            points_domain (ndarray): Sampled points in the domain.
            points_boundary (ndarray): Sampled points on the boundary.
            save_path (str, optional): File path to save the plot. Defaults to './solution.png'.
        """
        points = np.concatenate([points_domain, points_boundary], axis=0)
        values = np.concatenate([values_domain, values_boundary], axis=0)

        x, y, z = points[:, 0], points[:, 1], values
        xi = np.linspace(min(x), max(x), 2048)
        yi = np.linspace(min(y), max(y), 2048)

        c1, c2 = self.params[2]
        X, Y = np.meshgrid(xi, yi)
        theta = np.arctan2(X, Y)
        length = np.sqrt(X**2 + Y**2)
        r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)

        # Mask points inside the hole
        mask = r0 < length + 1e-7
        Z = scipy.interpolate.griddata((x, y), z, (X, Y), method="cubic")
        Z[mask] = -1  # Assign a special value for holes
        Z[np.isnan(Z)] = -1  # Handle NaNs

        # Normalize data for visualization
        Z_normalized = (255 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))).astype(
            np.uint8
        )

        # Save the solution plot
        plt.imsave(save_path, Z)
