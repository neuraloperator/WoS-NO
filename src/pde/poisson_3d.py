import numpy as np
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .pde_base import PDE
import math
import jax.numpy as jnp
import random as rd
from jax import random


class Poisson3D(PDE):
    """
    A class representing the 2D Poisson equation with Dirichlet boundary conditions.

    Attributes:
        domain_bounds (list of tuple): The spatial domain boundaries for the problem.
    """

    def __init__(
        self,
        domain_bounds=[(-1, 1), (-1, 1), (-1, 1)],
        dirichlet_freq=1.5,
        diffusion_freq=0.5,
        absorption_min=10.0,
        absorption_max=100.0,
    ):
        """
        Initializes the Poisson2D equation with the given domain bounds.

        Args:
            domain_bounds (list of tuple, optional): The boundaries of the domain. Defaults to [(-1, 1), (-1, 1)].
        """
        super().__init__(domain_bounds)
        self.dirichlet_freq = dirichlet_freq
        self.diffusion_freq = diffusion_freq
        self.absorption_min = absorption_min
        self.absorption_max = absorption_max

    def source(self, x):
        """
        Defines the source term f(x) in the Poisson equation.

        Args:
            x (ndarray): Spatial coordinates.

        Returns:
            float: The computed source term value at x.
        """
        a = np.pi * self.dirichlet_freq
        alpha, gradient = self.diffusion(x, gradient=True)
        sigma = self.absorption(x)

        b = 2.0 * a
        c = 3.0 * a
        sin_ax = np.sin(a * x[0])
        cos_ax = np.cos(a * x[0])
        sin_by = np.sin(b * x[1])
        cos_by = np.cos(b * x[1])
        sin_cz = np.sin(c * x[2])
        cos_cz = np.cos(c * x[2])

        u = sin_ax * cos_by + (1.0 - cos_ax) * (1.0 - sin_by) + sin_cz**2
        d2u_dx2 = (cos_ax * (1 - sin_by) - sin_ax * cos_by) * a * a
        d2u_dy2 = ((1 - cos_ax) * sin_by - sin_ax * cos_by) * b * b
        d2u_dz2 = 2 * (cos_cz * cos_cz - sin_cz * sin_cz) * c * c
        d2u = d2u_dx2 + d2u_dy2 + d2u_dz2

        du = np.asarray(
            [
                cos_ax * cos_by + sin_ax * (1 - sin_by) * a,
                -(sin_ax * sin_by + (1 - cos_ax) * cos_by) * b,
                2 * sin_cz * cos_cz * c,
            ]
        )
        return -alpha * d2u - gradient @ du + sigma * u

    def dirichlet_boundary(self, x):
        """
        Defines the Dirichlet boundary condition u(x).

        Args:
            df (float): Dirichlet Frequency.
            x (ndarray): Spatial coordinates.

        Returns:
            float: Boundary value at x.
        """
        k = np.pi * self.dirichlet_freq
        b = 2.0 * k
        c = 3.0 * k
        sinAx = math.sin(k * x[0])
        cosAx = math.cos(k * x[0])
        sinBy = math.sin(b * x[1])
        cosBy = math.cos(b * x[1])
        sinCz = math.sin(c * x[2])
        u = sinAx * cosBy + (1.0 - cosAx) * (1.0 - sinBy) + sinCz * sinCz
        u *= 1
        return u

    def analytic_expression(self, x):
        a = np.pi * self.dirichlet_freq
        b = 2.0 * a
        c = 3.0 * a
        sin_ax = np.sin(a * x[0])
        cos_ax = np.cos(a * x[0])
        sin_by = np.sin(b * x[1])
        cos_by = np.cos(b * x[1])
        sin_cz = np.sin(c * x[2])
        cos_cz = np.cos(c * x[2])

        g = sin_ax * cos_by + (1.0 - cos_ax) * (1.0 - sin_by) + sin_cz**2
        return g

    def diffusion(self, x, gradient=False, laplacian=False):
        """Computes the diffusion coefficient and optionally its gradient and Laplacian."""
        a = 4.0 * np.pi * self.diffusion_freq
        b = 3.0 * np.pi * self.diffusion_freq
        sin_ax = np.sin(a * x[0])
        cos_ax = np.cos(a * x[0])
        sin_by = np.sin(b * x[1])
        cos_by = np.cos(b * x[1])
        alpha = np.exp(-x[1] ** 2 + cos_ax * sin_by)
        if gradient == True and laplacian == False:
            gradient = [
                alpha * (-sin_ax * sin_by * a),
                alpha * (-2.0 * x[1] + cos_ax * cos_by * b),
                0,
            ]
            return alpha, np.asarray(gradient)
        elif laplacian == True and gradient == True:
            gradient = [
                alpha * (-sin_ax * sin_by * a),
                alpha * (-2.0 * x[1] + cos_ax * cos_by * b),
                0,
            ]
            d2alphadx2 = gradient[0] * (-sin_ax * sin_by * a) + alpha * (
                -cos_ax * sin_by * a * a
            )
            d2alphady2 = gradient[1] * (-2 * x[1] + cos_ax * cos_by * b) + alpha * (
                -2 - cos_ax * sin_by * b * b
            )
            lap = d2alphadx2 + d2alphady2
            return alpha, np.asarray(gradient), lap
        return alpha

    def absorption(self, x):
        """Computes the absorption coefficient."""
        return self.absorption_min + (self.absorption_max - self.absorption_min) * (
            1.0 + 0.5 * np.sin(2.0 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1])
        )

    def neumann_boundary(self, x):
        return

    def compute_sdf_bvh(self, mesh, points):
        """
        Compute the SDF of a set of points relative to a mesh using a BVH for efficiency.

        Args:
            mesh: A trimesh.Trimesh object.
            points: JAX array of shape [N, 3], points to compute SDF for.

        Returns:
            sdf: JAX array of shape [N], signed distances for each point.
        """
        if mesh is None:
            raise ValueError("Mesh is not loaded. Please load a mesh first.")
        # Build a BVH using trimesh
        closest_points, distances, triangle_indices = mesh.nearest.on_surface(points)

        # Determine the sign of the SDF (inside or outside the mesh)
        signs = (
            jnp.array(mesh.contains(points), dtype=jnp.float32) * (-2) + 1
        )  # Inside is -1, outside is +1

        # Compute signed distances
        sdf = distances * signs
        return sdf

    def compute_sdf(self, mesh, points):
        """
        Compute the signed distance function for a set of points given a mesh.

        Args:
            mesh: A trimesh.Trimesh object.
            points: JAX array of shape [N, 3].

        Returns:
            sdf: JAX array of shape [N], signed distances for each point.
        """
        if mesh is None:
            raise ValueError("Mesh is not loaded. Please load a mesh first.")
        return self.compute_sdf_bvh(mesh, points)
        # return compute_signed_distance(points, vertices, faces, mesh)

    def sample_points(self, key, n_points, bounds):
        """
        Sample N points in a 3D bounding box.

        Args:
            key: JAX random key.
            n_points: Number of points to sample.
            bounds: Tuple of (min_bound, max_bound), each a (3,) array.

        Returns:
            points: Array of shape [N, 3] containing sampled points.
        """
        min_bound, max_bound = bounds
        points = random.uniform(
            key, shape=(n_points, 3), minval=min_bound, maxval=max_bound
        )
        return points

    def sample_domain_points(self, mesh, bounds=None, n_points=100, n_iters=100):
        """
        Generates points inside the mesh domain.

        Returns:
            domain_points: Points inside the domain (SDF < 0).
            np.ndarray: Points inside the domain (inside the mesh).
        """
        domain_points = None
        if mesh is None:
            raise ValueError("Mesh is not loaded. Please load a mesh first.")
        if bounds is None:
            bounds = (jnp.array(mesh.bounds[0]), jnp.array(mesh.bounds[1]))
        iter = 0
        while (
            domain_points is None or domain_points.shape[0] < n_points
        ) and iter < n_iters:
            iter += 1
            key = random.PRNGKey(rd.randint(0, 52))
            points = self.sample_points(key, 2048, bounds)
            sdf_values = self.compute_sdf(mesh, points)
            domain_mask = sdf_values < 0
            if domain_points is None:
                domain_points = points[domain_mask]
            else:
                pts = points[domain_mask]
                domain_points = jnp.concatenate((domain_points, pts), axis=0)
                domain_points = jnp.unique(domain_points, axis=0)
            print("DOMAIN POINTS: ", domain_points.shape)
        if domain_points.shape[0] > n_points:
            domain_points = domain_points[:n_points]
        self.domain_points = domain_points
        return domain_points

    def sample_boundary_points(
        self, mesh, bounds=None, boundary_threshold=0.01, n_points=100, n_iters=100
    ):
        """
        Generates points inside the mesh domain.
        boundary_threshold: Distance threshold for identifying boundary points.

        Returns:
            boundary_points: Points near the boundary (|SDF| <= boundary_threshold).
            np.ndarray: Points inside the domain (inside the mesh).
        """
        boundary_points = None
        if mesh is None:
            raise ValueError("Mesh is not loaded. Please load a mesh first.")
        if bounds is None:
            bounds = (jnp.array(mesh.bounds[0]), jnp.array(mesh.bounds[1]))
        iter = 0
        while (
            boundary_points is None or boundary_points.shape[0] < n_points
        ) and iter < n_iters:
            iter += 1
            key = random.PRNGKey(rd.randint(0, 52))
            points = self.sample_points(key, 2048, bounds)
            sdf_values = self.compute_sdf(mesh, points)
            boundary_mask = jnp.abs(sdf_values) <= boundary_threshold
            if boundary_points is None:
                boundary_points = points[boundary_mask]
            else:
                pts = points[boundary_mask]
                boundary_points = jnp.concatenate((boundary_points, pts), axis=0)
                boundary_points = jnp.unique(boundary_points, axis=0)
            print("BOUNDARY POINTS: ", boundary_points.shape)
        if boundary_points.shape[0] > n_points:
            boundary_points = boundary_points[:n_points]
        self.boundary_points = boundary_points
        return boundary_points

    def plot_point_cloud_with_values(
        self, points, values, filename="point_cloud_with_values.png"
    ):
        """
        Plot a 3D point cloud with values defined over the points and save the plot.

        Args:
            points: (N, 3) array of point cloud coordinates.
            values: (N,) array of scalar values defined over the points.
            filename: File name to save the plot (default: "point_cloud_with_values.png").
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Normalize values for color mapping
        norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        # norm_values = np.asarray(values)
        colors = plt.cm.viridis(norm_values)  # Use a colormap (e.g., viridis)

        # Scatter plot
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            s=5,
            cmap="viridis",
            marker="o",
        )

        # Add color bar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label("Values")

        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("3D Point Cloud with Values")

        # Save and show the plot
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Point cloud visualization saved as {filename}")
        plt.show()

    def plot_mesh_with_values(self, mesh, values, filename="mesh_with_values.png"):
        """
        Plot a 3D mesh with values defined over the points and save it as a PNG.

        Args:
            mesh: A trimesh.Trimesh object.
            values: (N,) array of scalar values defined over the mesh vertices.
            filename: File name to save the plot.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Get the vertices and faces from the mesh
        vertices = mesh.vertices
        faces = mesh.faces

        # Normalize the values for color mapping
        norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        # norm_values = np.asarray(values)

        print(norm_values.shape)
        print(faces)

        # Create a Poly3DCollection for visualization
        face_colors = plt.cm.viridis(
            norm_values[faces].mean(axis=1)
        )  # Map average value of vertices in each face
        poly3d = Poly3DCollection(
            vertices[faces], facecolors=face_colors, edgecolor="k", linewidths=0.1
        )

        # Add the mesh to the plot
        ax.add_collection3d(poly3d)

        # Set limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("Mesh with Values")

        # Save and show the plot
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Mesh visualization saved as {filename}")
        # plt.show()

    def plot_solution(self, mesh, points, values, filename, plot_type):
        if plot_type == "point-cloud":
            self.plot_point_cloud_with_values(points, values, filename)
        elif plot_type == "mesh":
            self.plot_mesh_with_values(mesh, values, filename)

        else:
            raise Exception("Plot Type should be either 'point-cloud' or 'mesh'")
