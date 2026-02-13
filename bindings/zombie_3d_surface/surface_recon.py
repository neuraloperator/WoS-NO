import cv2
import numpy as np
import matplotlib.pyplot as plt
import zombie
import trimesh
import jax.numpy as jnp
import random as rd
from jax import random


class ShapeNetMesh:
    def __init__(self):
        self.faces = None
        self.vertices = None
        self.mesh = None

    def load_mesh(self, mesh_path, normalize_vertices=True):
        """
        Loads a mesh from a file and initializes vertices and faces.
        Can be called multiple times for different meshes.

        Args:
            mesh_path (str): Path to the mesh file.
        """
        mesh = trimesh.load(mesh_path)
        if hasattr(mesh, "geometry"):
            combined_mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            mesh = combined_mesh
        self.mesh = mesh

        self.vertices = mesh.vertices
        if normalize_vertices:
            # self.vertices = (self.vertices - self.vertices.min())/(self.vertices.max()-self.vertices.min())
            vertices = mesh.vertices
            mesh.vertices -= vertices.min()
            # mesh.vertices *= 2
            mesh.vertices /= vertices.max() - vertices.min()
            # mesh.vertices *=2
            # mesh.vertices -=1
            self.vertices = mesh.vertices
        self.faces = mesh.faces
        return self.mesh

    def get_mesh(self):
        """
        Returns the current mesh object.

        Returns:
            Trimesh.TriangleMesh: The current mesh.
        """
        return self.mesh

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
        self, mesh, bounds=None, boundary_threshold=0.001, n_points=100, n_iters=100
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


wost_data = {
    "solverType": "wost",
    "solver": {
        "boundaryCacheSize": 500,
        "domainCacheSize": 500,
        "nWalksForCachedSolutionEstimates": 1025,
        "nWalksForCachedGradientEstimates": 1025,
        "maxWalkLength": 1500,
        "epsilonShell": 1e-2,
        "minStarRadius": 1e-2,
        "radiusClampForKernels": 0,
        "ignoreDirichlet": False,
        "ignoreNeumann": False,
        "ignoreAbsorbingBoundaryContribution": False,
        "ignoreReflectingBoundaryContribution": False,
        "ignoreSource": False,
        "nWalks": 1024,
        "setpsBeforeApplyingTikhonov": 1024,
        "setpsBeforeUsingMaximalSpheres": 1024,
        "disableGradientAntitheticVariates": False,
        "disableGradientControlVariates": False,
        "useCosineSamplingForDirectionalDerivatives": False,
        "silhouettePrecision": 1e-3,
        "russianRouletteThreshold": 0.99,
        "useFiniteDifferencesForBoundaryDerivatives": False,
    },
    "modelProblem": {
        "geometry": "bunny_norm.obj",
        # "boundary": '/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/examples/karman/geometry_1cyl_long_open.obj',
        "absorptionCoeff": 0.0,
        "normalizeDomain": False,
        "flipOrientation": True,
        "isDoubleSided": False,
        "isWatertight": False,
    },
    "output": {
        "solutionFile": "./solutions/wost.png",
        "txtdir": "./solutions/",
        "gridRes": 300,
        "boundaryDistanceMask": 1e-3,
        "saveDebug": True,
        "saveColormapped": True,
        "colormap": "viridis",
        "colormapMinVal": 0.0,
        "colormapMaxVal": 1.0,
    },
}
import torch
import open3d as o3d

mesh = o3d.io.read_triangle_mesh(
    "/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/bindings/zombie_3d_surface/bunny_norm.obj"
)
# vertices = np.asarray(mesh.vertices)
# min_bound = vertices.min(axis=0)
# max_bound = vertices.max(axis=0)
# vertices -= min_bound  # shift min to 0
# scale = 1 / (max_bound - min_bound).max()
# vertices *= scale
# vertices *= 2
# vertices -= 1
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# # Extract edges from triangles
# lines = []
# for tri in mesh.triangles:
#     lines.append([tri[0], tri[1]])
#     lines.append([tri[1], tri[2]])
#     lines.append([tri[2], tri[0]])

# line_set = o3d.geometry.LineSet(
#     points=mesh.vertices,
#     lines=o3d.utility.Vector2iVector(lines)
# )

# o3d.visualization.draw_geometries([mesh, line_set])


mesh.compute_vertex_normals()

points = torch.from_numpy(np.asarray(mesh.vertices))
normals = torch.from_numpy(np.asarray(mesh.vertex_normals))
print(points.shape)
print(normals.shape)
print(normals)

# Step 2: Define the grid in [0, 1]^3
N = 20
lin = torch.linspace(0, 1, N)
grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing="ij")
grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # Shape: [N, N, N, 3]
divergence_grid = torch.zeros((N, N, N))
grid_points_flat = grid_points.reshape(-1, 3).requires_grad_(True)  # [N^3, 3]


# Step 3: Define vector field using RBF
def vector_field(x, sample_points, normals, sigma=0.1):
    diff = x[:, None, :] - sample_points[None, :, :]  # [Q, M, 3]
    r2 = (diff**2).sum(dim=-1)  # [Q, M]
    weights = torch.exp(-r2 / (sigma**2))  # [Q, M]
    V = (weights[..., None] * normals[None, :, :]).sum(dim=1)  # [Q, 3]
    return V


V = vector_field(grid_points_flat, points, normals)

# Step 4: Compute divergence using autograd
divergence = torch.zeros(grid_points_flat.shape[0])
for i in range(3):
    grad = torch.autograd.grad(
        V[:, i],
        grid_points_flat,
        grad_outputs=torch.ones_like(V[:, i]),
        create_graph=False,
        retain_graph=True,
    )[0]
    divergence += grad[:, i]

print(divergence.shape)
print(V.shape)
print(grid_points_flat.shape)
print(divergence_grid.shape)

for i in range(len(grid_points_flat)):
    point = grid_points_flat[i]
    # print(point)
    index = torch.round(point * 19).to(torch.int32)
    # print(index)
    divergence_grid[index[0].item()][index[1].item()][index[2].item()] = divergence[
        i
    ].item()

grid = divergence.view(N, N, N)
# print(divergence_grid)
# print(grid)
# print(divergence_grid[0][0][0])
# print(grid[0][0][0])


# Step 5: Function to map world coordinates to grid indices
def world_to_grid(xyz, N):
    indices = (xyz * (N - 1)).long()
    indices = torch.clamp(indices, 0, N - 1)
    return indices


# Example: map [0.5, 0.5, 0.5] to grid index
test_point = torch.tensor([[0.5, 0.5, 0.5]])
grid_index_example = world_to_grid(test_point, N)


trimeshobj = ShapeNetMesh()
trimeshobj.load_mesh(
    "/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/src/3d/bunny_norm.obj",
    normalize_vertices=False,
)
# trimeshobj.mesh.export('bunny_normalized.obj')
# exit()

domain_pts = np.asarray(
    trimeshobj.sample_domain_points(trimeshobj.mesh, n_points=100, n_iters=100)
)
boundary_pts = np.asarray(
    trimeshobj.sample_boundary_points(trimeshobj.mesh, n_points=100, n_iters=100)
)
# boundary_pts = trimeshobj.mesh.vertices
closest_points, distances, triangle_indices = trimeshobj.mesh.nearest.on_surface(
    grid_points_flat.detach().cpu().numpy()
)
signs = (
    np.array(
        trimeshobj.mesh.contains(grid_points_flat.detach().cpu().numpy()),
        dtype=jnp.float32,
    )
    * (-2)
    + 1
)
sdf = distances * signs
domain_mask = sdf < 0
boundary_mask = np.abs(sdf) <= 0.001
dpts = grid_points_flat.detach().cpu().numpy()[domain_mask]
bpts = grid_points_flat.detach().cpu().numpy()[boundary_mask]
points = np.vstack([domain_pts, points, dpts, bpts])
print(points.shape)
print(points.max(), points.min())

sceneConfig = wost_data["modelProblem"]
sceneConfig["sourceValue"] = (
    "/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/src/3d/sourceterm.png"
)
sceneConfig["isReflectingBoundary"] = "data/is_reflecting_boundary.pfm"
sceneConfig["absorbingBoundaryValue"] = "data/absorbing_boundary_value.pfm"
sceneConfig["reflectingBoundaryValue"] = "data/reflecting_boundary_value.pfm"
sceneConfig["sourceValue"] = "data/source_value.pfm"
solverConfig = wost_data["solver"]
outputConfig = wost_data["output"]

print(divergence_grid.max(), divergence_grid.min(), divergence_grid.mean())


scene = zombie.Scene3D(
    sceneConfig,
    "../../bindings/zombie_3d_surface/",
    divergence_grid.detach().cpu().numpy(),
)
# print(points

samples, p_arr, grad_arr = zombie.zombie3d(scene, solverConfig, outputConfig, points)

# print(p_arr)
print(len(points))

print(len(p_arr))


sdf = np.asarray(p_arr) * -1
print(sdf)
from scipy.interpolate import griddata

grid_x, grid_y, grid_z = np.meshgrid(
    np.linspace(0, 1, N), np.linspace(0, 1, N), np.linspace(0, 1, N), indexing="ij"
)
grid_coords = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)  # (N^3, 3)

from scipy.spatial import cKDTree

tree = cKDTree(points)
dist, idx = tree.query(grid_coords)
radius = 0.01  # or tune this based on mesh size
sdf_grid_flat = np.where(
    dist < radius, sdf[idx], 1.0  # fill value for points far from surface
)
sdf_grid = sdf_grid_flat.reshape(N, N, N)
print(sdf_grid)
from skimage.measure import marching_cubes

# Extract surface where sdf = 0
level = 0.5 * (np.nanmin(sdf_grid) + np.nanmax(sdf_grid))
verts, faces, normals, values = marching_cubes(sdf_grid, level=0.0)
import trimesh

mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
mesh.export("output.obj")
exit()
