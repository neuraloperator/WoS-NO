import cv2
import numpy as np
import matplotlib.pyplot as plt
import zombie
import zombie_bindings
import trimesh
import jax.numpy as jnp
import random as rd
from jax import random
import math

dirichletFreq = 1.5
diffusionFreq = 0.5


def diffusion(x, gradient=False, laplacian=False):
    """Computes the diffusion coefficient and optionally its gradient and Laplacian."""
    a = 4.0 * np.pi * diffusionFreq
    b = 3.0 * np.pi * diffusionFreq
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


def solution(x):
    a = np.pi * dirichletFreq
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
        self, mesh, bounds=None, boundary_threshold=0.05, n_points=100, n_iters=100
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
            boundary_mask = jnp.abs(sdf_values) < boundary_threshold
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
    "solverType": "bvc_dt",
    "solver": {
        "solverType": "bvc_dt",
        "boundaryCacheSize": 50000,
        "domainCacheSize": 50000,
        "nWalksForCachedSolutionEstimates": 1025,
        "nWalksForCachedGradientEstimates": 1025,
        "maxWalkLength": 1024,
        "epsilonShell": 1e-3,
        "epsilonShellForAbsorbingBoundary": 1e-3,
        "minStarRadius": 1e-3,
        "radiusClampForKernels": 0,
        "ignoreDirichlet": False,
        "ignoreNeumann": True,
        "ignoreAbsorbingBoundaryContribution": False,
        "ignoreReflectingBoundaryContribution": True,
        "ignoreSource": False,
        "nWalks": 1024,
        "setpsBeforeApplyingTikhonov": 1024,
        "setpsBeforeUsingMaximalSpheres": 1024,
        "disableGradientAntitheticVariates": False,
        "disableGradientControlVariates": False,
        "useCosineSamplingForDirectionalDerivatives": False,
        "silhouettePrecision": 1e-3,
        "russianRouletteThreshold": 0.8,
        "splittingThreshold": 1.2,
        "useFiniteDifferencesForBoundaryDerivatives": True,
    },
    "modelProblem": {
        "geometry": "spot_normalized.obj",
        # "boundary": '/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/examples/karman/geometry_1cyl_long_open.obj',
        "absorptionCoeff": 0.0,
        "normalizeDomain": False,
        "flipOrientation": False,
        "isDoubleSided": False,
        "isWatertight": False,
        "domainIsWatertight": True,
        "useSdfForAbsorbingBoundary": False,
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
    "/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/bindings/zombie_3d_delta/spot_normalized.obj"
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


trimeshobj = ShapeNetMesh()
trimeshobj.load_mesh(
    "/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/bindings/zombie_3d_delta/spot_normalized.obj",
    normalize_vertices=False,
)
# trimeshobj.mesh.export('spot_normalized.obj')
# exit()

domain_pts = np.asarray(
    trimeshobj.sample_domain_points(trimeshobj.mesh, n_points=1024, n_iters=100)
)
# boundary_pts = np.asarray(trimeshobj.sample_boundary_points(trimeshobj.mesh, n_points=1024, n_iters=10))
# boundary_pts = trimeshobj.mesh.vertices
points = np.vstack([domain_pts, points.numpy()])
# torch.save(torch.from_numpy(points), "points.pt")
# points = torch.load('points.pt').numpy()
# points = np.vstack([points, boundary_pts])
# points = domain_pts
# points = boundary_pts
# points = points.numpy()
print(points.shape)
print(points.max(axis=0), points.min(axis=0))

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

scene = zombie.Scene3D(
    sceneConfig,
    "../../bindings/zombie_3d_delta/",
    np.array([0.0], dtype=np.float32),
    np.array([0.0], dtype=np.float32),
    np.array([0.0], dtype=np.float32),
    np.array([0.0], dtype=np.float32),
)

# print(points


samples, p_arr, grad_arr = zombie.zombie3d(scene, solverConfig, outputConfig, points)


# print(p_arr)
print(len(points))

print(len(p_arr))
print(p_arr[:10])
gt = []
for point in points:
    gt.append(solution(point))
gt = np.array(gt)
print(gt[:10])
print(points[:10])


from torch.nn import MSELoss

loss_fn = MSELoss()
loss = loss_fn(torch.tensor(p_arr), torch.tensor(gt))
print("Loss: ", loss.item())
exit()
solverConfig["solverType"] = "dtrack"
samples, p_arr_dt, grad_arr = zombie.zombie3d(scene, solverConfig, outputConfig, points)
from torch.nn import MSELoss

loss_fn = MSELoss()
loss = loss_fn(torch.tensor(p_arr_dt), torch.tensor(gt))
print("DeltaTack Loss: ", loss.item())
print(p_arr[:10], p_arr_dt[:10])
print("Running Wiremesh version...")

bound_verts = torch.from_numpy(np.asarray(mesh.vertices))
vertices_sorted = bound_verts[
    np.lexsort((bound_verts[:, 2], bound_verts[:, 1], bound_verts[:, 0]))
]
vertices_u = []
geometry_string = []
edges_u = []
for ind, (x, y, z) in enumerate(vertices_sorted):

    geometry_string.append(f"v {x} {y} {z}\n")
    vertices_u.append([x, y, z])
for segment in range(len(vertices_sorted) - 2):

    geometry_string.append(f"l {segment+2} {segment+1}\n")
    edges_u.append((segment, segment + 1))

# scene = zombie_bindings.Scene3DVar(sceneConfig, [0.0], [0.0], [0.0], [0.0], geometry_string)

# samples, p_arr,grad_arr = zombie_bindings.wost_3dvar(scene, solverConfig, outputConfig, samples)

# print(len(points))

# print(len(p_arr))
# print(p_arr[:10])
# gt = []
# for point in points:
#     gt.append(solution(point))
# gt = np.array(gt)
# print(gt[:10])


# from torch.nn import MSELoss
# loss_fn = MSELoss()
# loss = loss_fn(torch.tensor(p_arr), torch.tensor(gt))
# print("Loss: ", loss.item())
