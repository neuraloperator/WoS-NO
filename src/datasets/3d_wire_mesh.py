import zombie_bindings
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.interpolate
import os
import sympy as sp
import math
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from point_sampling_3d import PointSampler3D


def plot_point_cloud_with_values(
    points, values, filename="point_cloud_with_values.png"
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


def plot_mesh_with_values(mesh, values, filename="mesh_with_values.png"):
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


def dirichlet(x):
    k = np.pi * dirichlet_freq
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


dirichlet_freq = 1.5


# 2. Diffusion Coefficient
def diffusion(x, gradient=False, laplacian=False):
    """Computes the diffusion coefficient and optionally its gradient and Laplacian."""
    a = 4.0 * np.pi * diffusion_freq
    b = 3.0 * np.pi * diffusion_freq
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
    if laplacian == True and gradient == True:
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


absorption_min = 10.0
absorption_max = 100.0
diffusion_freq = 0.5
dirichlet_freq = 1.5


def absorption(x):
    """Computes the absorption coefficient."""
    return absorption_min + (absorption_max - absorption_min) * (
        1.0 + 0.5 * np.sin(2.0 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1])
    )


# 4. Source Function
def source(x):
    """Computes the source function as in the original C++ logic."""
    a = np.pi * dirichlet_freq
    alpha, gradient = diffusion(x, gradient=True)
    sigma = absorption(x)

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


wost_data = {
    "solverType": "wost",
    "solver": {
        "boundaryCacheSize": 200,
        "domainCacheSize": 200,
        "nWalksForCachedSolutionEstimates": 1025,
        "nWalksForCachedGradientEstimates": 1025,
        "maxWalkLength": 1024,
        "epsilonShell": 1e-2,
        "minStarRadius": 1e-2,
        "radiusClampForKernels": 0,
        "ignoreDirichlet": False,
        "ignoreNeumann": True,
        "ignoreSource": False,
        "nWalks": 1024,
        "setpsBeforeApplyingTikhonov": 1024,
        "setpsBeforeUsingMaximalSpheres": 1024,
        "disableGradientAntitheticVariates": False,
        "disableGradientControlVariates": False,
        "useCosineSamplingForDirectionalDerivatives": False,
        "silhouettePrecision": 1e-3,
        "russianRouletteThreshold": 0.99,
    },
    "scene": {
        "boundary": "/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/src/3d/geometry.obj",
        # "boundary": '/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/examples/karman/geometry_1cyl_long_open.obj',
        "absorptionCoeff": 0.0,
        "normalizeDomain": False,
        "flipOrientation": False,
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


def analytic_solution(point):
    a = np.pi * 1.5
    b = 2.0 * a
    c = 3.0 * a
    sin_ax = np.sin(a * point[0])
    cos_ax = np.cos(a * point[0])
    sin_by = np.sin(b * point[1])
    cos_by = np.cos(b * point[1])
    sin_cz = np.sin(c * point[2])
    cos_cz = np.cos(c * point[2])

    g = sin_ax * cos_by + (1.0 - cos_ax) * (1.0 - sin_by) + sin_cz**2
    return g


# input_data = torch.load('metawos_large.pt')
files = sorted(
    list(os.listdir("/home/hviswan/Documents/surface_modelling/3D/Shapenet"))
)
losses = []
counter = 0
tensor = []
rejected_counter = 0
filecounter = 0
for file in files:
    filecounter += 1
    filename = "wos_input_tensors/" + file
    fileid = file.split(".")[0] + ".obj"
    fi = fileid.split(".")[0] + ".pt"

    if fi in list(os.listdir("MC_GINO_INPUT_TENSORS")):
        print(f"{filecounter} FILE ALREADY IN DIR. CONTINUING")
        ten = torch.load(f"MC_GINO_INPUT_TENSORS/{fi}")
        tensor.append(ten)
        continue
    obj_path = "/home/hviswan/Documents/surface_modelling/3D/Shapenet/" + fileid
    sampler = PointSampler3D()
    mesh = sampler.load_mesh(obj_path)
    if filecounter < 2911:
        continue

    if mesh.vertices.shape[0] < 1500:
        continue
    try:
        print(f"FILE {filecounter} SAMPLING DOMAIN")
        domain_points = sampler.generate_domain_points(n_points=1024, n_iters=5)
        print(f"FILE {filecounter} SAMPLING BOUNDARY")
        boundary_points = sampler.generate_boundary_points(n_points=1024, n_iters=10)
    except:
        continue
    # boundary_points = sampler.generate_boundary_points(n_points=1024)
    print(domain_points.shape)

    # print(boundary_points.shape)
    print(mesh.vertices.shape)
    if domain_points.shape[0] < 100:
        continue
    if domain_points.shape[0] < 450 and boundary_points.shape[0] < (1024 - 450):
        continue
    # input_instance = torch.load(filename)
    loss = torch.nn.MSELoss()
    nwalks = [1e3]
    counter += 1

    for walk in nwalks:
        wost_data["solver"]["nWalks"] = walk
        # print(wost_data['solver'])
        instance_losses = 0
        input_instance = {}

        boundary_points = np.asarray(boundary_points)

        sceneConfig = wost_data["scene"]
        sceneConfig["sourceValue"] = (
            "/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/src/3d/sourceterm.png"
        )
        sceneConfig["boundary"] = (
            f"/home/hviswan/Documents/Neural-Monte-Carlo-Fluid-Simulation/src/3d/geometry_{0}.obj"
        )
        solverConfig = wost_data["solver"]
        outputConfig = wost_data["output"]
        absorptionc = [1, 2]
        diffusionc = [1, 2, 3]
        dirichletc = [1, 2, 3]
        generalc = [1, 2, 3]
        geometry_string = []
        vertices_sorted = boundary_points[
            np.lexsort(
                (boundary_points[:, 2], boundary_points[:, 1], boundary_points[:, 0])
            )
        ]
        vertices_u = []
        edges_u = []
        for ind, (x, y, z) in enumerate(vertices_sorted):

            geometry_string.append(f"v {x} {y} {z}\n")
            vertices_u.append([x, y, z])
        for segment in range(len(vertices_sorted) - 2):

            geometry_string.append(f"l {segment+2} {segment+1}\n")
            edges_u.append((segment, segment + 1))

        scene = zombie_bindings.Scene3DVar(
            sceneConfig,
            list(absorptionc),
            list(diffusionc),
            list(dirichletc),
            list(generalc),
            list(geometry_string),
        )
        samples = torch.vstack(
            (
                torch.from_numpy(np.asarray(domain_points).astype(np.float32)),
                torch.from_numpy(np.asarray(boundary_points)),
            )
        )

        idx = torch.randperm(samples.shape[0])

        samples = samples[idx]

        samples = samples.numpy().tolist()

        loss = torch.nn.MSELoss()
        ls = 0.0

        samples, p_arr, grad_arr = zombie_bindings.wost_3dvar(
            scene, solverConfig, outputConfig, samples
        )

        p_arr = torch.from_numpy(np.asarray(p_arr).astype(np.float32))
        # print(list(p_arr.numpy()))
        # gt = torch.load('VALUES_GT.pt')[:1024]

        # print(list(gt.numpy()))
        p = list(p_arr.numpy())

        gt = []
        preds_domain = []
        for i in range(len(samples)):
            # g = solution[i]

            g = analytic_solution(samples[i])
            diff = abs(p[i] - g)
            preds_domain.append(p[i])
            gt.append(g)
        gt = torch.from_numpy(np.asarray(gt).astype(np.float32))

        preds_domain = torch.from_numpy(np.asarray(preds_domain).astype(np.float32))

        points = torch.from_numpy(np.asarray(samples).astype(np.float32))
        if loss(preds_domain, gt).item() < 0.01:
            print(
                "SHAPES: ",
                preds_domain.shape,
                len(samples),
                " LOSS: ",
                loss(preds_domain, gt).item(),
            )
            losses.append(loss(preds_domain, gt).item())
            input_instance["source_terms"] = np.asarray(
                [source(point) for point in samples]
            )
            input_instance["diffusion_term"] = np.asarray(
                [diffusion(point) for point in samples]
            )
            input_instance["absorption_term"] = np.asarray(
                [absorption(point) for point in samples]
            )

            input_instance["solution"] = gt
            input_instance["geometry"] = geometry_string
            input_instance["points"] = samples
            train_distances = []
            train_bc = []
            for point in samples:
                x = point[0]
                y = point[1]
                z = point[2]
                distances = np.sqrt(
                    np.square(boundary_points[:, 0] - x)
                    + np.square(boundary_points[:, 1] - y)
                    + np.square(boundary_points[:, 2] - z)
                )
                min_distance = np.min(distances)
                min_index = np.where(distances == min_distance)[0][0]
                closest_boundary_point = (
                    boundary_points[min_index, 0],
                    boundary_points[min_index, 1],
                    boundary_points[min_index, 2],
                )
                train_distances.append(min_distance)
                train_bc.append(analytic_solution(closest_boundary_point))
            input_instance["dist"] = train_distances
            input_instance["closest_bc"] = train_bc
            fi = fileid.split(".")[0] + ".pt"
            torch.save(input_instance, "MC_GINO_INPUT_TENSORS/" + fi)
            tensor.append(input_instance)
            samples = np.asarray(samples)

        else:
            print(loss(preds_domain, gt))
            print(preds_domain)
            print(gt)

            print(
                f"Counter: {counter}, REJECTED SHAPE: ",
                " VERTICES: ",
                points.shape[0],
                " BOUNDARY POINTS: ",
                boundary_points.shape,
                " Rejected Counter: ",
                rejected_counter + 1,
                " Average Loss: ",
                sum(losses) / (len(losses) + 0.0004),
            )
            rejected_counter += 1
print("WRITING TO OBJ")
with open("linear_poisson_3d_gino.obj", "wb") as f:
    pickle.dump(tensor, f)
print("WRITTEN TO OBJ")
