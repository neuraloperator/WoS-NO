import torch
import numpy as np
import pickle
import os
import sys
import trimesh

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pde import Poisson3D
from solvers.wos import WOSPoisson3DSolver
import yaml


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
            mesh.vertices *= 2
            mesh.vertices -= 1
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


def generate_data(data_path="../data/Shapenet"):
    """
    Generates a dataset by solving the 2D Poisson equation for a given number of samples.

    Args:
        num_samples (int): Number of Poisson equation solutions to generate.

    Returns:
        list: A dataset containing solution values, source terms, boundary conditions, and geometry information.
    """
    files = os.listdir(data_path)
    if os.path.exists("../data/3Dtensors/") == False:
        os.mkdir(os.path.join("../data/", "3Dtensors"))
    tensor = []
    losses = []
    if os.path.exists("../data/rejected.pt") == False:

        rejected = []
    else:
        rej = torch.load("../data/rejected.pt", weights_only=False)
        rejected = rej["rejected"]
    rejected_counter = 0
    for id in range(len(files)):
        if "obj" not in files[id]:
            continue
        if files[id] in rejected:
            rejected_counter += 1
            print("Rejected Counter: ", rejected_counter)
            continue
        path = os.path.join(data_path, files[id])
        mesh_obj = ShapeNetMesh()
        fileid = files[id].split(".")[0] + ".obj"
        fi = fileid.split(".")[0] + ".pt"

        if fi in list(os.listdir("../data/3Dtensors")):
            print(f"{id} FILE ALREADY IN DIR. CONTINUING")
            ten = torch.load(f"../data/3Dtensors/{fi}", weights_only=False)
            tensor.append(ten)
            continue

        mesh = mesh_obj.load_mesh(path)

        poisson_3d_obj = Poisson3D(
            absorption_min=0, absorption_max=1.0, dirichlet_freq=1.5, diffusion_freq=0.2
        )
        if mesh.vertices.shape[0] > 40000:
            print("Skipping Mesh. Too many Vertices")
            rejected_counter += 1
            rejected.append(files[id])
            continue
        if (
            mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min() < 0.1
            or mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min() < 0.1
            or mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min() < 0.1
        ):
            print("Skipping Mesh. Too Narrow")
            rejected_counter += 1
            rejected.append(files[id])
            continue
        try:
            print("SAMPLING DOMAIN POINTS")
            domain_points = poisson_3d_obj.sample_domain_points(
                mesh, n_points=2048, n_iters=5
            )
            print(domain_points.shape)

            if domain_points.shape[0] > 2046:
                print("SAMPLING BOUNDARY POINTS")
                boundary_points = poisson_3d_obj.sample_boundary_points(
                    mesh, n_points=1024, n_iters=15
                )
        except:
            print("INSIDE EXCEPT")
            rejected_counter += 1
            rejected.append(files[id])
            continue

        print("Number of Vertices: ", mesh.vertices.shape)
        if domain_points.shape[0] < 2048:
            print("Rejected due to insufficient domain points")
            rejected_counter += 1
            rejected.append(files[id])
            continue
        if boundary_points.shape[0] < 1024:
            print("Rejected due to insufficient boundary points")
            rejected_counter += 1
            rejected.append(files[id])
            continue

        domain_vals = np.asarray(
            [poisson_3d_obj.analytic_expression(x) for x in domain_points]
        )
        bound_vals = np.asarray(
            [poisson_3d_obj.analytic_expression(x) for x in boundary_points]
        )
        gt_solution = np.concatenate((domain_vals, bound_vals), axis=0)
        mesh_solution = np.asarray(
            [poisson_3d_obj.analytic_expression(x) for x in mesh.vertices]
        )

        with open("../configs/zombie_poisson_3d.yaml", "r") as file:
            data = yaml.safe_load(file)
        wos_solver = WOSPoisson3DSolver(data)
        points = np.concatenate((domain_points, boundary_points), axis=0)
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
        wos_estimate, grad_arr = wos_solver.solve(
            points, coefs=None, geometry=geometry_string
        )
        wos_estimate = np.asarray(wos_estimate)
        loss_fn = torch.nn.MSELoss()
        vals = np.concatenate((domain_vals, bound_vals), axis=0)

        loss_value = loss_fn(
            torch.from_numpy(vals), torch.from_numpy(wos_estimate)
        ).item()
        if loss_value < 7e-3:
            input_instance = {}
            print("SHAPES: ", domain_vals.shape, len(points), " LOSS: ", loss_value)
            losses.append(loss_value)
            input_instance["source_terms"] = np.asarray(
                [poisson_3d_obj.source(point) for point in points]
            )
            input_instance["diffusion_term"] = np.asarray(
                [poisson_3d_obj.diffusion(point) for point in points]
            )
            input_instance["absorption_term"] = np.asarray(
                [poisson_3d_obj.absorption(point) for point in points]
            )
            print(
                "SOURCE MAX: ",
                input_instance["source_terms"].max(),
                input_instance["source_terms"].min(),
            )
            print(
                "DIFFUSION TERMS: ",
                input_instance["diffusion_term"].max(),
                input_instance["diffusion_term"].min(),
            )
            print(
                "ABSORPTION: ",
                input_instance["absorption_term"].max(),
                input_instance["absorption_term"].min(),
            )
            print("SOLUTION: ", gt_solution.max(), gt_solution.min())
            input_instance["solution"] = gt_solution
            input_instance["geometry"] = geometry_string
            input_instance["points"] = points
            train_distances = []
            train_bc = []

            for point in points:
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
                train_bc.append(
                    poisson_3d_obj.analytic_expression(closest_boundary_point)
                )
            input_instance["dist"] = train_distances
            input_instance["closest_bc"] = train_bc
            fi = fileid.split(".")[0] + ".pt"
            print(fi)
            torch.save(input_instance, "../data/3Dtensors/" + fi)
            tensor.append(input_instance)
        else:
            print(loss_value)

            print("NAN LOSS")
            print("NAN VALUES: ", torch.isnan(torch.from_numpy(vals)).sum())
            print("NAN WOS: ", torch.isnan(torch.from_numpy(wos_estimate)).sum())

            print(
                f"Counter: {id+1}, REJECTED SHAPE: ",
                " VERTICES: ",
                points.shape[0],
                " BOUNDARY POINTS: ",
                boundary_points.shape,
                " Rejected Counter: ",
                rejected_counter + 1,
                " Average Loss: ",
                sum(losses) / (len(losses) + 0.0004),
            )
            rejected.append(files[id])
            torch.save({"rejected": rejected}, "../data/rejected.pt")
            rejected_counter += 1

    print(files)
    print("WRITING TO OBJ")
    with open("../data/linear_poisson_3d_gino.obj", "wb") as f:
        pickle.dump(tensor, f)
    print("WRITTEN TO OBJ")


if __name__ == "__main__":
    # Check if the user is hong
    # generate_data(data_path = '/media/hviswan/Data1/Shapenet')
    generate_data(data_path="/home/hong/datasets/ShapeNet")
    # generate_data(data_path='../data/Shapenet/')
