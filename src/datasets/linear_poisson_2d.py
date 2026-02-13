import torch
import numpy as np
from pathlib import Path
import random

random.seed(42)
import pickle

from torch.utils.data import DataLoader, Dataset
from src.solvers.wos import WOSPoisson2DSolver
import yaml
from omegaconf import DictConfig, OmegaConf

path = Path(__file__).resolve().parent.joinpath("data")


def generate_latent_queries(query_res, pad=0, domain_lims=[[-1.4, 1.4], [-1.4, 1.4]]):
    """Generate a regular grid of query points for PINO training"""
    oneDMeshes = []
    for lower, upper in domain_lims:
        oneDMesh = np.linspace(lower, upper, query_res)
        if pad > 0:
            start = np.linspace(lower - pad / query_res, lower, pad + 1)
            stop = np.linspace(upper, upper + pad / query_res, pad + 1)
            oneDMesh = np.concatenate([start, oneDMesh, stop])
        oneDMeshes.append(oneDMesh)
    grid = np.stack(np.meshgrid(*oneDMeshes, indexing="xy"))  # c, x, y, z(?)
    grid = torch.from_numpy(grid.astype(np.float32))
    latent_queries = grid.permute(*list(range(1, len(domain_lims) + 1)), 0)
    return latent_queries


class LinearPoisson2DDataset(Dataset):
    def __init__(
        self,
        data_path,
        wos_config,
        query_res=48,
        domain_padding=0,
        encode=True,
        val_on_same_instance=False,
        n_train=1,
        n_test=1,
        n_points=1024,
        logger=True,
        isTrain=True,
        train_pc=0.8,
        is_wos=True,
        use_grad=True,
        **kwargs,
    ):
        try:
            with open(data_path, "rb") as f:
                dataset = pickle.load(f)
                if logger:
                    print("Dictionary Loaded Successfully")
        except FileNotFoundError:
            print(f"Error: The file was not found.")
            raise FileNotFoundError(f"The file {data_path} does not exist.")

        data_size = len(dataset)

        train_end = int(train_pc * data_size)

        if n_train > train_end:
            n_train = train_end
        if logger and isTrain:
            print(f"n_train={n_train}")
        if n_test > data_size - train_end:
            n_test = data_size - train_end
        if logger and not isTrain:
            print(f"n_test={n_test}")

        random.shuffle(dataset)
        if isTrain:
            self.data = dataset[:n_train]
        else:

            if val_on_same_instance:
                self.data = dataset[:n_train]
            else:
                self.data = dataset[train_end : train_end + n_test]

        self.length = len(self.data)
        self.query_res = query_res
        self.domain_padding = domain_padding
        self.encode = encode
        self.val_on_same_instance = val_on_same_instance
        self.n_train = n_train
        self.n_test = n_test
        self.n_points = n_points
        self.logger = logger
        self.is_wos = is_wos
        self.use_grad = use_grad
        print("Generating Latent Queries for training...")
        # Generate query grid for PINO training
        self.latent_queries = generate_latent_queries(
            query_res=query_res, pad=domain_padding
        )

        # if wos_config is a DictConfig, convert it to a dict
        if self.is_wos:
            if isinstance(wos_config, DictConfig):
                wos_config = OmegaConf.to_container(wos_config)
            print("Initializing WoS Solver with config")
            self.wos_solver = WOSPoisson2DSolver(wos_config)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        instance = self.data[idx]

        n_points = self.n_points
        # points = random.sample(range(len(instance['train_points_boundary'])), n_points)
        # points = range(1024)

        n_points = self.n_points
        points = range(min(n_points, len(instance["train_points_boundary"])))
        # Load geometry data
        boundary_points = torch.tensor(
            instance["train_points_boundary"][points], dtype=torch.float32
        )
        domain_points = torch.tensor(
            instance["train_points_domain"][points], dtype=torch.float32
        )
        input_geom = torch.vstack((boundary_points, domain_points))

        # WOS Scene Setup

        if self.is_wos:
            geometry_string = [
                instance["geometry"][ind] for ind in range(len(instance["geometry"]))
            ]
            input_points = input_geom.detach().cpu().numpy()
            samples, p_arr, grad_arr = self.wos_solver.solve(
                input_points, instance["coefs"], geometry_string
            )
            p_arr = torch.from_numpy(np.asarray(p_arr).astype(np.float32)).unsqueeze(1)
            grad_arr = torch.from_numpy(
                np.asarray(grad_arr).astype(np.float32)
            ).unsqueeze(1)
        else:
            print("SETTING WOS TO ZERO")
            p_arr = torch.zeros(n_points, 1)
            grad_arr = torch.zeros(n_points, 1)

        ##
        ## TODO: This part is needed for training PINO and DeepRitz!
        if self.use_grad:
            source_terms_boundary = torch.tensor(
                instance["train_source_terms_boundary"][points], dtype=torch.float32
            )
            source_terms_domain = torch.tensor(
                instance["train_source_terms_domain"][points], dtype=torch.float32
            )
            source_terms = torch.cat(
                (source_terms_boundary, source_terms_domain)
            ).unsqueeze(-1)

            # Boundary conditions
            bc_boundary = torch.tensor(
                instance["train_bc_boundary"][points], dtype=torch.float32
            )
            bc_domain = torch.tensor(
                instance["train_bc_domain"][points], dtype=torch.float32
            )
            boundary_conditions = torch.cat((bc_boundary, bc_domain)).unsqueeze(-1)

            # Distances to boundary (for domain points)
            distances_domain = torch.tensor(
                instance["train_distances_domain"][points], dtype=torch.float32
            )
            distances = torch.cat(
                (torch.zeros(len(points)), distances_domain)
            ).unsqueeze(-1)

            num_boundary = len(bc_boundary)

            # Combine features for PINO training
            # Format: [boundary_conditions, distances, source_terms]
            f = torch.cat((boundary_conditions, distances, source_terms), dim=-1)

            # Separate boundary and domain points for PINO loss computation
            boundary_points_pino = boundary_points
            domain_points_pino = domain_points

            # Query points for PINO training (regular grid)
            output_queries = input_geom
        else:

            source_terms_boundary = torch.tensor(
                np.asarray(instance["train_source_terms_boundary"])[points],
                dtype=torch.float32,
            )
            source_terms_domain = torch.tensor(
                np.asarray(instance["train_source_terms_domain"])[points],
                dtype=torch.float32,
            )
            bc_boundary = torch.tensor(
                np.asarray(instance["train_bc_boundary"])[points], dtype=torch.float32
            )
            bc_domain = torch.tensor(
                np.asarray(instance["train_bc_domain"])[points], dtype=torch.float32
            )
            distances_domain = torch.tensor(
                np.asarray(instance["train_distances_domain"])[points],
                dtype=torch.float32,
            )
            # print(f"b_boundary.shape: {bc_boundary.shape}")
            num_boundary = len(bc_boundary)
            f_f = torch.cat((source_terms_boundary, source_terms_domain)).unsqueeze(-1)
            f_g = torch.cat((bc_boundary, bc_domain)).unsqueeze(-1)
            f_dist = torch.cat((torch.zeros(len(points)), distances_domain)).unsqueeze(
                -1
            )
            f = torch.cat((f_g, f_dist, f_f), dim=-1)

            output_queries = input_geom

        # Ground truth solution (if available)
        if (
            "val_values_boundary" in instance.keys()
            and "val_values_domain" in instance.keys()
        ):
            y = torch.cat(
                (
                    torch.tensor(
                        instance["val_values_boundary"][points], dtype=torch.float32
                    ),
                    torch.tensor(
                        instance["val_values_domain"][points], dtype=torch.float32
                    ),
                )
            ).unsqueeze(-1)
        else:
            print("SETTING GT TO ZERO")
            exit()
            y = torch.tensor([None])

        # f_f = souce terms
        # f_g = boundary conditions
        # f_dist = distances to the boundary
        # f_f = torch.cat((torch.tensor(instance['train_source_terms_boundary'][points], dtype=torch.float32), torch.tensor(instance['train_source_terms_domain'][points], dtype=torch.float32))).unsqueeze(dim=-1)
        # f_g = torch.cat((torch.tensor(instance['train_bc_boundary'][points], dtype=torch.float32), torch.tensor(instance['train_bc_domain'][points], dtype=torch.float32))).unsqueeze(dim=-1)
        # f_dist = torch.cat((torch.zeros(n_points), torch.tensor(instance['train_distances_domain'][points], dtype=torch.float32))).unsqueeze(dim=-1)
        # #p_arr_cache = torch.from_numpy(np.asarray(p_arr_cache).astype(np.float32)).unsqueeze(1)
        # f = torch.cat((f_g, f_dist, f_f), dim=-1)
        # Use diff points for output
        # out_p = torch.cat((torch.tensor(instance['val_points_boundary'][points], dtype=torch.float32), torch.tensor(instance['val_points_domain'][points], dtype=torch.float32)))
        # if('val_values_boundary' in instance.keys() and 'val_values_domain' in instance.keys()):
        #     y = torch.cat((torch.tensor(instance['val_values_boundary'][points], dtype=torch.float32), torch.tensor(instance['val_values_domain'][points], dtype=torch.float32))).unsqueeze(-1)
        # else:
        #     y = [None]

        output_queries.requires_grad_(self.use_grad)

        data_dict = {
            "x": f,
            "geom_str": instance["geometry"],  # Geometry string
            "coefs": instance["coefs"],  # Problem coefficients
            "input_geom": input_geom,  # All input points
            "latent_queries": self.latent_queries,  # Regular grid for PINO
            "output_queries": output_queries,  # Query points for PINO
            "output_source_terms_domain": source_terms_domain,
            "y": y,
            "wos_estimate": p_arr,
            "wos_grad": grad_arr,
            "wos_var": [],
            "num_boundary": num_boundary,
        }

        # Convert any JAX arrays in coefs to PyTorch tensors
        if "coefs" in data_dict:
            coefs = data_dict["coefs"]
            for key, value in coefs.items():
                if hasattr(value, "device_buffer") or str(type(value)).startswith(
                    "<class 'jaxlib."
                ):
                    coefs[key] = torch.from_numpy(np.asarray(value).astype(np.float32))

        # Convert any remaining JAX arrays in the data_dict
        for key, value in data_dict.items():
            if hasattr(value, "device_buffer") or str(type(value)).startswith(
                "<class 'jaxlib."
            ):
                data_dict[key] = np.asarray(value)

        return data_dict
