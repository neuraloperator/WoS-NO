import torch
import numpy as np
from pathlib import Path
import random

random.seed(42)
import pickle

from torch.utils.data import DataLoader, Dataset
from src.solvers.wos import WOSPoisson2DSolverBVC
import yaml

path = Path(__file__).resolve().parent.joinpath("data")


def generate_latent_queries(query_res, pad=0, domain_lims=[[-1.4, 1.4], [-1.4, 1.4]]):
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


class LinearPoisson2DDatasetBVC(Dataset):
    def __init__(
        self,
        data_path,
        wos_config_path,
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
    ):
        try:
            with open(data_path, "rb") as f:
                dataset = pickle.load(f)
                if logger:
                    print("Dictionary Loaded Successfully")
        except FileNotFoundError:
            print(f"Error: The file was not found.")

        data_size = len(dataset)

        train_end = int(train_pc * data_size)

        if n_train > train_end:
            n_train = train_end
        if logger and isTrain:
            print(f"{n_train=}")
        if n_test > data_size - train_end:
            n_test = data_size - train_end
        if logger and not isTrain:
            print(f"{n_test=}")

        random.shuffle(dataset)
        if isTrain:
            self.data = dataset[:n_train]
        else:

            if val_on_same_instance:
                self.data = dataset[:n_train]
            else:
                self.data = dataset[train_end : train_end + n_test]

        self.length = len(self.data)

        self.wos_estimate = []
        self.query_res = query_res
        self.domain_padding = domain_padding
        self.encode = encode
        self.val_on_same_instance = val_on_same_instance
        self.n_train = n_train
        self.n_test = n_test
        self.n_points = n_points
        self.logger = logger

        self.latent_queries = generate_latent_queries(
            query_res=query_res, pad=domain_padding
        )
        with open(wos_config_path, "r") as file:
            wos_config = yaml.safe_load(file)
        self.wos_solver = WOSPoisson2DSolverBVC(wos_config)
        self.bcache = [
            [([0.1, 0.1], [0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "None")]
            for _ in range(self.length)
        ]
        self.dcache = [
            [([0.1, 0.1], [0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "None")]
            for _ in range(self.length)
        ]
        self.ncache = [
            [([0.1, 0.1], [0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "None")]
            for _ in range(self.length)
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        instance = self.data[idx]

        n_points = self.n_points
        points = random.sample(range(len(instance["train_points_boundary"])), n_points)
        points = range(1024)
        # Load Data
        input_geom = torch.vstack(
            (
                torch.tensor(
                    instance["train_points_boundary"][points], dtype=torch.float32
                ),
                torch.tensor(
                    instance["train_points_domain"][points], dtype=torch.float32
                ),
            )
        )

        # WOS Scene Setup
        geometry_string = [
            instance["geometry"][ind] for ind in range(len(instance["geometry"]))
        ]
        input_points = input_geom.detach().cpu().numpy()
        samples, p_arr, grad_arr, bcache, dcache, ncache = self.wos_solver.solve(
            input_points,
            instance["coefs"],
            geometry_string,
            self.bcache[idx],
            self.dcache[idx],
            self.ncache[idx],
            usecache=True,
        )
        print(bcache)
        print(dcache)
        print(ncache)
        exit()

        p_arr = torch.from_numpy(np.asarray(p_arr).astype(np.float32)).unsqueeze(1)
        grad_arr = torch.from_numpy(np.asarray(grad_arr).astype(np.float32)).unsqueeze(
            1
        )

        # f_f = torch.cat((torch.tensor(instance['train_source_terms_boundary'][points], dtype=torch.float32), torch.tensor(instance['train_source_terms_domain'][points], dtype=torch.float32))).unsqueeze(dim=-1)
        f_f = p_arr
        f_g = torch.cat(
            (
                torch.tensor(
                    instance["train_bc_boundary"][points], dtype=torch.float32
                ),
                torch.tensor(instance["train_bc_domain"][points], dtype=torch.float32),
            )
        ).unsqueeze(dim=-1)
        f_dist = torch.cat(
            (
                torch.zeros(n_points),
                torch.tensor(
                    instance["train_distances_domain"][points], dtype=torch.float32
                ),
            )
        ).unsqueeze(dim=-1)

        # p_arr_cache = torch.from_numpy(np.asarray(p_arr_cache).astype(np.float32)).unsqueeze(1)
        f = torch.cat((f_g, f_dist, f_f), dim=-1)

        # Use diff points for output
        out_p = torch.cat(
            (
                torch.tensor(
                    instance["val_points_boundary"][points], dtype=torch.float32
                ),
                torch.tensor(
                    instance["val_points_domain"][points], dtype=torch.float32
                ),
            )
        )
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
            y = [None]

        data_dict = {
            "f": f,
            "geom_str": instance["geometry"],
            "coefs": instance["coefs"],
            "input_geom": input_geom,
            "latent_queries": self.latent_queries,
            "output_queries": input_geom,
            "y": y,
            "wos_estimate": p_arr,
            "wos_grad": grad_arr,
            "wos_var": [],
        }
        return data_dict
