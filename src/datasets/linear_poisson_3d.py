import torch
import numpy as np
from pathlib import Path
import random

random.seed(42)
import pickle

from torch.utils.data import DataLoader, Dataset
from src.solvers.wos import WOSPoisson3DSolver
from omegaconf import DictConfig, OmegaConf


path = Path(__file__).resolve().parent.joinpath("data")


def generate_latent_queries(
    query_res, pad=0, domain_lims=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
):
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


def get_grad_laplacian(x: torch.Tensor):

    # TODO: Where is it from?
    """
    Computes the gradient and Laplacian of the function alpha(x,y) = exp(-y^2 + cos(a*x) * sin(b*y))
    where a = 4.0 * pi * diffusion_freq and b = 3.0 * pi * diffusion_freq
    The function is defined on a 2D grid with x in [-1, 1] and y in [-1, 1].
    The diffusion frequency is set to 0.2.
    """

    diffusion_freq = 0.2
    a = 4.0 * np.pi * diffusion_freq
    b = 3.0 * np.pi * diffusion_freq
    sin_ax = torch.sin(a * x[:, 0])
    cos_ax = torch.cos(a * x[:, 0])
    sin_by = torch.sin(b * x[:, 1])
    cos_by = torch.cos(b * x[:, 1])
    alpha = torch.exp(-x[:, 1] ** 2 + cos_ax * sin_by)

    gradient = [
        alpha * (-sin_ax * sin_by * a),
        alpha * (-2.0 * x[:, 1] + cos_ax * cos_by * b),
        torch.zeros_like(x[:, 0]),
    ]
    d2alphadx2 = gradient[0] * (-sin_ax * sin_by * a) + alpha * (
        -cos_ax * sin_by * a * a
    )
    d2alphady2 = gradient[1] * (-2 * x[:, 1] + cos_ax * cos_by * b) + alpha * (
        -2 - cos_ax * sin_by * b * b
    )
    lap = d2alphadx2 + d2alphady2
    return gradient, lap


class LinearPoisson3DDataset(Dataset):
    def __init__(
        self,
        data_path,
        # wos_config_path,
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

        clean_data = []
        for idx in range(len(dataset)):
            data_elem = dataset[idx]["points"]
            if (
                data_elem[:, 0].max() - data_elem[:, 0].min() > 0.4
                and data_elem[:, 1].max() - data_elem[:, 1].min() > 0.4
                and data_elem[:, 2].max() - data_elem[:, 2].min() > 0.4
            ):
                clean_data.append(dataset[idx])
        dataset = clean_data
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
        self.use_grad = use_grad
        self.latent_queries = generate_latent_queries(
            query_res=query_res,
            pad=domain_padding,
            domain_lims=[[-1, 1], [-1, 1], [-1, 1]],
        )
        self.is_wos = is_wos
        if self.is_wos:
            if isinstance(wos_config, DictConfig):
                wos_config = OmegaConf.to_container(wos_config)
            self.wos_solver = WOSPoisson3DSolver(wos_config)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        instance = self.data[idx]

        n_points = self.n_points
        # Load Data
        input_geom = torch.tensor(instance["points"], dtype=torch.float32)

        # WOS Scene Setup
        geometry_string = [
            instance["geometry"][ind] for ind in range(len(instance["geometry"]))
        ]
        input_points = input_geom.detach().cpu().numpy()
        # self.latent_queries = generate_latent_queries(query_res=self.query_res,
        #                                             pad=self.domain_padding,
        #                                             domain_lims=[[input_geom.min(),input_geom.max()],
        #                                                         [input_geom.min(),input_geom.max()],
        #                                                         [input_geom.min(),input_geom.max()]]
        #                                             )
        # print(input_geom[:,0].min(),input_geom[:,0].max())
        # print(input_geom[:,1].min(),input_geom[:,1].max())
        # print(input_geom[:,2].min(),input_geom[:,2].max())
        p_arr, grad_arr = self.wos_solver.solve(
            input_points, coefs=None, geometry=geometry_string
        )

        p_arr = torch.from_numpy(np.asarray(p_arr).astype(np.float32)).unsqueeze(1)
        # grad_arr = torch.from_numpy(np.asarray(grad_arr).astype(np.float32)).unsqueeze(1)

        f_f = torch.tensor(instance["source_terms"], dtype=torch.float32).unsqueeze(
            dim=-1
        )
        f_g = torch.tensor(instance["closest_bc"], dtype=torch.float32).unsqueeze(
            dim=-1
        )
        f_dist = torch.tensor(instance["dist"], dtype=torch.float32).unsqueeze(dim=-1)
        f_diff = torch.tensor(
            instance["diffusion_term"], dtype=torch.float32
        ).unsqueeze(dim=-1)
        f_abs = torch.tensor(
            instance["absorption_term"], dtype=torch.float32
        ).unsqueeze(dim=-1)

        # self.latent_queries = generate_latent_queries(query_res=self.query_res,
        #                                             pad=self.domain_padding,
        #                                             domain_lims=[[input_geom.min(),input_geom.max()],
        #                                                         [input_geom.min(),input_geom.max()],
        #                                                         [input_geom.min(),input_geom.max()]])

        grad, lap = get_grad_laplacian(input_geom)

        num_boundary = len(instance["closest_bc"])
        grad_arr = torch.cat(
            (grad[0].unsqueeze(-1), grad[1].unsqueeze(-1), grad[2].unsqueeze(-1)),
            dim=-1,
        )

        grad_arr = torch.linalg.norm(grad_arr, dim=-1).unsqueeze(-1)

        lap = lap.unsqueeze(-1)
        lap = lap * 0.5 / f_diff
        grad_arr = grad_arr * 0.25 / f_diff**2

        # f_f = f_f*(1/f_diff**0.5)

        f_abs = f_abs / f_diff + lap - grad_arr
        bound = f_abs.max() - f_abs.min()
        # f_abs = f_abs/bound
        # f_abs = 1 - f_abs
        # f_abs = f_abs * (f_diff**0.5)

        # f_f = f_f * f_abs
        f_f = f_f / (f_diff**0.5)

        f_abs = 1 - f_abs / bound
        f_abs = f_abs * (f_diff**0.5)

        f_diff = 1 / (f_diff**0.5)

        # p_arr_cache = torch.from_numpy(np.asarray(p_arr_cache).astype(np.float32)).unsqueeze(1)
        f = torch.cat((f_g, f_dist, f_f, f_abs, f_diff), dim=-1)

        # Use diff points for output
        out_p = torch.tensor(instance["points"][:n_points], dtype=torch.float32)
        if (
            "val_values_boundary" in instance.keys()
            and "val_values_domain" in instance.keys()
        ):
            y = torch.cat(
                (
                    torch.tensor(instance["val_values_boundary"], dtype=torch.float32),
                    torch.tensor(instance["val_values_domain"], dtype=torch.float32),
                )
            ).unsqueeze(-1)
        else:
            y = torch.tensor(instance["solution"], dtype=torch.float32).unsqueeze(-1)

        # if(input_geom.isnan().any() or y.isnan().any() or f.isnan().any()):
        #     print("NAN DETECTED")
        #     print("INPUT GEOM: ", input_geom)
        #     print("Y: ", y)
        #     print("F: ", f)
        #     print("GRAD ARR: ", grad_arr)
        #     raise ValueError("NAN DETECTED")
        output_queries = input_geom
        output_queries.requires_grad_(self.use_grad)

        data_dict = {
            "x": f,
            "geom_str": instance["geometry"],
            "coefs": [],
            "input_geom": input_geom,
            "latent_queries": self.latent_queries,
            "output_queries": output_queries,
            "output_source_terms_domain": f_f,
            "y": y,
            "wos_estimate": p_arr,
            "wos_var": [],
            "num_boundary": num_boundary,
        }
        return data_dict
