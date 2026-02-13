import torch
import numpy as np
import pickle
import os
import sys

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from solvers.fenics import Poisson2DSolver


def generate_data(num_samples=2000):
    """
    Generates a dataset by solving the 2D Poisson equation for a given number of samples.

    Args:
        num_samples (int): Number of Poisson equation solutions to generate.

    Returns:
        list: A dataset containing solution values, source terms, boundary conditions, and geometry information.
    """
    dataset = []

    for i in range(num_samples):
        print(f"Generating Instance: {i}")
        solver = Poisson2DSolver(resolution=12)
        solution = solver.solve_poisson(seed=i)
        bound_val, dom_val, bound_pt, dom_pt = solver.get_pointwise_solution(solution)
        params = solver.params
        source_params, bc_params, geo_params = params

        print("Computing Source Terms")
        # Compute source terms at domain and boundary points
        source_terms_domain = np.array(
            [solver.poisson_pde.source(source_params, x) for x in dom_pt]
        )
        source_terms_boundary = np.array(
            [solver.poisson_pde.source(source_params, x) for x in bound_pt]
        )

        # Compute boundary distances and closest boundary conditions
        # distances_boundary = np.zeros(len(bound_pt))
        closest_boundary_cond_domain = []
        closest_boundary_cond_boundary = bound_val
        distances_domain = []

        print("Computing Distances")
        for point in dom_pt:
            x, y = point
            distances = np.sqrt(
                np.square(bound_pt[:, 0] - x) + np.square(bound_pt[:, 1] - y)
            )
            # min_distance = np.min(distances)
            # min_index = np.where(distances == min_distance)[0]
            min_index = np.argmin(distances)
            min_distance = distances[min_index]
            closest_boundary_point = (bound_pt[min_index, 0], bound_pt[min_index, 1])
            distances_domain.append(min_distance)
            # print(f"distances.shape: {distances.shape}")
            # print(f"dist: {distances[np.where(distances == min_distance)]}")
            # print(f"min_distance: {min_distance}")
            # print(f"idx: {np.where(distances == min_distance)}")
            # print(f"bound_pt.shape: {bound_pt.shape}")
            # print(f"min_index: {min_index}")
            # print(f"closest_boundary_point: {closest_boundary_point}")
            # if len(min_index) > 1:
            #     breakpoint()
            closest_boundary_cond_domain.append(solution(closest_boundary_point))

        closest_boundary_cond_domain = np.array(closest_boundary_cond_domain)
        distances_domain = np.array(distances_domain)

        print("Generating Eval Points")
        # Sample evaluation points and compute corresponding solutions and source terms
        eval_pt_dom, eval_pt_bound = solver.sample_points(params, n_points=1024)
        eval_source_domain = np.array(
            [solver.poisson_pde.source(source_params, x) for x in eval_pt_dom]
        )
        eval_source_boundary = np.array(
            [solver.poisson_pde.source(source_params, x) for x in eval_pt_bound]
        )

        eval_soln_domain = np.array([solution(x) for x in eval_pt_dom])
        eval_soln_bound = np.array([solution(x) for x in eval_pt_bound])

        # Store coefficient parameters
        r0 = 1.0
        c1, c2 = geo_params
        beta_i, mu_i_1, mu_i_2 = (
            np.array(source_params[:, 2]),
            np.array(source_params[:, 0]),
            np.array(source_params[:, 1]),
        )
        b_i = np.array(bc_params)
        print(c1, c2, beta_i, mu_i_1, mu_i_2, b_i)
        coefs = {
            "seed": i + 1,
            "c1": c1,
            "c2": c2,
            "r0": r0,
            "beta": torch.tensor(beta_i),
            "mu_1": torch.tensor(mu_i_1),
            "mu_2": torch.tensor(mu_i_2),
            "b": torch.tensor(b_i),
        }

        # Generate OBJ file data for geometry representation
        obj_data = []
        points = torch.from_numpy(np.asarray(bound_pt)).detach().cpu().numpy()
        for x, y in points:
            obj_data.append(f"v {x} {y} 0.0\n")
        for segment in range(len(points) - 2):
            obj_data.append(f"l {segment + 2} {segment + 1}\n")

        # Append the generated sample to the dataset
        dataset.append(
            {
                "train_points_boundary": bound_pt,
                "train_values_boundary": bound_val,
                "train_source_terms_boundary": source_terms_boundary,
                "train_bc_boundary": closest_boundary_cond_boundary,
                "train_points_domain": dom_pt,
                "train_values_domain": dom_val,
                "train_distances_domain": distances_domain,
                "train_source_terms_domain": source_terms_domain,
                "train_bc_domain": closest_boundary_cond_domain,
                "val_points_boundary": eval_pt_bound,
                "val_values_boundary": eval_soln_bound,
                "val_source_terms_boundary": eval_source_boundary,
                "val_points_domain": eval_pt_dom,
                "val_values_domain": eval_soln_domain,
                "val_source_terms_domain": eval_source_domain,
                "coefs": coefs,
                "geometry": obj_data,
            }
        )

    return dataset


if __name__ == "__main__":
    dataset = generate_data(num_samples=2000)

    with open("../data/linear_poisson_2d.obj", "wb") as f:
        pickle.dump(dataset, f)
