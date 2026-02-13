import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle


class ZombieSolver:
    def __init__(self, config):
        # Initialize configurations from the config dict
        self.config = config["wost_data"]

        self.solution = None
        self.metrics = None
        dataset_path = config["datasetPath"]
        self.load_dataset(dataset_path)

        # Call a method to initialize the solver, if needed
        self.initialize_solver()

    def initialize_solver(self):
        """
        Set up the solver with configurations.
        This can include setting the grid size, boundary conditions, etc.
        """
        self.scene_config = self.config["scene"]
        self.solver_config = self.config["solver"]
        self.output_config = self.config["output"]
        pass

    def load_dataset(self, dataset_path):
        """
        Load the dataset from a given path (could be pt tensor or obj file).
        The dataset should include ground truth solutions for comparison.
        """
        # Here, you might use torch.load or other methods depending on the dataset type
        # For simplicity, assuming it's a torch tensor for now
        if ".pt" in dataset_path:
            self.dataset = torch.load(dataset_path)
        else:
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
            self.dataset = dataset

    def compute_metrics(self, ground_truth):
        """
        Compute the L1 and L2 losses based on the ground truth solutions.
        """
        l1_loss = torch.mean(torch.abs(self.solution - ground_truth))
        l2_loss = torch.sqrt(torch.mean((self.solution - ground_truth) ** 2))
        self.metrics = {"L1": l1_loss.item(), "L2": l2_loss.item()}
        return self.metrics

    def plot_solution(self):
        """
        Plot the solution, either 2D or 3D, depending on the problem.
        """
        if len(self.solution.shape) == 2:
            plt.imshow(self.solution, cmap="hot", interpolation="nearest")
            plt.title("2D Solution")
            plt.colorbar()
            plt.show()
        elif len(self.solution.shape) == 3:
            # Here we would use a method to visualize 3D data
            # For simplicity, just plotting a 2D slice of the solution
            plt.imshow(
                self.solution[0], cmap="hot", interpolation="nearest"
            )  # Plotting the first slice
            plt.title("3D Solution (Slice)")
            plt.colorbar()
            plt.show()

    def run_solver(self):
        """
        Run the solver on the dataset. This is a general method, which will be overridden
        in the derived classes to handle specific solver implementations.
        """
        pass
