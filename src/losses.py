import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.nn as nn
from neuralop.losses import LpLoss, WeightedSumLoss

from typing import Union

# Import PINO losses
try:
    from .losses.pino_loss import PINOPoissonLoss, PINOCombinedLoss, create_pino_loss
except ImportError:
    # Fallback if the module doesn't exist yet
    PINOPoissonLoss = None
    PINOCombinedLoss = None
    create_pino_loss = None


def diffusion(x, gradient=False, laplacian=False):
    """Computes the diffusion coefficient and optionally its gradient and Laplacian."""
    diffusion_freq = 0.5
    a = 4.0 * torch.pi * diffusion_freq
    b = 3.0 * torch.pi * diffusion_freq
    scalar = torch.ones(x.shape[0]).to(x.device)
    a = scalar * a
    b = scalar * b
    sin_ax = torch.sin(a * x[:, 0])
    cos_ax = torch.cos(a * x[:, 0])
    sin_by = torch.sin(b * x[:, 1])
    cos_by = torch.cos(b * x[:, 1])
    alpha = torch.exp(-x[:, 1] ** 2 + cos_ax * sin_by)
    if gradient == True and laplacian == False:
        gradient = [
            alpha * (-sin_ax * sin_by * a),
            alpha * (-2.0 * x[:, 1] + cos_ax * cos_by * b),
            torch.zeros(x.shape[0]).to(x.device),
        ]
        return alpha, torch.stack(gradient).to(x.device)
    elif laplacian == True and gradient == True:
        gradient = [
            alpha * (-sin_ax * sin_by * a),
            alpha * (-2.0 * x[:, 1] + cos_ax * cos_by * b),
            torch.zeros(x.shape[0]).to(x.device),
        ]
        d2alphadx2 = gradient[0] * (-sin_ax * sin_by * a) + alpha * (
            -cos_ax * sin_by * a * a
        )
        d2alphady2 = gradient[1] * (-2 * x[:, 1] + cos_ax * cos_by * b) + alpha * (
            -2 * torch.ones(x.shape[0]).to(x.device) - cos_ax * sin_by * b * b
        )
        lap = d2alphadx2 + d2alphady2
        return alpha, torch.stack(gradient).to(x.device), lap
    return alpha


def absorption(x):
    """Computes the absorption coefficient."""
    absorption_min = 10.0
    absorption_max = 100.0
    scalar = torch.ones(x.shape[0]).to(x.device)
    absorption_min = scalar * absorption_min
    absorption_max = scalar * absorption_max
    return absorption_min + (absorption_max - absorption_min) * (
        torch.ones(x.shape[0]).to(x.device)
        + 0.5
        * torch.sin(2.0 * torch.pi * x[:, 0])
        * torch.cos(0.5 * torch.pi * x[:, 1])
    )


def source(x):
    """
    Defines the source term f(x) in the Poisson equation.

    Args:
        x (ndarray): Spatial coordinates.

    Returns:
        float: The computed source term value at x.
    """
    dirichlet_freq = 1.5
    a = torch.pi * dirichlet_freq
    alpha, gradient = diffusion(x, gradient=True)
    sigma = absorption(x)

    b = 2.0 * a
    c = 3.0 * a
    scalar = torch.ones(x.shape[0]).to(x.device)
    a = scalar * a
    b = scalar * b
    c = scalar * c
    sin_ax = torch.sin(a * x[:, 0])
    cos_ax = torch.cos(a * x[:, 0])
    sin_by = torch.sin(b * x[:, 1])
    cos_by = torch.cos(b * x[:, 1])
    sin_cz = torch.sin(c * x[:, 2])
    cos_cz = torch.cos(c * x[:, 2])

    u = (
        sin_ax * cos_by
        + (torch.ones(x.shape[0]).to(x.device) - cos_ax)
        * (torch.ones(x.shape[0]).to(x.device) - sin_by)
        + sin_cz**2
    )
    d2u_dx2 = (
        (cos_ax * (torch.ones(x.shape[0]).to(x.device) - sin_by) - sin_ax * cos_by)
        * a
        * a
    )
    d2u_dy2 = (
        ((torch.ones(x.shape[0]).to(x.device) - cos_ax) * sin_by - sin_ax * cos_by)
        * b
        * b
    )
    d2u_dz2 = 2 * (cos_cz * cos_cz - sin_cz * sin_cz) * c * c
    d2u = d2u_dx2 + d2u_dy2 + d2u_dz2

    du = torch.stack(
        [
            cos_ax * cos_by
            + sin_ax * (torch.ones(x.shape[0]).to(x.device) - sin_by) * a,
            -(sin_ax * sin_by + (torch.ones(x.shape[0]).to(x.device) - cos_ax) * cos_by)
            * b,
            2 * sin_cz * cos_cz * c,
        ]
    )
    # print(alpha.shape, d2u.shape, gradient.shape, du.shape, sigma.shape, u.shape)
    # print((alpha*d2u).shape, (sigma*u).shape)
    return -alpha * d2u - (gradient * du).sum(dim=0) + sigma * u


class NonLinearPoissonInteriorLoss(object):
    """
    Implements loss for variable coefficient Poisson equation:
        ∇·(k(x)∇u) - α(x)u = -f(x)

    Parameters
    ----------
    method : Literal['autograd']
    loss : Callable (e.g., torch.nn.functional.mse_loss)
    """

    def __init__(self, method="autograd", loss=F.mse_loss, dim=2):
        super().__init__()
        assert method == "autograd"
        assert dim in [2, 3]
        self.method = method
        self.loss = loss
        self.dim = dim
        self.is_3d = dim == 3

    def autograd(self, u, output_queries, k, alpha, f, num_boundary):
        """
        u: model prediction at query points (shape: [1, N, 1])
        output_queries: coordinates of points (shape: [1, N, dim])
        k: thermal conductivity (shape: [1, N, 1])
        alpha: reaction coefficient (shape: [1, N, 1])
        f: source term (shape: [1, N, 1])
        """

        # Interior points only
        num_domain = u.shape[1] - num_boundary
        if len(k.shape) == 2:
            k = k.unsqueeze(-1)
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(-1)
        if len(f.shape) == 2:
            f = f.unsqueeze(-1)

        coords = output_queries[:, :num_domain, :]
        u.requires_grad_(True)
        u = u[:, :num_domain, :]
        k = k[:, :num_domain, :]
        alpha = alpha[:, :num_domain, :]
        f = f[:, :num_domain, :]
        grad_u = grad(
            outputs=u.sum(), inputs=output_queries, create_graph=True, retain_graph=True
        )[0][:, :num_domain, :]

        # Compute k * grad_u
        k_grad_u = k * grad_u  # (1, N, dim)

        # Compute divergence ∇·(k∇u)
        div_k_grad_u = torch.zeros_like(u)
        for d in range(self.dim):
            partial = grad(
                k_grad_u[:, :, d].sum(),
                output_queries,
                create_graph=True,
                retain_graph=True,
            )[0][:, :num_domain, d]
            div_k_grad_u += partial.unsqueeze(-1)

        # Full PDE LHS
        lhs = div_k_grad_u - alpha * u
        rhs = -f

        loss = self.loss(lhs, rhs)

        return loss

    def __call__(self, y_pred, **kwargs):
        return self.autograd(u=y_pred, **kwargs)


class PoissonBoundaryLoss(object):
    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.loss = loss
        self.counter = 0

    def __call__(
        self,
        y_pred,
        num_boundary,
        y,
        output_queries,
        out_sub_level=1,
        boundary_first=True,
        **kwargs,
    ):
        if boundary_first:
            num_boundary = int(num_boundary.item() * out_sub_level)
            boundary_pred = y_pred.squeeze(0).squeeze(-1)[:num_boundary]
            y_bound = y.squeeze(0).squeeze(-1)[:num_boundary]
        else:
            num_boundary = int(num_boundary.item() * out_sub_level)
            num_domain = y_pred.shape[1] - num_boundary
            boundary_pred = y_pred.squeeze(0).squeeze(-1)[num_domain:]
            y_bound = y.squeeze(0).squeeze(-1)[num_domain:]

        assert boundary_pred.shape == y_bound.shape
        return self.loss(boundary_pred, y_bound)


class LinearPoissonInteriorLoss(object):
    """
    LinearPoissonInteriorLoss computes the loss on the interior points of model outputs
    according to Poisson's equation in nd: ∇·∇u(x) = f(x)

    Parameters
    ----------
    method : Literal['autograd'] only (for now)
        How to compute derivatives for equation loss.

        * If 'autograd', differentiates using torch.autograd.grad. This can be used with outputs with any irregular
        point cloud structure.
    loss: Callable, optional
        Base loss class to compute distances between expected and true values,
        by default torch.nn.functional.mse_loss
    """

    def __init__(self, method: str = "autograd", loss=F.mse_loss, dim=2):
        super().__init__()
        self.method = method
        self.loss = loss
        self.dim = dim

        self.is_3d = self.dim == 3
        assert self.dim in [2, 3]

    def autograd(
        self, u, output_queries, output_source_terms_domain, num_boundary, **kwargs
    ):
        """
        Compute loss between the left-hand side and right-hand side of
        nonlinear Poisson's equation: ∇·∇u(x) = f(x)

        u: torch.Tensor | dict
            output of the model.

            * If output_queries is passed to the model as a dict, this will be a
            dict of outputs provided over the points at each value in output_queries.
            Each tensor will be shape (batch, n_points, 2).

            * If a tensor, u will be of shape (batch, num_boundary + num_interior, 2), where
            u[:, 0:num_boundary, :] are boundary points and u[:, num_boundary:, :] are interior points.
        output_queries: torch.Tensor | dict
            output queries provided to the model. If provided as a dict of tensors,
            u will also be returned as a dict keyed the same way. If provided as a tensor,
            u will be a tensor of the same shape except for number of channels. If a tensor,
            output_queries[:, 0:num_boundary, :] are boundary points and output_queries[:, num_boundary:, :]
            are interior points.
        output_source_terms_domain: torch.Tensor
            source terms f(x) defined for this specific instance of Poisson's equation.

        """

        if isinstance(output_queries, dict):
            output_queries_domain = output_queries["domain"]
            u_prime = grad(
                outputs=u.sum(),
                inputs=output_queries_domain,
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            # We only care about U defined over the interior. Grab it now if the entire U is passed.
            output_queries_domain = None
            u.requires_grad_(True)
            u = u[:, num_boundary:, ...]
            u_prime = grad(
                outputs=u.sum(),
                inputs=output_queries,
                create_graph=True,
                retain_graph=True,
            )[0][:, num_boundary:, :]

        u_x = u_prime[:, :, 0]
        u_y = u_prime[:, :, 1]
        if self.is_3d:
            u_z = u_prime[:, :, 2]

        # compute second derivatives
        if output_queries_domain is not None:
            u_xx = grad(
                outputs=u_x.sum(),
                inputs=output_queries_domain,
                create_graph=True,
                retain_graph=True,
            )[0][:, :, 0]
            u_yy = grad(
                outputs=u_y.sum(),
                inputs=output_queries_domain,
                create_graph=True,
                retain_graph=True,
            )[0][:, :, 1]
            if self.is_3d:
                u_zz = grad(
                    outputs=u_z.sum(),
                    inputs=output_queries_domain,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, :, 1]
        else:
            u_xx = grad(
                outputs=u_x.sum(),
                inputs=output_queries,
                create_graph=True,
                retain_graph=True,
            )[0][:, num_boundary:, 0]
            u_yy = grad(
                outputs=u_y.sum(),
                inputs=output_queries,
                create_graph=True,
                retain_graph=True,
            )[0][:, num_boundary:, 1]
            if self.is_3d:
                u_zz = grad(
                    outputs=u_z.sum(),
                    inputs=output_queries,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, num_boundary:, 1]
        u_xx = u_xx.squeeze(0)
        u_yy = u_yy.squeeze(0)
        if self.is_3d:
            u_zz = u_zz.squeeze(0)
        u = u.squeeze([0, -1])

        # compute LHS of the Poisson equation
        laplacian = u_xx + u_yy
        if self.is_3d:
            laplacian += u_zz

        assert u_xx.shape == u_yy.shape
        if self.is_3d:
            assert u_zz.shape == u_yy.shape

        left_hand_side = laplacian
        output_source_terms_domain = output_source_terms_domain.squeeze(0)

        assert left_hand_side.shape == output_source_terms_domain.shape
        loss = self.loss(left_hand_side, output_source_terms_domain)

        assert not u_prime.isnan().any()
        assert not u_yy.isnan().any()
        assert not u_xx.isnan().any()

        del u_xx, u_yy, u_x, u_y, left_hand_side
        if self.is_3d:
            assert not u_zz.isnan().any()
            del u_zz

        return loss

    def __call__(self, y_pred, **kwargs):
        if self.method == "autograd":
            return self.autograd(u=y_pred, **kwargs)
        elif self.method == "finite_difference":
            raise NotImplementedError()
        else:
            raise NotImplementedError()


class LinearPoissonEqnLoss(object):
    """LinearPoissonEqnLoss computes a weighted sum of equation loss computed on the interior points of a model's output
    and a boundary loss computed on the boundary points.

    Parameters
    ----------
    boundary_weight : float
        weight by which to multiply boundary loss
    interior_weight : float
        weight by which to multiply interior loss
    diff_method : Literal['autograd', 'finite_difference'], optional
        method to use to compute derivatives, by default 'autograd'
    base_loss : Callable, optional
        base loss class to use inside equation and boundary loss, by default F.mse_loss
    """

    def __init__(
        self,
        boundary_weight: float,
        interior_weight: float,
        diff_method: str = "autograd",
        base_loss=F.mse_loss,
    ):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.boundary_loss = PoissonBoundaryLoss(loss=base_loss)

        self.interior_weight = interior_weight
        self.interior_loss = LinearPoissonInteriorLoss(
            method=diff_method, loss=base_loss
        )

    def __call__(self, out, y, **kwargs):
        if isinstance(out, dict):
            interior_loss = self.interior_weight * self.interior_loss(
                out["domain"], **kwargs
            )
            bc_loss = self.boundary_weight * self.boundary_loss(
                out["boundary"], y=y["boundary"], **kwargs
            )
        else:
            interior_loss = self.interior_weight * self.interior_loss(out, **kwargs)
            bc_loss = self.boundary_weight * self.boundary_loss(out, y=y, **kwargs)
        # else:
        #     raise ValueError("Wrong value for output is given!")
        if kwargs.get("return_individual_losses", False):
            return interior_loss + bc_loss, interior_loss, bc_loss
        else:
            return interior_loss + bc_loss

    def __str__(self):
        return (
            f"Interior Loss: {self.interior_loss}, Boundary Loss: {self.boundary_loss}"
        )


class NonLinearPoissonEqnLoss(object):
    """LinearPoissonEqnLoss computes a weighted sum of equation loss computed on the interior points of a model's output
    and a boundary loss computed on the boundary points.

    Parameters
    ----------
    boundary_weight : float
        weight by which to multiply boundary loss
    interior_weight : float
        weight by which to multiply interior loss
    diff_method : Literal['autograd', 'finite_difference'], optional
        method to use to compute derivatives, by default 'autograd'
    base_loss : Callable, optional
        base loss class to use inside equation and boundary loss, by default F.mse_loss
    """

    def __init__(
        self,
        boundary_weight: float,
        interior_weight: float,
        diff_method: str = "autograd",
        base_loss=F.mse_loss,
    ):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.boundary_loss = PoissonBoundaryLoss(loss=base_loss)

        self.interior_weight = interior_weight
        self.interior_loss = NonLinearPoissonInteriorLoss(
            method=diff_method, loss=base_loss, dim=3
        )

    def __call__(self, out, y, **kwargs):

        if isinstance(out, dict):
            interior_loss = self.interior_weight * self.interior_loss(
                out["domain"], **kwargs
            )
            bc_loss = self.boundary_weight * self.boundary_loss(
                out["boundary"], y=y["boundary"], **kwargs
            )
        else:

            output_queries = kwargs.get("output_queries", None)

            diffusion_term = kwargs.get("diffusion", None).detach()
            absorption_term = kwargs.get("absorption", None).detach()
            source_term = kwargs.get("source", None)

            interior_loss = self.interior_weight * self.interior_loss(
                y_pred=out,
                output_queries=output_queries,
                k=diffusion_term,
                alpha=absorption_term,
                f=source_term,
                num_boundary=1024,
            )
            bc_loss = self.boundary_weight * self.boundary_loss(
                out, y=y, boundary_first=False, **kwargs
            )
        # else:
        #     raise ValueError("Wrong value for output is given!")
        if kwargs.get("return_individual_losses", False):
            return interior_loss + bc_loss, interior_loss, bc_loss
        else:
            return interior_loss + bc_loss

    def __str__(self):
        return (
            f"Interior Loss: {self.interior_loss}, Boundary Loss: {self.boundary_loss}"
        )


def DeepRitz(
    # model: nn.Module,
    ux_domain: torch.Tensor,
    ux_bound: torch.Tensor,
    source_y: torch.Tensor,
    bound_y: torch.Tensor,
    # source_fn: Callable,  # domain function
    # bound_fn: Callable,  # boundary function
    x_domain: torch.Tensor,
    x_bound: torch.Tensor,
):
    """Summary
    Estimate Deep-Ritz method loss function

    g = Int_{\Omega} 1/2 |grad u|^2 - f u dx

    Args:
        u (nn.Module): neural network function estimator
        f (Callable): domain function
        x (torch.Tensor): input data

    Imported from: https://github.com/bizoffermark/neural_wos/blob/master/wos/utils/losses.py
    """
    assert source_y.shape[0] == x_domain.shape[0]
    assert bound_y.shape[0] == x_bound.shape[0]
    x_domain.requires_grad_(True)
    # ux = model(x_domain)
    fux = source_y * ux_domain
    du_dx = torch.autograd.grad(ux_domain.sum(), x_domain, create_graph=True)[0]
    norm_du_dx = (du_dx**2).sum(dim=-1)
    x_domain.requires_grad_(False)

    # ux_bound = model(x_bound)
    interior_loss = torch.mean(1 / 2 * norm_du_dx + fux)
    bound_loss = torch.mean((ux_bound - bound_y) ** 2)

    return interior_loss, bound_loss


class DeepRitzLoss(object):
    def __init__(
        self,
        interior_weight: float,
        boundary_weight: float,
        diff_method: str = "autograd",
    ):
        super().__init__()
        self.interior_weight = interior_weight
        self.boundary_weight = boundary_weight
        self.method = diff_method

    def autograd(
        self,
        out: Union[torch.Tensor, dict],
        y: Union[torch.Tensor, dict],
        output_queries: Union[torch.Tensor, dict],
        f: torch.Tensor,  # [source_fn, bound_fn, dist_fn]
        num_boundary: int = 0,
        **kwargs,
    ):
        if isinstance(out, dict):
            ux_domain = out["domain"]
            ux_bound = out["boundary"]
        else:
            ux_domain = out[:, num_boundary:, ...]
            ux_bound = out[:, :num_boundary, ...]

            # x_bound = output_queries['boundary']
            # x_bound = output_queries[:, :num_boundary]

        if isinstance(y, dict):
            source_y = y["domain"]
            bound_y = y["boundary"]
        else:
            source_y = y[:, num_boundary:, ...]
            bound_y = y[:, :num_boundary, ...]

        fux = source_y * ux_domain
        if isinstance(output_queries, dict):
            x_domain = output_queries["domain"]
            du_dx = torch.autograd.grad(ux_domain.sum(), x_domain, create_graph=True)[0]
        else:
            du_dx = torch.autograd.grad(
                ux_domain.sum(), output_queries, create_graph=True
            )[0][:, num_boundary:, ...]
        # if du_dx is None:
        # return torch.tensor(0.0), torch.tensor(0.0)
        norm_du_dx = (du_dx**2).sum(dim=-1)
        # x_domain.requires_grad_(False)

        interior_loss = torch.mean(1 / 2 * norm_du_dx + fux)
        bound_loss = torch.mean((ux_bound - bound_y) ** 2)

        interior_loss *= self.interior_weight
        bound_loss *= self.boundary_weight
        if kwargs.get("return_individual_losses", False):
            return interior_loss + bound_loss, interior_loss, bound_loss
        else:
            return interior_loss + bound_loss

    def __call__(
        self,
        out: Union[torch.Tensor, dict],
        y: Union[torch.Tensor, dict],
        output_queries: Union[torch.Tensor, dict],
        f: torch.Tensor,  # [source_fn, bound_fn, dist_fn]
        num_boundary: int = 0,
        **kwargs,
    ):
        if self.method == "autograd":
            return self.autograd(out, y, output_queries, f, num_boundary, **kwargs)
        elif self.method == "finite_difference":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __str__(self):
        return f"Interior Loss: {self.interior_weight}, Boundary Loss: {self.boundary_weight}"


class NonlinearDeepRitzLoss(object):
    """
    Nonlinear DeepRitz solves the Poisson equation ∇·(k(x)∇u) - α(x)u = -f(x)
    by definining the energy functional J(u) = int(1/2 k |∇u|^2 - α(x)u + f(x)u
    """

    def __init__(
        self,
        interior_weight: float,
        boundary_weight: float,
        diff_method: str = "autograd",
    ):
        super().__init__()
        self.interior_weight = interior_weight
        self.boundary_weight = boundary_weight
        self.method = diff_method

    def autograd(
        self,
        u: Union[torch.Tensor, dict],
        y: Union[torch.Tensor, dict],
        output_queries: Union[torch.Tensor, dict],
        f: torch.Tensor,  # [source_fn, bound_fn, dist_fn]
        k: torch.Tensor,
        alpha: torch.Tensor,
        num_boundary: int = 0,
        **kwargs,
    ):
        num_domain = u.shape[1] - num_boundary
        if isinstance(u, dict):
            ux_domain = u["domain"]
            ux_bound = u["boundary"]
        else:
            ux_domain = u[:, :num_domain, ...]
            ux_bound = u[:, num_domain:, ...]

        if isinstance(y, dict):
            source_y = y["domain"]
            bound_y = y["boundary"]
        else:
            source_y = y[:, :num_domain, ...]
            bound_y = y[:, num_domain:, ...]

        fux = source_y * ux_domain
        if isinstance(output_queries, dict):
            x_domain = output_queries["domain"]
            du_dx = torch.autograd.grad(ux_domain.sum(), x_domain, create_graph=True)[0]
        else:
            du_dx = torch.autograd.grad(
                ux_domain.sum(), output_queries, create_graph=True
            )[0][:, :num_domain, ...]
        # if du_dx is None:
        # return torch.tensor(0.0), torch.tensor(0.0)
        norm_du_dx = (du_dx**2).sum(dim=-1)
        # x_domain.requires_grad_(False)

        k = k[:, :num_domain, :]
        alpha = alpha[:, :num_domain, :]

        interior_loss = torch.mean(
            1 / 2 * norm_du_dx * k - 1 / 2 * alpha * ux_domain**2 + fux
        )
        bound_loss = torch.mean((ux_bound - bound_y) ** 2)

        interior_loss *= self.interior_weight
        bound_loss *= self.boundary_weight
        if kwargs.get("return_individual_losses", False):
            return interior_loss + bound_loss, interior_loss, bound_loss
        else:
            return interior_loss + bound_loss

    def __call__(
        self,
        out: Union[torch.Tensor, dict],
        y: Union[torch.Tensor, dict],
        output_queries: Union[torch.Tensor, dict],
        f: torch.Tensor = None,  # [source_fn, bound_fn, dist_fn]
        k: torch.Tensor = None,
        alpha: torch.Tensor = None,
        num_boundary: int = 1024,
        **kwargs,
    ):
        if self.method == "autograd":
            # diffusion_term = diffusion(output_queries.squeeze(0)).unsqueeze(0)
            # absorption_term = absorption(output_queries.squeeze(0)).unsqueeze(0)
            if f is None:
                source_term = kwargs.get("source", None)
                f = source_term
            if k is None:
                diffusion_term = kwargs.get("diffusion", None).detach()
                k = diffusion_term
            if alpha is None:
                absorption_term = kwargs.get("absorption", None).detach()
                alpha = absorption_term
            return self.autograd(
                out, y, output_queries, f, k, alpha, num_boundary=1024, **kwargs
            )
        elif self.method == "finite_difference":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __str__(self):
        return f"Interior Loss: {self.interior_weight}, Boundary Loss: {self.boundary_weight}"


# class WoSPINOLoss(object):
#     """
#     WoSPINOLoss computes the loss for the PINO model in the context of the WoS (Weighted Operator Smoothing) framework.

#     Parameters
#     ----------
#     interior_weight : float
#         Weight for the interior loss.
#     boundary_weight : float
#         Weight for the boundary loss.
#     diff_method : str, optional
#         Method to compute derivatives, by default 'autograd'.
#     base_loss : Callable, optional
#         Base loss function to use, by default F.mse_loss.
#     """

#     def __init__(self,
#                  wos_weight=1.0,
#                  interior_weight=1.0,
#                  boundary_weight=1.0,
#                  diff_method='autograd',
#                  base_loss=F.mse_loss,
#                  d=2):
#         super().__init__()
#         self.wos_weight = wos_weight
#         self.interior_weight = interior_weight
#         self.boundary_weight = boundary_weight
#         self.diff_method = diff_method
#         self.base_loss = base_loss

#         self.pino_loss = LinearPoissonEqnLoss(
#             interior_weight=interior_weight,
#             boundary_weight=boundary_weight,
#             diff_method=diff_method,
#             base_loss=base_loss
#         )

#         self.wos_loss = LpLoss(d=d, p=2)

#     def __call__(self, out, y, **kwargs):
#         """
#         Computes the total loss for the WoSPINO model, which includes both the PINO loss and the WOS loss.

#         Parameters
#         ----------
#         out : dict or torch.Tensor
#             Model outputs. If a dict, it should contain 'domain' and 'boundary' keys.
#         y : dict or torch.Tensor
#             Ground truth values. If a dict, it should contain 'domain' and 'boundary' keys.
#         x_domain : torch.Tensor, optional
#             Input queries for the domain, by default None.
#         x_bound : torch.Tensor, optional
#             Input queries for the boundary, by default None.
#         model : nn.Module, optional
#             The model to use for predictions, by default None.

#         **kwargs : dict
#             Additional keyword arguments for the loss computation.

#         Returns
#         torch.Tensor
#             The total loss value.
#         """

#         pino_loss  = self.pino_loss(out, y, **kwargs)
#         wos_loss = self.wos_weight * self.wos_loss(out['domain'], y['domain'])

#         return pino_loss + wos_loss
