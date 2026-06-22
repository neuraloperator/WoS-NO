"""
Euler-Maruyama SDE / BSDE solver.

Ported from ``wos/solvers/sde_solver.py`` of the neural walk-on-spheres
codebase (https://github.com/bizoffermark/neural_wos) and adapted to WoS-NO:

    * the heavy ``BaseSolver`` machinery (wandb, metric/plot helpers, hydra
      instantiation of everything) is dropped; this class only needs a
      :class:`~src.solvers.sde.base_pde.BasePDEEquation` and a handful of
      hyper-parameters, so it can be built directly inside a loss object;
    * an explicit ``mode`` selects the stochastic representation:

        - ``"elliptic"``  : hitting-time representation. Walks each path until
          it is absorbed at the boundary, accumulating the source integral
          along the way and adding the Dirichlet value at the exit point.
          This is the original neural_wos behaviour.

        - ``"parabolic"`` : fixed-horizon Feynman-Kac representation for
          ``d_t u = (sigma^2 / 2) Laplacian(u) + f`` with terminal/boundary
          data ``g``. Each path is integrated for a fixed number of time
          steps (or until it exits the spatial domain), accumulating the
          source and evaluating ``g`` at the terminal position.

The optional control-variate (BSDE martingale) term ``- sigma * <grad u, dW>``
is retained: when a pointwise-differentiable ``grad_and_model`` callable is
supplied it lowers the estimator variance without introducing bias. For neural
*operator* weak supervision the model is not pointwise-callable, so the loss
leaves it off and uses the plain unbiased Monte-Carlo estimate as the target.

The stochastic process is ``dX = sigma * dW`` (Brownian motion with diffusivity
``sigma^2 / 2``); ``sigma = sqrt(2)`` recovers the standard Laplacian
``-Delta u = f`` normalisation used by the WoS estimator.
"""

from typing import Callable

import torch
from torch import nn

from .base_pde import BasePDEEquation


class SDESolver:
    """Monte-Carlo solver for elliptic and parabolic PDEs via SDE paths."""

    def __init__(
        self,
        pde: BasePDEEquation,
        n_traj: int = 128,
        max_step: int = 1000,
        time_step: float = 1e-2,
        sigma: float = 2**0.5,
        mode: str = "elliptic",
        horizon: float = None,
        nn_target: bool = False,
        n_traj_max_per_shard: int = None,
        device: str = None,
    ) -> None:
        assert mode in ("elliptic", "parabolic"), f"unknown mode {mode!r}"
        self.pde = pde
        self.n_traj = n_traj
        self.max_step = max_step
        self.time_step = time_step
        self.sigma = sigma
        self.mode = mode
        # number of time steps for the parabolic horizon; defaults to max_step
        if horizon is not None:
            self.n_time_steps = max(int(round(horizon / time_step)), 1)
        else:
            self.n_time_steps = max_step
        self.horizon = horizon
        self.nn_target = nn_target
        # shard the trajectories to cap peak memory
        self.n_traj_max_per_shard = n_traj_max_per_shard or n_traj
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        xs: torch.Tensor,
        model: nn.Module = None,
        grad_and_model: Callable = None,
        value_fn: Callable = None,
        return_std: bool = False,
        evaluate_mode: bool = True,
    ):
        """Estimate ``u`` at the query points ``xs`` (shape ``[N, d]``).

        Parameters
        ----------
        value_fn : Callable, optional
            value function ``w(s, X_s) ~= value_fn(s, x)`` evaluated along the
            paths (signature ``(t: float, x: Tensor[N,d]) -> Tensor[N,1]``).
            Required for *semilinear* (nonlinear) problems whose PDE exposes a
            ``driver(t, x, y)``: the reaction integral
            ``int f(s, X_s, w(s,X_s)) ds`` needs ``w`` along the path. For an
            operator/Picard scheme this is the current model / iterate
            prediction (detached). When the PDE is linear (no ``driver``) this
            is ignored and the linear source ``domain_fn`` is used instead.

        Returns ``mu`` (``evaluate_mode=True``), ``(xs, mu)``
        (``evaluate_mode=False``), or ``(xs, mu, std)`` (``return_std=True``).
        """
        if self.mode == "elliptic":
            return self._evaluate_elliptic(
                xs, model, grad_and_model, return_std, evaluate_mode
            )
        return self._evaluate_parabolic(
            xs, model, grad_and_model, value_fn, return_std, evaluate_mode
        )

    # ------------------------------------------------------------------ #
    # elliptic: walk to the boundary (hitting time)
    # ------------------------------------------------------------------ #
    def _evaluate_elliptic(self, xs, model, grad_and_model, return_std, evaluate_mode):
        n_sample, n_dim = xs.shape
        xs = xs.to(self.device)
        n_shard = max(self.n_traj // self.n_traj_max_per_shard, 1)

        val_mean = torch.zeros([n_sample, 1], device=self.device)
        val_mean2 = torch.zeros([n_sample, 1], device=self.device)

        for i in range(n_shard):
            if i == n_shard - 1:
                n_traj_shard = self.n_traj - i * self.n_traj_max_per_shard
            else:
                n_traj_shard = self.n_traj_max_per_shard

            x_it = xs.repeat(1, n_traj_shard).view(-1, n_dim)
            val = torch.zeros([x_it.shape[0], 1], device=self.device)
            dist = torch.ones(x_it.shape[0], device=self.device) * 1e2
            conv = torch.zeros(x_it.shape[0], dtype=torch.bool, device=self.device)
            not_conv = torch.logical_not(conv)

            for _ in range(self.max_step):
                _, dist[not_conv] = self.pde.get_closest_bound_fn(x_it[not_conv])
                conv = self.pde.bound_test_fn(d=dist)
                not_conv = torch.logical_not(conv)
                if not_conv.sum() == 0:
                    break
                dW = (
                    torch.randn_like(x_it[not_conv]) * self.time_step**0.5
                )
                # accumulate the source integral
                val[not_conv] -= (
                    2 / self.sigma**2 * self.time_step
                    * self.pde.domain_fn(x_it[not_conv])
                )
                # optional control variate (BSDE martingale term)
                if grad_and_model is not None:
                    grad, _ = grad_and_model(x_it[not_conv])
                    val[not_conv] -= self.sigma * (grad * dW).sum(dim=-1, keepdim=True)
                x_it[not_conv] += dW * self.sigma  # Euler-Maruyama step

            # terminal contribution
            if self.nn_target and model is not None:
                x_closest, _ = self.pde.get_closest_bound_fn(x_it[conv])
                val[conv] += self.pde.bound_fn(x_closest)
                val[not_conv] += model(x_it[not_conv])
            else:
                x_closest, _ = self.pde.get_closest_bound_fn(x_it)
                val += self.pde.bound_fn(x_closest)

            shard_sum = val.view(-1, n_traj_shard).sum(dim=-1, keepdim=True)
            val_mean += shard_sum
            if return_std:
                val_mean2 += val.view(-1, n_traj_shard).pow(2).sum(dim=-1, keepdim=True)

        mu = val_mean / self.n_traj
        return self._pack(xs, mu, val_mean2, return_std, evaluate_mode)

    # ------------------------------------------------------------------ #
    # parabolic: fixed-horizon Feynman-Kac
    # ------------------------------------------------------------------ #
    def _evaluate_parabolic(
        self, xs, model, grad_and_model, value_fn, return_std, evaluate_mode
    ):
        n_sample, n_dim = xs.shape
        xs = xs.to(self.device)
        n_shard = max(self.n_traj // self.n_traj_max_per_shard, 1)

        # A nonlinear (semilinear) PDE exposes a driver f(t, x, y); the reaction
        # integral then needs the solution value along the path, supplied by
        # value_fn. With no driver/value_fn we fall back to the linear source.
        is_semilinear = value_fn is not None and hasattr(self.pde, "driver")

        val_mean = torch.zeros([n_sample, 1], device=self.device)
        val_mean2 = torch.zeros([n_sample, 1], device=self.device)

        for i in range(n_shard):
            if i == n_shard - 1:
                n_traj_shard = self.n_traj - i * self.n_traj_max_per_shard
            else:
                n_traj_shard = self.n_traj_max_per_shard

            x_it = xs.repeat(1, n_traj_shard).view(-1, n_dim)
            val = torch.zeros([x_it.shape[0], 1], device=self.device)
            # a path is "active" until it either reaches the horizon or exits
            # the spatial domain (lateral Dirichlet boundary)
            absorbed = torch.zeros(
                x_it.shape[0], dtype=torch.bool, device=self.device
            )

            t = 0.0
            for _ in range(self.n_time_steps):
                active = torch.logical_not(absorbed)
                if active.sum() == 0:
                    break
                # absorb paths that have left the spatial domain
                _, dist = self.pde.get_closest_bound_fn(x_it[active])
                on_bound = self.pde.bound_test_fn(d=dist)
                if on_bound.any():
                    idx_active = active.nonzero(as_tuple=True)[0]
                    hit = idx_active[on_bound]
                    x_closest, _ = self.pde.get_closest_bound_fn(x_it[hit])
                    val[hit] += self.pde.bound_fn(x_closest)
                    absorbed[hit] = True
                    active = torch.logical_not(absorbed)
                    if active.sum() == 0:
                        break
                dW = torch.randn_like(x_it[active]) * self.time_step**0.5
                # reaction / source integral over [t, T]
                if is_semilinear:
                    # nonlinear Feynman-Kac: int f(s, X_s, w(s, X_s)) ds, with
                    # w along the path provided by value_fn(t, x) (model/iterate)
                    y = value_fn(t, x_it[active])
                    val[active] += self.time_step * self.pde.driver(
                        t, x_it[active], y
                    )
                else:
                    val[active] += self.time_step * self.pde.domain_fn(x_it[active])
                if grad_and_model is not None:
                    grad, _ = grad_and_model(x_it[active])
                    val[active] -= self.sigma * (grad * dW).sum(dim=-1, keepdim=True)
                x_it[active] += dW * self.sigma
                t += self.time_step

            # terminal condition g at the horizon for paths still active
            active = torch.logical_not(absorbed)
            if active.any():
                if self.nn_target and model is not None:
                    val[active] += model(x_it[active])
                else:
                    val[active] += self.pde.bound_fn(x_it[active])

            shard_sum = val.view(-1, n_traj_shard).sum(dim=-1, keepdim=True)
            val_mean += shard_sum
            if return_std:
                val_mean2 += val.view(-1, n_traj_shard).pow(2).sum(dim=-1, keepdim=True)

        mu = val_mean / self.n_traj
        return self._pack(xs, mu, val_mean2, return_std, evaluate_mode)

    # ------------------------------------------------------------------ #
    def _pack(self, xs, mu, val_mean2, return_std, evaluate_mode):
        if return_std:
            std = (val_mean2 / self.n_traj - mu**2).clamp_min(0.0) ** 0.5
            return xs, mu, std
        if evaluate_mode:
            return mu
        return xs, mu
