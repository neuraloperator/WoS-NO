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

        - ``"parabolic"`` : fixed-horizon Feynman-Kac representation for
          ``d_t u = (sigma^2 / 2) Laplacian(u) + f`` with terminal/boundary
          data ``g``. Each path is integrated for a fixed number of time
          steps (or until it exits the spatial domain), accumulating the
          source and evaluating ``g`` at the terminal position.

Sign conventions (they differ between the two modes, as is standard):

    * elliptic  : ``-Laplacian(u) = f``, so ``domain_fn`` returns ``f`` and the
      Feynman-Kac weight on the accumulated source is ``sigma^2 / 2``;
    * parabolic : ``d_t u = (sigma^2 / 2) Laplacian(u) + f``, so ``domain_fn``
      returns the source ``f`` appearing in that form and its weight is 1.

Because the driving process ``dX = sigma dW`` has *constant* diffusivity, the
Euler-Maruyama update ``X_{k+1} = X_k + sigma sqrt(dt) Z`` samples the exact
law of Brownian motion at the grid times -- there is no path-discretisation
error in the forward state. The ``dt`` error in parabolic mode comes from the
BACKWARD half of the same Euler-Maruyama scheme -- the ``dt * f`` accumulation of
the driver (Bouchard-Touzi / Zhang), which is O(dt). This is one scheme, not a
choice of integration rule, so no order knob is offered.

Accuracy options (parabolic mode)
---------------------------------
``antithetic``
    pair each path with its sign-flipped Brownian increments. Unbiased, and
    cancels the odd part of the integrand's fluctuation.
``grad_fn``
    time-aware control variate ``grad_fn(t, x) -> grad_x w``. Subtracting the
    martingale ``int sigma <grad w, dW>`` leaves the mean unchanged (it is a
    martingale) while removing most of the variance; with an exact gradient the
    estimator becomes near-deterministic. This is the BSDE/diffusion-loss
    direction: ``grad_and_model`` is the older, time-unaware spelling.

``sigma = sqrt(2)`` recovers the standard Laplacian ``-Delta u = f``
normalisation used by the WoS estimator. When ``sigma`` is not given it is taken
from the PDE (``pde.sigma``) if present, so a PDE and its solver cannot silently
disagree about the diffusivity.
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
        sigma: float = None,
        mode: str = "elliptic",
        horizon: float = None,
        nn_target: bool = False,
        n_traj_max_per_shard: int = None,
        antithetic: bool = False,
        device: str = None,
    ) -> None:
        assert mode in ("elliptic", "parabolic"), f"unknown mode {mode!r}"
        self.pde = pde
        self.n_traj = n_traj
        self.max_step = max_step
        self.time_step = time_step
        # take the diffusivity from the PDE unless explicitly overridden, so a
        # PDE built with a non-default sigma cannot be silently ignored
        self.sigma = sigma if sigma is not None else getattr(pde, "sigma", 2**0.5)
        self.mode = mode
        # number of time steps for the parabolic horizon; defaults to max_step
        if horizon is not None:
            self.n_time_steps = max(int(round(horizon / time_step)), 1)
        else:
            self.n_time_steps = max_step
        self.horizon = horizon
        self.nn_target = nn_target
        self.antithetic = antithetic
        # shard the trajectories to cap peak memory
        self.n_traj_max_per_shard = n_traj_max_per_shard or n_traj
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # sharding helper
    # ------------------------------------------------------------------ #
    def _shard_sizes(self):
        """Trajectory counts per shard, none exceeding ``n_traj_max_per_shard``.

        Ceil-divides so the cap is a real bound (a floor-division would let the
        final shard carry the whole remainder and blow past it).
        """
        cap = self.n_traj_max_per_shard
        sizes = []
        remaining = self.n_traj
        while remaining > 0:
            sizes.append(min(cap, remaining))
            remaining -= sizes[-1]
        return sizes

    def _draw_dw(self, n_sample, n_traj_shard, n_dim):
        """Brownian increments for one shard, shaped ``[n_sample * n_traj, d]``.

        With ``antithetic`` the first half of each point's trajectories carries
        ``Z`` and the second half ``-Z``. The row layout matches
        ``xs.repeat(1, m).view(-1, d)`` (trajectory index is the fast axis), so
        trajectory ``j`` is paired with ``j + m / 2`` at the same query point.
        """
        if self.antithetic and n_traj_shard % 2 == 0:
            h = n_traj_shard // 2
            z = torch.randn(n_sample, h, n_dim, device=self.device)
            z = torch.cat([z, -z], dim=1).reshape(-1, n_dim)
        else:
            z = torch.randn(n_sample * n_traj_shard, n_dim, device=self.device)
        return z * self.time_step**0.5

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        xs: torch.Tensor,
        model: nn.Module = None,
        grad_and_model: Callable = None,
        value_fn: Callable = None,
        grad_fn: Callable = None,
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
            ``int f(s, X_s, w(s,X_s)) ds`` needs ``w`` along the path. It must
            be genuinely time-dependent -- passing a time-frozen field is a
            large, silent bias that grows with the accumulated reaction. When
            the PDE is linear (no ``driver``) this is ignored and the linear
            source ``domain_fn`` is used instead.
        grad_fn : Callable, optional
            control variate ``(t, x) -> grad_x w`` of shape ``[N, d]``. See the
            module docstring; ``grad_and_model`` is the time-unaware alias kept
            for backwards compatibility.

        Returns ``mu`` (``evaluate_mode=True``), ``(xs, mu)``
        (``evaluate_mode=False``), or ``(xs, mu, std)`` (``return_std=True``).
        """
        if grad_fn is None and grad_and_model is not None:
            grad_fn = lambda t, x: grad_and_model(x)[0]  # noqa: E731
        if self.mode == "elliptic":
            return self._evaluate_elliptic(
                xs, model, grad_fn, return_std, evaluate_mode
            )
        return self._evaluate_parabolic(
            xs, model, grad_fn, value_fn, return_std, evaluate_mode
        )

    # ------------------------------------------------------------------ #
    # elliptic: walk to the boundary (hitting time)
    # ------------------------------------------------------------------ #
    def _evaluate_elliptic(self, xs, model, grad_fn, return_std, evaluate_mode):
        n_sample, n_dim = xs.shape
        xs = xs.to(self.device)

        val_mean = torch.zeros([n_sample, 1], device=self.device)
        val_mean2 = torch.zeros([n_sample, 1], device=self.device)

        # Feynman-Kac weight for `-Laplacian(u) = f` under dX = sigma dW, whose
        # generator is (sigma^2 / 2) Laplacian:  u = E[g + int (sigma^2/2) f ds]
        src_weight = self.sigma**2 / 2

        for n_traj_shard in self._shard_sizes():
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
                dW = torch.randn_like(x_it[not_conv]) * self.time_step**0.5
                # accumulate the source integral
                val[not_conv] += (
                    src_weight * self.time_step
                    * self.pde.domain_fn(x_it[not_conv])
                )
                # optional control variate (BSDE martingale term)
                if grad_fn is not None:
                    grad = grad_fn(0.0, x_it[not_conv])
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

            val_mean += val.view(-1, n_traj_shard).sum(dim=-1, keepdim=True)
            if return_std:
                val_mean2 += val.view(-1, n_traj_shard).pow(2).sum(dim=-1, keepdim=True)

        mu = val_mean / self.n_traj
        return self._pack(xs, mu, val_mean2, return_std, evaluate_mode)

    # ------------------------------------------------------------------ #
    # parabolic: fixed-horizon Feynman-Kac
    # ------------------------------------------------------------------ #
    def _evaluate_parabolic(
        self, xs, model, grad_fn, value_fn, return_std, evaluate_mode
    ):
        n_sample, n_dim = xs.shape
        xs = xs.to(self.device)
        dt = self.time_step
        n_steps = self.n_time_steps
        T = dt * n_steps

        # A nonlinear (semilinear) PDE exposes a driver f(t, x, y); the reaction
        # integral then needs the solution value along the path, supplied by
        # value_fn. With no driver/value_fn we fall back to the linear source.
        is_semilinear = value_fn is not None and hasattr(self.pde, "driver")

        # Euler-Maruyama backward step: dt * f at the opening node of each
        # interval (non-anticipating).

        val_mean = torch.zeros([n_sample, 1], device=self.device)
        val_mean2 = torch.zeros([n_sample, 1], device=self.device)

        for n_traj_shard in self._shard_sizes():
            x_it = xs.repeat(1, n_traj_shard).view(-1, n_dim)
            val = torch.zeros([x_it.shape[0], 1], device=self.device)
            # a path is "active" until it either reaches the horizon or exits
            # the spatial domain (lateral Dirichlet boundary)
            absorbed = torch.zeros(
                x_it.shape[0], dtype=torch.bool, device=self.device
            )

            t = 0.0
            for k in range(n_steps):
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

                dW = self._draw_dw(n_sample, n_traj_shard, n_dim)[active]

                # reaction / source integral over [t, T]
                if is_semilinear:
                    # nonlinear Feynman-Kac: int f(s, X_s, w(s, X_s)) ds, with
                    # w along the path provided by value_fn(t, x)
                    y = value_fn(t, x_it[active])
                    val[active] += dt * self.pde.driver(t, x_it[active], y)
                else:
                    val[active] += dt * self.pde.domain_fn(x_it[active])
                if grad_fn is not None:
                    # Ito integral: the integrand must be non-anticipating, so
                    # the Kac factor is taken at the LEFT endpoint of the step
                    grad = grad_fn(t, x_it[active])
                    val[active] -= (
                        self.sigma * (grad * dW).sum(dim=-1, keepdim=True)
                    )
                x_it[active] += dW * self.sigma
                t += dt

            # terminal condition g at the horizon for paths still active
            active = torch.logical_not(absorbed)
            if active.any():
                if self.nn_target and model is not None:
                    g_T = model(x_it[active])
                else:
                    g_T = self.pde.bound_fn(x_it[active])
                val[active] += g_T

            val_mean += val.view(-1, n_traj_shard).sum(dim=-1, keepdim=True)
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
