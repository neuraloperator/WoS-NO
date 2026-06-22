"""
Lightweight PDE abstraction for the SDE / BSDE solver.

Ported from ``wos/pdes/base_pde.py`` of the neural walk-on-spheres codebase
(https://github.com/bizoffermark/neural_wos) and trimmed to the interface the
Euler-Maruyama walker actually consumes:

    * ``domain_fn(x)``           -> source term f(x)
    * ``bound_fn(x)``            -> Dirichlet data g(x) (also used as the
                                    terminal condition for parabolic problems)
    * ``get_closest_bound_fn(x)``-> (projection onto boundary, distance)
    * ``bound_test_fn(x, d)``    -> boolean mask of absorbed walks

A walker that only needs these four callables works unchanged for both the
elliptic (hitting-time) and parabolic (fixed-horizon Feynman-Kac) regimes, so
the same abstraction backs both modes of :class:`SDESolver`.

Two concrete, fully analytic examples are provided so the BSDE loss is runnable
out of the box and unit-testable without external geometry/data:

    * :class:`BoxPoissonPDE`     -> ``-Delta u = f`` on a box, Dirichlet g
    * :class:`AnnulusLaplacePDE` -> ``-Delta u = 0`` on an annulus, Dirichlet g
"""

from typing import List, Tuple

import torch

from .bound import (
    find_closest_point_on_annulus,
    find_closest_point_on_cube,
    test_rect_bound,
)


class BasePDEEquation:
    """Interface assumed by :class:`SDESolver`.

    The PDE is taken to be of the form::

        L[u](x) = 0      for x in the interior   (encoded by ``domain_fn``)
        u(x)    = g(x)   for x on the boundary   (``bound_fn``)

    For parabolic problems ``bound_fn`` doubles as the terminal condition
    ``u(T, x) = g(x)`` and ``is_time_dependent`` is set to ``True``.
    """

    is_time_dependent: bool = False

    def __init__(self, n_dim: int, stop_tol: float) -> None:
        self.n_dim = n_dim
        self.stop_tol = stop_tol

    def domain_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Source term f(x). Shape ``[N, 1]``."""
        raise NotImplementedError

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Dirichlet / terminal data g(x). Shape ``[N, 1]``."""
        raise NotImplementedError

    def get_closest_bound_fn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (closest boundary point, distance-to-boundary)."""
        raise NotImplementedError

    def bound_test_fn(
        self, x: torch.Tensor = None, d: torch.Tensor = None
    ) -> torch.Tensor:
        """Boolean mask of points within ``stop_tol`` of the boundary."""
        if d is None:
            d = self.get_closest_bound_fn(x)[1]
        return d < self.stop_tol

    @staticmethod
    def get_true_solution_fn(x: torch.Tensor) -> torch.Tensor:
        """Analytic solution, if known (used only for diagnostics)."""
        raise NotImplementedError


class SemilinearParabolicPDE(BasePDEEquation):
    """Semilinear parabolic PDE in terminal-value (BSDE) form::

        d_t w + (sigma^2 / 2) Laplacian(w) + f(t, x, w) = 0,   w(T, x) = g(x)

    solved on the whole space (no lateral spatial boundary by default). The
    nonlinear Feynman-Kac representation is

        w(t, x) = E[ g(X_T) + int_t^T f(s, X_s, w(s, X_s)) ds ],   dX = sigma dW

    which is a *plain* BSDE: the driver ``f`` depends on the value ``w`` (=Y)
    only, not on its gradient (=Z), so no gradient/policy network is needed
    (this is what distinguishes it from the forward-backward deep-BSDE/FBSDE
    solver). The path value ``w(s, X_s)`` is supplied to the solver via
    ``value_fn`` (the operator/MLP being trained, or a Picard iterate).

    Subclasses override :meth:`driver` (the reaction f) and :meth:`bound_fn`
    (the terminal data g).
    """

    is_time_dependent = True
    is_semilinear = True

    def __init__(
        self,
        n_dim: int,
        stop_tol: float = 1e-3,
        sigma: float = 2**0.5,
        terminal_fn=None,
    ) -> None:
        super().__init__(n_dim=n_dim, stop_tol=stop_tol)
        self.sigma = sigma
        self._terminal_fn = terminal_fn

    def driver(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Nonlinear reaction f(t, x, y). Default 0 recovers the linear heat eq."""
        return torch.zeros_like(y)

    def domain_fn(self, x: torch.Tensor) -> torch.Tensor:
        # no explicit linear source by default (reaction lives in `driver`)
        return torch.zeros([x.shape[0], 1], device=x.device)

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self._terminal_fn is not None:
            return self._terminal_fn(x)
        raise NotImplementedError("provide a terminal_fn or override bound_fn")

    def get_closest_bound_fn(self, x: torch.Tensor):
        # whole space: report an effectively infinite distance so the walker
        # never absorbs early and always integrates the full time horizon
        d = torch.full((x.shape[0],), 1e30, device=x.device)
        return x, d


class AllenCahnND(SemilinearParabolicPDE):
    """Allen-Cahn benchmark from E, Han & Jentzen (2017), arXiv:1706.04702.

        d_t u = Laplacian(u) + u - u^3,   u(0, x) = g(x),   x in R^d

    Posed in terminal-value (BSDE) form via the time reversal t -> T - t::

        d_t w + Laplacian(w) + (w - w^3) = 0,   w(T, x) = g(x)

    so the diffusion is ``dX = sqrt(2) dW`` (sigma = sqrt(2), generator = Lap),
    the driver is ``f(w) = w - w^3`` (depends on w only -> plain BSDE), and the
    terminal data is ``g(x) = 1 / (2 + 0.4 ||x||^2)``.

    Reference: with d = 100, T = 0.3, the benchmark value is u(0, 0) ~= 0.0528.
    """

    def __init__(
        self, n_dim: int = 100, stop_tol: float = 1e-3, sigma: float = 2**0.5
    ) -> None:
        super().__init__(n_dim=n_dim, stop_tol=stop_tol, sigma=sigma)

    def driver(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - y**3

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        r2 = (x**2).sum(dim=-1, keepdim=True)
        return 1.0 / (2.0 + 0.4 * r2)


class BoxPoissonPDE(BasePDEEquation):
    """``-Delta u = f`` on a box ``[lo, hi]^d`` with Dirichlet data.

    Defaults reproduce the DeepRitz benchmark ``-Delta u = 2 d`` with
    ``u = |x|^2`` on the boundary, whose exact solution is ``u(x) = |x|^2``.
    Override ``source_fn`` / ``boundary_fn`` (callables taking ``x``) to plug in
    a different instance without subclassing.
    """

    def __init__(
        self,
        n_dim: int = 2,
        stop_tol: float = 1e-3,
        min_max: List[float] = [0.0, 1.0],
        source_fn=None,
        boundary_fn=None,
    ) -> None:
        super().__init__(n_dim=n_dim, stop_tol=stop_tol)
        self.min_max = list(min_max)
        self._source_fn = source_fn
        self._boundary_fn = boundary_fn

    def domain_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self._source_fn is not None:
            return self._source_fn(x)
        return 2.0 * self.n_dim * torch.ones([x.shape[0], 1], device=x.device)

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self._boundary_fn is not None:
            return self._boundary_fn(x)
        return (x**2).sum(dim=-1, keepdim=True)

    def get_closest_bound_fn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return find_closest_point_on_cube(x, self.min_max)

    def domain_test_fn(self, x: torch.Tensor) -> torch.Tensor:
        in_box = test_rect_bound(x, self.min_max, self.stop_tol)
        return torch.logical_and(in_box, torch.logical_not(self.bound_test_fn(x)))

    @staticmethod
    def get_true_solution_fn(x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum(dim=-1, keepdim=True)


class AnnulusLaplacePDE(BasePDEEquation):
    """``-Delta u = 0`` on an annulus ``r_low <= |x| <= r_high``."""

    def __init__(
        self,
        n_dim: int = 2,
        stop_tol: float = 1e-3,
        r_low: float = 2.0,
        r_high: float = 4.0,
        source_fn=None,
        boundary_fn=None,
    ) -> None:
        super().__init__(n_dim=n_dim, stop_tol=stop_tol)
        self.r_low = r_low
        self.r_high = r_high
        self._source_fn = source_fn
        self._boundary_fn = boundary_fn

    def domain_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self._source_fn is not None:
            return self._source_fn(x)
        return torch.zeros([x.shape[0], 1], device=x.device)

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self._boundary_fn is not None:
            return self._boundary_fn(x)
        # log-kernel harmonic profile, =0 on the inner ring, =1 on the outer
        r = torch.norm(x, dim=-1, keepdim=True)
        return (torch.log(r / self.r_low) / torch.log(
            torch.tensor(self.r_high / self.r_low, device=x.device)
        ))

    def get_closest_bound_fn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return find_closest_point_on_annulus(x, self.r_low, self.r_high)
