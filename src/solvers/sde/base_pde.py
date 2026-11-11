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
    ``value_fn``, which must be time-dependent.

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


class AllenCahnFamily(SemilinearParabolicPDE):
    """A *parametric family* of Allen-Cahn instances for operator learning.

        d_t w + (sigma^2 / 2) Laplacian(w) + lam * (w - w^3) = 0,  w(T, x) = g(x)

    solved on the whole space (terminal-value / BSDE form), where each family
    member is defined by:

        * a terminal condition ``g``, a mixture of ``K`` radial bumps

              g(x) = sum_k  amp_k / (offset_k + width_k * ||x - center_k||^2)

          which generalises the single fixed profile ``1 / (2 + 0.4 ||x||^2)``
          of :class:`AllenCahnND` (recovered by K=1, amp=1, offset=2, width=0.4,
          center=0). Amplitudes are drawn large enough that ``g`` reaches ~1,
          i.e. into the regime where the ``w^3`` term actually bites and the
          problem is meaningfully nonlinear (not a disguised heat equation);

        * a reaction strength ``lam`` scaling the nonlinear driver.

    An operator is trained to regress ``g (and lam) -> w(0, .)``. The bump
    parametrisation is a finite, exact description of ``g``, so it doubles as
    the operator's branch/conditioning input via :meth:`param_vector`.

    Works in any spatial dimension (only ``||x - center||`` is used), so the
    same class backs the 2-D and 3-D training families.
    """

    def __init__(
        self,
        n_dim: int,
        centers: torch.Tensor,
        amps: torch.Tensor,
        offsets: torch.Tensor,
        widths: torch.Tensor,
        lam: float = 1.0,
        sigma: float = 2**0.5,
        stop_tol: float = 1e-3,
    ) -> None:
        super().__init__(n_dim=n_dim, stop_tol=stop_tol, sigma=sigma)
        # bump params: centers [K, d], amps/offsets/widths [K]
        self.centers = torch.as_tensor(centers, dtype=torch.float32)
        self.amps = torch.as_tensor(amps, dtype=torch.float32).reshape(-1)
        self.offsets = torch.as_tensor(offsets, dtype=torch.float32).reshape(-1)
        self.widths = torch.as_tensor(widths, dtype=torch.float32).reshape(-1)
        self.lam = float(lam)
        self.n_bumps = self.centers.shape[0]

    def driver(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.lam * (y - y**3)



    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        # g(x) = sum_k amp_k / (offset_k + width_k ||x - c_k||^2)
        centers = self.centers.to(x.device)            # [K, d]
        amps = self.amps.to(x.device)                  # [K]
        offsets = self.offsets.to(x.device)            # [K]
        widths = self.widths.to(x.device)              # [K]
        # ||x - c_k||^2 : [N, K]
        diff = x.unsqueeze(1) - centers.unsqueeze(0)   # [N, K, d]
        r2 = (diff**2).sum(dim=-1)                      # [N, K]
        terms = amps / (offsets + widths * r2)         # [N, K]
        return terms.sum(dim=-1, keepdim=True)         # [N, 1]

    def param_vector(self) -> torch.Tensor:
        """Flat parameter vector describing this instance (operator branch input).

        Layout per bump: ``[amp, offset, width, center(d)]`` concatenated over
        the K bumps, followed by ``[lam]``. Length ``K * (3 + d) + 1``.
        """
        per_bump = torch.cat(
            [
                self.amps.reshape(-1, 1),
                self.offsets.reshape(-1, 1),
                self.widths.reshape(-1, 1),
                self.centers.reshape(self.n_bumps, self.n_dim),
            ],
            dim=-1,
        ).reshape(-1)
        return torch.cat([per_bump, torch.tensor([self.lam])])


def sample_allen_cahn_family(
    n_dim: int,
    n_bumps=(1, 8),
    amp_range: Tuple[float, float] = (0.4, 1.5),
    offset_range: Tuple[float, float] = (1.8, 3.2),
    width_range: Tuple[float, float] = (0.3, 1.2),
    center_range: Tuple[float, float] = (-1.0, 1.0),
    lam_range: Tuple[float, float] = (0.5, 2.0),
    sigma: float = 2**0.5,
    signed: bool = False,
    generator: torch.Generator = None,
) -> AllenCahnFamily:
    """Draw one random :class:`AllenCahnFamily` instance.

    ``n_bumps`` may be a fixed ``int`` or an inclusive ``(lo, hi)`` range, in
    which case the number of radial bumps is itself sampled per instance -- so
    the family varies in the *structure* of ``g`` (how many features it has),
    not just the bump parameters. Because the operator sees ``g`` sampled on a
    point cloud (not the bump parameters), a variable bump count needs no change
    to the operator.

    ``signed=True`` gives each bump a random +/- sign, so ``g`` takes both signs
    and the solution develops genuine Allen-Cahn *interfaces* between the
    ``u=+1`` and ``u=-1`` phases. This is required at long horizons: with only
    positive ``g`` the strong accumulated reaction pulls every instance to the
    single ``u=+1`` attractor, collapsing the solution map to a constant
    (``w(0,.) ~ 1`` regardless of ``g``); signed ``g`` keeps the map
    ``g -> w`` non-degenerate. See :func:`rescale_to_unit_horizon`.

    Amplitude / offset ranges are chosen so the terminal ``g`` (single-bump
    peak ``~ amp / offset``, mildly higher where bumps overlap) reaches ~0.8-1.1:
    firmly into the nonlinear (``w^3``) regime -- at ``w~0.9`` the reaction
    ``w - w^3`` is ~30% of ``w`` -- while keeping the solution below the
    ``w = 1`` stable equilibrium, where the slow ``1/r^2`` tail of ``g`` would
    otherwise make the whole-space value sensitive to the far field. With more
    overlapping bumps the peak can climb, so the amplitude band is kept modest.
    In this range the stochastic (BSDE) target and an independent FEM solve of
    the equivalent forward IVP agree to a few percent (see
    ``scripts/fem_crosscheck.py``).
    """

    def _u(lo, hi, shape):
        return lo + (hi - lo) * torch.rand(shape, generator=generator)

    if isinstance(n_bumps, (tuple, list)):
        lo, hi = int(n_bumps[0]), int(n_bumps[1])
        n_bumps = int(torch.randint(lo, hi + 1, (1,), generator=generator).item())

    centers = _u(*center_range, (n_bumps, n_dim))
    # normalise per-bump amplitude by (2/K)^0.75 (anchored so K=2 is unchanged):
    # more bumps -> more, individually smaller features, so the solution stays
    # in the controlled ~<=1 regime (p90 peak ~1.0) instead of climbing with the
    # bump count. Empirically tuned to keep the FEM<->BSDE agreement.
    amps = _u(*amp_range, (n_bumps,)) * (2.0 / n_bumps) ** 0.75
    if signed:
        signs = torch.where(
            torch.rand(n_bumps, generator=generator) < 0.5, -1.0, 1.0)
        amps = amps * signs
    offsets = _u(*offset_range, (n_bumps,))
    widths = _u(*width_range, (n_bumps,))
    lam = float(_u(*lam_range, (1,)).item())
    return AllenCahnFamily(
        n_dim=n_dim,
        centers=centers,
        amps=amps,
        offsets=offsets,
        widths=widths,
        lam=lam,
        sigma=sigma,
    )


def rescale_to_unit_horizon(inst: "AllenCahnFamily", T: float) -> "AllenCahnFamily":
    """Diffusive rescaling of a horizon-``T`` instance to a unit horizon.

    With ``tau = t / T`` and ``xi = x / sqrt(T)`` the Allen-Cahn terminal-value
    problem on ``[0, T]`` maps to an equivalent problem on ``[0, 1]``::

        d_tau w + Lap_xi w + (T lam)(w - w^3) = 0,   w(1, xi) = g(sqrt(T) xi)

    (diffusion unchanged: ``dxi = sqrt(2) dW_tau``). The returned instance is
    that unit-horizon problem, so training/solving runs with ``horizon=1`` and a
    handful of steps regardless of how large ``T`` is -- the physical horizon
    enters only through the amplified reaction ``T lam`` and the rescaled ``g``.

    ``g(sqrt(T) xi) = sum_k a_k / (b_k + w_k T ||xi - c_k / sqrt(T)||^2)``, i.e.
    ``centers -> c_k / sqrt(T)``, ``widths -> w_k T``, ``amps``/``offsets`` kept.
    The physical solution is recovered by ``w(t, x) = w_tilde(t / T, x / sqrt(T))``.
    """
    s = T ** 0.5
    return AllenCahnFamily(
        n_dim=inst.n_dim,
        centers=inst.centers / s,
        amps=inst.amps,
        offsets=inst.offsets,
        widths=inst.widths * T,
        lam=inst.lam * T,
        sigma=inst.sigma,
    )


class BoxPoissonPDE(BasePDEEquation):
    """``-Delta u = f`` on a box ``[lo, hi]^d`` with Dirichlet data.

    Defaults give the quadratic test problem with exact solution
    ``u(x) = |x|^2``: since ``Delta |x|^2 = 2 d``, the source in ``-Delta u = f``
    form is ``f = -2 d``, and the Dirichlet data is ``g = |x|^2``.

    Override ``source_fn`` / ``boundary_fn`` (callables taking ``x``) to plug in
    a different instance without subclassing. ``source_fn`` must return ``f`` in
    the ``-Delta u = f`` convention that :class:`~src.solvers.sde.SDESolver`
    assumes in elliptic mode.
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
        # f = -Delta u = -2d for the exact solution u = |x|^2
        return -2.0 * self.n_dim * torch.ones([x.shape[0], 1], device=x.device)

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
