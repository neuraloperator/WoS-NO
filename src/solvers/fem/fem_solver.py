"""First-class finite-element (scikit-fem) solver for the Allen-Cahn family, for
online ground-truth / supervision-target generation -- interchangeable with the
Monte-Carlo :class:`~src.solvers.sde.SDESolver`.

Solves the forward reaction-diffusion IVP equivalent to the terminal-value
(BSDE) problem. For a horizon-``T`` Allen-Cahn instance rescaled to unit horizon
(see :func:`~src.solvers.sde.rescale_to_unit_horizon`),

    d_tau u = Laplacian(u) + lam (u - u^3),   u(0, xi) = g(xi),   tau in [0, 1],

with the terminal-value solution recovered by ``w(t, .) = u_forward(1 - t, .)``.

Scalability
-----------
Every family instance shares the SAME rectangular mesh -- only the initial data
``g`` and the reaction strength ``lam`` change. So the mesh, mass/stiffness
assembly, and the ``(M + dt*K)`` LU factorisation are built ONCE in the
constructor and reused for every ``solve`` call; solving an instance is then just
cheap back-substitution time-stepping. This makes online FEM-supervised training
practical (no per-instance meshing -- the cost that cripples naive FEM datasets).

Backward-Euler diffusion + explicit group-FEM reaction (``M @ f(u_nodal)``).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.sparse.linalg import splu
from skfem import (Basis, BilinearForm, ElementTetP1, ElementTriP1, MeshTet,
                   MeshTri)
from skfem.helpers import dot, grad


@BilinearForm
def _mass(u, v, _):
    return u * v


@BilinearForm
def _stiff(u, v, _):
    return dot(grad(u), grad(v))


class GridField:
    """Multilinear interpolation of a nodal field on a regular ``[-L, L]^d`` grid
    via ``grid_sample`` (GPU) -- a cheap pointwise ``field(x)`` callable."""

    def __init__(self, values_grid: torch.Tensor, L: float, dim: int, device):
        self.L, self.dim = L, dim
        g = values_grid.to(device)
        # grid_sample input layout: 2D [1,1,H(y),W(x)], 3D [1,1,D(z),H(y),W(x)]
        self.field = (g.permute(1, 0) if dim == 2 else g.permute(2, 1, 0))[None, None]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gcoord = (x / self.L).clamp(-1 + 1e-6, 1 - 1e-6)
        N = gcoord.shape[0]
        shape = (1, N, 1, 2) if self.dim == 2 else (1, N, 1, 1, 3)
        out = torch.nn.functional.grid_sample(
            self.field, gcoord.view(*shape), align_corners=True,
            padding_mode="border")
        return out.view(N, 1)


class SpacetimeField:
    """Space-time field from ``K`` GridField slices at ascending times; linear in
    time between slices. ``__call__(t, x)`` with scalar ``t``."""

    def __init__(self, fields, times):
        order = sorted(range(len(times)), key=lambda i: times[i])
        self.fields = [fields[i] for i in order]
        self.times = [float(times[i]) for i in order]

    def __call__(self, t, x):
        ts = self.times
        if t <= ts[0]:
            return self.fields[0](x)
        if t >= ts[-1]:
            return self.fields[-1](x)
        import bisect
        j = bisect.bisect_right(ts, t) - 1
        a = (t - ts[j]) / (ts[j + 1] - ts[j])
        return (1 - a) * self.fields[j](x) + a * self.fields[j + 1](x)


class AllenCahnFEMSolver:
    """FEM solver for the Allen-Cahn forward IVP; mesh/assembly/factorisation
    built once and reused across instances (see module docstring)."""

    def __init__(self, dim: int, L: float = 4.5, n: int = None,
                 dt: float = 2e-3, device: str = None):
        self.dim = dim
        self.L = L
        self.n = n or (201 if dim == 2 else 41)
        self.dt = dt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        t0 = time.perf_counter()
        axis = np.linspace(-L, L, self.n)
        if dim == 2:
            mesh = MeshTri.init_tensor(axis, axis)
            basis = Basis(mesh, ElementTriP1())
        elif dim == 3:
            mesh = MeshTet.init_tensor(axis, axis, axis)
            basis = Basis(mesh, ElementTetP1())
        else:
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        self.basis = basis
        self.M = _mass.assemble(basis)
        self.K = _stiff.assemble(basis)
        self.lu = splu((self.M + dt * self.K).tocsc())     # reused every solve
        self.dofs = basis.doflocs                          # [dim, Ndof]
        self._x0 = torch.tensor(self.dofs.T, dtype=torch.float32)
        self._idx = np.rint((self.dofs + L) / (2 * L) * (self.n - 1)).astype(np.int64)
        self.setup_time = time.perf_counter() - t0

    def _to_field(self, u_nodal):
        grid = np.zeros((self.n,) * self.dim, dtype=np.float64)
        grid[tuple(self._idx)] = u_nodal
        return GridField(torch.tensor(grid, dtype=torch.float32),
                         self.L, self.dim, self.device)

    def solve(self, inst, T: float = 1.0, n_slices: int = 0,
              return_timing: bool = False):
        """Solve one instance's forward IVP to horizon ``T``.

        Returns the final :class:`GridField` (``u_forward(T, .)`` == ``w(0, .)``).
        With ``n_slices > 0`` also returns a :class:`SpacetimeField` of the
        forward solution ``u_forward(sigma, .)`` sampled at ``n_slices`` times
        (so ``w(t, .) = spacetime(1 - t/T, .)`` in unit-horizon coords).
        """
        u = inst.bound_fn(self._x0).squeeze(-1).numpy().astype(np.float64)
        lam = float(inst.lam)
        n_steps = int(round(T / self.dt))
        save_at, slices, sigmas = set(), [], []
        if n_slices:
            save_at = {int(round(j * n_steps / (n_slices - 1)))
                       for j in range(n_slices)}
            if 0 in save_at:
                slices.append(self._to_field(u)); sigmas.append(0.0)

        t0 = time.perf_counter()
        for step in range(n_steps):
            u = self.lu.solve(self.M @ u + self.dt * (self.M @ (lam * (u - u ** 3))))
            if n_slices and (step + 1) in save_at:
                slices.append(self._to_field(u))
                sigmas.append((step + 1) * self.dt / T)     # tau in [0,1]
        solve_time = time.perf_counter() - t0

        field = self._to_field(u)
        out = [field]
        if n_slices:
            out.append(SpacetimeField(slices, sigmas))
        if return_timing:
            out.append(solve_time)
        return out[0] if len(out) == 1 else tuple(out)
