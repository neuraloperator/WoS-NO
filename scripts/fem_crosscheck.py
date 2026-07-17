"""Independent finite-element (scikit-fem) cross-check of the Allen-Cahn family
BSDE target, in 2-D and 3-D.

Why FEM (scikit-fem) and not FEniCS/dolfin: legacy FEniCS/dolfin is not
pip-installable in this environment; scikit-fem is a genuine finite-element
library and serves the same role -- an independent, *non-Monte-Carlo* classical
solver to validate the stochastic (BSDE) target against.

Equivalence used
----------------
The family target is ``w(0, x)`` of the terminal-value problem

    d_t w + Laplacian(w) + lam (w - w^3) = 0,   w(T, x) = g(x).

Under the time reversal ``tau = T - t`` this is the forward IVP

    d_tau u = Laplacian(u) + lam (u - u^3),   u(0, x) = g(x),

with ``w(0, x) = u(T, x)``. We solve the forward IVP with FEM (backward-Euler
diffusion, explicit group-FEM reaction) on a large box with natural (zero-flux)
boundaries -- big enough that the boundary does not influence the near-origin
region over the short horizon T -- and compare ``u(T, .)`` against the BSDE
Monte-Carlo target on query points near the origin.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import time

import numpy as np
import torch
from scipy.sparse.linalg import splu
from skfem import (Basis, BilinearForm, ElementTetP1, ElementTriP1, MeshTet,
                   MeshTri)
from skfem.helpers import dot, grad

# torch-only sde subpackage (skip heavy src/__init__ that needs fenics/jax)
_ROOT = Path(__file__).resolve().parents[1]
try:
    from src.solvers.sde import SDESolver, sample_allen_cahn_family
except ModuleNotFoundError:
    for _n, _p in [("src", _ROOT / "src"), ("src.solvers", _ROOT / "src/solvers")]:
        _m = types.ModuleType(_n)
        _m.__path__ = [str(_p)]
        sys.modules[_n] = _m
    from src.solvers.sde import SDESolver, sample_allen_cahn_family


@BilinearForm
def mass(u, v, _):
    return u * v


@BilinearForm
def stiff(u, v, _):
    return dot(grad(u), grad(v))


class GridField:
    """Fast multilinear interpolation of a nodal FEM field defined on a regular
    tensor grid, via ``grid_sample`` (GPU). Replaces slow scipy point-location.
    """

    def __init__(self, values_grid: torch.Tensor, L: float, dim: int, device):
        self.L = L
        self.dim = dim
        self.device = device
        if dim == 2:
            # grid_sample input [1,1,H(y),W(x)]; values_grid indexed [ix,iy]
            self.field = values_grid.permute(1, 0)[None, None].to(device)
        else:
            # [1,1,D(z),H(y),W(x)]; values_grid indexed [ix,iy,iz]
            self.field = values_grid.permute(2, 1, 0)[None, None].to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        g = (x / self.L).clamp(-1 + 1e-6, 1 - 1e-6)   # [N, d], order (x,y[,z])
        N = g.shape[0]
        if self.dim == 2:
            grid = g.view(1, N, 1, 2)
            out = torch.nn.functional.grid_sample(
                self.field, grid, align_corners=True, padding_mode="border")
            return out.view(N, 1)
        grid = g.view(1, N, 1, 1, 3)
        out = torch.nn.functional.grid_sample(
            self.field, grid, align_corners=True, padding_mode="border")
        return out.view(N, 1)


def fem_solve(inst, dim, device, L=4.0, n=None, T=0.3, dt=0.003,
              return_timing=False, n_slices=0):
    """Forward Allen-Cahn IVP via FEM; returns a GridField for u(T, .).

    With ``return_timing=True`` also returns ``(t_mesh, t_solve)``: ``t_mesh`` is
    the mesh generation + basis/DOF setup + matrix assembly + factorisation cost
    (the setup a mesh-free BSDE solver never pays), ``t_solve`` the time-stepping.

    With ``n_slices > 0`` also returns ``(slice_fields, slice_taus)``: ``n_slices``
    GridFields of the forward solution ``u_forward`` sampled at evenly spaced
    forward times ``sigma in [0, T]`` (mapped to ``tau = sigma / T`` in [0,1]).
    These give the *true* space-time field for a controlled value_fn: the
    terminal-value solution is ``w(t, .) = u_forward(T - t*T, .)``.
    """
    if n is None:
        n = 161 if dim == 2 else 33
    axis = np.linspace(-L, L, n)

    t0 = time.perf_counter()
    if dim == 2:
        mesh = MeshTri.init_tensor(axis, axis)
        basis = Basis(mesh, ElementTriP1())
    else:
        mesh = MeshTet.init_tensor(axis, axis, axis)
        basis = Basis(mesh, ElementTetP1())

    M = mass.assemble(basis)
    K = stiff.assemble(basis)
    A = (M + dt * K).tocsc()
    lu = splu(A)
    t_mesh = time.perf_counter() - t0

    # initial condition u0 = g at DOF locations (P1 -> nodal)
    dofs = basis.doflocs                      # [dim, Ndof]
    x0 = torch.tensor(dofs.T, dtype=torch.float32)
    u = inst.bound_fn(x0).squeeze(-1).numpy().astype(np.float64)
    idx = np.rint((dofs + L) / (2 * L) * (n - 1)).astype(np.int64)  # [dim, Ndof]

    def to_field(u_nodal):
        grid = np.zeros((n,) * dim, dtype=np.float64)
        grid[tuple(idx)] = u_nodal
        return GridField(torch.tensor(grid, dtype=torch.float32), L, dim, device)

    n_steps = int(round(T / dt))
    slice_fields, slice_sigmas = [], []
    if n_slices:
        save_at = {int(round(j * n_steps / (n_slices - 1))) for j in range(n_slices)}
        if 0 in save_at:
            slice_fields.append(to_field(u)); slice_sigmas.append(0.0)

    t0 = time.perf_counter()
    lam = inst.lam
    for step in range(n_steps):
        reaction = lam * (u - u**3)           # group-FEM: nodal f(u)
        rhs = M @ u + dt * (M @ reaction)
        u = lu.solve(rhs)
        if n_slices and (step + 1) in save_at:
            slice_fields.append(to_field(u)); slice_sigmas.append((step + 1) * dt)
    t_solve = time.perf_counter() - t0

    field = to_field(u)
    out = [field]
    if return_timing:
        out += [t_mesh, t_solve]
    if n_slices:
        out.append((slice_fields, [s / T for s in slice_sigmas]))   # taus in [0,1]
    return out[0] if len(out) == 1 else tuple(out)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"FEM (scikit-fem) cross-check of BSDE target | device={dev}\n")
    for dim in (2, 3):
        gen = torch.Generator().manual_seed(2024)
        rels, rmses = [], []
        n_inst = 8 if dim == 2 else 5
        for i in range(n_inst):
            inst = sample_allen_cahn_family(n_dim=dim, n_bumps=(1, 8), generator=gen)
            field = fem_solve(inst, dim, dev)

            # query points near origin (where the box boundary is irrelevant)
            xs = (torch.rand(200, dim, generator=gen) * 2 - 1).to(dev)
            fem_vals = field(xs).cpu()

            # BSDE target with the FEM field as value_fn, so the driver reaction
            # along the path matches -> this isolates the *representation* error
            # (whole-space MC vs bounded FEM), not a self-training gap.
            def value_fn(t, x, field=field):
                return field(x)

            solver = SDESolver(
                pde=inst, n_traj=1024, time_step=0.015, horizon=0.3, sigma=2**0.5,
                mode="parabolic", n_traj_max_per_shard=1024, device=dev,
            )
            bsde_vals = solver.evaluate(
                xs, value_fn=value_fn, evaluate_mode=True
            ).cpu()

            diff = fem_vals - bsde_vals
            rel = (diff.norm() / bsde_vals.norm().clamp_min(1e-8)).item()
            rmse = diff.pow(2).mean().sqrt().item()
            rels.append(rel)
            rmses.append(rmse)
            print(f"  [d={dim}] inst {i}: K={inst.n_bumps} lam={inst.lam:.2f}  "
                  f"FEM<->BSDE rel L2={rel:.4f}  RMSE={rmse:.4f}  "
                  f"(FEM range {fem_vals.min():.3f}..{fem_vals.max():.3f})", flush=True)

        r = torch.tensor(rels)
        print(f"  => d={dim}: FEM vs BSDE  rel L2 = {r.mean():.4f} +/- {r.std():.4f}"
              f"   mean RMSE = {torch.tensor(rmses).mean():.4f}\n")


if __name__ == "__main__":
    main()
