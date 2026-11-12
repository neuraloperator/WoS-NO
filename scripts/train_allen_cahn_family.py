"""Train a GINO neural operator on a *family* of Allen-Cahn PDEs with BSDE weak
supervision (data-free), for 2-D and 3-D.

The family is :class:`src.solvers.sde.AllenCahnFamily`: a semilinear parabolic
Allen-Cahn instance

    d_t w + Laplacian(w) + lam (w - w^3) = 0,   w(T, x) = g(x),   dX = sqrt(2) dW

whose terminal condition ``g`` is a random mixture of radial bumps and whose
reaction strength ``lam`` is random. The operator learns ``(g, lam) -> w(0, .)``.
The stochastic (BSDE) target is validated against an independent FEM solve to
~1% by ``scripts/fem_crosscheck.py``.

Operator: GINO
--------------
``neuralop.models.gino.GINO`` (the WoS-NO operator), built from
``configs/model/gino.yaml`` (2-D) / ``gino3d.yaml`` (3-D). The input function
``(g, lam)`` is sampled on a *fixed* input point cloud as feature channels
``[g(x_i), lam]``; because the geometry (input cloud, latent grid, query points)
is shared across the batch, a whole batch of instances is a single GINO forward
(GINO's supported batching regime).

Supervision (data-free)
-----------------------
For each instance the target is built on the fly by the nonlinear
Feynman-Kac / BSDE estimator (:class:`SDESolver`, ``mode="parabolic"``). The
driver needs the solution value along the path; we supply it as the operator's
*own current prediction*, read off a regular grid via fast ``grid_sample``
interpolation (:class:`GridField`) -- Picard-in-the-loop / deep-BSDE-style
self-supervision. No precomputed dataset, no mesh.

Metric
------
Held-out fixed-point residual: on fresh instances, relative L2 between the
operator and a high-trajectory BSDE target that uses the operator as
``value_fn`` -- the weak-form PDE residual.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

# torch-only sde subpackage (skip heavy src/__init__ that needs fenics/jax)
_ROOT = Path(__file__).resolve().parents[1]
try:
    from src.solvers.sde import SDESolver, sample_allen_cahn_family
except ModuleNotFoundError:
    import sys, types

    for _n, _p in [("src", _ROOT / "src"), ("src.solvers", _ROOT / "src/solvers")]:
        _m = types.ModuleType(_n)
        _m.__path__ = [str(_p)]
        sys.modules[_n] = _m
    from src.solvers.sde import SDESolver, sample_allen_cahn_family

import torch
from torch import nn
from neuralop.models.gino import GINO


# --------------------------------------------------------------------------- #
# fast interpolation of a nodal field on a regular grid (for value_fn)
# --------------------------------------------------------------------------- #
class GridField:
    """Multilinear interpolation of values on a regular ``[-L, L]^d`` grid via
    ``grid_sample`` -- turns the operator's grid prediction into a cheap
    pointwise ``value_fn(x)`` for the SDE walker."""

    def __init__(self, values_grid: torch.Tensor, L: float, dim: int):
        self.L, self.dim = L, dim
        if dim == 2:                       # grid_sample input [1,1,H(y),W(x)]
            self.field = values_grid.permute(1, 0)[None, None]
        else:                              # [1,1,D(z),H(y),W(x)]
            self.field = values_grid.permute(2, 1, 0)[None, None]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        g = (x / self.L).clamp(-1 + 1e-6, 1 - 1e-6)     # coords order (x,y[,z])
        N = g.shape[0]
        shape = (1, N, 1, 2) if self.dim == 2 else (1, N, 1, 1, 3)
        out = nn.functional.grid_sample(
            self.field.to(x.device), g.view(*shape),
            align_corners=True, padding_mode="border")
        return out.view(N, 1)


# --------------------------------------------------------------------------- #
# GINO construction / geometry
# --------------------------------------------------------------------------- #
def build_gino(dim: int, device, in_radius=None, out_radius=None):
    """Build GINO from the repo config; returns (model, in_channels).

    We use the ``gino.yaml`` architecture for *both* dimensions (4 FNO layers;
    the ``gino3d.yaml`` preset is a lower-capacity regime tuned for another
    problem and badly underfits this family), adapting only the dimensionality:
    ``gno_coord_dim`` and the FNO mode list. ``in_channels`` stays at the config
    value (the GNO kernel-MLP output dim is tied to it); features are zero-padded
    to that width. The GNO radii are overridden to our domain scale -- the config
    defaults assume a unit-ish domain, so on a ``[-2,2]^d`` cloud they gather
    almost no neighbours. The *in* radius must exceed the input-cloud spacing,
    the *out* radius the latent-grid spacing; these differ, so set separately.
    """
    cfg = yaml.safe_load(open(_ROOT / "configs/model/gino.yaml"))
    cfg.pop("_target_", None)
    for k, v in list(cfg.items()):        # YAML 'None'/'null' strings -> None
        if isinstance(v, str) and v in ("None", "null"):
            cfg[k] = None
    cfg["gno_coord_dim"] = dim
    cfg["fno_n_modes"] = [12] * dim if dim == 3 else cfg["fno_n_modes"]
    if in_radius is not None:
        cfg["in_gno_radius"] = in_radius
    if out_radius is not None:
        cfg["out_gno_radius"] = out_radius
    return GINO(**cfg).to(device), int(cfg["in_channels"])


def make_geometry(dim, device, n_in=None, grid_res=None, grid_L=3.0,
                  latent_res=None, latent_L=2.0, generator=None):
    """Fixed geometry shared across the batch: input cloud, latent grid, and the
    regular grid used to read the operator field for value_fn. Also returns a
    GNO interaction radius scaled to the latent-grid spacing."""
    if n_in is None:
        n_in = 1024 if dim == 2 else 4096
    if grid_res is None:
        grid_res = 40 if dim == 2 else 20
    if latent_res is None:
        latent_res = 32 if dim == 2 else 24

    # input point cloud where the input function (g, lam) is sampled
    input_geom = ((torch.rand(n_in, dim, generator=generator) * 2 - 1) * latent_L)
    input_geom = input_geom.to(device)

    # latent FNO grid
    ax = torch.linspace(-latent_L, latent_L, latent_res)
    latent_queries = torch.stack(
        torch.meshgrid(*([ax] * dim), indexing="ij"), dim=-1).to(device)

    # value_fn readout grid (flattened in [ix,iy(,iz)] order -> reshape-able)
    gx = torch.linspace(-grid_L, grid_L, grid_res)
    field_grid = torch.stack(
        torch.meshgrid(*([gx] * dim), indexing="ij"), dim=-1
    ).reshape(-1, dim).to(device)

    # in-GNO gathers input-cloud points (spacing ~ domain / n_in^(1/d)); out-GNO
    # gathers latent-grid points (spacing = domain / (latent_res-1)). Scale each
    # radius to its own spacing so every query gathers a handful of neighbours.
    in_spacing = 2 * latent_L / n_in ** (1.0 / dim)
    latent_spacing = 2 * latent_L / (latent_res - 1)
    in_radius = 1.8 * in_spacing
    out_radius = 1.8 * latent_spacing
    return (input_geom, latent_queries, field_grid, grid_res, grid_L,
            in_radius, out_radius)


def instance_features(inst, input_geom, in_channels):
    """Input-function channels on the fixed cloud: [g(x_i), lam, 0...].

    Point coordinates are already supplied to GINO through ``input_geom``, so
    channels beyond ``g`` and ``lam`` are zero-padded to the config width.
    """
    feat = torch.zeros(input_geom.shape[0], in_channels, device=input_geom.device)
    feat[:, 0:1] = inst.bound_fn(input_geom)                   # g(x_i)
    feat[:, 1] = inst.lam
    return feat


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def make_solver(pde, n_traj, device, horizon=0.3, time_step=0.015):
    return SDESolver(
        pde=pde, n_traj=n_traj, time_step=time_step, horizon=horizon,
        sigma=2**0.5, mode="parabolic", n_traj_max_per_shard=n_traj, device=device)


def train(dim, steps, device, seed=0):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    (input_geom, latent_queries, field_grid, gres, gL,
     in_radius, out_radius) = make_geometry(dim, device, generator=gen)

    model, in_channels = build_gino(
        dim, device=device, in_radius=in_radius, out_radius=out_radius)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    G = field_grid.shape[0]
    B, Q, n_bumps = 8, 64, (1, 8)        # bump count sampled per instance
    n_traj = 64 if dim == 2 else 96      # more paths in 3D to cut target noise

    ig = input_geom.unsqueeze(0)             # [1, N_in, d]
    lq = latent_queries.unsqueeze(0)         # [1, res..., d]

    t0 = time.time()
    for step in range(steps):
        insts = [sample_allen_cahn_family(n_dim=dim, n_bumps=n_bumps, generator=gen)
                 for _ in range(B)]
        feats = torch.stack(
            [instance_features(m, input_geom, in_channels) for m in insts])  # [B,N_in,C]

        # query points shared across the batch, plus the field-readout grid
        xq = ((torch.rand(Q, dim, generator=gen) * 2 - 1)).to(device)
        out_queries = torch.cat([field_grid, xq], dim=0).unsqueeze(0)   # [1, G+Q, d]

        out = model(input_geom=ig, latent_queries=lq,
                    output_queries=out_queries, x=feats)                 # [B, G+Q, 1]
        out_grid, out_q = out[:, :G], out[:, G:]                         # predictions

        opt.zero_grad()
        loss = 0.0
        for b, inst in enumerate(insts):
            field = GridField(out_grid[b].detach().reshape((gres,) * dim), gL, dim)
            solver = make_solver(inst, n_traj, device)
            with torch.no_grad():
                # clamp the (possibly untrained) operator field to the physical
                # range so the y^3 reaction term cannot blow the target up.
                target = solver.evaluate(
                    xq, value_fn=lambda t, x, f=field: f(x).clamp(-1.5, 1.5),
                    evaluate_mode=True)
            loss = loss + nn.functional.mse_loss(out_q[b], target)
        loss = loss / B
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"  [d={dim}] step {step:4d}/{steps}  loss={loss.item():.5e}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.1f}s)", flush=True)

    return model, (ig, lq, field_grid, gres, gL, input_geom, in_channels)


@torch.no_grad()
def evaluate_residual(model, geom, dim, device, n_instances=16, Q=256,
                      n_traj=512, n_bumps=(1, 8), seed=123):
    ig, lq, field_grid, gres, gL, input_geom, in_channels = geom
    G = field_grid.shape[0]
    gen = torch.Generator().manual_seed(seed)
    rels = []
    for _ in range(n_instances):
        inst = sample_allen_cahn_family(n_dim=dim, n_bumps=n_bumps, generator=gen)
        feats = instance_features(inst, input_geom, in_channels).unsqueeze(0)  # [1,N_in,C]
        xq = ((torch.rand(Q, dim, generator=gen) * 2 - 1)).to(device)
        out_queries = torch.cat([field_grid, xq], dim=0).unsqueeze(0)
        out = model(input_geom=ig, latent_queries=lq,
                    output_queries=out_queries, x=feats)
        field = GridField(out[0, :G].reshape((gres,) * dim), gL, dim)
        solver = make_solver(inst, n_traj, device)
        target = solver.evaluate(
            xq, value_fn=lambda t, x, f=field: f(x).clamp(-1.5, 1.5),
            evaluate_mode=True)
        pred = out[0, G:]
        rels.append((pred - target).norm().item() / target.norm().clamp_min(1e-8).item())
    rels = torch.tensor(rels)
    return rels.mean().item(), rels.std().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"GINO + BSDE weak supervision | device={args.device}  steps={args.steps}")
    for dim in args.dims:
        print(f"\n=== Allen-Cahn family, dim={dim} (GINO) ===")
        model, geom = train(dim, args.steps, args.device)
        mean, std = evaluate_residual(model, geom, dim, args.device)
        print(f"  --> held-out BSDE fixed-point residual (rel L2): "
              f"{mean:.4f} +/- {std:.4f}")


if __name__ == "__main__":
    main()
