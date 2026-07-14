"""Train a *time-dependent* GINO operator w_theta(t, x) on the Allen-Cahn family
with BSDE weak supervision, over the whole space-time slab [0, T] x R^d.

Unlike ``train_allen_cahn_family.py`` (which learns only the t=0 slice
w(0, .)), this learns the full solution field. Time enters GINO through its
adaptive-instance-norm scalar input (``fno_norm='ada_in'``, ``ada_in=t``): one
forward at a given ``ada_in=t`` returns that whole spatial slice w(t, .).

BSDE target for a query (t0, x): roll Euler-Maruyama paths from t0 toward T,

    w(t0, x) = E[ g(X_T) + int_{t0}^T lam (w - w^3)(s, X_s) ds ],   dX = sqrt(2) dW,

with w along the path read from the operator itself (Picard-in-the-loop). The
reaction value w(s, X_s) is interpolated in (t, x) from a few operator time
slices on the spatial grid (:class:`SpacetimeField`), so value_fn is cheap.

Neural termination (``--sub-steps k``): instead of integrating to the true
horizon T, stop after k steps and bootstrap the terminal value with
model(t0 + k*dt, X) -- a TD estimate. Cheaper / lower-variance per target, at
the cost of bootstrapping bias. With ``--sub-steps 0`` (default) each path runs
to T and terminates at the exact g(X_T) (fully unbiased).
"""

from __future__ import annotations

import argparse
import bisect
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(_ROOT / "scripts"))
try:
    from src.solvers.sde import sample_allen_cahn_family
except ModuleNotFoundError:
    import types
    for _n, _p in [("src", _ROOT / "src"), ("src.solvers", _ROOT / "src/solvers")]:
        _m = types.ModuleType(_n); _m.__path__ = [str(_p)]; sys.modules[_n] = _m
    from src.solvers.sde import sample_allen_cahn_family

import torch
from torch import nn
from neuralop.models.gino import GINO
import train_allen_cahn_family as T   # GridField, make_geometry, instance_features

BUMPS = (1, 8)
T_HORIZON = 0.3
DT = 0.015
SIGMA = 2 ** 0.5


# --------------------------------------------------------------------------- #
# time-aware GINO (time via ada_in scalar)
# --------------------------------------------------------------------------- #
def build_gino_time(dim, device, in_radius, out_radius):
    cfg = yaml.safe_load(open(_ROOT / "configs/model/gino.yaml"))
    cfg.pop("_target_", None)
    for k, v in list(cfg.items()):
        if isinstance(v, str) and v in ("None", "null"):
            cfg[k] = None
    cfg["gno_coord_dim"] = dim
    cfg["fno_n_modes"] = [12] * dim if dim == 3 else cfg["fno_n_modes"]
    # adaptive instance norm so the scalar ada_in (time) conditions the FNO
    cfg["fno_norm"] = "ada_in"
    cfg["fno_ada_in_dim"] = 1
    cfg["fno_ada_in_features"] = 8
    cfg["out_gno_pos_embed_type"] = "transformer"   # sinusoidal time embedding
    cfg["in_gno_radius"] = in_radius
    cfg["out_gno_radius"] = out_radius
    return GINO(**cfg).to(device), int(cfg["in_channels"])


def op(model, ig, lq, oq, feats, t, device):
    """Query the operator at time t: returns [B, N, 1] = w(t, oq)."""
    return model(input_geom=ig, latent_queries=lq, output_queries=oq, x=feats,
                 ada_in=torch.tensor([float(t)], device=device))


# --------------------------------------------------------------------------- #
# space-time value field (interpolate operator prediction in (t, x))
# --------------------------------------------------------------------------- #
class SpacetimeField:
    """Operator field w(t, x) from K spatial grid slices at ascending times.

    Linear in t between slices, multilinear in x within a slice (grid_sample).
    ``t`` is a python float (all paths share the absolute time at a rollout
    step), so bracketing is a cheap bisect.
    """

    def __init__(self, slice_grids, times, L, dim):
        self.fields = [T.GridField(g, L, dim) for g in slice_grids]
        self.times = list(times)

    def __call__(self, t, x):
        ts = self.times
        if t <= ts[0]:
            return self.fields[0](x)
        if t >= ts[-1]:
            return self.fields[-1](x)
        j = bisect.bisect_right(ts, t) - 1
        a = (t - ts[j]) / (ts[j + 1] - ts[j])
        return (1 - a) * self.fields[j](x) + a * self.fields[j + 1](x)


# --------------------------------------------------------------------------- #
# space-time BSDE rollout (whole-space, optional neural termination)
# --------------------------------------------------------------------------- #
def bsde_rollout(inst, xq, t_start, field, n_traj, sub_steps, device):
    """Monte-Carlo estimate of w(t_start, xq). field = value/terminal callable."""
    Q, d = xq.shape
    x = xq.repeat_interleave(n_traj, 0).to(device)          # [Q*n_traj, d]
    val = torch.zeros(x.shape[0], 1, device=device)
    n_to_h = int(round((T_HORIZON - t_start) / DT))
    n_steps = n_to_h if not sub_steps else min(sub_steps, n_to_h)
    t = t_start
    for _ in range(n_steps):
        y = field(t, x).clamp(-1.5, 1.5)
        val += DT * inst.driver(t, x, y)                    # reaction integral
        x = x + SIGMA * torch.randn_like(x) * DT ** 0.5     # Euler-Maruyama
        t += DT
    if n_steps == n_to_h:
        val += inst.bound_fn(x)                             # exact terminal g(X_T)
    else:
        val += field(t, x).clamp(-1.5, 1.5)                 # neural termination
    return val.view(Q, n_traj, 1).mean(dim=1)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def make_slice_times(t_start, sub_steps, k_slices):
    hi = T_HORIZON if not sub_steps else min(t_start + sub_steps * DT, T_HORIZON)
    hi = max(hi, t_start + 1e-4)
    return [t_start + (hi - t_start) * j / (k_slices - 1) for j in range(k_slices)]


def train(dim, steps, device, sub_steps=0, n_traj=48, k_slices=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    (input_geom, latent_queries, field_grid, gres, gL,
     in_radius, out_radius) = T.make_geometry(dim, device, generator=gen)
    model, in_ch = build_gino_time(dim, device, in_radius, out_radius)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    ig, lq = input_geom.unsqueeze(0), latent_queries.unsqueeze(0)
    G = field_grid.shape[0]
    B, Q = 8, 64
    geom = (ig, lq, field_grid, gres, gL, input_geom, in_ch)

    t0 = time.time()
    for step in range(steps):
        insts = [sample_allen_cahn_family(n_dim=dim, n_bumps=BUMPS, generator=gen)
                 for _ in range(B)]
        feats = torch.stack([T.instance_features(m, input_geom, in_ch) for m in insts])
        # one query time per step; over steps this covers [0, T)
        t_start = float(torch.rand(1, generator=gen).item()) * (T_HORIZON - DT)
        xq = ((torch.rand(Q, dim, generator=gen) * 2 - 1)).to(device)

        # operator value field: K time slices on the spatial grid (detached)
        slice_times = make_slice_times(t_start, sub_steps, k_slices)
        with torch.no_grad():
            slices = [op(model, ig, lq, field_grid.unsqueeze(0), feats, s, device)
                      for s in slice_times]                      # each [B, G, 1]

        pred = op(model, ig, lq, xq.unsqueeze(0), feats, t_start, device)  # [B,Q,1]

        opt.zero_grad()
        loss = 0.0
        for b in range(B):
            grids_b = [slices[j][b].reshape((gres,) * dim) for j in range(k_slices)]
            field_b = SpacetimeField(grids_b, slice_times, gL, dim)
            with torch.no_grad():
                tgt = bsde_rollout(insts[b], xq, t_start, field_b,
                                   n_traj, sub_steps, device)
            loss = loss + nn.functional.mse_loss(pred[b], tgt)
        loss = loss / B
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"  [d={dim} sub={sub_steps}] step {step:4d}/{steps} "
                  f"t0={t_start:.3f} loss={loss.item():.4e} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return model, geom


@torch.no_grad()
def evaluate(model, geom, dim, device, eval_times=(0.0, 0.15), n_instances=12,
             Q=256, n_traj=512, seed=123):
    """Fixed-point residual at several times: rel L2 between the operator and a
    full-horizon (unbiased) BSDE target that uses the operator as value_fn."""
    ig, lq, field_grid, gres, gL, input_geom, in_ch = geom
    out = {}
    for te in eval_times:
        gen = torch.Generator().manual_seed(seed)
        rels = []
        for _ in range(n_instances):
            inst = sample_allen_cahn_family(n_dim=dim, n_bumps=BUMPS, generator=gen)
            feats = T.instance_features(inst, input_geom, in_ch).unsqueeze(0)
            xq = ((torch.rand(Q, dim, generator=gen) * 2 - 1)).to(device)
            slice_times = make_slice_times(te, 0, 5)         # full horizon
            slices = [op(model, ig, lq, field_grid.unsqueeze(0), feats, s, device)[0]
                      for s in slice_times]
            grids = [s.reshape((gres,) * dim) for s in slices]
            field = SpacetimeField(grids, slice_times, gL, dim)
            tgt = bsde_rollout(inst, xq, te, field, n_traj, 0, device)
            pred = op(model, ig, lq, xq.unsqueeze(0), feats, te, device)[0]
            rels.append((pred - tgt).norm().item()
                        / tgt.norm().clamp_min(1e-8).item())
        r = torch.tensor(rels)
        out[te] = (r.mean().item(), r.std().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sub-steps", type=int, default=0,
                    help="0 = full-horizon (unbiased); k>0 = neural termination")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"space-time GINO w(t,x) + BSDE | device={args.device} "
          f"steps={args.steps} sub_steps={args.sub_steps}")
    for dim in args.dims:
        print(f"\n=== Allen-Cahn family, dim={dim} (space-time GINO) ===")
        model, geom = train(dim, args.steps, args.device, sub_steps=args.sub_steps)
        res = evaluate(model, geom, dim, args.device)
        for te, (m, s) in res.items():
            print(f"  residual at t={te:.2f}: {m:.4f} +/- {s:.4f}")


if __name__ == "__main__":
    main()
