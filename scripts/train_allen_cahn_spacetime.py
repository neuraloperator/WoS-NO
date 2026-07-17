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
import math
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(_ROOT / "scripts"))
try:
    from src.solvers.sde import sample_allen_cahn_family, rescale_to_unit_horizon
except ModuleNotFoundError:
    import types
    for _n, _p in [("src", _ROOT / "src"), ("src.solvers", _ROOT / "src/solvers")]:
        _m = types.ModuleType(_n); _m.__path__ = [str(_p)]; sys.modules[_n] = _m
    from src.solvers.sde import sample_allen_cahn_family, rescale_to_unit_horizon

import torch
from torch import nn
from neuralop.models.gino import GINO
import train_allen_cahn_family as T   # GridField, make_geometry, instance_features

BUMPS = (1, 8)
T_HORIZON = 0.3
DT = 0.015
SIGMA = 2 ** 0.5
N_ROLLOUT_STEPS = 20          # steps over the (unit) training horizon

# --- adaptive rollout discretisation ------------------------------------- #
# The Euler-Maruyama reaction integral is under-resolved when the (effective,
# post-rescaling) reaction coefficient lam_eff = T*lam is large: the per-step
# increment is dt * lam_eff * max|u-u^3|. We instead pick the step count so that
# this increment stays <= KAPPA, i.e. dt <= KAPPA / (lam_eff * MAX_REACT), and
# clamp to [N_MIN, N_MAX]. This keeps the target accurate for *arbitrary* T
# (more, finer steps as the problem stiffens) at almost no extra cost -- the
# rollout is cheap next to the GINO forwards.
MAX_REACT = 2.0 / (3.0 * 3.0 ** 0.5)      # max of |u - u^3| on [-1, 1] (~0.385)
KAPPA = 0.15                              # target max per-step reaction increment
N_MIN = 20
N_MAX = 80


def rollout_n_steps(lam_eff: float, horizon_len: float) -> int:
    """Adaptive Euler-Maruyama step count for a rollout of length ``horizon_len``
    under effective reaction ``lam_eff`` (keeps per-step reaction <= KAPPA)."""
    n = math.ceil(horizon_len * abs(lam_eff) * MAX_REACT / KAPPA)
    return int(min(max(n, N_MIN), N_MAX))

# physical horizon + whether g is sign-varying; set once in main(). Training
# always runs in the (possibly rescaled) unit horizon, so T_HORIZON/DT above
# are overwritten by configure_horizon() when T_PHYS != 0.3.
T_PHYS = 0.3
SIGNED = False
# Bound the operator output to the Allen-Cahn invariant region via 1.1*tanh.
# Auto-enabled when rescaling (long horizon): the stiff reaction (T*lam large)
# otherwise lets an unbounded early prediction feed a runaway target. Off for
# the validated T=0.3 path so those results are unchanged.
BOUND_OUTPUT = False
OUT_SCALE = 1.1


# half-width of the query/supervision domain. MUST cover where the solution
# structure lives, and MUST be the same for training, eval and FEM comparison --
# otherwise the operator is supervised on one region and scored on another. For
# the rescaled long-horizon problem the interfaces spread to |xi|~2.5, so query
# points, eval points and FEM error are all defined on [-QUERY_L, QUERY_L]^d.
QUERY_L = 1.0


def configure_horizon(t_phys):
    """When solving a horizon-``t_phys`` problem via diffusive rescaling, training
    runs on the unit horizon; set the rollout horizon/step accordingly."""
    global T_HORIZON, DT, T_PHYS, QUERY_L
    T_PHYS = float(t_phys)
    if abs(T_PHYS - 0.3) < 1e-9:
        T_HORIZON, DT, QUERY_L = 0.3, 0.3 / N_ROLLOUT_STEPS, 1.0
    else:
        T_HORIZON, DT, QUERY_L = 1.0, 1.0 / N_ROLLOUT_STEPS, 2.5   # rescaled


def sample_instance(dim, gen):
    """Sample a family instance; rescale to unit horizon when T_PHYS != 0.3."""
    inst = sample_allen_cahn_family(n_dim=dim, n_bumps=BUMPS, signed=SIGNED,
                                    generator=gen)
    if abs(T_PHYS - 0.3) >= 1e-9:
        inst = rescale_to_unit_horizon(inst, T_PHYS)
    return inst


# --------------------------------------------------------------------------- #
# time-aware GINO (time via ada_in scalar)
# --------------------------------------------------------------------------- #
def build_gino_time(dim, device, in_radius, out_radius,
                    fno_hidden=None, fno_modes=None, fno_layers=None):
    cfg = yaml.safe_load(open(_ROOT / "configs/model/gino.yaml"))
    cfg.pop("_target_", None)
    for k, v in list(cfg.items()):
        if isinstance(v, str) and v in ("None", "null"):
            cfg[k] = None
    cfg["gno_coord_dim"] = dim
    cfg["fno_n_modes"] = [12] * dim if dim == 3 else cfg["fno_n_modes"]
    if fno_modes is not None:
        cfg["fno_n_modes"] = [fno_modes] * dim
    if fno_hidden is not None:
        cfg["fno_hidden_channels"] = fno_hidden
    if fno_layers is not None:
        cfg["fno_n_layers"] = fno_layers
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
    out = model(input_geom=ig, latent_queries=lq, output_queries=oq, x=feats,
                ada_in=torch.tensor([float(t)], device=device))
    if BOUND_OUTPUT:
        out = OUT_SCALE * torch.tanh(out)      # enforce |w| <~ 1 (invariant region)
    return out


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
def rollout_t_end(t_start, sub_frac):
    """End time of a rollout from ``t_start``. ``sub_frac`` in (0,1) rolls a fixed
    sub-horizon of length ``sub_frac * T`` before terminating; <=0 or >=1 rolls to
    the true terminal T. Returns (t_end, exact_terminal).

    Crucially, when ``t_start`` is within one sub-horizon of T the rollout REACHES
    T and terminates at the true g(X_T) (exact=True). This GROUNDS the neural
    termination: the bootstrap w_theta(X_tau, tau) is valid (it's the DP identity),
    but only the true-terminal cases pin the iteration to the correct fixed point
    rather than a trivial self-consistent one (e.g. w==0). Grounding at late times
    then propagates backward through the bootstrap chain as the operator learns."""
    if sub_frac <= 0.0 or sub_frac >= 1.0:
        return T_HORIZON, True
    t_end = min(t_start + sub_frac * T_HORIZON, T_HORIZON)
    return t_end, t_end >= T_HORIZON - 1e-9


def bsde_rollout(inst, xq, t_start, field, n_traj, sub_frac, device, n_steps=None,
                 antithetic=True):
    """Monte-Carlo estimate of w(t_start, xq). field = value/terminal callable.

    The number of Euler-Maruyama steps adapts to the effective reaction
    stiffness (``rollout_n_steps``), so the target stays accurate for any T.
    Pass ``n_steps`` to override the adaptive count (used by the coarse-fine
    study to pin the discretisation).

    ``antithetic=True`` pairs each Brownian increment ``dW`` with its mirror
    ``-dW`` (negatively correlated paths), a free variance-reduction that cuts
    the MC noise of the target -- tightening the weak-supervision signal.
    """
    Q, d = xq.shape
    anti = antithetic and n_traj % 2 == 0
    x = xq.repeat_interleave(n_traj, 0).to(device)          # [Q*n_traj, d]
    val = torch.zeros(x.shape[0], 1, device=device)
    t_end, exact = rollout_t_end(t_start, sub_frac)
    if n_steps is None:
        n_steps = rollout_n_steps(inst.lam, t_end - t_start)
    dt = (t_end - t_start) / n_steps
    sqrt_dt = dt ** 0.5
    t = t_start
    half = n_traj // 2
    # Clamp the path value to the Allen-Cahn invariant region [-1, 1]. This is
    # not a heuristic: |u| <= 1 is preserved by the dynamics (maximum principle)
    # and the reaction f(u)=lam(u-u^3) vanishes at u=+/-1. Clamping there keeps
    # the +/-1 phases stable, so an early wrong operator cannot drive the
    # accumulated reaction integral to blow up -- essential once rescaling makes
    # the reaction coefficient (T*lam) large (a 1.5 clamp diverges at T=10).
    for _ in range(n_steps):
        y = field(t, x).clamp(-1.0, 1.0)
        val += dt * inst.driver(t, x, y)                    # reaction integral
        if anti:                                            # antithetic Brownian
            dwh = torch.randn(Q, half, d, device=device)
            dW = torch.cat([dwh, -dwh], dim=1).reshape(Q * n_traj, d)
        else:
            dW = torch.randn_like(x)
        x = x + SIGMA * dW * sqrt_dt                        # Euler-Maruyama
        t += dt
    if exact:
        val += inst.bound_fn(x)                             # exact terminal g(X_T)
    else:
        val += field(t_end, x).clamp(-1.0, 1.0)            # neural termination
    return val.view(Q, n_traj, 1).mean(dim=1)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def make_slice_times(t_start, sub_frac, k_slices):
    hi, _ = rollout_t_end(t_start, sub_frac)
    hi = max(hi, t_start + 1e-4)
    return [t_start + (hi - t_start) * j / (k_slices - 1) for j in range(k_slices)]


def train(dim, steps, device, sub_frac=0.0, n_traj=32, k_slices=3, seed=0,
          wb=None, eval_every=0, eval_times=(0.0,), lr=1e-3, ema_decay=0.99,
          terminal_weight=0.0, fno_hidden=None, fno_modes=None, fno_layers=None):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    # rescaled long-horizon interfaces span wider than the T=0.3 bumps -> widen
    rescaled = abs(T_PHYS - 0.3) >= 1e-9
    geo_kw = dict(latent_L=3.0, grid_L=4.0) if rescaled else {}
    (input_geom, latent_queries, field_grid, gres, gL,
     in_radius, out_radius) = T.make_geometry(dim, device, generator=gen, **geo_kw)
    model, in_ch = build_gino_time(dim, device, in_radius, out_radius,
                                   fno_hidden=fno_hidden, fno_modes=fno_modes,
                                   fno_layers=fno_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # EMA target network: the value_fn along the rollout reads from a slowly
    # updated copy of the operator, NOT the live weights. This breaks the
    # Picard-in-the-loop self-reference (target = f(self)) that destabilises the
    # nonlinear (w-dependent) BSDE target under a stiff reaction -- the operator
    # regresses against a nearly-fixed target instead of chasing itself.
    import copy
    target_model = copy.deepcopy(model)
    for p in target_model.parameters():
        p.requires_grad_(False)
    target_model.eval()
    use_ema = ema_decay is not None and ema_decay > 0

    def value_model():
        return target_model if use_ema else model

    ig, lq = input_geom.unsqueeze(0), latent_queries.unsqueeze(0)
    G = field_grid.shape[0]
    B, Q = 8, 64
    geom = (ig, lq, field_grid, gres, gL, input_geom, in_ch)

    t0 = time.time()
    for step in range(steps):
        insts = [sample_instance(dim, gen) for _ in range(B)]
        feats = torch.stack([T.instance_features(m, input_geom, in_ch) for m in insts])
        # one query time per step; over steps this covers [0, T)
        t_start = float(torch.rand(1, generator=gen).item()) * (T_HORIZON - DT)
        xq = ((torch.rand(Q, dim, generator=gen) * 2 - 1) * QUERY_L).to(device)

        # operator value field: K time slices from the TARGET (EMA) network
        slice_times = make_slice_times(t_start, sub_frac, k_slices)
        with torch.no_grad():
            slices = [op(value_model(), ig, lq, field_grid.unsqueeze(0), feats, s, device)
                      for s in slice_times]                      # each [B, G, 1]

        pred = op(model, ig, lq, xq.unsqueeze(0), feats, t_start, device)  # [B,Q,1]

        opt.zero_grad()
        loss = 0.0
        for b in range(B):
            grids_b = [slices[j][b].reshape((gres,) * dim) for j in range(k_slices)]
            field_b = SpacetimeField(grids_b, slice_times, gL, dim)
            with torch.no_grad():
                tgt = bsde_rollout(insts[b], xq, t_start, field_b,
                                   n_traj, sub_frac, device)
            loss = loss + nn.functional.mse_loss(pred[b], tgt)
        loss = loss / B
        # terminal (time-boundary) PINO loss: pin w(T, .) = g exactly. Cheap
        # (one forward, no rollout) and a strong anchor -- grounds the solution
        # directly instead of relying only on g(X_T) inside the rollout.
        loss_term = torch.zeros((), device=device)
        if terminal_weight > 0:
            predT = op(model, ig, lq, xq.unsqueeze(0), feats, T_HORIZON, device)
            gT = torch.stack([m.bound_fn(xq) for m in insts])
            loss_term = nn.functional.mse_loss(predT, gT)
            loss = loss + terminal_weight * loss_term
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # EMA update of the target network
        if use_ema:
            with torch.no_grad():
                for tp, p in zip(target_model.parameters(), model.parameters()):
                    tp.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)
                for tb, b_ in zip(target_model.buffers(), model.buffers()):
                    tb.copy_(b_)

        n_roll = rollout_n_steps(insts[0].lam,
                                 rollout_t_end(t_start, sub_frac)[0] - t_start)
        elapsed = time.time() - t0
        if wb is not None:
            wb.log({"train/loss": loss.item(), "train/n_roll": n_roll,
                    "train/loss_terminal": loss_term.item(), "train/t_start": t_start,
                    "elapsed_min": elapsed / 60,
                    "lr": sched.get_last_lr()[0]}, step=step)
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            print(f"  [d={dim} subfrac={sub_frac:.2f}] step {step:4d}/{steps} "
                  f"t0={t_start:.3f} n_roll={n_roll} loss={loss.item():.4e} "
                  f"({elapsed:.1f}s)", flush=True)

        # periodic held-out unbiased residual -> wandb curve (decide budget by eye)
        if eval_every and wb is not None and step > 0 and (
                step % eval_every == 0 or step == steps - 1):
            res = evaluate(model, geom, dim, device, eval_times=eval_times,
                           n_instances=4, Q=64, n_traj=96)
            wb.log({f"val/residual_t{te:.2f}": m for te, (m, _) in res.items()},
                   step=step)
            print(f"    [eval @ {step}] " + " ".join(
                f"res(t={te:.2f})={m:.4f}" for te, (m, _) in res.items()), flush=True)
    return model, geom


def _log_and_eval(model, geom, dim, device, step, steps, loss_val, extra, wb,
                  eval_every, eval_times, t0, tag=""):
    """Shared per-step wandb log + periodic held-out unbiased residual eval."""
    elapsed = time.time() - t0
    if wb is not None:
        wb.log({"train/loss": loss_val, "elapsed_min": elapsed / 60, **extra}, step=step)
    if step % max(1, steps // 20) == 0 or step == steps - 1:
        print(f"  [d={dim}{tag}] step {step:4d}/{steps} loss={loss_val:.4e} "
              f"({elapsed:.1f}s)", flush=True)
    if eval_every and wb is not None and step > 0 and (
            step % eval_every == 0 or step == steps - 1):
        res = evaluate(model, geom, dim, device, eval_times=eval_times,
                       n_instances=4, Q=64, n_traj=96)
        wb.log({f"val/residual_t{te:.2f}": m for te, (m, _) in res.items()}, step=step)
        print(f"    [eval @ {step}] " + " ".join(
            f"res(t={te:.2f})={m:.4f}" for te, (m, _) in res.items()), flush=True)


def train_buffered(dim, steps, device, sub_frac=0.0, n_traj=16, k_slices=3, seed=0,
                   wb=None, eval_every=0, eval_times=(0.0,), lr=1e-3,
                   pool_size=64, nX=256, nT=8, buf_cap=20, terminal_weight=0.0,
                   warmup=0, fem_target=False,
                   fno_hidden=None, fno_modes=None, fno_layers=None,
                   refresh_per_step=2):
    """Like :func:`train` but with an nwos-style running-mean target BUFFER.

    A FIXED pool of instances, query grid and time grid is used so each
    (instance, time, point) target cell has a stable identity. Every visit folds
    a fresh MC target into a running mean (capped at ``buf_cap`` so recent, more
    accurate estimates dominate once the operator has moved on -- our target is
    operator-dependent, unlike nwos's linear WoS). The operator regresses against
    the LOW-NOISE buffered target while the rollout value_fn stays the LIVE
    operator (self-consistent) -- this damps the noise-driven Picard oscillation
    without the stale-value_fn bias that an EMA target network introduces."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    rescaled = abs(T_PHYS - 0.3) >= 1e-9
    geo_kw = dict(latent_L=3.0, grid_L=4.0) if rescaled else {}
    (input_geom, latent_queries, field_grid, gres, gL,
     in_radius, out_radius) = T.make_geometry(dim, device, generator=gen, **geo_kw)
    model, in_ch = build_gino_time(dim, device, in_radius, out_radius,
                                   fno_hidden=fno_hidden, fno_modes=fno_modes,
                                   fno_layers=fno_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ig, lq = input_geom.unsqueeze(0), latent_queries.unsqueeze(0)
    G = field_grid.shape[0]
    geom = (ig, lq, field_grid, gres, gL, input_geom, in_ch)

    # fixed pool: instances, features, shared query grid, time grid
    pool = [sample_instance(dim, gen) for _ in range(pool_size)]
    pool_feats = torch.stack([T.instance_features(m, input_geom, in_ch) for m in pool])
    Xq = ((torch.rand(nX, dim, generator=gen) * 2 - 1) * QUERY_L).to(device)
    Tq = torch.linspace(0.0, T_HORIZON - DT, nT).tolist()
    # running-mean target buffer + visit counts
    buf = torch.zeros(pool_size, nT, nX, 1, device=device)
    cnt = torch.zeros(pool_size, nT, dtype=torch.long)

    # FEM-supervised comparison: precompute the fine-FEM space-time field for
    # every pool instance and regress the operator directly against it (the
    # accuracy CEILING for this architecture, at the cost of a full FEM dataset
    # -- meshing + solve per instance, which BSDE weak supervision avoids).
    fem_w = None
    if fem_target:
        from src.solvers.fem import AllenCahnFEMSolver
        t_gen = time.time()
        # coarser dt for target generation (still far more accurate than the
        # operator's ~0.15 error) -> feasible online FEM-supervised
        fem_solver = AllenCahnFEMSolver(dim, n=141, dt=1e-2, device=device)  # mesh ONCE
        fem_w = [fem_solver.solve(pool[p], T=1.0, n_slices=11)[1]
                 for p in range(pool_size)]                     # SpacetimeField each
        fem_gen_time = time.time() - t_gen                      # accumulates below
        fem_solves = pool_size
        print(f"  [d={dim}] FEM dataset: {pool_size} solves in "
              f"{fem_gen_time:.1f}s (mesh built once in {fem_solver.setup_time:.2f}s; "
              f"data-gen cost BSDE avoids)", flush=True)

    B = 8
    # Warm start: pretrain w_theta(t, .) = g(.) for all t (cheap, no rollout).
    # Allen-Cahn is BISTABLE (+/-1 both stable, both self-consistent BSDE fixed
    # points), so plain BSDE training can collapse to the WRONG phase regardless
    # of g. Seeding the field to g everywhere puts every instance in the correct
    # sign(g) basin, so the subsequent reaction evolves it to the right phase.
    if warmup > 0:
        gpool = torch.stack([pool[p].bound_fn(Xq) for p in range(pool_size)])  # [P,nX,1]
        for w in range(warmup):
            idx = torch.randint(0, pool_size, (B,), generator=gen).tolist()
            tw_ = float(torch.rand(1, generator=gen).item()) * T_HORIZON
            pred = op(model, ig, lq, Xq.unsqueeze(0), pool_feats[idx], tw_, device)
            loss = nn.functional.mse_loss(pred, gpool[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        print(f"  [d={dim}] warmup {warmup} steps done (fit w=g), "
              f"final warmup loss={loss.item():.4e}", flush=True)

    # rolling replay buffer: how many pool slots to REPLACE with fresh instances
    # each step. The buffer still UPDATES (running mean) within a slot's lifetime
    # -- but slots are continuously refreshed, so over training the operator sees
    # ~unlimited instances (no fixed pool -> no overfitting wall) while keeping
    # the buffer's variance reduction. refresh_per_step slots live ~pool/refresh
    # steps each, accumulating a few MC estimates before eviction.
    fem_solver_live = None
    if fem_target and refresh_per_step > 0:
        from src.solvers.fem import AllenCahnFEMSolver
        fem_solver_live = AllenCahnFEMSolver(dim, n=141, dt=1e-2, device=device)

    t0 = time.time()
    for step in range(steps):
        # replace (evict + insert fresh) some slots -> rolling infinite-instance
        if refresh_per_step > 0 and step > 0:
            for _ in range(refresh_per_step):
                s = int(torch.randint(0, pool_size, (1,), generator=gen).item())
                pool[s] = sample_instance(dim, gen)
                pool_feats[s] = T.instance_features(pool[s], input_geom, in_ch)
                buf[s].zero_(); cnt[s].zero_()
                if fem_w is not None:
                    _tf = time.time()
                    fem_w[s] = fem_solver_live.solve(pool[s], T=1.0, n_slices=11)[1]
                    fem_gen_time += time.time() - _tf; fem_solves += 1

        idx = torch.randint(0, pool_size, (B,), generator=gen).tolist()
        j = int(torch.randint(0, nT, (1,), generator=gen).item())
        t = Tq[j]
        feats = pool_feats[idx]                                   # [B, N_in, C]

        # value-field slices only needed for the BSDE rollout target
        if fem_w is None:
            slice_times = make_slice_times(t, sub_frac, k_slices)
            with torch.no_grad():
                slices = [op(model, ig, lq, field_grid.unsqueeze(0), feats, s, device)
                          for s in slice_times]
        pred = op(model, ig, lq, Xq.unsqueeze(0), feats, t, device)   # [B, nX, 1]

        opt.zero_grad()
        loss = 0.0
        for bi, p in enumerate(idx):
            with torch.no_grad():
                if fem_w is not None:                # FEM-supervised target
                    target = fem_w[p](1.0 - t, Xq)
                else:                                # BSDE weak-supervision target
                    grids_b = [slices[k][bi].reshape((gres,) * dim) for k in range(k_slices)]
                    field_b = SpacetimeField(grids_b, slice_times, gL, dim)
                    tgt = bsde_rollout(pool[p], Xq, t, field_b, n_traj, sub_frac, device)
                    n = min(int(cnt[p, j].item()), buf_cap)   # nwos running mean
                    buf[p, j] = (tgt + n * buf[p, j]) / (n + 1)
                    cnt[p, j] += 1
                    target = buf[p, j].clone()
            loss = loss + nn.functional.mse_loss(pred[bi], target)
        loss = loss / B
        # terminal (time-boundary) PINO loss: pin w(T, .) = g exactly.
        loss_term = torch.zeros((), device=device)
        if terminal_weight > 0:
            predT = op(model, ig, lq, Xq.unsqueeze(0), feats, T_HORIZON, device)
            gT = torch.stack([pool[p].bound_fn(Xq) for p in idx])
            loss_term = nn.functional.mse_loss(predT, gT)
            loss = loss + terminal_weight * loss_term
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        _log_and_eval(model, geom, dim, device, step, steps, loss.item(),
                      {"train/n_roll": rollout_n_steps(pool[idx[0]].lam,
                                                       rollout_t_end(t, sub_frac)[0] - t),
                       "lr": sched.get_last_lr()[0], "train/loss_terminal": loss_term.item(),
                       "buf/mean_count": float(cnt.float().mean())},
                      wb, eval_every, eval_times, t0, tag=" buffered")
    if fem_w is not None:
        print(f"  [d={dim}] TOTAL FEM data-gen: {fem_solves} solves, "
              f"{fem_gen_time:.0f}s ({fem_gen_time/max(fem_solves,1):.2f}s/solve) "
              f"-- the cost BSDE weak supervision pays ZERO of", flush=True)
    return model, geom


@torch.no_grad()
def evaluate(model, geom, dim, device, eval_times=(0.0, 0.15), n_instances=6,
             Q=96, n_traj=128, seed=123):
    """Fixed-point residual at several times: rel L2 between the operator and a
    full-horizon (unbiased) BSDE target that uses the operator as value_fn."""
    ig, lq, field_grid, gres, gL, input_geom, in_ch = geom
    out = {}
    for te in eval_times:
        gen = torch.Generator().manual_seed(seed)
        rels = []
        for _ in range(n_instances):
            inst = sample_instance(dim, gen)
            feats = T.instance_features(inst, input_geom, in_ch).unsqueeze(0)
            xq = ((torch.rand(Q, dim, generator=gen) * 2 - 1) * QUERY_L).to(device)
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


@torch.no_grad()
def evaluate_vs_fem(model, geom, dim, device, eval_times=(0.0,), n_instances=8,
                    Q=256, seed=321):
    """TRUE relative L2 error of the operator vs a fine-FEM ground truth.

    The rescaled forward FEM gives w(t, .) = u_forward(1 - t, .); we compare the
    operator prediction w_theta(t, .) against it (not the self-consistent BSDE
    residual). This is the honest accuracy metric."""
    from src.solvers.fem import AllenCahnFEMSolver
    ig, lq, field_grid, gres, gL, input_geom, in_ch = geom
    fem = AllenCahnFEMSolver(dim, device=device)               # mesh built once
    out = {te: [] for te in eval_times}
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_instances):
        inst = sample_instance(dim, gen)                       # already rescaled
        feats = T.instance_features(inst, input_geom, in_ch).unsqueeze(0)
        xq = ((torch.rand(Q, dim, generator=gen) * 2 - 1) * QUERY_L).to(device)   # match train
        _, stf = fem.solve(inst, T=1.0, n_slices=11)           # stf(sigma)=u_forward
        for te in eval_times:
            fem_w = stf(1.0 - te, xq)                           # w(te,.)=u_fwd(1-te,.)
            pred = op(model, ig, lq, xq.unsqueeze(0), feats, te, device)[0]
            out[te].append((pred - fem_w, fem_w))
    # POOLED (variance-normalised) rel-L2 = sqrt(sum ||err||^2 / sum ||fem||^2):
    # one global denominator, so signed instances whose solution norm ~0 cannot
    # blow up the ratio (unlike mean per-instance rel-L2). Also report abs RMSE.
    res = {}
    for te, v in out.items():
        errs = torch.cat([e.reshape(-1) for e, _ in v])
        fems = torch.cat([f.reshape(-1) for _, f in v])
        nrmse = (errs.pow(2).sum() / fems.pow(2).sum().clamp_min(1e-12)).sqrt().item()
        rmse = errs.pow(2).mean().sqrt().item()
        res[te] = (nrmse, rmse)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sub-frac", type=float, default=None,
                    help="fraction of remaining horizon to roll before neural "
                         "termination (bootstrap terminal with the operator); "
                         "0 = full-horizon unbiased. Default: 0.25 when rescaling "
                         "(keeps rollouts short at large T), else 0.")
    ap.add_argument("--n-traj", type=int, default=16,
                    help="walk trajectories per query. Kept small: the coarse-fine "
                         "autoresearch shows MC variance is amortized by SGD, so "
                         "n_traj~8-16 matches n_traj=128 in trained accuracy "
                         "(see scripts/autoresearch_coarse_fine.py).")
    ap.add_argument("--T", type=float, default=0.3,
                    help="physical horizon; T!=0.3 trains via diffusive rescaling")
    ap.add_argument("--signed", action="store_true",
                    help="sign-varying g (needed at long T to avoid collapse)")
    ap.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    ap.add_argument("--project", default="wos-no-spacetime")
    ap.add_argument("--tag", default=None, help="wandb tag for grouping (e.g. a sweep)")
    ap.add_argument("--save-model", default=None,
                    help="path prefix to save the trained model + geometry per dim")
    ap.add_argument("--eval-every", type=int, default=25,
                    help="log held-out unbiased residual to wandb every N steps")
    ap.add_argument("--k-slices", type=int, default=3,
                    help="operator time-slices for the value_fn (more resolves fast "
                         "time-varying dynamics for full-horizon rollouts)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", action="store_true",
                    help="nwos-style running-mean target buffer over a fixed pool "
                         "(low-noise, stable targets; live self-consistent value_fn)")
    ap.add_argument("--pool-size", type=int, default=64)
    ap.add_argument("--refresh-per-step", type=int, default=2,
                    help="rolling replay buffer: pool slots replaced with FRESH "
                         "instances each step (0 = fixed pool; >0 = ~infinite "
                         "instances over training, no overfitting wall)")
    ap.add_argument("--buf-cap", type=int, default=20,
                    help="max visits averaged (EMA-like window once saturated)")
    ap.add_argument("--terminal-weight", type=float, default=0.0,
                    help="weight of the terminal/boundary PINO loss ||w(T,.)-g||^2 "
                         "(direct anchor of the time-boundary condition)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="pretrain w(t,.)=g for all t before BSDE (seeds the correct "
                         "sign(g) phase; Allen-Cahn is bistable so BSDE alone can "
                         "collapse to the wrong +/-1 phase)")
    ap.add_argument("--fem-target", action="store_true",
                    help="FEM-supervised baseline: regress against precomputed FEM "
                         "fields instead of BSDE targets (accuracy ceiling; pays a "
                         "full FEM dataset-gen cost that BSDE avoids)")
    ap.add_argument("--fno-hidden", type=int, default=None, help="FNO hidden channels")
    ap.add_argument("--fno-modes", type=int, default=None, help="FNO modes per dim")
    ap.add_argument("--fno-layers", type=int, default=None, help="FNO layers")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="EMA target-network decay for the value_fn. Default OFF: a "
                         "STALE value field biases the nonlinear reaction integral "
                         "(the target needs the CORRECT w along the path), so the "
                         "self-consistent live value_fn is required for the fixed "
                         "point to be the true solution. Damp oscillation via --lr "
                         "and --n-traj instead, not by breaking self-consistency.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    global SIGNED, BOUND_OUTPUT
    SIGNED = args.signed
    configure_horizon(args.T)
    rescaled = abs(args.T - 0.3) >= 1e-9
    BOUND_OUTPUT = rescaled          # bound output in the stiff long-horizon case
    # Neural termination by default when rescaling: at large T a full rollout is
    # deep/stiff (adaptive steps -> N_MAX); rolling only a short fraction and
    # bootstrapping the terminal with the operator keeps each target cheap. The
    # bootstrap is self-consistent (TD): it converges as the operator learns.
    sub_frac = args.sub_frac if args.sub_frac is not None else (0.25 if rescaled else 0.0)
    print(f"space-time GINO w(t,x) + BSDE | device={args.device} "
          f"steps={args.steps} sub_frac={sub_frac} n_traj={args.n_traj} "
          f"T={args.T} signed={SIGNED} "
          f"{'(rescaled unit-horizon, neural-termination)' if rescaled else ''}")
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
    # residual eval times, expressed in the (rescaled) training horizon
    eval_times = (0.0, 0.5 * T_HORIZON)
    for dim in args.dims:
        print(f"\n=== Allen-Cahn family, dim={dim} (space-time GINO) ===")
        wb = None
        if use_wandb:
            wb = wandb.init(
                project=args.project, reinit=True,
                tags=[args.tag] if args.tag else None,
                name=f"T{args.T:g}-d{dim}-{'signed' if SIGNED else 'pos'}-nt{args.n_traj}"
                     f"{'-buf' if args.buffer else ''}"
                     f"-tw{args.terminal_weight:g}",
                config=dict(dim=dim, T=args.T, signed=SIGNED, steps=args.steps,
                            terminal_weight=args.terminal_weight,
                            sub_frac=sub_frac, n_traj=args.n_traj, kappa=KAPPA,
                            bound_output=BOUND_OUTPUT, rescaled=rescaled,
                            lr=args.lr, ema_decay=args.ema_decay, k_slices=args.k_slices,
                            buffer=args.buffer, pool_size=args.pool_size,
                            buf_cap=args.buf_cap))
            print(f"  wandb: {wb.url}", flush=True)
        if args.buffer:
            model, geom = train_buffered(
                dim, args.steps, args.device, sub_frac=sub_frac, n_traj=args.n_traj,
                k_slices=args.k_slices, wb=wb, eval_every=args.eval_every,
                eval_times=eval_times, lr=args.lr, pool_size=args.pool_size,
                buf_cap=args.buf_cap, terminal_weight=args.terminal_weight,
                warmup=args.warmup, fem_target=args.fem_target,
                fno_hidden=args.fno_hidden, fno_modes=args.fno_modes,
                fno_layers=args.fno_layers, refresh_per_step=args.refresh_per_step)
        else:
            model, geom = train(dim, args.steps, args.device, sub_frac=sub_frac,
                                n_traj=args.n_traj, k_slices=args.k_slices, wb=wb,
                                eval_every=args.eval_every, eval_times=eval_times,
                                lr=args.lr, ema_decay=args.ema_decay,
                                terminal_weight=args.terminal_weight)
        res = evaluate(model, geom, dim, args.device, eval_times=eval_times)
        for te, (m, s) in res.items():
            tp = te * args.T / T_HORIZON      # report physical time
            print(f"  BSDE residual t_phys={tp:.3f} (tau={te/T_HORIZON:.2f}): "
                  f"{m:.4f} +/- {s:.4f}")
            if wb is not None:
                wb.summary[f"final/residual_t{te:.2f}"] = m
        # TRUE relative error vs fine FEM ground truth
        ferr = evaluate_vs_fem(model, geom, dim, args.device, eval_times=eval_times)
        for te, (nrmse, rmse) in ferr.items():
            tp = te * args.T / T_HORIZON
            print(f"  FEM error t_phys={tp:.3f} (tau={te/T_HORIZON:.2f}): "
                  f"nRMSE(pooled relL2)={nrmse:.4f}  RMSE={rmse:.4f}")
            if wb is not None:
                wb.summary[f"final/fem_nrmse_t{te:.2f}"] = nrmse
                wb.summary[f"final/fem_rmse_t{te:.2f}"] = rmse
        if args.save_model:
            path = f"{args.save_model}_d{dim}.pt"
            torch.save({"state_dict": model.state_dict(), "dim": dim,
                        "T": args.T, "signed": SIGNED, "seed": 0}, path)
            print(f"  saved model -> {path}", flush=True)
        if wb is not None:
            wb.finish()


if __name__ == "__main__":
    main()
