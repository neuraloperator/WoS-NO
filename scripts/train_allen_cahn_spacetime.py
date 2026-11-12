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
# Floor on the step count. The Euler-Maruyama scheme for the BSDE is O(dt) in
# dt, so the floor has to stay high enough for that first-order error: at 8 steps
# it measures ~3x worse than at 20 against an exact radial reference.
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
# Finite-difference step for SPATIAL derivatives of the operator. GINO's output
# is piecewise constant in the query coordinate, so a step must span several
# cells: sweeping eps, |dw|/eps is 2.97 at 1e-3 (noise amplification), 0.944 at
# 1e-2, then settles at 0.256 (0.1) and 0.219 (0.3). 0.25 sits on that plateau
# and matches the value-grid spacing (0.154 in 2-D, 0.462 in 3-D).
FD_EPS = 0.25
# The price of that step is truncation: a central second difference carries an
# O(eps^2 d^4w/dx^4) error, i.e. the Laplacian and the Dirichlet energy are
# smoothed over ~1.6 grid cells (eps 0.25 vs spacing 0.154 in 2-D). It cannot be
# reduced -- below the plateau the piecewise-constant response takes over and the
# derivative collapses toward 0 -- so it is a standing bias of the PINO and
# DeepRitz arms, not of the BSDE or FEM ones, and belongs in their error budget.
#
# Step for the TIME derivative. Time enters through ada_in, a smooth conditioning
# scalar with no query-grid staircase, so unlike FD_EPS this step is limited from
# BELOW by float noise, not from below by the staircase. Sweeping h at a fixed
# state (RMS of d_t w over 8 instances x 256 points):
#   h      1e-5    3e-5    1e-4    3e-4    1e-3    3e-3    5e-3    1e-2    3e-2
#   RMS    0.01105 0.00717 0.00674 0.00658 0.00661 0.00659 0.00659 0.00659 0.00660
# so everything from 1e-3 to 3e-2 agrees to 0.02% and below 1e-4 cancellation
# takes over (68% high at 1e-5). 5e-3 sits in the middle of that plateau.
FD_EPS_T = 5e-3
# Profiling: one training step in PROF_EVERY is timed with CUDA syncs, so the
# breakdown of where an arm's budget goes costs ~nothing. PROFILE holds the last
# snapshot and is updated live during training.
PROF_EVERY = 20
PROFILE = {}
# Scale on the cv="bsde" martingale increment. The regressed Z_k already IS
# sigma grad w, so the increment Z.dW = Z xi sqrt(dt) needs no extra sigma: 1.0
# is correct. The first version multiplied by SIGMA as well; measured at
# n_traj=4096 that gave target variance x0.71 (2-D) / x1.17 (3-D, worse than no
# control variate) against x0.35 / x0.49 at 1.0. Kept as a knob for A/B only.
BSDE_CV_SCALE = 1.0
# Time knots at which Z is regressed for cv="bsde"; None = every step. Between
# knots the fitted (intercept, slope) is held. Fewer knots = less memory traffic
# (the statistics are reductions over the whole stored path), at some loss of
# variance reduction. Measured at n_traj=4096 (scripts/cv_knots_check.py):
#   knots  1      2      4      10     20(all)
#   2-D    1.000  0.895  0.327  0.225  0.241
#   3-D    1.000  0.880  0.337  0.292  0.288
# knots=1 does NOTHING under antithetic sampling: a Z identical for both
# partners multiplies xi on one path and -xi on the other and the pair's
# martingale cancels exactly, so the whole effect of the control variate comes
# from Z's variation along the path. 5 keeps most of the reduction at ~1/4 of
# the per-knot statistics traffic; the GPU probe measures the throughput.
BSDE_CV_KNOTS = 5
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


class SpacetimeGradField:
    """``grad_x w(t, x)`` by central differences on the SAME operator slices.

    The rollout's value field is already read off ``k_slices`` grid slices, so
    the spatial gradient costs one ``torch.gradient`` per slice on tensors that
    already exist -- no extra operator forward. That makes the BSDE martingale
    control variate essentially free here (see ``bsde_rollout``).
    """

    def __init__(self, slice_grids, times, L, dim):
        h = 2.0 * L / (slice_grids[0].shape[0] - 1)
        self.slices = [[T.GridField(c, L, dim)
                        for c in torch.gradient(g, spacing=h)]
                       for g in slice_grids]
        self.times = list(times)
        self.dim = dim

    def _at(self, k, x):
        return torch.cat([f(x) for f in self.slices[k]], dim=-1)   # [N, d]

    def __call__(self, t, x):
        ts = self.times
        if t <= ts[0]:
            return self._at(0, x)
        if t >= ts[-1]:
            return self._at(len(ts) - 1, x)
        j = bisect.bisect_right(ts, t) - 1
        a = (t - ts[j]) / (ts[j + 1] - ts[j])
        return (1 - a) * self._at(j, x) + a * self._at(j + 1, x)


# --------------------------------------------------------------------------- #
# space-time BSDE rollout (whole-space, optional neural termination)
# --------------------------------------------------------------------------- #
def invariant_bound(inst, pts):
    """max(1, ||g||_inf) for one instance: the Allen-Cahn maximum-principle bound.

    For d_t w + Lap w + lam (w - w^3) = 0 with w(T, .) = g, any M >= 1 with
    M >= ||g||_inf is a supersolution (the reaction is <= 0 there), so |w| <= M
    on the whole slab. ``g`` is a sum of radial bumps
    ``amp_k / (offset_k + width_k ||x - c_k||^2)``, so its supremum sits at or
    between the bump centres; evaluating there and on ``pts`` (the value grid the
    driver is read off) brackets it tightly. The loose alternative
    ``sum_k amp_k / offset_k`` reaches ~4 for 8 well-separated bumps and would
    switch the clamp off in exactly the runs that need it.
    """
    with torch.no_grad():
        cand = [inst.bound_fn(pts).abs().max()]
        c = inst.centers.to(pts.device)
        if c.numel():
            cand.append(inst.bound_fn(c).abs().max())
        return max(1.0, float(torch.stack(cand).max()))


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
                 antithetic=True, grad_field=None, cv="bsde", return_grad=True,
                 return_zcoef=False, bound=1.0):
    """Monte-Carlo estimate of the BSDE solution at (t_start, xq).

    Returns the PAIR (value, grad w). An FBSDE solution has two components,
    Y and Z_t = sigma grad w(t, X_t), and both are estimated from the SAME
    paths -- Z is not an optional extra and is never imported from elsewhere.
    Pass ``return_grad=False`` only for call sites that genuinely want the value
    alone. Z is the projection of the payoff onto the first Brownian
    increment, E[Y xi_0] = Z_0 sqrt(dt) (Bismut-Elworthy-Li); antithetic pairing
    makes it a central difference, which is what makes it usable at these path
    counts.

    ``cv`` selects the control variate. "bsde" (default) estimates Z from the
    paths themselves by cross-fitted regression -- no external gradient. "off"
    disables it. The remaining modes take Z from a supplied ``grad_field`` and
    exist only for comparison: "auto" fits its scale by least squares, a float
    forces one (1.0 is the textbook martingale control variate, which measurement
    shows over-subtracts because the grid gradient is ~2x too long in 3-D).

    "bsde" needs enough paths for the regression: measured variance against no
    control variate is 1.00/1.61 (2-D/3-D) at n_traj=2048 but 0.38/0.72 at 4096
    and 0.44/0.31 at 8192. Below ~2048 paths the regression noise dominates and
    it is worse than doing nothing.

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
    # the martingale is accumulated SEPARATELY rather than subtracted inline, so
    # its coefficient can be fitted after the paths are known
    mart = torch.zeros(x.shape[0], 1, device=device)
    xi0 = None
    t_end, exact = rollout_t_end(t_start, sub_frac)
    if n_steps is None:
        n_steps = rollout_n_steps(inst.lam, t_end - t_start)
    dt = (t_end - t_start) / n_steps
    sqrt_dt = dt ** 0.5
    t = t_start
    half = n_traj // 2
    # one simulation produces BOTH components: the path is kept so the backward
    # sweep can read Z off the same increments that produced Y
    keep = (cv == "bsde")
    if keep:                                     # written in-loop, no stacking
        PX = torch.empty(n_steps, Q * n_traj, d, device=device)
        PXI = torch.empty(n_steps, Q * n_traj, d, device=device)
        PH = torch.empty(n_steps, Q * n_traj, 1, device=device)
    # Clamp the path value to the Allen-Cahn invariant region. Not a heuristic:
    # |u| <= max(1, ||g||_inf) is preserved by the dynamics (maximum principle,
    # since lam(M - M^3) <= 0 for M >= 1), and clamping there keeps the phases
    # stable so an early wrong operator cannot drive the accumulated reaction
    # integral to blow up -- essential once rescaling makes T*lam large.
    #
    # ``bound`` is that constant and MUST be the instance's own max(1, ||g||_inf)
    # (see :func:`invariant_bound`). It was hardcoded to 1.0, which is below the
    # true bound wherever g peaks above 1 (this family draws peaks at 0.8-1.1):
    # there the clamp replaced a negative reaction lam(u - u^3) with zero and
    # biased every label slightly high near the peaks.
    for k_step in range(n_steps):
        y = field(t, x).clamp(-bound, bound)
        hk = dt * inst.driver(t, x, y)              # Euler backward step
        val += hk
        if anti:                                            # antithetic Brownian
            dwh = torch.randn(Q, half, d, device=device)
            dW = torch.cat([dwh, -dwh], dim=1).reshape(Q * n_traj, d)
        else:
            dW = torch.randn_like(x)
        if xi0 is None:
            xi0 = dW.clone()                    # first increment -> Z estimate
        if keep:
            PX[k_step] = x                       # X_k BEFORE the update
            PXI[k_step] = dW
            PH[k_step] = hk
        if grad_field is not None:
            # martingale control variate, integrand at the LEFT endpoint so it
            # stays non-anticipating and the expectation is exactly zero
            mart += SIGMA * sqrt_dt * (grad_field(t, x) * dW).sum(
                dim=-1, keepdim=True)
        x = x + SIGMA * dW * sqrt_dt                        # Euler-Maruyama
        t += dt
    if exact:
        val += inst.bound_fn(x)                             # terminal g(X_T)
    else:
        val += field(t_end, x).clamp(-bound, bound)         # neural termination

    Yq = val.view(Q, n_traj, 1)
    zcoef = None
    # Z_0 = E[Y xi_0] / sqrt(dt), the projection of the payoff onto the first
    # increment. Cheap: xi_0 is already drawn and Y already accumulated.
    grad_w = ((Yq * xi0.view(Q, n_traj, d)).mean(dim=1) / (sqrt_dt * SIGMA)
              if return_grad else None)

    if cv == "bsde":
        # BOTH solution components from ONE simulation, no external gradient.
        #
        # Z_k = sigma grad w(t_k, X_k) is a conditional expectation on the paths
        # that produced Y:  E[Y_future xi_k | X_k] = sqrt(dt) Z_k, so regressing
        # Y_future xi_k / sqrt(dt) on X_k estimates Z_k itself. Only the FUTURE
        # payoff enters (driver terms before k are F_k-measurable: pure noise).
        #
        # Cost is memory traffic over the stored path, not arithmetic, so:
        # the path is written into preallocated buffers (no stack copies), the
        # two cross-fit groups are CONTIGUOUS blocks of antithetic pairs on a
        # (partner, pair) view (no gathers; partners are exact negations, so a
        # pair is never split), slopes are per-component (no solve), statistics
        # are taken at BSDE_CV_KNOTS time knots and held between them, and the
        # martingale is accumulated without materialising Z over the path.
        N_t = n_steps
        G = 2 if anti else 1
        P = n_traj // G
        X = PX.view(N_t, Q, G, P, d)
        XI = PXI.view(N_t, Q, G, P, d)
        H = PH.view(N_t, Q, G, P, 1)
        yT = (inst.bound_fn(x) if exact
              else field(t_end, x).clamp(-bound, bound)).view(1, Q, G, P, 1)
        Yf = yT + torch.flip(torch.cumsum(torch.flip(H, [0]), 0), [0])  # future payoff
        K = N_t if BSDE_CV_KNOTS is None else max(1, min(N_t, int(BSDE_CV_KNOTS)))
        knots = torch.linspace(0, N_t - 1, K, device=device).round().long()
        # step -> most recent knot at or before it
        jmap = torch.searchsorted(knots, torch.arange(N_t, device=device), right=True) - 1
        jmap = jmap.clamp_min(0)
        h2 = P // 2
        grp = (slice(0, h2), slice(h2, P))
        M2 = torch.zeros(Q, G, P, 1, device=device)
        a0 = []
        for src, dst in ((grp[0], grp[1]), (grp[1], grp[0])):
            xs = X[knots][..., src, :]                                # [K,Q,G,hg,d]
            var, mu = torch.var_mean(xs, dim=(2, 3), keepdim=True, unbiased=False)
            yk = Yf[knots][..., src, :]
            ts = (yk - yk.mean(dim=(2, 3), keepdim=True)) * XI[knots][..., src, :] / sqrt_dt
            a = ts.mean(dim=(2, 3), keepdim=True)                      # [K,Q,1,1,d]
            b = ((xs - mu) * (ts - a)).mean(dim=(2, 3), keepdim=True) / var.clamp_min(1e-8)
            A, B, MU = a[jmap], b[jmap], mu[jmap]                      # [N_t,Q,1,1,d]
            xi_d, x_d = XI[..., dst, :], X[..., dst, :]
            # sum_k sqrt(dt) < a + b (x - mu), xi >  without forming Z
            M2[:, :, dst] = (sqrt_dt * ((A * xi_d).sum(-1, keepdim=True)
                                        + (B * (x_d - MU) * xi_d).sum(-1, keepdim=True))
                             ).sum(dim=0)
            a0.append(a[0])
        out = (Yq - BSDE_CV_SCALE * M2.view(Q, n_traj, 1)).mean(dim=1)
        if return_zcoef:
            # Z_k(x) ~ a_k + b_k (x - mu_k) per query point and knot, fitted on
            # ALL paths (no cross-fit: consumers apply it to INDEPENDENT paths,
            # e.g. the diffusion loss), as (a, b, mu) each [K, Q, d]
            xs = X[knots]
            var, mu = torch.var_mean(xs, dim=(2, 3), keepdim=True, unbiased=False)
            yk = Yf[knots]
            ts = (yk - yk.mean(dim=(2, 3), keepdim=True)) * XI[knots] / sqrt_dt
            a = ts.mean(dim=(2, 3), keepdim=True)
            b = ((xs - mu) * (ts - a)).mean(dim=(2, 3), keepdim=True) / var.clamp_min(1e-8)
            zcoef = (a.view(K, Q, d), b.view(K, Q, d), mu.view(K, Q, d))
        if return_grad:
            # knot 0 is the query point (all paths coincide there), so the
            # intercept IS Z(t, x_q); average the two cross-fit groups, /sigma
            grad_w = (0.5 * (a0[0] + a0[1])).view(Q, d) / SIGMA
        del PX, PXI, PH, X, XI, H, Yf
    elif grad_field is None or cv == "off":
        out = Yq.mean(dim=1)
    elif cv == "auto":
        # Coefficient 1 is exact only when grad w is the TRUE solution gradient
        # (then Y - M is deterministic). It is not: measured Var(M)/Var(Y) is
        # 2.2-2.5 and corr(Y, M) only 0.42 (2-D) / 0.86 (3-D), so subtracting all
        # of M over-corrects -- it TRIPLES the target variance in 2-D. The
        # least-squares coefficient Cov(Y,M)/Var(M) is what a simulation-native
        # Z estimate would give, and independently agrees with it: in 3-D the
        # Malliavin projection puts |grad w| at 0.52x the grid value, matching
        # the fitted 0.539. Measured variance vs no control variate:
        # 2-D 3.03 -> 0.87, 3-D 0.83 -> 0.27.
        #
        # Cross-fitted, because E[M] = 0 needs the integrand independent of the
        # increment it multiplies and a beta fitted on the same paths is not.
        # The split divides PAIRS and never splits one: antithetic partners are
        # exact negations, so one member determines the other.
        Mq = mart.view(Q, n_traj, 1)
        if anti and n_traj >= 8:
            half = n_traj // 2
            a = torch.arange(half // 2, device=device)
            b = torch.arange(half // 2, half, device=device)
            groups = (torch.cat([a, a + half]), torch.cat([b, b + half]))
        else:
            cut = n_traj // 2
            groups = (torch.arange(cut, device=device),
                      torch.arange(cut, n_traj, device=device))
        out = torch.zeros(Q, 1, device=device)
        for src, dst in (groups, groups[::-1]):
            ys, ms = Yq[:, src], Mq[:, src]
            mc = ms - ms.mean(dim=1, keepdim=True)
            beta = ((ys - ys.mean(dim=1, keepdim=True)) * mc).mean(dim=1) / \
                mc.pow(2).mean(dim=1).clamp_min(1e-12)
            out = out + (Yq[:, dst] - beta.unsqueeze(1) * Mq[:, dst]).mean(dim=1)
        out = out / 2
    else:
        out = (Yq - float(cv) * mart.view(Q, n_traj, 1)).mean(dim=1)
    if return_zcoef:
        return out, grad_w, zcoef
    return (out, grad_w) if return_grad else out


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def make_slice_times(t_start, sub_frac, k_slices):
    hi, _ = rollout_t_end(t_start, sub_frac)
    hi = max(hi, t_start + 1e-4)
    return [t_start + (hi - t_start) * j / (k_slices - 1) for j in range(k_slices)]


def train(dim, steps, device, sub_frac=0.0, n_traj=32, k_slices=3, seed=0,
          wb=None, eval_every=0, eval_times=(0.0,), lr=1e-3, ema_decay=0.99,
          terminal_weight=0.0, fno_hidden=None, fno_modes=None, fno_layers=None, grid_res=None):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    # rescaled long-horizon interfaces span wider than the T=0.3 bumps -> widen
    rescaled = abs(T_PHYS - 0.3) >= 1e-9
    geo_kw = dict(latent_L=3.0, grid_L=4.0) if rescaled else {}
    # Resolution of the grid the BSDE value field is read off. Profiling a step
    # (scripts/profile_bsde_step.py): the k_slices GINO forwards over this grid
    # are 65% of a 3-D BSDE step, while the rollout is only 5% -- so this, not the
    # rollout, is the throughput knob. Measured against the exact radial arbiter,
    # gres=20 already sits at the no-grid floor (3.03e-3 vs 3.00e-3) and gres=14
    # costs +18% target bias for a 3x cheaper readout, which buys ~1.75x more
    # optimiser steps. Unused by the FEM arm (it reads fem_w, not this grid).
    if grid_res is not None:
        geo_kw["grid_res"] = grid_res
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
                tgt, _ = bsde_rollout(insts[b], xq, t_start, field_b,
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


def _fd_stencil(model, ig, lq, xq, feats, tt, eps):
    """Operator at xq and at xq +- eps e_k for every k, from ONE forward.

    GINO's FNO latent does not depend on the output queries, so evaluating the
    2d+1 stencil points as separate forwards recomputes it 2d+1 times AND keeps
    2d+1 autograd graphs alive for the backward: 22.99 GiB in 3-D, CUDA OOM.
    Stacking the points into a single ``output_queries`` tensor computes the
    latent once; only the out-GNO, the cheap part, sees (2d+1)N points.

    Returns ``(w, plus, minus)`` with ``w`` [B,N,1] at xq and ``plus[k]``,
    ``minus[k]`` [B,N,1] at xq +- eps e_k.
    """
    N, d = xq.shape
    shifts = [torch.zeros_like(xq)]
    for k in range(d):
        e = torch.zeros_like(xq); e[:, k] = eps
        shifts += [e, -e]
    Xs = torch.cat([xq + sft for sft in shifts], dim=0)         # [(2d+1)N, d]
    out = model(input_geom=ig, latent_queries=lq, output_queries=Xs.unsqueeze(0),
                x=feats, ada_in=tt)
    if BOUND_OUTPUT:
        out = OUT_SCALE * torch.tanh(out)
    parts = out.split(N, dim=1)                                  # 2d+1 x [B,N,1]
    return parts[0], parts[1::2], parts[2::2]


def pino_residual(model, ig, lq, xq, feats, t, inst_lams, device, eps=FD_EPS,
                  eps_t=FD_EPS_T):
    """PDE residual of the Allen-Cahn terminal-value form.

        R = d_t w + Laplacian(w) + lam (w - w^3)          (sigma = sqrt(2))

    One structural caveat, shared with DeepRitz and NOT with the BSDE or FEM
    arms: the residual is imposed only on the query box [-QUERY_L, QUERY_L]^d
    with an anchor at t = T, and a backward parabolic problem on a bounded box
    needs LATERAL data as well. Information reaches an interior point from all
    of space (the paths of the BSDE arm leave the box and terminate on the exact
    g; the FEM reference solves a larger domain), so this objective does not
    determine the solution near the box edge -- and the score is taken on that
    same box. Widening the collocation box would need a margin of ~2 sigma
    sqrt(T) ~ 2.8 beyond QUERY_L = 2.5, past the grid_L = 4.0 the operator's
    input geometry covers, so it is a property of the arm rather than a knob.

    The SPATIAL derivatives are central differences over a step that spans several
    query cells, not autograd. This is not a shortcut -- it is what makes the
    residual a residual. GINO's output is piecewise constant in the query
    coordinate (a line profile is flat for ~7 samples at spacing 0.005, then
    jumps), so its pointwise spatial derivative is ~0 almost everywhere: measured
    against the operator's own finite difference, autograd is 182-498x too small
    for grad w and 840-1568x too small for Laplacian(w). Autograd is CORRECT --
    the function really is flat between jumps -- but that derivative carries no
    information about the field the operator represents. Feeding it into the
    residual leaves d_t w + lam (w - w^3): the reaction ODE, with the diffusion
    silently absent.

    Each finite difference is an operator forward and so is differentiable in the
    weights exactly as autograd is; 2d+1 forwards replaces one forward plus a
    double backward, which is comparable in cost.

    The TIME derivative is a central difference too, for a different reason.
    ada_in is a single scalar per forward, so ``autograd.grad(w.sum(), t)``
    returns ONE number -- the sum of d_t w over every point and instance -- which
    then broadcasts to the whole residual. Measured at the training batch
    (B=8, N=256): that scalar is 3.23 while the true per-point d_t w has RMS
    0.0140, so it was 231x too large and carried ~100% of the residual energy,
    leaving Laplacian(w) and the reaction under 1% of the loss. The arm was
    minimising a global constant, which is why its normalisation scale came out
    at 35-95 instead of ~0.01 and why tuning drove terminal_weight to 1000+.
    An earlier check cleared this by comparing autograd against a finite
    difference AFTER summing both, so it validated the derivative and not the
    aggregation.

    Forward-mode AD would give the per-point derivative in one pass but is
    unavailable here: torch_scatter's segment_csr is a C++ autograd Function and
    functorch refuses to trace it. Two extra N-point forwards at t +- FD_EPS_T
    cost ~+40% (2-D) / +28% (3-D) against the existing (2d+1)N stencil forward,
    and they replace a double backward, so the step is barely more expensive.

    ``inst_lams`` is the per-instance reaction strength, shape [B, 1, 1].
    """
    tt = torch.tensor([float(t)], device=device)
    w, plus, minus = _fd_stencil(model, ig, lq, xq, feats, tt, eps)   # one forward
    lap = sum((wp - 2.0 * w + wm) / eps ** 2 for wp, wm in zip(plus, minus))
    # per-point d_t w, clipped to [0, T] so the stencil never leaves the slab
    # (one-sided at the ends, hence the division by the realised span)
    tp = min(float(t) + eps_t, float(T_HORIZON))
    tm = max(float(t) - eps_t, 0.0)
    oq = xq.unsqueeze(0)
    dwdt = (op(model, ig, lq, oq, feats, tp, device)
            - op(model, ig, lq, oq, feats, tm, device)) / (tp - tm)
    return dwdt + lap + inst_lams * (w - w ** 3)


def diffusion_path_loss(model, ig, lq, xq, feats, t, insts, zcoefs, device,
                        n_paths=8, use_checkpoint=False):
    """Nuesken-Richter diffusion loss along Euler-Maruyama paths, Monte Carlo
    averaged, with Z from the BSDE regression and u_theta evaluated exactly.

    For the terminal-value Allen-Cahn form, Ito along X_{n+1} = X_n + sigma
    sqrt(dt) xi_n gives, for the true solution,

        u(t_{n+1}, X_{n+1}) - u(t_n, X_n) = -h(u) dt + <Z_n, dW_n>,   Z = sigma grad u,

    so a candidate u_theta has the per-step residual

        R_n = u_theta(t_{n+1}, X_{n+1}) - u_theta(t_n, X_n) + h(u_theta(t_n, X_n)) dt
              - <Z_n(X_n), dW_n>,

    and the loss is E |R_n|^2 / dt averaged over paths, steps and query points.

    Three choices, each replacing something the first version did wrong:
      * MONTE CARLO over ``n_paths`` paths from every query point and over all
        N_t Euler-Maruyama steps to T -- the same discretisation as the BSDE
        rollout, no separate step-length knob. The first version drew ONE
        increment per point per step and left the averaging to SGD.
      * u_theta is evaluated EXACTLY at the path points, with gradients, one
        stacked forward per time level. No value grid, no interpolation. The
        forward SDE does not depend on the instance, so all B instances share
        the path points and the cost is N_t forwards per training step.
      * Z comes from the BSDE regression (``bsde_rollout(..., return_zcoef=True)``:
        a linear model a + b (x - mu) per query point and time knot, fitted on
        4096 paths), NOT from a derivative of the network. On this operator the
        pointwise spatial derivative is ~0 (piecewise-constant output), and the
        user's directive is that Z is a BSDE quantity.
    The last increment ends at the exact terminal datum f(X_T), which is what
    pins the value; the standard terminal anchor is added by the caller.

    ``use_checkpoint`` recomputes each forward in the backward pass so only one
    forward's activations are alive at a time (needed in 3-D: 20 retained
    graphs would exceed the card).
    """
    Q, d = xq.shape
    B = feats.shape[0]
    t0, T_end = float(t), float(T_HORIZON)
    lam_max = max(float(i.lam) for i in insts)
    N_t = rollout_n_steps(lam_max, T_end - t0)
    dt = (T_end - t0) / N_t
    sq = dt ** 0.5
    L = n_paths
    lams = torch.tensor([float(i.lam) for i in insts], device=device).view(B, 1, 1)
    a, b, mu = zcoefs                                           # each [B, K, Q, d]
    K = a.shape[1]

    def fwd(pts, tt):
        o = model(input_geom=ig, latent_queries=lq, output_queries=pts.unsqueeze(0),
                  x=feats, ada_in=tt)
        return OUT_SCALE * torch.tanh(o) if BOUND_OUTPUT else o

    def u_at(pts, tval):
        tt = torch.tensor([tval], device=device)
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(fwd, pts, tt, use_reentrant=False)
        return fwd(pts, tt)

    x = xq.unsqueeze(0).expand(L, Q, d).reshape(L * Q, d)       # shared paths
    u_prev = u_at(x, t0)                                        # [B, L*Q, 1]
    loss = torch.zeros((), device=device)
    for n in range(N_t):
        tn = t0 + n * dt
        j = min(int(n * (K - 1) / max(N_t - 1, 1)), K - 1)      # knot in use
        xi = torch.randn(L, Q, d, device=device)
        xv = x.view(1, L, Q, d)
        Z = a[:, j].unsqueeze(1) + b[:, j].unsqueeze(1) * (xv - mu[:, j].unsqueeze(1))
        mart = (Z * xi.unsqueeze(0)).sum(dim=-1) * sq            # [B, L, Q]
        x_next = x + SIGMA * sq * xi.reshape(L * Q, d)
        if n < N_t - 1:
            u_next = u_at(x_next, tn + dt)
        else:                                                   # exact terminal
            u_next = torch.stack([i.bound_fn(x_next) for i in insts])
        up = u_prev.view(B, L, Q)
        R = u_next.view(B, L, Q) - up + lams * (up - up ** 3) * dt - mart
        loss = loss + R.pow(2).mean() / dt
        u_prev, x = u_next, x_next
    return loss / N_t


def deepritz_loss(model, ig, lq, xq, feats, r_n, r_np1, inst_lams, f_n, device):
    """One minimizing-movement (JKO) step of the Ginzburg--Landau gradient flow.

    Allen-Cahn is the L2 gradient flow of
        E[v] = int  1/2 |grad v|^2 + (lam/4) (1 - v^2)^2  dx,
    since delta E / delta v = -Lap v - lam (v - v^3) = -Lap v - h(v). Discretising
    the flow implicitly gives, for a step dr from a known slice v_n,
        v_{n+1} = argmin_v  ||v - v_n||^2 / (2 dr) + E[v],
    whose stationarity condition is exactly backward Euler for the flow. Deep Ritz
    takes that functional directly as the loss, with Monte-Carlo quadrature over x.

    Three properties distinguish this from the PINO residual. It needs only FIRST
    derivatives of the network (the energy has |grad v|^2, not Lap v), so no double
    backward. It minimises an energy rather than a squared residual, which is
    better conditioned. The exact JKO sequence also decreases E monotonically,
    which would rule out settling in the wrong phase of the bistable potential --
    but that argument does NOT carry to this implementation: for n > 0 the
    previous slice ``v_prev`` is the operator's own detached output, not the
    converged minimiser of the previous step, so the chain is bootstrapped and
    inherits the same self-reference the BSDE label has. Only n = 0 is anchored
    (on the exact terminal datum). The lateral-condition caveat in
    :func:`pino_residual` applies here too.

    Time runs FORWARD here: r = T - t, so v(r) = u(., T - r) and r = 0 is the known
    terminal datum f. ``f_n`` supplies the r=0 slice exactly when n = 0; otherwise
    the previous slice comes from the operator itself (detached).
    """
    tt = torch.tensor([float(T_HORIZON - r_np1)], device=device)

    def _v(x):
        o = model(input_geom=ig, latent_queries=lq, output_queries=x.unsqueeze(0),
                  x=feats, ada_in=tt)
        return OUT_SCALE * torch.tanh(o) if BOUND_OUTPUT else o

    # v and |grad v|^2 from ONE stacked forward (see _fd_stencil). Central
    # differences over a cell-spanning step, not autograd: the operator is
    # piecewise constant in the query coordinate, so autograd gives ~0 and the
    # Dirichlet term would vanish, leaving only the double-well potential whose
    # minimisers are v = +/-1 -- the drift toward u=1 seen with longer budgets.
    v, plus, minus = _fd_stencil(model, ig, lq, xq, feats, tt, FD_EPS)
    grad_sq = sum(((vp - vm) / (2.0 * FD_EPS)) ** 2 for vp, vm in zip(plus, minus))

    if f_n is not None:                        # r_n = 0: exact terminal datum
        v_prev = f_n
    else:
        with torch.no_grad():
            v_prev = op(model, ig, lq, xq.detach().unsqueeze(0), feats,
                        T_HORIZON - r_n, device)
    dr = max(r_np1 - r_n, 1e-8)
    move = (v - v_prev).pow(2).mean() / (2.0 * dr)
    energy = (0.5 * grad_sq
              + 0.25 * inst_lams * (1.0 - v ** 2) ** 2).mean()
    return move + energy


def train_buffered(dim, steps, device, sub_frac=0.0, n_traj=16, k_slices=3, seed=0,
                   wb=None, eval_every=0, eval_times=(0.0,), lr=1e-3,
                   pool_size=64, nX=256, nT=8, buf_cap=20, terminal_weight=0.0,
                   warmup=0, fem_target=False,
                   fno_hidden=None, fno_modes=None, fno_layers=None,
                   refresh_per_step=2, budget_s=None, fem_n=None,
                   grid_res=None, refresh_frac=None, control_variate=False,
                   pino_target=False, deepritz=False, n_jko=8,
                   pino_normalise=True, diffusion=False, diff_paths=8,
                   lr_sched="flat", lr_min_frac=0.02, ema_decay=0.0,
                   eval_cb=None, eval_wall_s=None, clamp=0.0, label_ema=False,
                   resample_xq=False):
    """Like :func:`train` but with an nwos-style running-mean target BUFFER.

    ``clamp`` is the LABEL clamp, as a multiple of each instance's
    max(1, ||g||_inf) (0 = off). The driver's value field is always clamped to
    that bound inside :func:`bsde_rollout`, independently of this.

    ``resample_xq`` draws a FRESH query set every step for the DeepRitz, PINO and
    diffusion branches. Those objectives are residuals and energies, which are
    normally Monte-Carlo integrated over fresh collocation points; pinning them
    to one set of nX points for thousands of steps invites overfitting the
    residual at those points, and both arms were measured turning UPWARD in the
    second half of the budget. The fixed set is a requirement of the BSDE branch
    ONLY -- its running-mean buffer is indexed by (instance, time, point) cell,
    so the points must keep their identity -- so this flag is ignored there
    rather than silently corrupting the buffer.

    A FIXED pool of instances, query grid and time grid is used so each
    (instance, time, point) target cell has a stable identity. Every visit folds
    a fresh MC target into a running mean (capped at ``buf_cap`` so recent, more
    accurate estimates dominate once the operator has moved on -- our target is
    operator-dependent, unlike nwos's linear WoS). The operator regresses against
    the LOW-NOISE buffered target while the rollout value_fn stays the LIVE
    operator (self-consistent) -- this damps the noise-driven Picard oscillation
    without the stale-value_fn bias that an EMA target network introduces."""
    # budget clock starts HERE, before any FEM dataset generation below, so an
    # equal-wall-clock comparison charges FEM supervision for the meshing +
    # solves it needs and BSDE weak supervision does not.
    t_budget0 = time.time()
    res_scale = None      # PINO residual normalisation, fixed once per run
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    rescaled = abs(T_PHYS - 0.3) >= 1e-9
    geo_kw = dict(latent_L=3.0, grid_L=4.0) if rescaled else {}
    # Resolution of the grid the BSDE value field is read off. Profiling a step
    # (scripts/profile_bsde_step.py): the k_slices GINO forwards over this grid
    # are 65% of a 3-D BSDE step, while the rollout is only 5% -- so this, not the
    # rollout, is the throughput knob. Measured against the exact radial arbiter,
    # gres=20 already sits at the no-grid floor (3.03e-3 vs 3.00e-3) and gres=14
    # costs +18% target bias for a 3x cheaper readout, which buys ~1.75x more
    # optimiser steps. Unused by the FEM arm (it reads fem_w, not this grid).
    if grid_res is not None:
        geo_kw["grid_res"] = grid_res
    (input_geom, latent_queries, field_grid, gres, gL,
     in_radius, out_radius) = T.make_geometry(dim, device, generator=gen, **geo_kw)
    model, in_ch = build_gino_time(dim, device, in_radius, out_radius,
                                   fno_hidden=fno_hidden, fno_modes=fno_modes,
                                   fno_layers=fno_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    # lr_sched="cosine_budget": decay the learning rate with the WALL-CLOCK
    # fraction of the budget, down to lr_min_frac*lr. The step-based cosine
    # above has T_max=steps=1e6 under a budget, i.e. it is flat for the whole
    # run: every arm so far trained at a constant 1e-3 and stopped mid-noise.
    # ema_decay>0: keep an exponential moving average of the WEIGHTS and return
    # it as the trained model. Not the EMA target network the docstring rejects
    # (the rollout's value field still comes from the live model); it only
    # smooths what is evaluated, which matters because the error was measured
    # to oscillate ~3x between evaluations at these budgets.
    ema_model = None
    if ema_decay and ema_decay > 0:
        import copy
        ema_model = copy.deepcopy(model)
        for pe in ema_model.parameters():
            pe.requires_grad_(False)
    ig, lq = input_geom.unsqueeze(0), latent_queries.unsqueeze(0)
    G = field_grid.shape[0]
    geom = (ig, lq, field_grid, gres, gL, input_geom, in_ch)

    # fixed pool: instances, features, shared query grid, time grid
    pool = [sample_instance(dim, gen) for _ in range(pool_size)]
    pool_feats = torch.stack([T.instance_features(m, input_geom, in_ch) for m in pool])
    # per-instance max(1, ||g||_inf): the constant the rollout clamps the driver's
    # value field to. One g evaluation on the value grid, so it costs nothing.
    pool_bnd = [invariant_bound(m, field_grid) for m in pool]
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
        # resolution is the FEM arm's other budget knob: in 3-D n=41 costs 31.7s
        # of setup + 2.98s/solve vs 1.3s + 0.38s at n=25, so a finer grid can eat
        # the whole budget before training starts.
        fem_n_ = fem_n or (141 if dim == 2 else 41)
        fem_solver = AllenCahnFEMSolver(dim, n=fem_n_, dt=1e-2, device=device)
        # solve to T_HORIZON, not 1.0: the SpacetimeField index is NORMALISED
        # forward time, so w(t,.) = u_forward(T_H - t) = stf(1 - t/T_H). Solving
        # to 1.0 while training on a 0.3 horizon supervises the operator against
        # forward times [0.715, 1.0] of the wrong problem. The two coincide only
        # in the rescaled unit-horizon setting (T_HORIZON == 1).
        fem_w = [fem_solver.solve(pool[p], T=T_HORIZON, n_slices=11)[1]
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

    # Replacement is naturally a per-slot PROBABILITY, not an absolute count: at
    # fixed refresh_per_step a larger pool turns each slot over more slowly
    # (2/64 = 3.1% per step vs 2/256 = 0.78%), so pool size and turnover rate move
    # together and cannot be attributed separately. refresh_frac pins the
    # probability and scales the count with the pool.
    if refresh_frac is not None:
        refresh_per_step = max(int(round(refresh_frac * pool_size)), 1)
        print(f"  [d={dim}] replacement p={refresh_frac:.4f}/slot/step -> "
              f"refresh_per_step={refresh_per_step} of pool {pool_size}", flush=True)

    # rolling replay buffer: how many pool slots to REPLACE with fresh instances
    # each step. The buffer still UPDATES (running mean) within a slot's lifetime
    # -- but slots are continuously refreshed, so over training the operator sees
    # ~unlimited instances (no fixed pool -> no overfitting wall) while keeping
    # the buffer's variance reduction. refresh_per_step slots live ~pool/refresh
    # steps each, accumulating a few MC estimates before eviction.
    fem_solver_live = None
    if fem_target and refresh_per_step > 0:
        from src.solvers.fem import AllenCahnFEMSolver
        fem_solver_live = AllenCahnFEMSolver(
            dim, n=(fem_n or (141 if dim == 2 else 41)), dt=1e-2, device=device)

    def _apply_update():
        """Gradient clip, optimiser step, lr schedule, weight EMA -- EVERY arm.

        This used to sit in the BSDE branch alone, and the data-free branches
        ``continue`` before reaching it, so ``lr_sched`` and ``ema_decay`` were
        accepted and silently ignored for DeepRitz, PINO and the diffusion loss:
        they always ran the step-based CosineAnnealingLR with T_max = steps =
        1e6, which is flat to within 0.01% over the 3k-11k steps a budget buys.
        The wall-clock cosine was worth 1.2-1.6x to the BSDE arm, so while it
        was reachable from one branch only, any "BSDE with schedule vs baseline"
        row compared two different optimisers.
        """
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if lr_sched == "cosine_budget" and budget_s is not None:
            frac = min(1.0, (time.time() - t_budget0) / budget_s)
            cur = lr * (lr_min_frac + (1.0 - lr_min_frac)
                        * 0.5 * (1.0 + math.cos(math.pi * frac)))
            for gp in opt.param_groups:
                gp["lr"] = cur
        else:
            sched.step()
        if ema_model is not None:
            with torch.no_grad():
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(ema_decay).add_(pm.detach(), alpha=1.0 - ema_decay)
                for be, bm in zip(ema_model.buffers(), model.buffers()):
                    be.copy_(bm)

    t0 = time.time()
    last_eval_el = 0.0     # budget-time of the last eval_cb call
    prof = {"refresh": 0.0, "labels": 0.0, "iter": 0.0, "eval": 0.0, "steps": 0}
    _cuda = str(device).startswith("cuda")

    def _mark(sync=True):
        if sync and _cuda:
            torch.cuda.synchronize()
        return time.time()

    def _end_of_step(do_prof, t_iter0, step):
        """Tail every arm must run: the timed evaluation callback and the sampled
        profile. The data-free branches return early, so they have to call this
        or their convergence curves would be empty."""
        nonlocal t_budget0, last_eval_el
        ev = 0.0
        if eval_cb is not None and eval_wall_s and budget_s is not None:
            el = time.time() - t_budget0
            if el - last_eval_el >= eval_wall_s:
                te0 = time.time()
                eval_cb(ema_model if ema_model is not None else model, el,
                        (ig, lq, field_grid, gres, gL, input_geom, in_ch))
                ev = time.time() - te0
                t_budget0 += ev          # evaluation is not charged to the budget
                prof["eval"] += ev
                last_eval_el = el
        if do_prof:
            prof["iter"] += _mark() - t_iter0 - ev
            prof["steps"] += 1
        if prof["steps"]:
            # live counters the evaluation callback reads, so a convergence curve
            # can use TRAINING work as its x-axis instead of elapsed time:
            # opt_steps is exact, core_s is opt_steps x the sampled cost of one
            # step with validation excluded (labels and pool refresh reported
            # separately, since for BSDE the labels ARE the work).
            per = prof["iter"] / prof["steps"]
            lab = prof["labels"] / prof["steps"]
            ref = prof["refresh"] / prof["steps"]
            PROFILE.update(prof, opt_steps=step + 1, core_s=(step + 1) * (per - lab - ref),
                           train_s=(step + 1) * per, labels_s=(step + 1) * lab)

    for step in range(steps):
        # equal-wall-clock mode: stop on time rather than step count, so BSDE and
        # FEM supervision are compared on compute spent, not iterations taken.
        # Measured from t_budget0 (function entry), so the FEM dataset generation
        # above is inside the budget -- the fairness condition.
        if budget_s is not None and time.time() - t_budget0 >= budget_s:
            break
        do_prof = (step % PROF_EVERY == 0)
        _iter_t0 = _mark(do_prof)
        # replace (evict + insert fresh) some slots -> rolling infinite-instance
        if refresh_per_step > 0 and step > 0:
            for _ in range(refresh_per_step):
                s = int(torch.randint(0, pool_size, (1,), generator=gen).item())
                pool[s] = sample_instance(dim, gen)
                pool_feats[s] = T.instance_features(pool[s], input_geom, in_ch)
                pool_bnd[s] = invariant_bound(pool[s], field_grid)
                buf[s].zero_(); cnt[s].zero_()
                if fem_w is not None:
                    _tf = time.time()
                    fem_w[s] = fem_solver_live.solve(
                        pool[s], T=T_HORIZON, n_slices=11)[1]
                    fem_gen_time += time.time() - _tf; fem_solves += 1

        if do_prof:
            prof["refresh"] += _mark() - _iter_t0
        idx = torch.randint(0, pool_size, (B,), generator=gen).tolist()
        j = int(torch.randint(0, nT, (1,), generator=gen).item())
        t = Tq[j]
        feats = pool_feats[idx]                                   # [B, N_in, C]

        # Value-field slices feed the BSDE rollout target and the diffusion
        # residual. DeepRitz and PINO never read them, and computing them anyway
        # charged those arms k_slices FULL-GRID forwards on every step.
        if fem_w is None and not (deepritz or pino_target):
            _l0 = _mark(do_prof)
            slice_times = make_slice_times(t, sub_frac, k_slices)
            # Target network: read the driver's value field off the EMA copy of
            # the weights rather than the live ones. The label already depends on
            # the operator it trains, and a live field lets a wobble amplify
            # itself within a few steps; an EMA field moves ~200x slower, which
            # is the standard cure for exactly this feedback (as in Q-learning).
            lab_model = ema_model if (label_ema and ema_model is not None) else model
            with torch.no_grad():
                slices = [op(lab_model, ig, lq, field_grid.unsqueeze(0), feats, s, device)
                          for s in slice_times]
                # No clamp here. The driver's value field is already clamped to
                # the instance's max(1, ||g||_inf) INSIDE bsde_rollout, on every
                # read, and interpolation is linear so it cannot leave a range
                # its inputs are inside. Clamping the slices first at any bound
                # >= that one is exactly a no-op -- which is what the earlier
                # "field clamp" arm turned out to be. ``clamp`` now controls the
                # one thing that is not already bounded: the LABEL.
            if do_prof:
                prof["labels"] += _mark() - _l0
        # Collocation points for the data-free branches. The BSDE/FEM path must
        # keep Xq (buffer cell identity), so this never reaches it.
        xq_s = Xq
        if resample_xq and (deepritz or pino_target or diffusion):
            xq_s = ((torch.rand(nX, dim, generator=gen) * 2 - 1) * QUERY_L).to(device)

        opt.zero_grad()
        if pino_target and pino_normalise and res_scale is None:
            # Fix the residual's SCALE once, instead of hand-weighting the anchor
            # against it. The raw residual starts around 91 while the anchor
            # ||u(T,.)-f||^2 starts around 1, which is why tuning drove
            # terminal_weight to 1000 -- and a 1000x term makes the landscape
            # dominated by one loss, saturates clip_grad_norm every step, and was
            # measured to diverge on 1 seed in 3 at full budget (0.097/0.072/0.993)
            # even though it looked stable at the short screening budget.
            # Dividing by the initial RMS puts both terms at O(1), so
            # terminal_weight ~ 1 is meaningful and lr transfers from the other
            # arms. It rescales the objective only -- the minimiser is unchanged.
            # pino_residual needs autograd internally, so this probe cannot run
            # under no_grad; take the value and drop the graph instead
            lam0 = torch.tensor([pool[p].lam for p in idx],
                                device=device).view(-1, 1, 1)
            R0 = pino_residual(model, ig, lq, xq_s, feats, t, lam0, device)
            res_scale = float(R0.pow(2).mean().detach().clamp_min(1e-12))
            del R0
            print(f"  [d={dim}] PINO residual scale at init = "
                  f"{res_scale:.3e} (normalising)", flush=True)

        if deepritz:
            lams = torch.tensor([pool[p].lam for p in idx], device=device
                                ).view(-1, 1, 1)
            dr = T_HORIZON / n_jko
            n = int(torch.randint(0, n_jko, (1,), generator=gen).item())
            r_n, r_np1 = n * dr, (n + 1) * dr
            f_n = (torch.stack([pool[p].bound_fn(xq_s) for p in idx])
                   if n == 0 else None)
            loss = deepritz_loss(model, ig, lq, xq_s, feats, r_n, r_np1, lams,
                                 f_n, device)
            loss_term = torch.zeros((), device=device)
            if terminal_weight > 0:
                predT = op(model, ig, lq, xq_s.unsqueeze(0), feats, T_HORIZON,
                           device)
                gT = torch.stack([pool[p].bound_fn(xq_s) for p in idx])
                loss_term = nn.functional.mse_loss(predT, gT)
                loss = loss + terminal_weight * loss_term
            loss.backward()
            _apply_update()
            _log_and_eval(model, geom, dim, device, step, steps, loss.item(),
                          {"lr": opt.param_groups[0]["lr"],
                           "train/loss_terminal": loss_term.item()},
                          wb, eval_every, eval_times, t0, tag=" deepritz")
            _end_of_step(do_prof, _iter_t0, step)
            continue

        if diffusion:
            # Nuesken-Richter along paths. Z for the residual comes from the
            # BSDE regression on 4096 paths per instance (no gradient, the
            # value field read off the grid slices as in the BSDE arm); the
            # residual itself evaluates u_theta exactly at its own paths.
            insts = [pool[p] for p in idx]
            za, zb, zmu = [], [], []
            _l0 = _mark(do_prof)
            with torch.no_grad():
                for bi, p in enumerate(idx):
                    grids_b = [slices[k][bi].reshape((gres,) * dim) for k in range(k_slices)]
                    field_b = SpacetimeField(grids_b, slice_times, gL, dim)
                    _, _, (ca, cb, cmu) = bsde_rollout(pool[p], xq_s, t, field_b, n_traj,
                                                       sub_frac, device, cv="bsde",
                                                       return_zcoef=True,
                                                       bound=pool_bnd[p])
                    za.append(ca); zb.append(cb); zmu.append(cmu)
            if do_prof:
                prof["labels"] += _mark() - _l0
            zcoefs = (torch.stack(za), torch.stack(zb), torch.stack(zmu))  # [B,K,Q,d]
            loss = diffusion_path_loss(model, ig, lq, xq_s, feats, t, insts, zcoefs,
                                       device, n_paths=diff_paths,
                                       use_checkpoint=(dim == 3))
            loss_term = torch.zeros((), device=device)
            if terminal_weight > 0:
                predT = op(model, ig, lq, xq_s.unsqueeze(0), feats, T_HORIZON, device)
                gT = torch.stack([pool[p].bound_fn(xq_s) for p in idx])
                loss_term = nn.functional.mse_loss(predT, gT)
                loss = loss + terminal_weight * loss_term
            loss.backward()
            _apply_update()
            _log_and_eval(model, geom, dim, device, step, steps, loss.item(),
                          {"lr": opt.param_groups[0]["lr"],
                           "train/loss_terminal": loss_term.item()},
                          wb, eval_every, eval_times, t0, tag=" diffusion")
            _end_of_step(do_prof, _iter_t0, step)
            continue

        if pino_target:
            lams = torch.tensor([pool[p].lam for p in idx], device=device
                                ).view(-1, 1, 1)
            R = pino_residual(model, ig, lq, xq_s, feats, t, lams, device)
            loss = R.pow(2).mean()
            if pino_normalise and res_scale:
                loss = loss / res_scale
            loss_term = torch.zeros((), device=device)
            if terminal_weight > 0:
                predT = op(model, ig, lq, xq_s.unsqueeze(0), feats, T_HORIZON, device)
                gT = torch.stack([pool[p].bound_fn(xq_s) for p in idx])
                loss_term = nn.functional.mse_loss(predT, gT)
                loss = loss + terminal_weight * loss_term
            loss.backward()
            _apply_update()
            _log_and_eval(model, geom, dim, device, step, steps, loss.item(),
                          {"lr": opt.param_groups[0]["lr"],
                           "train/loss_terminal": loss_term.item()},
                          wb, eval_every, eval_times, t0, tag=" pino")
            _end_of_step(do_prof, _iter_t0, step)
            continue

        # Computed HERE, not before the branches: DeepRitz, PINO and the
        # diffusion loss never read `pred`, so evaluating it up front spent one
        # full nX-point forward per step on a tensor those arms discard -- pure
        # throughput loss for them under an equal-wall-clock budget.
        pred = op(model, ig, lq, Xq.unsqueeze(0), feats, t, device)   # [B, nX, 1]
        loss = 0.0
        for bi, p in enumerate(idx):
            with torch.no_grad():
                _l0 = _mark(do_prof)
                if fem_w is not None:                # FEM-supervised target
                    target = fem_w[p](1.0 - t / T_HORIZON, Xq)
                else:                                # BSDE weak-supervision target
                    grids_b = [slices[k][bi].reshape((gres,) * dim) for k in range(k_slices)]
                    field_b = SpacetimeField(grids_b, slice_times, gL, dim)
                    # BSDE martingale control variate. Z is a solution
                    # component and comes from the SAME paths as Y, by
                    # cross-fitted regression -- not from finite differences on
                    # the operator's grid. E[M] = 0 exactly, so it removes
                    # variance without biasing the target.
                    # Z from the same paths is used INSIDE the rollout, as the
                    # control variate (cv="bsde"), and nowhere else -- by
                    # decision (2026-09-07): the returned gradient estimate is
                    # not a training label.
                    bnd = pool_bnd[p]
                    tgt, _ = bsde_rollout(
                        pool[p], Xq, t, field_b, n_traj, sub_frac, device,
                        cv="bsde" if control_variate else "off", bound=bnd)
                    # The true Y obeys |Y| <= max(1, ||g||_inf), but the MC
                    # estimate need not: it sums driver increments read off a
                    # possibly-wrong operator. clamp > 0 projects the label back
                    # onto clamp * that bound (1.0 = the bound itself).
                    if clamp > 0:
                        tgt = tgt.clamp(-clamp * bnd, clamp * bnd)
                    n = min(int(cnt[p, j].item()), buf_cap)   # nwos running mean
                    buf[p, j] = (tgt + n * buf[p, j]) / (n + 1)
                    cnt[p, j] += 1
                    target = buf[p, j].clone()
                if do_prof:
                    prof["labels"] += _mark() - _l0
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
        _apply_update()
        _log_and_eval(model, geom, dim, device, step, steps, loss.item(),
                      {"train/n_roll": rollout_n_steps(pool[idx[0]].lam,
                                                       rollout_t_end(t, sub_frac)[0] - t),
                       "lr": opt.param_groups[0]["lr"], "train/loss_terminal": loss_term.item(),
                       "buf/mean_count": float(cnt.float().mean())},
                      wb, eval_every, eval_times, t0, tag=" buffered")
        _end_of_step(do_prof, _iter_t0, step)
    if fem_w is not None:
        print(f"  [d={dim}] TOTAL FEM data-gen: {fem_solves} solves, "
              f"{fem_gen_time:.0f}s ({fem_gen_time/max(fem_solves,1):.2f}s/solve) "
              f"-- the cost BSDE weak supervision pays ZERO of", flush=True)
    if prof["steps"]:
        core = prof["iter"] - prof["labels"] - prof["refresh"]
        per = prof["iter"] / prof["steps"]
        PROFILE.update(prof, opt_steps=step + 1, train_s=(step + 1) * per,
                       labels_s=(step + 1) * prof["labels"] / prof["steps"],
                       core=core, frac_core=core / max(prof["iter"], 1e-9),
                       frac_labels=prof["labels"] / max(prof["iter"], 1e-9),
                       frac_refresh=prof["refresh"] / max(prof["iter"], 1e-9),
                       ms_per_step=1e3 * prof["iter"] / prof["steps"],
                       ms_core_per_step=1e3 * core / prof["steps"])
        print(f"  [d={dim}] step profile over {prof['steps']} sampled steps: "
              f"{1e3 * prof['iter'] / prof['steps']:.1f} ms/step, "
              f"core {100 * core / max(prof['iter'], 1e-9):.0f}%, "
              f"labels {100 * prof['labels'] / max(prof['iter'], 1e-9):.0f}%, "
              f"pool refresh {100 * prof['refresh'] / max(prof['iter'], 1e-9):.0f}%"
              f" (evaluation, off the clock: {prof['eval']:.0f}s)", flush=True)
        if wb is not None:
            try:
                wb.summary.update({f"prof/{k}": v for k, v in PROFILE.items()})
            except Exception:
                pass
    if eval_cb is not None and budget_s is not None:           # final point
        eval_cb(ema_model if ema_model is not None else model,
                time.time() - t_budget0, (ig, lq, field_grid, gres, gL, input_geom, in_ch))
    return (ema_model if ema_model is not None else model), geom


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
            tgt, _ = bsde_rollout(inst, xq, te, field, n_traj, 0, device)
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
        _, stf = fem.solve(inst, T=T_HORIZON, n_slices=11)     # stf(sigma)=u_forward
        for te in eval_times:
            fem_w = stf(1.0 - te / T_HORIZON, xq)               # w(t,.)=u_fwd(T_H-t,.)
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


_EVAL_FEM = {}


def _eval_fem_solver(dim, device):
    """Reference FEM solver for scoring, built once per (dim, device).

    The mesh/assembly/LU factorisation costs ~32 s at the 3-D default resolution
    and does not depend on the instance, so rebuilding it inside every scoring
    call would dominate the measurement."""
    from src.solvers.fem import AllenCahnFEMSolver
    key = (dim, str(device))
    if key not in _EVAL_FEM:
        _EVAL_FEM[key] = AllenCahnFEMSolver(dim, device=device)
    return _EVAL_FEM[key]


@torch.no_grad()
def evaluate_spacetime(model, geom, dim, device, n_instances=8, n_times=16,
                       Q=256, seed=321, n_slices=101, per_time=False,
                       mode="random", n_grid=8):
    """Relative L2 of the operator over the whole space-time slab.

    ``evaluate_vs_fem`` scores a handful of *fixed* times, which reports the
    accuracy of a few slices rather than of the field the operator actually
    represents. Here the evaluation times are drawn at random in ``[0, 1]``, so
    the metric averages over time as well as space and cannot be flattered by a
    lucky choice of slice.

    Reported error is the AVERAGE RELATIVE ERROR over instances: for each
    instance the relative L2 is taken over its own (t, x) samples,
    ``||err_i|| / ||w_i||``, and those per-instance ratios are averaged. Every
    instance counts equally.

    (A single pooled denominator over all instances would instead weight each
    instance by its solution magnitude. That form is only needed to protect
    against instances whose solution norm is near zero, which requires signed
    ``g``; ``SIGNED`` is False here, so all instances have comparable norms and
    the plain average is both safe and the quantity actually wanted. The std
    across instances is returned alongside so the spread is visible.)

    Ground truth is the rescaled forward FEM, ``w(t, .) = u_forward(1 - t, .)``,
    sampled at ``n_slices`` slices and interpolated linearly in time. Slices are
    kept dense (101, vs the 11 used during training logging) so that the
    interpolation error of the *reference* stays well below the operator error
    it is meant to measure.

    Returns ``(pooled_rel_l2, rmse)``, plus a ``{t: (rel, rmse)}`` breakdown
    when ``per_time`` is set -- the breakdown is worth looking at because the
    problem is not uniformly hard in time: at ``t = 1`` the solution is exactly
    the known terminal data ``g``, while ``t = 0`` carries the full accumulated
    reaction.
    """
    ig, lq, field_grid, gres, gL, input_geom, in_ch = geom
    fem = _eval_fem_solver(dim, device)                   # cached across calls
    gen = torch.Generator().manual_seed(seed)

    # Time units. The operator's ada_in is the SAME clock the training grid uses,
    # i.e. t in [0, T_HORIZON] -- NOT a normalised [0, 1]. The FEM field must
    # therefore be solved to T_HORIZON (not 1.0), and its slice index is the
    # normalised forward time, w(t, .) = u_forward(T_H - t) = stf(1 - t/T_H).
    # Getting this wrong queries the operator far outside the times it was ever
    # trained on and compares it against the wrong horizon; it only coincides in
    # the rescaled unit-horizon setting where T_HORIZON == 1.
    T_H = float(T_HORIZON)

    # Evaluation times. Training visits a FIXED grid of nT times
    # (``Tq = linspace(0, T_H - DT, nT)``), so "random" additionally probes the
    # times in between -- which is the honest test of the learned field, but is
    # also strictly harder than the grid. ``mode`` lets both be reported so
    # time-interpolation error can be separated from overall error.
    grid = torch.linspace(0.0, T_H - float(DT), n_grid).tolist()

    def draw_times():
        if mode == "grid":                    # exactly the trained times
            return list(grid)
        if mode == "midgrid":                 # worst case: midway between them
            return [0.5 * (grid[i] + grid[i + 1]) for i in range(len(grid) - 1)]
        return (torch.rand(n_times, generator=gen) * T_H).tolist()

    per_inst, by_t = [], {}
    for _ in range(n_instances):
        errs_all, fems_all = [], []
        inst = sample_instance(dim, gen)
        feats = T.instance_features(inst, input_geom, in_ch).unsqueeze(0)
        _, stf = fem.solve(inst, T=T_H, n_slices=n_slices)
        times = draw_times()
        for te in times:
            xq = ((torch.rand(Q, dim, generator=gen) * 2 - 1)
                  * QUERY_L).to(device)
            fem_w = stf(1.0 - te / T_H, xq)               # w(t,.)=u_fwd(T_H-t,.)
            pred = op(model, ig, lq, xq.unsqueeze(0), feats, te, device)[0]
            e = (pred - fem_w).reshape(-1)
            errs_all.append(e)
            fems_all.append(fem_w.reshape(-1))
            if per_time:
                by_t.setdefault(round(te, 2), []).append((e, fem_w.reshape(-1)))
        ei, fi = torch.cat(errs_all), torch.cat(fems_all)
        per_inst.append((ei.norm() / fi.norm().clamp_min(1e-12)).item())

    t_inst = torch.tensor(per_inst)
    rel = t_inst.mean().item()                     # average relative error
    sd = t_inst.std().item() if len(per_inst) > 1 else 0.0
    if not per_time:
        return rel, sd
    br = {}
    for te, v in sorted(by_t.items()):
        e = torch.cat([a for a, _ in v])
        f = torch.cat([b for _, b in v])
        br[te] = ((e.pow(2).sum() / f.pow(2).sum().clamp_min(1e-12)).sqrt().item(),
                  e.pow(2).mean().sqrt().item())
    return rel, rmse, br


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
