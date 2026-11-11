"""Validation suite for the Allen-Cahn BSDE target (``src/solvers/sde``).

Eight independent checks of the nonlinear Feynman-Kac estimator used as the
weak-supervision target, each isolating one possible source of error:

  T1  d=100 literature benchmark. Independent *radial* finite-difference solve
      of the equivalent forward IVP, cross-checked against the published
      ``u(0,0) ~= 0.0528`` of E, Han & Jentzen (2017, arXiv:1706.04702), then
      used as ``value_fn`` for the BSDE estimate at the same point. Anchors the
      estimator to an external number, in the dimension it is meant for.

  T2  value_fn time-consistency. The nonlinear Feynman-Kac representation needs
      ``w(s, X_s)`` along the path. ``scripts/fem_crosscheck.py`` supplies the
      time-*frozen* terminal field ``w(0, .)`` instead, which is not the same
      function. This runs the true space-time ``w(s, .)`` from FEM slices and
      reports both, so the size of that inconsistency is visible.

  T3  time-discretisation bias. rel L2 vs FEM as ``dt -> 0`` at fixed, large
      trajectory count. The left-Riemann reaction integral is first order, so
      the error must decay ~O(dt) until it reaches the Monte-Carlo floor.

  T4  Monte-Carlo convergence. Estimator spread vs ``n_traj``; separates noise
      from bias by comparing the measured standard error against the residual
      disagreement with FEM.

  T5  sigma handling, FEM-free. Diffusive rescaling ``y = x sqrt(2)/sigma`` maps
      a ``sigma``-instance onto an equivalent ``sqrt(2)``-instance, so the same
      solver must return the same values at corresponding points. Catches a
      mis-scaled diffusion or reaction coefficient.

  T6  rescale_to_unit_horizon. A horizon-T instance solved directly must match
      its unit-horizon rescaling evaluated at ``x / sqrt(T)``.

  T7  frozen-value_fn bias vs effective reaction ``lam * T``. T2 measures the
      penalty at one short horizon; this sweeps the accumulated reaction that
      amplifies it, marking the regime where a frozen value function stops being
      usable at all.

  T8  regression guards for the two solver bugs fixed alongside this suite: the
      elliptic Feynman-Kac source weight (wrong for every sigma != sqrt(2)) and
      the trajectory-shard memory cap.

Run: ``python scripts/validate_bsde_allen_cahn.py [--tests 1,2,3]``
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

# torch-only sde/fem subpackages (skip heavy src/__init__ needing fenics/jax)
_ROOT = Path(__file__).resolve().parents[1]
try:
    from src.solvers.sde import (AllenCahnFamily, AllenCahnND, SDESolver,
                                 rescale_to_unit_horizon,
                                 sample_allen_cahn_family)
except ModuleNotFoundError:
    for _n, _p in [("src", _ROOT / "src"), ("src.solvers", _ROOT / "src/solvers")]:
        _m = types.ModuleType(_n)
        _m.__path__ = [str(_p)]
        sys.modules[_n] = _m
    from src.solvers.sde import (AllenCahnFamily, AllenCahnND, SDESolver,
                                 rescale_to_unit_horizon,
                                 sample_allen_cahn_family)
from src.solvers.fem.fem_solver import AllenCahnFEMSolver

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PASS, FAIL = "PASS", "FAIL"
_results = []


def _report(name, ok, detail):
    _results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}", flush=True)


def rel_l2(a, b):
    """Relative L2 of ``a`` against reference ``b``."""
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


# --------------------------------------------------------------------------- #
# radial reference solver: forward IVP  d_tau u = nu Lap u + lam (u - u^3)
# --------------------------------------------------------------------------- #
class RadialAllenCahn:
    """Radially symmetric Allen-Cahn forward IVP in ``d`` dimensions.

    Solves ``d_tau u = nu [u_rr + (d-1)/r u_r] + lam (u - u^3)`` on ``r in [0, R]``
    with ``u_r(0) = 0`` and a zero-flux outer boundary, by backward-Euler
    diffusion + explicit reaction. Radially symmetric initial data stays
    radially symmetric, so this is an exact reduction of the full d-dimensional
    problem to one dimension -- independent of both FEM and Monte Carlo, and
    usable at ``d = 100`` where a mesh is impossible.

    At ``r = 0`` the singular term has the limit ``Lap u(0) = d u_rr(0)``, which
    with ``u_r(0) = 0`` discretises to ``2 d (u_1 - u_0) / dr^2``.
    """

    def __init__(self, d: int, R: float = 40.0, n: int = 4001,
                 nu: float = 1.0, dtau: float = 2e-4):
        self.d, self.R, self.n, self.nu, self.dtau = d, R, n, nu, dtau
        self.r = np.linspace(0.0, R, n)
        dr = self.r[1] - self.r[0]
        self.dr = dr

        rows, cols, vals = [], [], []
        # r = 0 : Lap u = d * 2 (u_1 - u_0) / dr^2
        rows += [0, 0]; cols += [0, 1]; vals += [-2.0 * d / dr**2, 2.0 * d / dr**2]
        for i in range(1, n - 1):
            c = (d - 1) / (2.0 * self.r[i] * dr)
            rows += [i, i, i]
            cols += [i - 1, i, i + 1]
            vals += [1.0 / dr**2 - c, -2.0 / dr**2, 1.0 / dr**2 + c]
        # zero-flux outer boundary: u_N = u_{N-1}  ->  Lap u ~ 0 there
        rows += [n - 1, n - 1]; cols += [n - 2, n - 1]; vals += [0.0, 0.0]
        lap = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        eye = sp.identity(n, format="csr")
        self.lu = spla.splu((eye - dtau * nu * lap).tocsc())

    def solve(self, g_radial, lam: float, T: float, n_slices: int = 0):
        """Evolve ``u(0, r) = g_radial(r)`` to ``tau = T``.

        Returns the final radial profile, plus (if ``n_slices``) a list of
        ``(tau, profile)`` snapshots at evenly spaced forward times.
        """
        u = np.asarray(g_radial(self.r), dtype=np.float64)
        n_steps = int(round(T / self.dtau))
        save_at = ({int(round(j * n_steps / (n_slices - 1))) for j in range(n_slices)}
                   if n_slices > 1 else set())
        snaps = [(0.0, u.copy())] if 0 in save_at else []
        for step in range(n_steps):
            u = self.lu.solve(u + self.dtau * lam * (u - u**3))
            if (step + 1) in save_at:
                snaps.append(((step + 1) * self.dtau, u.copy()))
        return (u, snaps) if n_slices else u


def radial_value_fn(solver: RadialAllenCahn, snaps, T: float):
    """``value_fn(t, x) -> w(t, x)`` from forward snapshots, for the BSDE solver.

    The terminal-value solution relates to the forward one by
    ``w(t, x) = u_forward(T - t, |x|)``; snapshots are interpolated linearly in
    time and in radius.
    """
    taus = np.array([s[0] for s in snaps])
    profiles = torch.tensor(np.stack([s[1] for s in snaps]), dtype=torch.float32)
    rgrid = torch.tensor(solver.r, dtype=torch.float32)

    def value_fn(t, x):
        tau = float(np.clip(T - t, taus[0], taus[-1]))
        j = int(np.clip(np.searchsorted(taus, tau) - 1, 0, len(taus) - 2))
        a = (tau - taus[j]) / (taus[j + 1] - taus[j])
        prof = ((1 - a) * profiles[j] + a * profiles[j + 1]).to(x.device)
        r = x.norm(dim=-1)
        # linear interpolation in r on the uniform grid
        rg = rgrid.to(x.device)
        pos = (r / solver.dr).clamp(0, len(rg) - 1 - 1e-4)
        i0 = pos.floor().long()
        w = (pos - i0).unsqueeze(-1)
        return ((1 - w) * prof[i0].unsqueeze(-1) + w * prof[i0 + 1].unsqueeze(-1))

    return value_fn


def spacetime_value_fn(st_field, T: float):
    """``value_fn(t, x)`` from a FEM :class:`SpacetimeField` of the forward solve.

    ``st_field`` is indexed by ``tau = forward_time / T`` in ``[0, 1]``, and
    ``w(t, .) = u_forward(T - t, .)``, hence ``tau = 1 - t / T``.
    """
    def value_fn(t, x):
        return st_field(1.0 - t / T, x)
    return value_fn


# --------------------------------------------------------------------------- #
# T1: d = 100 literature benchmark
# --------------------------------------------------------------------------- #
def test_1_benchmark_d100():
    print("\nT1  d=100 Allen-Cahn benchmark (E, Han & Jentzen 2017)")
    d, T, REF = 100, 0.3, 0.0528
    pde = AllenCahnND(n_dim=d)

    radial = RadialAllenCahn(d=d, R=40.0, n=4001, nu=1.0, dtau=2e-4)
    u_final, snaps = radial.solve(
        lambda r: 1.0 / (2.0 + 0.4 * r**2), lam=1.0, T=T, n_slices=61)
    u0 = float(u_final[0])
    _report("T1a radial-FD vs published u(0,0)",
            abs(u0 - REF) < 5e-4,
            f"radial FD = {u0:.5f}, published = {REF:.4f}, "
            f"abs diff = {abs(u0 - REF):.5f}")

    vf = radial_value_fn(radial, snaps, T)
    xs = torch.zeros(1, d, device=DEV)
    for dt in (0.01, 0.005):
        solver = SDESolver(pde=pde, n_traj=200_000, time_step=dt, horizon=T,
                           sigma=2**0.5, mode="parabolic",
                           n_traj_max_per_shard=20_000, device=DEV)
        _, mu, std = solver.evaluate(xs, value_fn=vf, return_std=True)
        v = float(mu.item())
        se = float(std.item()) / 200_000**0.5
        ok = abs(v - u0) < max(4 * se, 5e-4)
        _report(f"T1b BSDE(dt={dt}) vs radial-FD", ok,
                f"BSDE = {v:.5f} +/- {se:.5f} (1 s.e.), radial FD = {u0:.5f}, "
                f"diff = {v - u0:+.5f} = {abs(v - u0) / max(se, 1e-12):.1f} s.e.")


# --------------------------------------------------------------------------- #
# T2: value_fn time-consistency (true space-time vs frozen terminal field)
# --------------------------------------------------------------------------- #
def test_2_value_fn_consistency():
    print("\nT2  value_fn time-consistency (space-time w(s,.) vs frozen w(0,.))")
    T = 0.3
    for dim in (2, 3):
        fem = AllenCahnFEMSolver(dim=dim, L=4.5, dt=2e-3, device=DEV)
        gen = torch.Generator().manual_seed(2024)
        r_st, r_frozen = [], []
        for _ in range(4 if dim == 3 else 6):
            inst = sample_allen_cahn_family(n_dim=dim, n_bumps=(1, 8), generator=gen)
            field, st = fem.solve(inst, T=T, n_slices=41)
            xs = (torch.rand(200, dim, generator=gen) * 2 - 1).to(DEV)
            ref = field(xs).cpu()

            def make(vf):
                s = SDESolver(pde=inst, n_traj=4096, time_step=0.005, horizon=T,
                              sigma=2**0.5, mode="parabolic",
                              n_traj_max_per_shard=4096, device=DEV)
                return s.evaluate(xs, value_fn=vf, evaluate_mode=True).cpu()

            r_st.append(rel_l2(make(spacetime_value_fn(st, T)), ref))
            r_frozen.append(rel_l2(make(lambda t, x, f=field: f(x)), ref))

        m_st, m_fr = float(np.mean(r_st)), float(np.mean(r_frozen))
        _report(f"T2 d={dim} space-time value_fn beats frozen", m_st < m_fr,
                f"rel L2 vs FEM: space-time = {m_st:.4f}, frozen = {m_fr:.4f} "
                f"({m_fr / max(m_st, 1e-12):.1f}x worse)")
        _report(f"T2 d={dim} space-time value_fn accurate", m_st < 0.01,
                f"rel L2 = {m_st:.4f} (target < 0.01)")


# --------------------------------------------------------------------------- #
# T3: time-discretisation bias
# --------------------------------------------------------------------------- #
def test_3_dt_convergence():
    print("\nT3  time-discretisation bias (dt -> 0, fixed large n_traj)")
    T, dim = 0.3, 2
    fem = AllenCahnFEMSolver(dim=dim, L=4.5, dt=1e-3, device=DEV)
    gen = torch.Generator().manual_seed(7)
    inst = sample_allen_cahn_family(n_dim=dim, n_bumps=4, generator=gen)
    field, st = fem.solve(inst, T=T, n_slices=151)
    xs = (torch.rand(64, dim, generator=gen) * 2 - 1).to(DEV)
    ref = field(xs).cpu()
    vf = spacetime_value_fn(st, T)

    dts = [0.06, 0.03, 0.015, 0.0075, 0.00375]
    errs = []
    for dt in dts:
        s = SDESolver(pde=inst, n_traj=131_072, time_step=dt, horizon=T,
                      sigma=2**0.5, mode="parabolic",
                      n_traj_max_per_shard=8192, device=DEV)
        _, mu, std = s.evaluate(xs, value_fn=vf, return_std=True)
        se = (std.cpu() / 131_072**0.5).mean().item()
        e = rel_l2(mu.cpu(), ref)
        errs.append(e)
        print(f"      dt={dt:<8g} rel L2 = {e:.5f}   mean s.e. = {se:.6f}")

    # halving dt must reduce (or hold at the MC/FEM floor) the error
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    monotone = all(errs[i + 1] <= errs[i] * 1.15 for i in range(len(errs) - 1))
    _report("T3 error decreases as dt -> 0", monotone,
            f"rel L2 {errs[0]:.5f} -> {errs[-1]:.5f} over dt {dts[0]} -> {dts[-1]}; "
            f"successive ratios {[round(r, 2) for r in ratios]}")
    _report("T3 first-order rate visible on coarse dt", ratios[0] > 1.4,
            f"coarse-dt halving ratio = {ratios[0]:.2f} (O(dt) would be ~2)")


# --------------------------------------------------------------------------- #
# T4: Monte-Carlo convergence
# --------------------------------------------------------------------------- #
def test_4_mc_convergence():
    print("\nT4  Monte-Carlo convergence (n_traj)")
    T, dim = 0.3, 2
    fem = AllenCahnFEMSolver(dim=dim, L=4.5, dt=1e-3, device=DEV)
    gen = torch.Generator().manual_seed(11)
    inst = sample_allen_cahn_family(n_dim=dim, n_bumps=4, generator=gen)
    field, st = fem.solve(inst, T=T, n_slices=151)
    xs = (torch.rand(64, dim, generator=gen) * 2 - 1).to(DEV)
    ref = field(xs).cpu()
    vf = spacetime_value_fn(st, T)

    ns, ses, errs = [512, 2048, 8192, 32768, 131072], [], []
    for n in ns:
        s = SDESolver(pde=inst, n_traj=n, time_step=0.00375, horizon=T,
                      sigma=2**0.5, mode="parabolic",
                      n_traj_max_per_shard=min(n, 8192), device=DEV)
        _, mu, std = s.evaluate(xs, value_fn=vf, return_std=True)
        se = (std.cpu() / n**0.5).mean().item()
        e = rel_l2(mu.cpu(), ref)
        ses.append(se); errs.append(e)
        print(f"      n_traj={n:<8d} rel L2 = {e:.5f}   mean s.e. = {se:.6f}")

    # standard error must fall like 1/sqrt(n): 256x more paths -> ~16x smaller
    obs = ses[0] / ses[-1]
    exp = (ns[-1] / ns[0]) ** 0.5
    _report("T4 s.e. scales as 1/sqrt(n_traj)", 0.75 * exp < obs < 1.33 * exp,
            f"s.e. shrank {obs:.1f}x for {ns[-1] // ns[0]}x paths "
            f"(1/sqrt(n) predicts {exp:.1f}x)")
    _report("T4 residual error is bias-dominated at large n", errs[-1] > 0.5 * errs[-2],
            f"rel L2 floors at {errs[-1]:.5f} while s.e. is {ses[-1]:.6f} "
            f"-> remaining error is discretisation/FEM, not MC noise")


# --------------------------------------------------------------------------- #
# T5: sigma handling via diffusive rescaling (no FEM involved)
# --------------------------------------------------------------------------- #
def test_5_sigma_scaling():
    print("\nT5  sigma handling: diffusive rescaling invariance (FEM-free)")
    T, dim = 0.3, 2
    gen = torch.Generator().manual_seed(5)
    base = sample_allen_cahn_family(n_dim=dim, n_bumps=3, generator=gen)

    for sigma in (1.0, 2.0):
        # instance A: general sigma.  y = x * sqrt(2)/sigma maps it onto an
        # equivalent sqrt(2) instance with centers -> c*sqrt(2)/sigma,
        # widths -> w*sigma^2/2, since g(x) = g_tilde(y) requires |x - c|^2 =
        # (sigma^2/2)|y - c sqrt(2)/sigma|^2.
        k = 2**0.5 / sigma
        A = AllenCahnFamily(n_dim=dim, centers=base.centers, amps=base.amps,
                            offsets=base.offsets, widths=base.widths,
                            lam=base.lam, sigma=sigma)
        B = AllenCahnFamily(n_dim=dim, centers=base.centers * k, amps=base.amps,
                            offsets=base.offsets, widths=base.widths / k**2,
                            lam=base.lam, sigma=2**0.5)

        xs = (torch.rand(128, dim, generator=gen) * 2 - 1).to(DEV)
        ys = xs * k
        # the *same* driver value must be seen along corresponding paths; use a
        # matched analytic value_fn so only the diffusion scaling is tested
        vA = lambda t, x, A=A: A.bound_fn(x)
        vB = lambda t, y, B=B: B.bound_fn(y)

        kw = dict(n_traj=65_536, time_step=0.00375, horizon=T, mode="parabolic",
                  n_traj_max_per_shard=8192, device=DEV)
        sA = SDESolver(pde=A, sigma=sigma, **kw)
        sB = SDESolver(pde=B, sigma=2**0.5, **kw)
        _, muA, stdA = sA.evaluate(xs, value_fn=vA, return_std=True)
        _, muB, stdB = sB.evaluate(ys, value_fn=vB, return_std=True)
        se = ((stdA.cpu()**2 + stdB.cpu()**2) / 65_536).sqrt().mean().item()
        e = rel_l2(muA.cpu(), muB.cpu())
        _report(f"T5 sigma={sigma} matches rescaled sqrt(2) instance", e < 0.01,
                f"rel L2 = {e:.5f} (paired s.e. ~ {se:.5f}); "
                f"ranges A {muA.min():.3f}..{muA.max():.3f}, "
                f"B {muB.min():.3f}..{muB.max():.3f}")


# --------------------------------------------------------------------------- #
# T6: rescale_to_unit_horizon
# --------------------------------------------------------------------------- #
def test_6_unit_horizon_rescaling():
    print("\nT6  rescale_to_unit_horizon equivalence")
    dim, T = 2, 2.5
    gen = torch.Generator().manual_seed(3)
    inst = sample_allen_cahn_family(n_dim=dim, n_bumps=3, generator=gen)
    resc = rescale_to_unit_horizon(inst, T)

    xs = (torch.rand(128, dim, generator=gen) * 2 - 1).to(DEV)
    xis = xs / T**0.5

    # value_fn must also be expressed in each problem's own coordinates; the
    # terminal data is the natural matched choice (w(t,.) = g in the frozen
    # limit is not exact, but it is *identical* under the map, so any
    # disagreement is the rescaling, not the value function)
    kwA = dict(n_traj=65_536, time_step=T / 400, horizon=T, sigma=2**0.5,
               mode="parabolic", n_traj_max_per_shard=8192, device=DEV)
    kwB = dict(n_traj=65_536, time_step=1 / 400, horizon=1.0, sigma=2**0.5,
               mode="parabolic", n_traj_max_per_shard=8192, device=DEV)
    a = SDESolver(pde=inst, **kwA).evaluate(
        xs, value_fn=lambda t, x: inst.bound_fn(x), evaluate_mode=True).cpu()
    b = SDESolver(pde=resc, **kwB).evaluate(
        xis, value_fn=lambda t, x: resc.bound_fn(x), evaluate_mode=True).cpu()
    e = rel_l2(b, a)
    _report("T6 unit-horizon rescaling matches direct horizon-T solve", e < 0.01,
            f"rel L2 = {e:.5f}, ranges direct {a.min():.3f}..{a.max():.3f}, "
            f"rescaled {b.min():.3f}..{b.max():.3f}")


# --------------------------------------------------------------------------- #
# T7: how the frozen-value_fn bias scales with the effective reaction lam*T
# --------------------------------------------------------------------------- #
def test_7_frozen_bias_vs_horizon():
    """Severity of the time-frozen ``value_fn`` as a function of ``lam * T``.

    T2 measures the frozen-``value_fn`` penalty at the short horizon ``T = 0.3``,
    where ``w`` barely moves over the window so the penalty is mild. The
    accumulated reaction is what amplifies it, so this sweeps the effective
    strength ``lam_eff = lam * T`` using :func:`rescale_to_unit_horizon` -- which
    maps every horizon onto ``[0, 1]``, keeping one fixed FEM box valid for all
    of them (the diffusive reach in rescaled coordinates is horizon-independent).

    A frozen ``value_fn`` is only defensible if its error stays small at the
    ``lam_eff`` the training runs actually use.
    """
    print("\nT7  frozen-value_fn bias vs effective reaction lam*T")
    dim = 2
    fem = AllenCahnFEMSolver(dim=dim, L=4.5, dt=1e-3, device=DEV)
    gen = torch.Generator().manual_seed(19)
    base = sample_allen_cahn_family(n_dim=dim, n_bumps=3, generator=gen)
    xs = (torch.rand(64, dim, generator=gen) * 2 - 1).to(DEV)

    print(f"      base lam={base.lam:.2f}")
    print("      T      lam_eff   rel L2 space-time   rel L2 frozen   penalty")
    rows = []
    for T in (0.3, 1.0, 2.5, 6.0):
        inst = rescale_to_unit_horizon(base, T)
        field, st = fem.solve(inst, T=1.0, n_slices=101)
        ref = field(xs).cpu()

        def run(vf):
            s = SDESolver(pde=inst, n_traj=16_384, time_step=0.0025, horizon=1.0,
                          sigma=2**0.5, mode="parabolic",
                          n_traj_max_per_shard=8192, device=DEV)
            return s.evaluate(xs, value_fn=vf, evaluate_mode=True).cpu()

        e_st = rel_l2(run(spacetime_value_fn(st, 1.0)), ref)
        e_fr = rel_l2(run(lambda t, x, f=field: f(x)), ref)
        rows.append((T, base.lam * T, e_st, e_fr))
        print(f"      {T:<6g} {base.lam * T:<9.2f} {e_st:<19.4f} {e_fr:<15.4f} "
              f"{e_fr / max(e_st, 1e-12):.1f}x")

    _report("T7 space-time value_fn stays accurate at all lam*T",
            all(r[2] < 0.02 for r in rows),
            "space-time rel L2 = "
            + ", ".join(f"{r[2]:.4f}@lam_eff={r[1]:.1f}" for r in rows))
    grew = rows[-1][3] / max(rows[0][3], 1e-12)
    _report("T7 frozen value_fn degrades as lam*T grows (expected)", grew > 1.0,
            f"frozen rel L2 {rows[0][3]:.4f} (lam_eff={rows[0][1]:.1f}) -> "
            f"{rows[-1][3]:.4f} (lam_eff={rows[-1][1]:.1f}), {grew:.1f}x worse")


# --------------------------------------------------------------------------- #
# T8: regression guards for the two fixed solver bugs
# --------------------------------------------------------------------------- #
def test_8_regressions():
    """Guards against the bugs fixed alongside this suite.

    (a) The elliptic source weight is the Feynman-Kac factor ``sigma^2 / 2``. It
        was once ``2 / sigma^2``, which is numerically identical at the default
        ``sigma = sqrt(2)`` and wrong everywhere else -- so only a test at a
        non-default ``sigma`` can catch it. Holding ``sigma^2 dt`` fixed keeps the
        spatial step (and hence the absorption bias) identical across sigmas, so
        the three runs are directly comparable.

    (b) ``n_traj_max_per_shard`` is a real memory cap; a floor-division shard
        count let the final shard absorb the whole remainder and exceed it.
    """
    print("\nT8  regression guards (elliptic sigma weight, shard cap)")
    from src.solvers.sde import BoxPoissonPDE

    torch.manual_seed(0)
    xs = torch.rand(32, 2) * 0.6 + 0.2          # interior of [0,1]^2
    exact = (xs**2).sum(-1, keepdim=True)
    errs = {}
    for sig in (2**0.5, 1.0, 2.0):
        s = SDESolver(pde=BoxPoissonPDE(n_dim=2, stop_tol=1e-3), n_traj=4096,
                      max_step=200_000, time_step=1e-4 * (2.0 / sig**2),
                      sigma=sig, mode="elliptic", n_traj_max_per_shard=2048,
                      device=DEV)
        mu = s.evaluate(xs.to(DEV), evaluate_mode=True).cpu()
        errs[sig] = rel_l2(mu, exact)
    spread = max(errs.values()) / min(errs.values())
    _report("T8a elliptic solve is sigma-invariant", spread < 1.5,
            "rel L2 vs exact u=|x|^2: "
            + ", ".join(f"sigma={k:.3f} -> {v:.4f}" for k, v in errs.items())
            + f"  (spread {spread:.2f}x, must be ~1)")
    _report("T8b all sigmas accurate", max(errs.values()) < 0.05,
            f"worst rel L2 = {max(errs.values()):.4f}")

    bad = []
    for n_traj, cap in [(1000, 300), (500, 300), (128, 128), (7, 3)]:
        s = SDESolver(pde=BoxPoissonPDE(n_dim=2), n_traj=n_traj,
                      n_traj_max_per_shard=cap, device="cpu")
        sizes = s._shard_sizes()
        if sum(sizes) != n_traj or max(sizes) > cap:
            bad.append((n_traj, cap, sizes))
    _report("T8c shard sizes respect the cap and sum to n_traj", not bad,
            "all (n_traj, cap) combinations exact" if not bad else f"broken: {bad}")


TESTS = {
    1: test_1_benchmark_d100,
    2: test_2_value_fn_consistency,
    3: test_3_dt_convergence,
    4: test_4_mc_convergence,
    5: test_5_sigma_scaling,
    6: test_6_unit_horizon_rescaling,
    7: test_7_frozen_bias_vs_horizon,
    8: test_8_regressions,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="1,2,3,4,5,6,7,8")
    args = ap.parse_args()
    print(f"Allen-Cahn BSDE validation | device={DEV}")
    for k in [int(s) for s in args.tests.split(",")]:
        TESTS[k]()

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{'=' * 72}\n{len(_results) - n_fail}/{len(_results)} checks passed")
    for name, ok, detail in _results:
        if not ok:
            print(f"  FAIL {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
