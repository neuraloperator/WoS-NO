"""Do the methods separate once the problem gets harder?

Everything so far ran at T=0.3, i.e. accumulated reaction lam_eff = lam*T in
[0.15, 0.60] -- the mild regime, where the nonlinearity barely bites and the
solution stays close to its terminal datum. The consequence is a low "do nothing"
floor (0.0587 in 2-D: just echo the input f), which compresses every method
toward it and makes the benchmark less discriminating than its numbers suggest.

This sweeps the difficulty by raising T through the diffusive rescaling
(configure_horizon), which maps a horizon-T problem to the unit horizon with
reaction lam*T -- verified correct to 0.001 rel L2 against a direct horizon-2.5
solve (validation suite T6). Each method runs at ITS OWN best hyper-parameters
from the tuning sweeps, so no arm is handicapped by another's settings.

TWO trivial baselines are reported at every level, because at high lam_eff the
family can degenerate in two opposite ways and each has its own null model:

  u = f    echo the input. The floor at short horizon, where u has no time to
           move away from f.
  u = 1    the Allen-Cahn attractor. With positive-only g, a large accumulated
           reaction pulls every instance toward the single u=+1 well, which
           collapses the solution map to a constant -- the problem becomes EASY,
           not hard. If this baseline gets good, that level is not a valid
           difficulty increase and its scores mean nothing.

A method only demonstrates it learned the dynamics by beating BOTH.

Run: ``python scripts/difficulty_sweep.py --dim 2 --budget 300 --seeds 3``
"""

from __future__ import annotations

import argparse
import glob
import re
import json
import statistics
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_allen_cahn_spacetime as S  # noqa: E402
import wblog  # noqa: E402
import rescore_all as R  # noqa: E402
from fair_budget_comparison import save_ckpt  # noqa: E402

# each arm at its own tuned optimum (see tune_datafree.py sweeps, and the
# 64-instance rescore in scripts/rescore_d{2,3}.json which settled two of them:
# in 3-D plain pool=256 beats the control-variate version 0.0365 vs 0.0435, and
# the best 2-D FEM allocation is dataset=256, not 1024).
def tuned(arm, dim, default):
    """Best (lr, terminal_weight[, dl_frac]) from the matching tune JSON.

    The data-free arms were re-tuned after their spatial derivatives were fixed
    (autograd returned ~0 on this operator, so PINO minimised a reaction ODE and
    DeepRitz an energy with no Dirichlet term); their earlier optima describe
    the broken objectives. PINO's tunes were invalidated a SECOND time when its
    time derivative was fixed: ``autograd.grad(w.sum(), t)`` had returned the
    sum of d_t w over all points as one scalar, 231x too large and ~100% of the
    residual energy, so the grid was optimising a global constant and the
    optimum it found (terminal_weight 3000-10000, needed to outweigh that
    constant) means nothing for the corrected residual. Those files are parked
    as scripts/stale_tune_pino_*_brokendt.json, outside this glob. Reading the tune output here keeps a single source
    of truth and removes the hand-copy step that let the 3-D PINO config go
    stale once before.
    """
    import glob, json, math, statistics
    # base grid plus any boundary extensions (tune_*_fd_ext.json): the optimum
    # landed on the tw edge for every arm at least once, so the retrain chain
    # extends the grid when that happens and the results merge here
    files = sorted(glob.glob(f"scripts/tune_{arm}_{dim}d_fd*.json"))
    if not files:
        return default
    d = {}
    for f in files:
        d.update(json.loads(Path(f).read_text()))
    # Rank by mean + sd, not mean alone. With 2 seeds, min-of-means picks the
    # cell whose one lucky seed landed lowest: tonight PINO 3-D's nominal best
    # (0.1041 +/- 0.035) "beat" 0.1068 +/- 0.002, and DeepRitz 2-D's (0.0858 +/-
    # 0.034) "beat" 0.0956 +/- 0.006. Those cells underperform their neighbours
    # once more seeds are run, so the retrain would inherit a worse arm than the
    # grid actually found. mean + sd prefers the reliable cell when means tie
    # and only follows a large gap.
    best, bm = None, float("inf")
    for k, v in d.items():
        g = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
        if not g:
            continue
        score = statistics.mean(g) + (statistics.stdev(g) if len(g) > 1 else 0.0)
        if score < bm:
            best, bm = k, score
    if best is None:
        return default
    kv = dict(re.findall(r"(\w+)=([-\d.e+]+)", best))
    out = dict(default)
    out["lr"] = float(kv["lr"]); out["terminal_weight"] = float(kv["tw"])
    if "dl" in kv:
        out["dl_frac"] = float(kv["dl"])
    print(f"  [tuned] {arm} d={dim}: {best}  (mean+sd {bm:.4f})", flush=True)
    return out


def arms(dim, extra=()):
    dr = tuned("dr", dim, dict(lr=1e-3, terminal_weight=10.0 if dim == 2 else 100.0))
    # PINO defaults after the d_t fix: the residual is normalised by its RMS at
    # init (pino_normalise), which is now ~0.01 instead of 35-95, so both terms
    # of the objective are O(1) and terminal_weight ~ 1 is the right starting
    # point rather than the 1000+ the broken residual forced.
    pn = tuned("pino", dim, dict(lr=1e-3, terminal_weight=1.0))
    # Pathwise Nuesken-Richter arm (reformulated 2026-09-07 17:15): Monte Carlo
    # over paths and all Euler-Maruyama steps, u_theta exact at path points,
    # Z from the BSDE regression. Defaults from the rejected single-increment
    # version's tune until its own tune file exists.
    df = tuned("diffusion_path", dim, dict(lr=3e-4, terminal_weight=1000.0))
    a = {
        # BSDE with Y and Z both from the simulation (cv="bsde"). The earlier
        # grid-gradient control variate at coefficient 1 TRIPLED the 2-D target
        # variance; the cross-fitted simulation-native Z cuts it to 0.38 (2-D)
        # and 0.72 (3-D) at n_traj=4096, so it is on in both dimensions.
        "BSDE pool256": dict(pool_size=256, refresh_per_step=2, n_traj=4096,
                             control_variate=True, terminal_weight=1.0, lr=1e-3),
        # FEM-trained: ground truth as SUPERVISION -- the reference this whole
        # comparison exists to beat. refresh_per_step=0, NOT 2: pool turnover is
        # free for BSDE but for FEM every refreshed slot is a live solve inside
        # the budget (at refresh=2 the 2-D arm spent 244s of 300s on 4948 solves).
        "FEM  n=%d pool=256" % (141 if dim == 2 else 25):
            dict(fem_target=True, pool_size=256, refresh_per_step=0,
                 fem_n=141 if dim == 2 else 25, terminal_weight=1.0, lr=1e-3),
        "DeepRitz jko":    dict(pool_size=64, refresh_per_step=2, deepritz=True,
                                terminal_weight=dr["terminal_weight"], lr=dr["lr"]),
        "PINO residual":   dict(pool_size=64, refresh_per_step=2, pino_target=True,
                                terminal_weight=pn["terminal_weight"], lr=pn["lr"]),
        # Nuesken-Richter diffusion loss along paths: Z (from the BSDE regression)
        # in the OBJECTIVE, residual averaged over paths and E-M steps.
        ("Diffusion NR (paths)" if glob.glob(f"scripts/tune_diffusion_path_{dim}d_fd*.json")
         else "Diffusion NR (paths, untuned)"):
            dict(pool_size=64, refresh_per_step=2, diffusion=True, n_traj=4096,
                 diff_paths=8, terminal_weight=df["terminal_weight"], lr=df["lr"]),
    }
    if "nocv" in extra:
        # the A/B for the control variate at equal wall-clock: the GPU probe
        # measured 4.5x less target variance but only 45% (2-D) / 66% (3-D) of
        # the optimiser steps, and the one prior data point (tripling target
        # variance changed nothing in 2-D) says steps may matter more than noise
        a["BSDE pool256 no-CV"] = dict(pool_size=256, refresh_per_step=2, n_traj=4096,
                                       control_variate=False, terminal_weight=1.0, lr=1e-3)
    if "base_sched" in extra:
        # The wall-clock cosine schedule applied to the BASELINES. It was worth
        # 1.2-1.6x to BSDE, and until the optimiser tail was shared no data-free
        # arm could receive it at all (train_buffered accepted lr_sched and the
        # DeepRitz/PINO branches returned before reading it), so "BSDE sched" vs
        # "DeepRitz"/"PINO" was not an equal-optimiser comparison.
        a["DeepRitz jko sched"] = dict(a["DeepRitz jko"], lr_sched="cosine_budget")
        a["PINO residual sched"] = dict(a["PINO residual"], lr_sched="cosine_budget")
    if "colloc" in extra:
        # Is the FIXED collocation set handicapping the data-free arms? Xq is
        # drawn once and reused for the whole run because the BSDE branch's
        # running-mean buffer is indexed by (instance, time, point) cell and
        # needs stable point identity. DeepRitz and PINO have NO buffer -- their
        # objectives are a residual and an energy, normally Monte-Carlo
        # integrated over fresh points -- so for them the fixed set is pure
        # overfitting risk, and both were measured turning upward in the second
        # half of the budget (DeepRitz 0.055 -> 0.072 at 2-D T=0.3).
        #
        # Each method is tested at ITS OWN chosen configuration, and its
        # baseline is RE-RUN alongside: a separate fix removed a wasted
        # nX-point forward per step from these three branches, so the frozen
        # numbers were measured at lower throughput and are not a valid control.
        a["DeepRitz jko freshxq"] = dict(a["DeepRitz jko"], resample_xq=True)
        a["PINO residual sched freshxq"] = dict(
            a["PINO residual"], lr_sched="cosine_budget", resample_xq=True)
        # pool_size is the OTHER asymmetry: 64 here vs 256 for BSDE/FEM, never
        # tuned for these arms and (no buffer) expected to be a no-op. Measured
        # rather than argued.
        a["DeepRitz jko pool256"] = dict(a["DeepRitz jko"], pool_size=256)
        a["PINO residual sched pool256"] = dict(
            a["PINO residual"], lr_sched="cosine_budget", pool_size=256)
    if "k5" in extra:
        # Is the driver's TIME interpolation the limiter at stiff reaction? The
        # BSDE label reads u_theta along the path off k_slices grid slices spanning
        # [t, T], linear in time. 3 slices is coarse when u moves fast (T=3,
        # lam_eff up to 6), which is the one level where BSDE loses to FEM
        # (2-D 0.153 vs 0.108). 5 slices cost two extra grid forwards per step.
        a["BSDE pool256 k5"] = dict(pool_size=256, refresh_per_step=2, n_traj=4096,
                                    control_variate=True, terminal_weight=1.0, lr=1e-3,
                                    k_slices=5)
    # ---- optimising the BSDE(CV) arm, one lever at a time vs "BSDE pool256" ----
    base = dict(pool_size=256, refresh_per_step=2, n_traj=4096, control_variate=True,
                terminal_weight=1.0, lr=1e-3)
    if "opt_sched" in extra:
        # lr decays with the wall-clock fraction of the budget (was flat 1e-3)
        a["BSDE pool256 sched"] = dict(base, lr_sched="cosine_budget")
    if "opt_ema" in extra:
        # evaluate an EMA of the weights (decay 0.995 ~ 200-step window); the
        # rollout's value field still comes from the live model
        a["BSDE pool256 ema"] = dict(base, ema_decay=0.995)
    if "opt_all" in extra:
        # schedule + EMA + 5 driver slices (k5 helped in 3-D at stiff reaction)
        a["BSDE pool256 sched+ema+k5"] = dict(base, lr_sched="cosine_budget",
                                              ema_decay=0.995, k_slices=5)
    if "labelclamp" in extra:
        # Divergence repair. Two failure modes were measured at flat lr: a sudden
        # late blow-up (2-D T=0.3, 0.025 -> 1.50 in the last 25 s) and a collapse
        # into a constant well (2-D T=3, 0.193 -> 1.032 and stuck, the bistable
        # wrong phase). Both are the label feeding on the operator's own field.
        #
        # Named "labelclamp", not "clamp": the driver's value FIELD was already
        # clamped to the invariant region on every read inside bsde_rollout, so
        # the previous arm's slice clamp at 1.1 was exactly a no-op and only its
        # label clamp did anything. clamp=1.0 projects the label onto the
        # instance's own max(1, ||g||_inf), which the true solution obeys and a
        # Monte-Carlo sum over a wrong operator's driver does not.
        a["BSDE labelclamp"] = dict(base, clamp=1.0)
        a["BSDE sched labelclamp"] = dict(base, lr_sched="cosine_budget", clamp=1.0)
    if "tgtnet" in extra:
        # The end-of-run RISE (3-D T=0.3: every seed ends 1.2-2.0x above the
        # points before it) and the blow-ups share one cause: the label reads the
        # driver off the live operator. Compare against "opt_ema", which has the
        # same EMA weights but still labels from the live model, so the pair
        # isolates the target network from the smoothing.
        a["BSDE tgtnet"] = dict(base, ema_decay=0.995, label_ema=True)
        a["BSDE sched tgtnet"] = dict(base, lr_sched="cosine_budget",
                                      ema_decay=0.995, label_ema=True)
    if "opt2" in extra:
        # Round 2, all on top of the cosine schedule, all aimed at STEPS rather
        # than per-step noise: the 3-D curves were still descending at the end of
        # the budget, so throughput is the binding constraint there.
        sc = dict(base, lr_sched="cosine_budget")
        # the driver's value field is read off a gres^dim grid every step, which
        # the profile shows is the dominant cost in 3-D
        a["BSDE sched gres10"] = dict(sc, grid_res=10)
        # the control variate cuts target variance ~4x, so half the paths should
        # still be quieter than the frozen no-CV arm, at more steps
        a["BSDE sched traj2048"] = dict(sc, n_traj=2048)
        # pool turnover zeroes 2 x nT = 16 target-buffer cells per step while one
        # step fills only B = 8, so the running mean never accumulates (measured
        # mean visit count 0.5); at refresh 1 it can
        a["BSDE sched refresh1"] = dict(sc, refresh_per_step=1)
    if "lr3e-4" in extra:
        # Optimiser stability: at lr=1e-3 the FEM reference diverged in 1 of 3
        # 3-D seeds on deterministic targets and no-CV BSDE in 2 of 6 2-D runs.
        # Neither BSDE nor FEM ever had its lr tuned. Same three arms at 3e-4.
        fem_n = 141 if dim == 2 else 25
        a["BSDE pool256 lr3e-4"] = dict(pool_size=256, refresh_per_step=2, n_traj=4096,
                                        control_variate=True, terminal_weight=1.0, lr=3e-4)
        a["BSDE pool256 no-CV lr3e-4"] = dict(pool_size=256, refresh_per_step=2, n_traj=4096,
                                              control_variate=False, terminal_weight=1.0, lr=3e-4)
        a[f"FEM  n={fem_n} pool=256 lr3e-4"] = dict(fem_target=True, pool_size=256,
                                                    refresh_per_step=0, fem_n=fem_n,
                                                    terminal_weight=1.0, lr=3e-4)
    return a


@torch.no_grad()
def baselines(refs):
    """Relative error of the two trivial predictors on the SHARED references.

    Scored on exactly the instances, times and points every arm is scored on.
    This used to draw its own 8 instances and run its own FEM solves while the
    arms were scored on 64, so "beats the floor" compared two different samples
    of a metric whose per-instance spread is large (||w|| varies 3.6x in 3-D).
    Recomputed on all 64 the floor moves 0.89-1.24x, and at 2-D T=1 that is the
    difference between PINO sitting clearly above the null model (0.3098 vs a
    0.2442 floor) and level with it (vs 0.3027). Reading the two predictors off
    the cached references also drops the duplicate FEM solves entirely.
    """
    per_f, per_1 = [], []
    for inst, _times, xqs, rfs in refs:
        Ef, E1, Fv = [], [], []
        for xq, ref in zip(xqs, rfs):
            Ef.append((inst.bound_fn(xq) - ref).reshape(-1))
            E1.append((torch.ones_like(ref) - ref).reshape(-1))
            Fv.append(ref.reshape(-1))
        f = torch.cat(Fv).norm()
        per_f.append((torch.cat(Ef).norm() / f).item())
        per_1.append((torch.cat(E1).norm() / f).item())
    return (statistics.mean(per_f), statistics.mean(per_1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizons", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--grid-res", type=int, default=None)
    ap.add_argument("--n-times", type=int, default=16)
    ap.add_argument("--n-inst", type=int, default=64,
                    help="instances the reported mean is averaged over. 8 (the "
                         "old default) is far too few: the metric is a mean of "
                         "per-instance ||e||/||w|| and ||w|| varies 3.6x across "
                         "instances in 3-D, so one low-norm draw moved the "
                         "reported number 2.4x (see scripts/scorer_diff.py).")
    ap.add_argument("--ckpt-dir", default="checkpoints_difficulty",
                    help="save trained operators so a metric change can be "
                         "applied by RE-SCORING instead of retraining -- this "
                         "sweep previously saved nothing, so its T=1 and T=3 "
                         "numbers could not be corrected at all.")
    ap.add_argument("--extra-arms", nargs="*", default=[],
                    choices=["nocv", "k5", "lr3e-4", "opt_sched", "opt_ema",
                             "opt_all", "opt2", "labelclamp", "tgtnet",
                             "base_sched", "colloc"],
                    help="optional arms not in the default comparison")
    ap.add_argument("--only", default=None,
                    help="regex: run only arms whose name matches (e.g. a follow-up "
                         "arm alone, sharing the same references/baselines)")
    ap.add_argument("--out", default="scripts/difficulty_sweep.json")
    wblog.add_args(ap)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"difficulty sweep | d={args.dim} budget={args.budget}s "
          f"seeds={args.seeds} device={dev}")

    res = {}
    for T in args.horizons:
        S.configure_horizon(T)
        # the rescaled long-horizon problem is stiff; bound the output to the
        # invariant region, as the trainer's own main() does for T != 0.3
        S.BOUND_OUTPUT = abs(T - 0.3) >= 1e-9
        lam_lo, lam_hi = 0.5 * T, 2.0 * T
        # one reference set per horizon, shared by every arm and seed: the FEM
        # solves dominate evaluation cost, and sharing them also makes the
        # arm-to-arm comparison paired rather than confounded with the draw
        # references are deterministic (seed 321) and the FEM solves dominate
        # evaluation cost (64 x ~13 s per horizon in 3-D), so cache them: an arm
        # run later with --only scores on EXACTLY the same instances and points
        rpath = Path(args.ckpt_dir) / f"refs_d{args.dim}_T{T:g}_n{args.n_inst}.pt"
        refs = None
        if rpath.exists():
            try:
                refs = torch.load(rpath, map_location=dev, weights_only=False)
                print(f"  loaded {len(refs)} shared references from {rpath}", flush=True)
            except Exception as e:
                print(f"  reference cache unreadable ({e}); rebuilding", flush=True)
        if refs is None:
            print(f"  building {args.n_inst} shared references...", flush=True)
            refs = R.build_refs(args.dim, dev, args.n_inst, args.n_times, 256)
            try:
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(refs, rpath)
            except Exception as e:
                print(f"  (could not cache references: {e})", flush=True)
        # floors AFTER the references exist: they are scored on the same 64
        b_f, b_1 = baselines(refs)
        print(f"\n=== T={T:g}  lam_eff in [{lam_lo:.2f}, {lam_hi:.2f}]  "
              f"(T_HORIZON={S.T_HORIZON:g}, QUERY_L={S.QUERY_L:g}) ===")
        print(f"  baseline u=f : {b_f:.4f}     baseline u=1 : {b_1:.4f}")
        lvl = {"lam_eff": [lam_lo, lam_hi], "base_f": b_f, "base_1": b_1}
        for tag, kw in arms(args.dim, args.extra_arms).items():
            if args.only and not re.search(args.only, tag):
                continue
            vals = []
            for seed in range(args.seeds):
                wb = wblog.init(
                    args, f"d{args.dim}-T{T:g}-{tag}-seed{seed}", job_type=tag,
                    tags=[f"d{args.dim}", f"T{T:g}", tag],
                    config=dict(dim=args.dim, horizon=T, arm=tag, seed=seed,
                                budget_s=args.budget, grid_res=args.grid_res,
                                n_inst=args.n_inst,
                                lam_eff_lo=lam_lo, lam_eff_hi=lam_hi,
                                base_f=b_f, base_1=b_1, **kw))
                kw2 = dict(kw)
                fem_target = kw2.pop("fem_target", False)
                gres_arm = kw2.pop("grid_res", None)   # an arm may set its own
                if "dl_dt" in kw2:                 # stored as a horizon fraction
                    kw2["dl_dt"] = kw2["dl_dt"] * float(S.T_HORIZON)
                model, geom = S.train_buffered(
                    args.dim, steps=10**6, device=dev, seed=seed,
                    fem_target=fem_target, budget_s=args.budget,
                    grid_res=None if fem_target else (gres_arm or args.grid_res),
                    eval_every=0, wb=wb, **kw2)
                per = R.score(model, geom, args.dim, dev, refs)
                rel = statistics.mean(per)
                med = statistics.median(per)
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
                save_ckpt(model, Path(args.ckpt_dir) /
                          f"d{args.dim}_T{T:g}_{safe}_seed{seed}.pt",
                          dict(dim=args.dim, seed=seed, t_phys=T, arm=tag,
                               grid_res=None if fem_target else (gres_arm or args.grid_res),
                               rel=rel, median=med, budget_s=args.budget))
                # the floor is per-level, so ratio-to-floor is the only number
                # comparable ACROSS levels -- log both
                wblog.finish(wb, dict(rel=rel, median=med, base_f=b_f,
                                      base_1=b_1, floor=min(b_f, b_1),
                                      floor_over_rel=min(b_f, b_1) / rel
                                      if rel else None))
                vals.append(rel)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            m = statistics.mean(vals)
            floor = min(b_f, b_1)
            print(f"  {tag:<18} {m:.4f} +/- {sd:.4f}   "
                  f"{'BEATS' if m < floor else 'above'} floor "
                  f"({floor / m:.2f}x)   " + " ".join(f"{v:.3f}" for v in vals),
                  flush=True)
            lvl[tag] = vals
            res[f"T={T:g}"] = lvl
            Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")

    rows = []
    for lvl_key, lvl in res.items():
        fl = min(lvl["base_f"], lvl["base_1"])
        for tag in arms(args.dim, args.extra_arms):
            if tag not in lvl:
                continue
            v = lvl[tag]
            rows.append([lvl_key, lvl["lam_eff"][0], lvl["lam_eff"][1], tag,
                         statistics.mean(v),
                         statistics.stdev(v) if len(v) > 1 else 0.0,
                         lvl["base_f"], lvl["base_1"], fl,
                         fl / statistics.mean(v) if statistics.mean(v) else None])
    wblog.table(args, f"SUMMARY-difficulty-d{args.dim}",
                ["level", "lam_eff_lo", "lam_eff_hi", "arm", "rel", "sd",
                 "base_f", "base_1", "floor", "floor_over_rel"], rows,
                config=dict(dim=args.dim, budget_s=args.budget,
                            seeds=args.seeds, horizons=args.horizons))


if __name__ == "__main__":
    main()
