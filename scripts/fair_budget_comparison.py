"""Fixed-budget, best-pipeline-each comparison: BSDE vs FEM supervision.

Metric: AVERAGE RELATIVE ERROR. For each instance the relative L2 is taken over
its own (time, point) samples at random times in [0, T] and random spatial
queries; those per-instance ratios are averaged, so every instance counts
equally. Reported as mean +/- standard deviation across seeds. Nothing else.

Ground truth is a fine FEM solve, and only that (2-D n=201, 3-D n=41, dt=2e-3).
Its own absolute error, measured against an exact radial reduction, is 1.6e-4 in
2-D and 2.8e-3 in 3-D -- i.e. ~125x and ~8x below the operator errors it is used
to judge. Note it shares its discretisation family with the FEM-trained arm's
targets (same P1 scheme, same time stepping, same box), only finer.

The fairest comparison is not "same pipeline, different targets" -- it is "same
total budget, each method spending it the way that suits it best". The two
supervisions have structurally different cost models:

  * BSDE targets are generated online from the operator itself, so a FRESH
    instance every step is free. Its budget goes almost entirely into optimiser
    steps, and it never runs out of training instances.

  * FEM targets are exact but cost a PDE solve per instance (~0.15 s here). So
    FEM faces a real allocation choice: spend budget on a bigger dataset, or on
    more optimiser steps over a smaller one. Too small and it overfits a finite
    instance set (the test set is always fresh instances); too big and there is
    no time left to train.

Forcing FEM to re-solve fresh instances every step (``refresh_per_step > 0``)
puts it in BSDE's cost model, which no practitioner would choose -- it spent 96%
of a 300 s budget solving PDEs. Pinning it to one arbitrary dataset size is not
fair either. So this script SWEEPS FEM's allocation and reports its best, which
is the number BSDE has to beat.

Everything else is held identical: architecture, seed, total wall-clock, and the
random-t space-time evaluator. FEM's dataset generation happens inside its
budget (``train_buffered`` starts the budget clock before it).

Run: ``python scripts/fair_budget_comparison.py --budget 300 --seeds 2``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_allen_cahn_spacetime as S  # noqa: E402
import wblog  # noqa: E402


def save_ckpt(model, path, meta):
    """Persist a trained operator so it can be RE-SCORED without retraining.

    Not doing this is expensive: every change to the metric or the reference
    (pooled -> average relative error, adding/removing a ground truth) otherwise
    costs a full multi-hour retrain of every arm, and results already computed
    become unusable. The geometry is a deterministic function of (dim, seed,
    grid_res) via ``make_geometry(generator=Generator().manual_seed(seed))``, so
    storing those three plus the state dict is enough to rebuild exactly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_ckpt(path, device):
    """Rebuild (model, geom) from a checkpoint written by :func:`save_ckpt`."""
    ck = torch.load(path, map_location=device, weights_only=False)
    dim, seed, grid_res = ck["dim"], ck["seed"], ck.get("grid_res")
    S.configure_horizon(ck.get("t_phys", 0.3))
    gen = torch.Generator().manual_seed(seed)
    geo_kw = {"grid_res": grid_res} if grid_res is not None else {}
    (input_geom, latent_queries, field_grid, gres, gL,
     in_r, out_r) = S.T.make_geometry(dim, device, generator=gen, **geo_kw)
    model, in_ch = S.build_gino_time(dim, device, in_r, out_r)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    geom = (input_geom.unsqueeze(0), latent_queries.unsqueeze(0), field_grid,
            gres, gL, input_geom, in_ch)
    return model, geom, ck


def run_arm(dim, dev, budget, seed, n_traj, fem_target, pool_size, refresh,
            n_times, fem_n=None, grid_res=None, tag="", ckpt_dir=None, wb=None,
            terminal_weight=0.0, control_variate=False, pino=False,
            deepritz=False, lr=1e-3):
    t0 = time.time()
    model, geom = S.train_buffered(
        dim, steps=10**6, device=dev, n_traj=n_traj, seed=seed,
        fem_target=fem_target, budget_s=budget, pool_size=pool_size,
        refresh_per_step=refresh, fem_n=fem_n, grid_res=grid_res,
        terminal_weight=terminal_weight,
        control_variate=control_variate and not fem_target,
        pino_target=pino, deepritz=deepritz, lr=lr,
        eval_every=0, wb=wb)
    wall = time.time() - t0
    # average relative error over instances, plus its spread ACROSS instances
    rel, sd_inst = S.evaluate_spacetime(model, geom, dim, dev, n_instances=8,
                                        n_times=n_times, Q=256, mode="random")
    if ckpt_dir is not None:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
        save_ckpt(model, Path(ckpt_dir) / f"d{dim}_{safe}_seed{seed}.pt",
                  dict(dim=dim, seed=seed, grid_res=grid_res, t_phys=0.3,
                       arm=tag, n_traj=n_traj, fem_target=fem_target,
                       pool_size=pool_size, refresh=refresh, fem_n=fem_n,
                       budget_s=budget, rel=rel, sd_inst=sd_inst))
    return dict(seed=seed, wall_s=round(wall, 1), rel=rel, sd_inst=sd_inst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n-traj", type=int, default=4096,
                    help="paths per query. 4096 not 256: the rollout is only 5%% "
                         "of a 3-D step and is launch-bound, so 16x the paths "
                         "(4x less target NOISE -- the measured limiter) costs "
                         "+3.7%% step time. Above ~4096 it stops being free.")
    ap.add_argument("--grid-res", type=int, default=None,
                    help="value-field grid resolution for the BSDE arm (see "
                         "train_buffered); 14 in 3-D trades +18%% target bias for "
                         "~1.75x more optimiser steps")
    ap.add_argument("--n-times", type=int, default=16)
    ap.add_argument("--fem-configs", nargs="+", default=None,
                    help="FEM allocations as RES:POOL (e.g. 25:64 41:32). In 3-D "
                         "resolution is a second budget knob: n=41 costs 31.7s "
                         "setup + 2.98s/solve vs n=25 at 1.3s + 0.38s, so a finer "
                         "grid can consume the budget before training starts. "
                         "Default: 2-D 141:{64,256,1024}, 3-D 25:{64,256} + 41:32")
    ap.add_argument("--out", default="scripts/fair_budget_comparison.json")
    ap.add_argument("--terminal-weight", type=float, default=1.0,
                    help="weight of the time-boundary anchor ||w(T,.)-g||^2. "
                         "Applied to EVERY arm: the terminal data g is part of "
                         "the problem statement, not extra data, so withholding "
                         "it from the FEM arm would hand BSDE a free advantage. "
                         "Measured to cut worst-seed 3-D error 0.0512 -> 0.0340.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--deepritz-only", action="store_true",
                    help="run only the Deep Ritz / JKO arm")
    ap.add_argument("--pino-only", action="store_true",
                    help="run only the PINO arm")
    ap.add_argument("--fem-only", action="store_true",
                    help="run only the FEM arms. FEM is ground truth, so its "
                         "TRAINED arms are the reference row of the table, not "
                         "a competitor -- but that row still has to be measured "
                         "at the same budget as everything else.")
    ap.add_argument("--bsde-only", action="store_true",
                    help="run only the BSDE arms (FEM is ground truth; its "
                         "trained arms are excluded from the results table)")
    ap.add_argument("--control-variate", action="store_true",
                    help="BSDE arms only: subtract the martingale "
                         "sum sigma <grad w, dW>. Zero-mean by construction (dW "
                         "is independent of the left-endpoint integrand), so it "
                         "cannot bias the target; gradient is free from the value "
                         "slices. Measured 4.4x target-noise reduction alone.")
    ap.add_argument("--ckpt-dir", default="checkpoints",
                    help="where to save trained operators for later re-scoring")
    wblog.add_args(ap)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S.configure_horizon(0.3)

    print(f"fair fixed-budget comparison | d={args.dim} "
          f"budget={args.budget}s/arm seeds={args.seeds} device={dev}")

    arms = {
        # BSDE at its natural best: fresh instances every step (free for it)
        "BSDE  fresh-instances": dict(fem_target=False, pool_size=64, refresh=2),
    }
    if args.fem_configs:
        fem_cfgs = [tuple(int(x) for x in c.split(":")) for c in args.fem_configs]
    elif args.dim == 2:
        fem_cfgs = [(141, 64), (141, 256), (141, 1024)]
    else:
        fem_cfgs = [(25, 64), (25, 256), (41, 32)]
    for res, n in fem_cfgs:
        # FEM pays mesh/LU setup + n solves up front, inside the budget, then
        # trains on that fixed pool for whatever budget remains
        arms[f"FEM   n={res} dataset={n}"] = dict(
            fem_target=True, pool_size=n, refresh=0, fem_n=res)
    # Second BSDE arm: same replacement rate, 4x larger pool. refresh=0 would be
    # the wrong control -- a pool that never turns over can only overfit, which
    # conflates "fixed data" with "no replacement". Holding refresh=2 fixed and
    # varying only pool size isolates turnover RATE from pool SIZE, and cuts the
    # fraction of COLD slots (freshly inserted, zeroed target buffer) per batch 4x.
    arms["BSDE  pool=256 refresh=2"] = dict(fem_target=False, pool_size=256,
                                            refresh=2)
    # PINO: the OTHER data-free supervision. No dataset either, but it constrains
    # the PDE pointwise (exact autograd residual) instead of through a stochastic
    # representation -- the natural comparison for BSDE once FEM is used only as
    # ground truth.
    arms["PINO  residual"] = dict(fem_target=False, pool_size=64, refresh=2,
                                  pino=True)
    # Deep Ritz / JKO: the variational data-free arm. Minimises the
    # Ginzburg-Landau energy through a minimizing-movement step instead of a
    # residual, so it needs only FIRST derivatives and decreases the energy
    # monotonically -- it cannot settle in the wrong phase.
    arms["DEEPRITZ  jko"] = dict(fem_target=False, pool_size=64, refresh=2,
                                 deepritz=True)
    for k in arms:
        arms[k].setdefault("fem_n", None)
        arms[k].setdefault("pino", False)
        arms[k].setdefault("deepritz", False)

    if args.bsde_only:
        arms = {k: v for k, v in arms.items() if k.startswith("BSDE")}
    if args.fem_only:
        arms = {k: v for k, v in arms.items() if k.startswith("FEM")}
    if args.pino_only:
        arms = {k: v for k, v in arms.items() if k.startswith("PINO")}
    if args.deepritz_only:
        arms = {k: v for k, v in arms.items() if k.startswith("DEEPRITZ")}

    res = {}
    for tag, kw in arms.items():
        runs = []
        for seed in range(args.seeds):
            print(f"\n--- {tag} | seed {seed} ---", flush=True)
            wb = wblog.init(
                args, f"d{args.dim}-{tag}-seed{seed}", job_type=tag.split()[0],
                tags=[f"d{args.dim}", tag.split()[0]],
                config=dict(dim=args.dim, arm=tag, seed=seed,
                            budget_s=args.budget, n_traj=args.n_traj,
                            grid_res=args.grid_res,
                            terminal_weight=args.terminal_weight,
                            control_variate=args.control_variate,
                            lr=args.lr, **kw))
            r = run_arm(args.dim, dev, args.budget, seed, args.n_traj,
                        kw["fem_target"], kw["pool_size"], kw["refresh"],
                        args.n_times, kw["fem_n"],
                        None if kw["fem_target"] else args.grid_res,
                        tag=tag, ckpt_dir=args.ckpt_dir, wb=wb,
                        terminal_weight=args.terminal_weight,
                        control_variate=args.control_variate,
                        pino=kw["pino"], deepritz=kw["deepritz"], lr=args.lr)
            wblog.finish(wb, dict(rel=r["rel"], sd_inst=r["sd_inst"],
                                  wall_s=r["wall_s"]))
            runs.append(r)
            print(f"  -> rel L2 = {r['rel']:.4f}", flush=True)
        res[tag] = runs

    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {args.out}")

    print(f"\n{'=' * 78}")
    print(f"space-time rel L2 (random-t), equal {args.budget}s/arm, "
          f"{args.seeds} seeds")
    print(f"  {'arm':<24} {'rel L2':>22}")
    means = {}
    for tag, runs in res.items():
        v = [r["rel"] for r in runs]
        means[tag] = statistics.mean(v)

        def pm(x):
            sd = statistics.stdev(x) if len(x) > 1 else 0.0
            return f"{statistics.mean(x):.4f} +/- {sd:.4f}"

        print(f"  {tag:<24} {pm(v):>22}")

    fem = {k: v for k, v in means.items() if k.startswith("FEM")}
    bsde = means.get("BSDE  fresh-instances")
    if fem and bsde is not None:
        bk = min(fem, key=fem.get)
        spread = max(
            max(r["rel"] for r in res[k]) - min(r["rel"] for r in res[k])
            for k in (bk, "BSDE  fresh-instances"))
        gap = abs(bsde - fem[bk])
        winner = "BSDE" if bsde < fem[bk] else "FEM"
        line = (f"\n  best FEM allocation = {bk} ({fem[bk]:.4f})  vs  "
                f"BSDE {bsde:.4f}  ->  {winner} ahead "
                f"{max(bsde, fem[bk]) / max(min(bsde, fem[bk]), 1e-9):.2f}x")
        if gap < spread:
            line += f"  BUT gap {gap:.4f} < seed spread {spread:.4f} -> NOT resolved"
        print(line)

    rows = []
    for tag, runs in res.items():
        v = [r["rel"] for r in runs]
        rows.append([tag, statistics.mean(v),
                     statistics.stdev(v) if len(v) > 1 else 0.0,
                     min(v), max(v), len(v),
                     statistics.mean([r["wall_s"] for r in runs])])
    wblog.table(args, f"SUMMARY-fair-budget-d{args.dim}",
                ["arm", "rel", "sd", "min", "max", "n_seeds", "wall_s"], rows,
                config=dict(dim=args.dim, budget_s=args.budget,
                            seeds=args.seeds, n_traj=args.n_traj,
                            grid_res=args.grid_res,
                            control_variate=args.control_variate,
                            terminal_weight=args.terminal_weight),
                summary=dict(best_arm=min(means, key=means.get) if means else None,
                             best_rel=min(means.values()) if means else None))


if __name__ == "__main__":
    main()
