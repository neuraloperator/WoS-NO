"""Fixed-wall-clock comparison of four ways to solve the Allen-Cahn family:

    1. BSDE-direct   -- standalone nonlinear Feynman-Kac (Picard) per instance
    2. FEM-direct    -- coarse finite-element solve per instance
    3. GINO+BSDE     -- neural operator trained on ROUGH (low-n_traj) BSDE targets
    4. GINO+FEM      -- neural operator trained on a precomputed FEM dataset

Why BSDE is Picard: the driver f(w)=lam(w-w^3) depends on the solution value
along the path, so the Feynman-Kac representation is a fixed-point equation in
w; a standalone solver iterates it (Picard). GINO+BSDE instead uses the
operator's own current prediction as the path value (Picard-in-the-loop), so
unbiased *rough* estimates converge as their noise averages out over steps.

Fairness: each learned method gets the SAME wall-clock budget, and GINO+FEM's
budget is charged for generating its FEM training set (data-free BSDE is not).
Every method is scored against BOTH ground truths (fine FEM and high-n_traj
Picard-BSDE) on a shared held-out test set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
import train_allen_cahn_family as T          # build_gino, make_geometry, ...
import fem_crosscheck as F                    # fem_solve -> GridField
from src.solvers.sde import sample_allen_cahn_family

BUMPS = (1, 8)


# --------------------------------------------------------------------------- #
# standalone nonlinear BSDE solver (explicit Picard)
# --------------------------------------------------------------------------- #
def picard_bsde(inst, dim, device, field_grid, gres, gL, xq,
                n_traj=512, n_iter=6):
    """Solve one instance by Picard-iterating the nonlinear Feynman-Kac
    representation on a grid, then read the fixed point at xq."""
    solver = T.make_solver(inst, n_traj, device)
    vals = inst.bound_fn(field_grid)                       # w^0 = g on the grid
    for _ in range(n_iter):
        field = T.GridField(vals.detach().reshape((gres,) * dim), gL, dim)
        vals = solver.evaluate(
            field_grid, value_fn=lambda t, x, f=field: f(x).clamp(-1.5, 1.5),
            evaluate_mode=True)
    field = T.GridField(vals.reshape((gres,) * dim), gL, dim)
    return field(xq)


# --------------------------------------------------------------------------- #
# ground truths + scoring on a shared test set
# --------------------------------------------------------------------------- #
def build_test_set(dim, device, geom, M=16, Qt=256, seed=999):
    ig, lq, field_grid, gres, gL, input_geom, in_ch = geom
    gen = torch.Generator().manual_seed(seed)
    insts, feats, xq_list, gt_fem, gt_bsde = [], [], [], [], []
    for _ in range(M):
        inst = sample_allen_cahn_family(n_dim=dim, n_bumps=BUMPS, generator=gen)
        xq = ((torch.rand(Qt, dim, generator=gen) * 2 - 1)).to(device)
        insts.append(inst)
        feats.append(T.instance_features(inst, input_geom, in_ch))
        xq_list.append(xq)
        # GT 1: fine deterministic FEM
        fem_field = F.fem_solve(inst, dim, device,
                                n=(201 if dim == 2 else 41))
        gt_fem.append(fem_field(xq))
        # GT 2: high-trajectory Picard BSDE
        gt_bsde.append(picard_bsde(inst, dim, device, field_grid, gres, gL, xq,
                                   n_traj=2048, n_iter=6))
    return {"insts": insts, "feats": feats, "xq": xq_list,
            "gt_fem": gt_fem, "gt_bsde": gt_bsde}


def _rel(pred, gt):
    return ((pred - gt).norm() / gt.norm().clamp_min(1e-8)).item()


@torch.no_grad()
def score_operator(model, geom, test):
    ig, lq = geom[0], geom[1]
    ef, eb = [], []
    for feats, xq, gf, gb in zip(test["feats"], test["xq"],
                                 test["gt_fem"], test["gt_bsde"]):
        pred = model(input_geom=ig, latent_queries=lq,
                     output_queries=xq.unsqueeze(0), x=feats.unsqueeze(0))[0]
        ef.append(_rel(pred, gf))
        eb.append(_rel(pred, gb))
    return sum(ef) / len(ef), sum(eb) / len(eb)


def score_fn(pred_of_inst, test):
    """Score a per-instance predictor callable pred_of_inst(inst, xq)->[Qt,1]."""
    ef, eb = [], []
    for inst, xq, gf, gb in zip(test["insts"], test["xq"],
                                test["gt_fem"], test["gt_bsde"]):
        pred = pred_of_inst(inst, xq)
        ef.append(_rel(pred, gf))
        eb.append(_rel(pred, gb))
    return sum(ef) / len(ef), sum(eb) / len(eb)


# --------------------------------------------------------------------------- #
# GINO training under a wall-clock budget (BSDE rough targets, or FEM dataset)
# --------------------------------------------------------------------------- #
def train_gino_budget(dim, device, method, budget_s, geom, test,
                      in_radius, out_radius,
                      n_traj=32, fem_pool_size=64, fem_pool_n=None,
                      eval_every_s=None, seed=0):
    ig, lq, field_grid, gres, gL, input_geom, in_ch = geom
    G = field_grid.shape[0]
    B, Q = 8, 64
    if eval_every_s is None:
        eval_every_s = 20 if dim == 2 else 90
    if fem_pool_n is None:
        fem_pool_n = 121 if dim == 2 else 33

    gen = torch.Generator().manual_seed(seed)
    model, _ = T.build_gino(dim, device, in_radius=in_radius, out_radius=out_radius)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.time()
    pool = None
    pool_gen_s = pool_mesh_s = 0.0
    if method == "fem":
        # charge the budget for generating the FEM training set, incl. the mesh
        # generation + assembly + factorisation cost (which BSDE never pays)
        pool = []
        for _ in range(fem_pool_size):
            inst = sample_allen_cahn_family(n_dim=dim, n_bumps=BUMPS, generator=gen)
            fld, t_mesh, _ = F.fem_solve(inst, dim, device, n=fem_pool_n,
                                         return_timing=True)
            pool_mesh_s += t_mesh
            pool.append((T.instance_features(inst, input_geom, in_ch), fld))
        pool_gen_s = time.time() - t0

    history, step, next_eval = [], 0, eval_every_s
    while time.time() - t0 < budget_s:
        if method == "bsde":
            insts = [sample_allen_cahn_family(n_dim=dim, n_bumps=BUMPS, generator=gen)
                     for _ in range(B)]
            feats = torch.stack([T.instance_features(m, input_geom, in_ch)
                                 for m in insts])
        else:
            idx = torch.randint(0, len(pool), (B,), generator=gen).tolist()
            feats = torch.stack([pool[i][0] for i in idx])

        xq = ((torch.rand(Q, dim, generator=gen) * 2 - 1)).to(device)
        if method == "bsde":
            oq = torch.cat([field_grid, xq], dim=0).unsqueeze(0)
            out = model(input_geom=ig, latent_queries=lq, output_queries=oq, x=feats)
            out_grid, out_q = out[:, :G], out[:, G:]
        else:
            out_q = model(input_geom=ig, latent_queries=lq,
                          output_queries=xq.unsqueeze(0), x=feats)

        opt.zero_grad()
        loss = 0.0
        for b in range(B):
            if method == "bsde":
                fld = T.GridField(out_grid[b].detach().reshape((gres,) * dim), gL, dim)
                solver = T.make_solver(insts[b], n_traj, device)
                with torch.no_grad():
                    tgt = solver.evaluate(
                        xq, value_fn=lambda t, x, f=fld: f(x).clamp(-1.5, 1.5),
                        evaluate_mode=True)
            else:
                tgt = pool[idx[b]][1](xq)
            loss = loss + torch.nn.functional.mse_loss(out_q[b], tgt)
        loss = loss / B
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        el = time.time() - t0
        if el >= next_eval:
            ef, eb = score_operator(model, geom, test)
            history.append({"t": round(el, 1), "step": step,
                            "err_fem": ef, "err_bsde": eb})
            print(f"  [{method}] d={dim} t={el:5.0f}s step={step:4d} "
                  f"err_fem={ef:.4f} err_bsde={eb:.4f}", flush=True)
            next_eval += eval_every_s

    ef, eb = score_operator(model, geom, test)
    return {"method": f"GINO+{method.upper()}", "pool_gen_s": round(pool_gen_s, 1),
            "pool_mesh_s": round(pool_mesh_s, 1),
            "steps": step, "err_fem": ef, "err_bsde": eb, "history": history}


# --------------------------------------------------------------------------- #
def run_dim(dim, device, budget_s, seed=0):
    print(f"\n########## dim={dim}  budget={budget_s}s/method ##########", flush=True)
    gen = torch.Generator().manual_seed(seed)
    (input_geom, latent_queries, field_grid, gres, gL,
     in_radius, out_radius) = T.make_geometry(dim, device, generator=gen)
    _, in_ch = T.build_gino(dim, device, in_radius=in_radius, out_radius=out_radius)
    geom = (input_geom.unsqueeze(0), latent_queries.unsqueeze(0), field_grid,
            gres, gL, input_geom, in_ch)

    print("building test set (fine FEM + high-n_traj BSDE ground truths)...", flush=True)
    test = build_test_set(dim, device, geom)

    results = []

    # --- reference solver 1: FEM-direct (coarse), meshing time included ---
    t0 = time.time()
    coarse_n = 61 if dim == 2 else 25
    fields, mesh_s = {}, 0.0
    for inst in test["insts"]:
        fld, t_mesh, _ = F.fem_solve(inst, dim, device, n=coarse_n, return_timing=True)
        fields[id(inst)] = fld
        mesh_s += t_mesh
    fem_t = time.time() - t0
    ef, eb = score_fn(lambda inst, xq: fields[id(inst)](xq), test)
    results.append({"method": f"FEM-direct(n={coarse_n})", "solve_s": round(fem_t, 1),
                    "mesh_s": round(mesh_s, 1), "err_fem": ef, "err_bsde": eb})
    print(f"  [FEM-direct] total={fem_t:.1f}s (mesh={mesh_s:.1f}s) "
          f"err_fem={ef:.4f} err_bsde={eb:.4f}", flush=True)

    # --- reference solver 2: BSDE-direct (Picard) ---
    t0 = time.time()
    bd = {id(inst): picard_bsde(inst, dim, device, field_grid, gres, gL,
                                test["xq"][i], n_traj=256, n_iter=5)
          for i, inst in enumerate(test["insts"])}
    bsde_t = time.time() - t0
    ef, eb = score_fn(lambda inst, xq: bd[id(inst)], test)
    results.append({"method": "BSDE-direct(picard,256)", "solve_s": round(bsde_t, 1),
                    "err_fem": ef, "err_bsde": eb})
    print(f"  [BSDE-direct] solve={bsde_t:.1f}s err_fem={ef:.4f} err_bsde={eb:.4f}", flush=True)

    # --- learned: GINO+BSDE (rough) and GINO+FEM (dataset), equal wall-clock ---
    results.append(train_gino_budget(dim, device, "bsde", budget_s, geom, test,
                                     in_radius, out_radius))
    results.append(train_gino_budget(dim, device, "fem", budget_s, geom, test,
                                     in_radius, out_radius))

    print(f"\n=== dim={dim} summary (rel L2 vs each ground truth) ===")
    print(f"{'method':28s} {'vs FEM-GT':>10s} {'vs BSDE-GT':>11s}  notes")
    for r in results:
        if "solve_s" in r:
            note = f"solve {r['solve_s']}s"
            if r.get("mesh_s"):
                note += f" (mesh {r['mesh_s']}s)"
        else:
            note = f"{r['steps']} steps"
            if r.get("pool_gen_s"):
                note += (f", FEM pool {r['pool_gen_s']}s "
                         f"(mesh {r.get('pool_mesh_s', 0)}s)")
        print(f"{r['method']:28s} {r['err_fem']:10.4f} {r['err_bsde']:11.4f}  {note}")
    return {"dim": dim, "budget_s": budget_s, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--budget2d", type=int, default=200)
    ap.add_argument("--budget3d", type=int, default=700)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_ROOT / "scripts" / "compare_results.json"))
    args = ap.parse_args()

    print(f"4-way comparison | device={args.device}")
    allres = []
    for dim in args.dims:
        budget = args.budget2d if dim == 2 else args.budget3d
        allres.append(run_dim(dim, args.device, budget))
    Path(args.out).write_text(json.dumps(allres, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
