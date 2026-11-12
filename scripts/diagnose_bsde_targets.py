"""Frozen BSDE target variance, FEM discrepancy, and time-step sensitivity.

Uses seed-0 CV + cosine checkpoints and a fixed subset of cached FEM queries.
Raw/CV comparisons share paths; different time-step counts use independent
random streams. This diagnoses single-rollout targets, not the training buffer.
Run from the repository root; defaults are a small CPU-compatible diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import train_allen_cahn_spacetime as S


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    S.configure_horizon(ck["t_phys"])
    S.BOUND_OUTPUT = abs(ck["t_phys"] - 0.3) >= 1e-9
    kw = dict(latent_L=3.0, grid_L=4.0) if S.BOUND_OUTPUT else {}
    if ck.get("grid_res") is not None:
        kw["grid_res"] = ck["grid_res"]
    geom = S.T.make_geometry(ck["dim"], device,
                            generator=torch.Generator().manual_seed(ck["seed"]), **kw)
    inp, latent, grid, gres, length, rin, rout = geom
    model, channels = S.build_gino_time(ck["dim"], device, rin, rout)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, (inp.unsqueeze(0), latent.unsqueeze(0), grid, gres, length, inp, channels), ck


def summarize(samples, reference):
    # Each row is an independent MC repetition at the SAME fixed queries.
    a = samples.double()
    ref = reference.double()
    variance = a.var(0, unbiased=True).mean()
    mean_mse = (a.mean(0) - ref).square().mean()
    return dict(variance=variance.item(), target_sd=variance.sqrt().item(),
                mean_mc_se=(variance / len(a)).sqrt().item(),
                mean_rmse_vs_fem=mean_mse.sqrt().item(),
                mean_relative_l2=((a.mean(0) - ref).norm() / ref.norm()).item(),
                mean_mse_minus_mc_noise=(mean_mse - variance / len(a)).item())


@torch.no_grad()
def run_case(dim, horizon, args):
    path = ROOT / args.ckpt_dir / f"d{dim}_T{horizon:g}_BSDE_pool256_sched_seed{args.seed}.pt"
    model, geom, ck = load_model(path, args.device)
    ig, lq, grid, gres, length, inp, channels = geom
    refs_path = ROOT / args.ckpt_dir / f"refs_d{dim}_T{horizon:g}_n64.pt"
    refs = torch.load(refs_path, map_location=args.device, weights_only=False)
    observations, truth, predictions, details = {}, [], [], []
    for i, (inst, times, queries, values) in enumerate(refs[:args.instances]):
        features = S.T.instance_features(inst, inp, channels).unsqueeze(0)
        bound = S.invariant_bound(inst, grid)
        for j in range(min(args.times, len(times))):
            t = float(times[j])
            x = queries[j][:args.queries]
            truth.append(values[j][:args.queries].flatten().cpu())
            predictions.append(S.op(model, ig, lq, x.unsqueeze(0), features, t,
                                    args.device).flatten().cpu())
            ts = S.make_slice_times(t, 0.0, args.slices)
            grids = [S.op(model, ig, lq, grid.unsqueeze(0), features, s, args.device)[0]
                     .reshape((gres,) * dim) for s in ts]
            field = S.SpacetimeField(grids, ts, length, dim)
            steps = S.rollout_n_steps(inst.lam, S.T_HORIZON - t)
            details.append(dict(instance=i, time_index=j, t=t, lam=inst.lam,
                                bound=bound, base_steps=steps))
            for factor in args.factors:
                for cv in ("off", "bsde"):
                    samples = []
                    for r in range(args.repeats):
                        # Identical paths for raw/CV; independent across cases,
                        # queries/time blocks, refinement levels and repeats.
                        torch.manual_seed(20260911 + dim * 10000000 + int(horizon) * 1000000
                                          + i * 10000 + j * 1000 + factor * 100 + r)
                        y = S.bsde_rollout(inst, x, t, field, args.trajectories,
                                             0.0, args.device, n_steps=steps * factor,
                                             cv=cv, return_grad=False, bound=bound)
                        if y.shape != (len(x), 1) or not torch.isfinite(y).all():
                            raise ValueError(f"Invalid rollout target: shape={y.shape}")
                        samples.append(y.flatten().cpu())
                    observations.setdefault(f"x{factor}_{cv}", []).append(torch.stack(samples))
            print(f"d={dim} T={horizon:g} instance={i} time={t:.3f} done", flush=True)
    reference = torch.cat(truth)
    samples = {k: torch.cat(v, dim=1) for k, v in observations.items()}
    stats = {k: summarize(v, reference) for k, v in samples.items()}
    comparisons = {}
    for factor in args.factors:
        raw, cv = samples[f"x{factor}_off"], samples[f"x{factor}_bsde"]
        delta = (cv - raw).double()
        comparisons[f"x{factor}"] = dict(
            cv_raw_variance_ratio=stats[f"x{factor}_bsde"]["variance"] / stats[f"x{factor}_off"]["variance"],
            paired_mean_shift_rms=delta.mean(0).square().mean().sqrt().item(),
            paired_shift_mc_se_rms=(delta.var(0, unbiased=True).mean() / args.repeats).sqrt().item())
    base = samples[f"x{args.factors[0]}_bsde"].double()
    refinement = {}
    for factor in args.factors[1:]:
        fine = samples[f"x{factor}_bsde"].double()
        refinement[f"x{factor}"] = dict(
            mean_shift_rms=(fine.mean(0) - base.mean(0)).square().mean().sqrt().item(),
            independent_shift_mc_se_rms=((fine.var(0, unbiased=True) + base.var(0, unbiased=True))
                                         .mean() / args.repeats).sqrt().item())
    result = dict(checkpoint=str(path), checkpoint_arm=ck["arm"], geometry=dict(grid_res=gres, grid_L=length),
                  references=str(refs_path), query_blocks=details, metrics=stats,
                  cv_comparison=comparisons, refinement=refinement,
                  model_relative_l2=((torch.cat(predictions) - reference).norm() / reference.norm()).item())
    tensor_path = Path(args.out).with_suffix("").with_name(Path(args.out).stem + f"_d{dim}_T{horizon:g}.pt")
    torch.save(dict(samples=samples, reference=reference, predictions=torch.cat(predictions)), tensor_path)
    result["samples_path"] = str(tensor_path)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--horizons", type=float, nargs="+", default=[1, 3])
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--times", type=int, default=2)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=24)
    ap.add_argument("--trajectories", type=int, default=4096)
    ap.add_argument("--slices", type=int, default=3)
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt-dir", default="checkpoints_retrain")
    ap.add_argument("--out", default="scripts/bsde_target_diagnostic.json")
    args = ap.parse_args()
    if args.repeats < 2 or args.repeats > 100 or args.slices < 2 or min(args.factors) < 1:
        ap.error("require 2 <= repeats <= 100, slices >= 2, factors >= 1")
    if min(args.instances, args.times, args.queries, args.threads) < 1:
        ap.error("sample counts and threads must be positive")
    if args.trajectories < 8 or args.trajectories % 4:
        ap.error("trajectories must be a multiple of 4 and >= 8 for paired cross-fitting")
    torch.set_num_threads(args.threads)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = dict(config=vars(args), cases={})
    start = time.time()
    for dim in args.dims:
        for horizon in args.horizons:
            result = run_case(dim, horizon, args)
            report["cases"][f"d{dim}_T{horizon:g}"] = result
            report["elapsed_s"] = time.time() - start
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: result[k] for k in ("cv_comparison", "refinement")}), flush=True)


if __name__ == "__main__":
    main()
