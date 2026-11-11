"""Rescore separately saved Allen-Cahn runs on fresh shared FEM references.

No retraining. Cached per-instance references and per-checkpoint scores make
the evaluation resumable. Run from the repository root.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import train_allen_cahn_spacetime as S
from diagnose_bsde_targets import load_model
import rescore_all as R

ARMS = {"BSDE": "BSDE_pool256_sched", "DeepRitz": "DeepRitz_jko",
        "PINO": "PINO_residual_sched"}
_solver = None


def init_worker(dim):
    global _solver
    torch.set_num_threads(1)
    from src.solvers.fem import AllenCahnFEMSolver
    _solver = AllenCahnFEMSolver(dim, device="cpu")


def make_reference(task):
    dim, horizon, index, path, seed = task
    S.configure_horizon(horizon)
    gen = torch.Generator().manual_seed(seed + index)
    inst = S.sample_instance(dim, gen)
    times = (torch.rand(16, generator=gen) * S.T_HORIZON).tolist()
    xqs = [(torch.rand(256, dim, generator=gen) * 2 - 1) * S.QUERY_L for _ in times]
    _, field = _solver.solve(inst, T=S.T_HORIZON, n_slices=101)
    values = [field(1.0 - t / S.T_HORIZON, x) for t, x in zip(times, xqs)]
    if not all(torch.isfinite(v).all() for v in values):
        raise ValueError(f"Non-finite FEM reference {path}")
    payload = dict(dim=dim, horizon=horizon, index=index, seed=seed + index,
                   n=201 if dim == 2 else 41, dt=.002,
                   reference=(inst, times, xqs, values))
    temp = Path(path).with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    return index


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_report(report, path):
    path.write_text(json.dumps(report, indent=2) + "\n")
    rows = report["cases"]
    if not rows:
        return
    lines = [r"% Requires \usepackage{booktabs}.", r"\begin{table}[t]", r"\centering",
             r"\caption{Final relative $L^2$ error (\%) on 256 fresh held-out FEM instances per case, shared across methods and seeds. Each entry is the mean $\pm$ sample standard deviation across three training seeds of the mean per-instance relative error. Separately saved training runs are evaluated without retraining. Budgets exclude evaluation time. BSDE uses a martingale control variate; DeepRitz and PINO use fixed collocation points. BSDE and PINO use cosine learning-rate schedules; DeepRitz uses a flat learning rate. Bold denotes the lowest mean.}",
             r"\label{tab:" + path.stem.replace("_", "-") + "}", r"\begin{tabular}{ccrccc}", r"\toprule",
             r"Dimension & $T$ & Budget (s) & BSDE & DeepRitz & PINO \\", r"\midrule"]
    for name, c in rows.items():
        if not all(a in c.get("summary", {}) for a in ARMS):
            continue
        best = min(ARMS, key=lambda a: c["summary"][a]["mean"])
        cells = []
        for a in ARMS:
            s = c["summary"][a]
            text = f"{100*s['mean']:.2f} \\pm {100*s['seed_sd']:.2f}"
            cells.append("$" + (r"\mathbf{" + text + "}" if a == best else text) + "$")
        lines.append(f"{c['dim']}D & {c['horizon']:g} & {300 if c['dim']==2 else 400} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.with_suffix(".tex").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--horizons", type=float, nargs="+", default=[.3, 1.])
    ap.add_argument("--out", default="docs/allen_cahn_final_256.json")
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    config = dict(n_instances=256, n_times=16, queries_per_time=256,
                  seed_base=20260912, training_seeds=[0, 1, 2],
                  checkpoint_group="separate saved runs; not figure checkpoints",
                  device=args.device, reference_L=4.5, reference_dt=.002,
                  reference_n={"2": 201, "3": 41}, reference_slices=101)
    report = json.loads(out.read_text()) if out.exists() else dict(config=config, cases={})
    if report["config"] != config:
        raise ValueError("Existing report has a different evaluation configuration")
    for dim in (2, 3):
        gpu_fem = None
        if dim == 3:
            from paper_fem_gpu import BatchedFEM
            gpu_fem = BatchedFEM()
            report["reference_backend"] = dict(
                dim2="CPU LU", dim3="Cached CPU LU plus float64 GPU CG, relative residual <= 2e-10",
                validation="docs/allen_cahn_256_fem_audit.json")
        # FEM factorization is reused across both physical horizons.
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=init_worker, initargs=(dim,)) as executor:
            for horizon in args.horizons:
                case = f"d{dim}_T{horizon:g}"
                root = Path("checkpoints_retrain/paper_refs_256") / case
                root.mkdir(parents=True, exist_ok=True)
                paths = [root / f"{i:03d}.pt" for i in range(256)]
                if gpu_fem is not None:
                    S.configure_horizon(horizon)
                    missing = [i for i,p in enumerate(paths) if not p.exists()]
                    for begin in range(0, len(missing), 16):
                        indices = missing[begin:begin+16]
                        instances, sample_times, sample_x = [], [], []
                        for i in indices:
                            gen = torch.Generator().manual_seed(config["seed_base"] + i)
                            instances.append(S.sample_instance(dim, gen))
                            sample_times.append((torch.rand(16, generator=gen)*S.T_HORIZON).tolist())
                            sample_x.append([(torch.rand(256, dim, generator=gen)*2-1)*S.QUERY_L for _ in range(16)])
                        values = gpu_fem.solve(instances, sample_times, sample_x, S.T_HORIZON)
                        for j,i in enumerate(indices):
                            assert all(torch.isfinite(v).all() for v in values[j])
                            payload = dict(dim=dim, horizon=horizon, index=i, seed=config["seed_base"]+i,
                                n=41, dt=.002, linear_solver="float64 GPU CG",
                                reference=(instances[j],sample_times[j],sample_x[j],values[j]))
                            temp = paths[i].with_suffix(".tmp")
                            torch.save(payload,temp)
                            temp.replace(paths[i])
                        print(f"{case}: GPU references {256-len(missing)+min(begin+16,len(missing))}/256", flush=True)
                pending = [executor.submit(make_reference, (dim, horizon, i, str(p), config["seed_base"]))
                           for i, p in enumerate(paths) if not p.exists()]
                done = 256 - len(pending)
                print(f"{case}: {done}/256 cached references", flush=True)
                start = time.time()
                for f in as_completed(pending):
                    f.result()
                    done += 1
                    if done % 16 == 0:
                        print(f"{case}: references {done}/256 ({time.time()-start:.0f}s)", flush=True)
                refs = []
                for i, p in enumerate(paths):
                    data = torch.load(p, weights_only=False, map_location="cpu")
                    assert (data["dim"], data["horizon"], data["index"], data["seed"]) == (
                        dim, horizon, i, config["seed_base"] + i)
                    inst, times, xqs, values = data["reference"]
                    refs.append((inst, times, [x.to(args.device) for x in xqs],
                                 [v.to(args.device) for v in values]))
                c = report["cases"].setdefault(case, dict(dim=dim, horizon=horizon,
                       references=str(root), checkpoints={}, summary={}))
                for arm, stem in ARMS.items():
                    for seed in range(3):
                        key = f"{arm}_seed{seed}"
                        path = Path("checkpoints_retrain") / f"{case}_{stem}_seed{seed}.pt"
                        sha = digest(path)
                        if key in c["checkpoints"]:
                            assert c["checkpoints"][key]["sha256"] == sha
                            continue
                        model, geom, ck = load_model(path, args.device)
                        # Original scores are retained for provenance; the separate
                        # loader audit checks reconstruction against 64 references.
                        print(f"{case} {key}: scoring 256 references", flush=True)
                        per = []
                        for begin in range(0, 256, 16):
                            per.extend(R.score(model, geom, dim, args.device, refs[begin:begin+16]))
                            if (begin + 16) % 64 == 0:
                                print(f"{case} {key}: {begin+16}/256", flush=True)
                        assert len(per) == 256 and np.isfinite(per).all()
                        c["checkpoints"][key] = dict(path=str(path), sha256=sha,
                            original_64_score=ck.get("rel"), budget_s=ck.get("budget_s"),
                            mean=statistics.mean(per), per_instance=per)
                        save_report(report, out)
                        del model
                        torch.cuda.empty_cache()
                    means = [c["checkpoints"][f"{arm}_seed{s}"]["mean"] for s in range(3)]
                    per_seed = np.asarray([c["checkpoints"][f"{arm}_seed{s}"]["per_instance"] for s in range(3)])
                    # Reference sampling uncertainty, keeping all three seeds
                    # together for each sampled instance. Separate from seed SD.
                    rng = np.random.default_rng(76543)
                    bootstrap = per_seed.mean(0)[rng.integers(0, 256, (5000, 256))].mean(1)
                    c["summary"][arm] = dict(mean=statistics.mean(means), seed_sd=statistics.stdev(means),
                        seed_means=means, reference_bootstrap_ci95=np.quantile(bootstrap, [.025, .975]).tolist())
                    save_report(report, out)
                del refs
                print(f"{case} COMPLETE: {json.dumps(c['summary'])}", flush=True)
    print(f"ALL DONE: {out} and {out.with_suffix('.tex')}", flush=True)


if __name__ == "__main__":
    main()
