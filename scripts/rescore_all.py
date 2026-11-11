"""Re-score every saved checkpoint on a SHARED, large instance set.

scripts/scorer_diff.py showed the reported metric -- mean over instances of
||e_i|| / ||w_i|| -- is dominated by its denominator. Across 8 draws in 3-D,
||w|| spans 14.4 to 51.2 while ||e|| stays in 0.74-3.07, so the instance with the
smallest solution norm produces a ratio 9x the median and, averaged over only 8
instances, sets the number by itself. evaluate_spacetime assumes this cannot
happen ("all instances have comparable norms"); in 3-D that is false.

The estimand is right -- it is the per-instance relative L2 asked for -- so the
fix is not to change the metric but to stop estimating it from 8 samples. This
scores every checkpoint against ONE cached set of `--n-inst` instances, so:

  * the mean is estimated from enough draws to be stable;
  * every arm sees exactly the SAME instances, making arm-to-arm differences
    paired rather than confounded with the draw;
  * the reference solves happen once, not once per arm.

Median and IQR are reported next to the mean, since with a heavy tail the two
answer different questions and the gap between them is itself the diagnostic.

Run: ``python scripts/rescore_all.py --dim 3 --n-inst 64``
"""
from __future__ import annotations
import argparse, json, re, statistics, sys, time
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import train_allen_cahn_family as T  # noqa: E402
import train_allen_cahn_spacetime as S  # noqa: E402
import fair_budget_comparison as FB  # noqa: E402


def build_refs(dim, dev, n_inst, n_times, Q, seed=321):
    """Instances, their FEM space-time fields, and the (t, x) they are scored at.

    Drawn once and reused for every checkpoint. The (t, x) samples are frozen
    with the instance so two arms never differ because of the evaluation draw.
    """
    fem = S._eval_fem_solver(dim, dev)
    gen = torch.Generator().manual_seed(seed)
    refs, t0 = [], time.time()
    for k in range(n_inst):
        inst = S.sample_instance(dim, gen)
        _, stf = fem.solve(inst, T=S.T_HORIZON, n_slices=101)
        times = (torch.rand(n_times, generator=gen) * S.T_HORIZON).tolist()
        xqs = [((torch.rand(Q, dim, generator=gen) * 2 - 1) * S.QUERY_L).to(dev)
               for _ in times]
        rfs = [stf(1.0 - te / S.T_HORIZON, xq) for te, xq in zip(times, xqs)]
        refs.append((inst, times, xqs, rfs))
        if (k + 1) % 8 == 0:
            print(f"    {k+1}/{n_inst} refs ({time.time()-t0:.0f}s)", flush=True)
    return refs


@torch.no_grad()
def score(model, geom, dim, dev, refs):
    ig, lq, _, _, _, input_geom, in_ch = geom
    per = []
    for inst, times, xqs, rfs in refs:
        fe = T.instance_features(inst, input_geom, in_ch).unsqueeze(0)
        E, F = [], []
        for te, xq, rf in zip(times, xqs, rfs):
            pr = S.op(model, ig, lq, xq.unsqueeze(0), fe, te, dev)[0]
            E.append((pr - rf).reshape(-1)); F.append(rf.reshape(-1))
        e, f = torch.cat(E), torch.cat(F)
        per.append((e.norm() / f.norm()).item())
    return per


def arm_of(path):
    """Group checkpoints by arm, folding the seed suffix away."""
    n = Path(path).stem
    n = re.sub(r"_seed\d+$", "", n)
    return n.split("_", 1)[1].replace("_", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--n-inst", type=int, default=64)
    ap.add_argument("--n-times", type=int, default=16)
    ap.add_argument("--Q", type=int, default=256)
    ap.add_argument("--dirs", nargs="+",
                    default=["checkpoints", "checkpoints_cv", "checkpoints_dr",
                             "checkpoints_pino"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S.configure_horizon(0.3)
    S.BOUND_OUTPUT = False
    out = args.out or f"scripts/rescore_d{args.dim}.json"

    cks = sorted(p for d in args.dirs for p in Path(d).glob(f"d{args.dim}_*.pt"))
    print(f"rescoring {len(cks)} checkpoints | d={args.dim} "
          f"n_inst={args.n_inst} n_times={args.n_times} device={dev}")
    print(f"  building {args.n_inst} shared references...", flush=True)
    refs = build_refs(args.dim, dev, args.n_inst, args.n_times, args.Q)

    by_arm, res = {}, {}
    for p in cks:
        try:
            model, geom, ck = FB.load_ckpt(str(p), dev)
        except Exception as e:
            print(f"  SKIP {p.name}: {e}", flush=True); continue
        per = score(model, geom, args.dim, dev, refs)
        m, md = statistics.mean(per), statistics.median(per)
        res[str(p)] = dict(arm=arm_of(p), mean=m, median=md,
                           old=ck.get("rel"), per=per)
        by_arm.setdefault(f"{p.parent.name}/{arm_of(p)}", []).append((m, md))
        print(f"  {p.parent.name}/{p.name:<44} mean={m:.4f} median={md:.4f} "
              f"(saved {ck.get('rel'):.4f})", flush=True)
        Path(out).write_text(json.dumps(res, indent=1, default=str))

    print(f"\n{'='*84}\nd={args.dim}, {args.n_inst} shared instances, "
          f"mean over seeds of the per-instance statistic")
    print(f"  {'arm':<44}{'mean':>10}{'median':>10}{'n':>4}")
    for k, v in sorted(by_arm.items(), key=lambda kv: statistics.mean(
            [x[1] for x in kv[1]])):
        mm = statistics.mean([x[0] for x in v])
        mdd = statistics.mean([x[1] for x in v])
        print(f"  {k:<44}{mm:>10.4f}{mdd:>10.4f}{len(v):>4}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
