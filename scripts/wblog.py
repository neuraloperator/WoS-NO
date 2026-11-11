"""One place to wire experiment scripts to Weights & Biases.

Every sweep script here follows the same shape -- an outer product of
configurations, a few runs each, one `train_buffered` call per run -- so they can
share one logging contract instead of each growing its own flags:

  * logging is ON by default (``--no-wandb`` turns it off), because a run that is
    not recorded is a run that has to be repeated;
  * one wandb run per (config, seed), grouped by script and by config, so the
    per-step training curves stay separable while the group carries the sweep;
  * the final score lands in ``wandb.summary`` (not only ``log``), which is what
    the runs table and any later grouping/sorting actually read;
  * one extra "summary" run per script holds the aggregate table, so a sweep can
    be read off a single page without re-deriving means from the children.

A crashed or killed run must not take the sweep with it, so `init` degrades to
None on any wandb failure and every helper is a no-op on None.
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "wos-no"


def add_args(ap: argparse.ArgumentParser, project: str = "wos-no-ac") -> None:
    """Attach the standard wandb flags. Logging defaults to ON."""
    ap.add_argument("--no-wandb", action="store_true",
                    help="disable wandb logging (it is on by default)")
    ap.add_argument("--wandb", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--wandb-project", default=project)
    ap.add_argument("--wandb-group", default=None,
                    help=f"run group; defaults to the script name ({_SCRIPT})")
    ap.add_argument("--wandb-mode", default="online",
                    choices=["online", "offline", "disabled"])


def _base(args):
    return dict(project=getattr(args, "wandb_project", "wos-no-ac"),
                group=getattr(args, "wandb_group", None) or _SCRIPT,
                mode=getattr(args, "wandb_mode", "online"))


def init(args, name, config=None, job_type=None, tags=None):
    """Start a run, or return None if logging is off / wandb is unreachable."""
    if getattr(args, "no_wandb", False):
        return None
    try:
        import wandb
        b = _base(args)
        return wandb.init(project=b["project"], group=b["group"], mode=b["mode"],
                          name=name, job_type=job_type, tags=tags,
                          config=dict(config or {}), reinit=True,
                          settings=wandb.Settings(console="off"))
    except Exception as e:                       # never let logging kill a sweep
        print(f"  [wandb] init failed ({e}); continuing without logging",
              file=sys.stderr, flush=True)
        return None


def finish(wb, summary=None):
    """Write final scalars into wandb.summary and close the run."""
    if wb is None:
        return
    try:
        if summary:
            wb.summary.update({k: v for k, v in summary.items() if v is not None})
        wb.finish()
    except Exception as e:
        print(f"  [wandb] finish failed ({e})", file=sys.stderr, flush=True)


def table(args, name, columns, rows, key="results", config=None, summary=None):
    """Log one aggregate table as its own run, so a sweep reads off one page."""
    if getattr(args, "no_wandb", False):
        return
    try:
        import wandb
        wb = init(args, name, config=config, job_type="summary")
        if wb is None:
            return
        wb.log({key: wandb.Table(columns=list(columns),
                                 data=[list(r) for r in rows])})
        finish(wb, summary)
    except Exception as e:
        print(f"  [wandb] table failed ({e})", file=sys.stderr, flush=True)
