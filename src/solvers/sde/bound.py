"""
Geometry helpers for the SDE / BSDE solver.

Closest-boundary and boundary-test utilities, ported (and trimmed to the
functions needed here) from the neural walk-on-spheres codebase
(``wos/utils/bound.py`` in https://github.com/bizoffermark/neural_wos).

These let the Euler-Maruyama walker (i) measure the distance from a path
point to the domain boundary so it can decide whether a walk has been
absorbed, and (ii) project an absorbed point onto the boundary so the
Dirichlet value can be read off.
"""

from typing import List

import torch


def find_closest_point_on_cube(
    x: torch.Tensor, x_bound: List[float] = [0.0, 1.0], signed: bool = True
):
    """Closest point on an axis-aligned cube ``[x_bound[0], x_bound[1]]^d``.

    Projects the coordinate that is nearest to a face onto that face and
    returns the projected point together with the (unsigned) distance.
    """
    x_closest = x.clone()

    a = torch.tensor(x_bound[0], device=x.device)
    b = torch.tensor(x_bound[1], device=x.device)
    min_val, min_idx = (x - a).min(dim=-1)
    max_val, max_idx = (b - x).min(dim=-1)
    idx = torch.where(min_val < max_val, min_idx, max_idx)
    val = torch.where(min_val < max_val, a, b)
    dist_closest = torch.where(min_val < max_val, min_val, max_val)
    x_closest[torch.arange(0, x.size(0), dtype=torch.long), idx] = val

    return x_closest, dist_closest


def find_closest_point_on_annulus(
    x: torch.Tensor, r_low: float = 2.0, r_high: float = 4.0, signed: bool = True
):
    """Closest point on an annulus with radii ``r_low`` and ``r_high``.

    The distance is naturally signed/positive inside the annulus.
    """
    r = torch.norm(x, dim=-1)
    mid = (r_low + r_high) / 2.0

    low_idx = r < mid
    high_idx = torch.logical_not(low_idx)

    x_out = torch.empty_like(x)
    x_out[low_idx] = x[low_idx] / r[low_idx].unsqueeze(-1) * r_low
    x_out[high_idx] = x[high_idx] / r[high_idx].unsqueeze(-1) * r_high

    dist_closest = torch.empty_like(r)
    dist_closest[low_idx] = r[low_idx] - r_low
    dist_closest[high_idx] = r_high - r[high_idx]

    return x_out, dist_closest


def test_rect_bound(
    x: torch.Tensor, min_max: List[float], stop_tol: float = 1e-4
) -> torch.Tensor:
    """Boolean mask: which rows of ``x`` are strictly inside the box."""
    if isinstance(min_max, list):
        assert len(min_max) == 2
        min_max = torch.tensor(min_max, device=x.device)
        min_max = torch.ones([x.shape[-1], 2], device=x.device) * min_max
    min_val, max_val = min_max[:, 0] + stop_tol, min_max[:, 1] - stop_tol
    within_bounds = (x > min_val) & (x < max_val)
    return within_bounds.all(dim=-1)
