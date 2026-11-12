"""Network inputs and GeoSPSR targets for every mesh in split.json, on the 63^3 periodic grid.

Each mesh gets one file DATA/<name>.npz with
  source   divergence of the kernel-smoothed oriented points (input)
  dist     distance to the nearest sampled point (input)
  density  kernel-smoothed density of the sampled points (input)
  draws    GeoSPSR posterior draws, 32 per training mesh and 8 per test mesh (training targets)
  mean     GeoSPSR posterior mean, the reference solution (used only for scoring)

usage: python prepare_data.py [--meshes meshes] [--out data] [--workers 4]
Each worker needs about 2.5 GB of memory.
"""
import argparse
import multiprocessing as mp
import os
import time
import traceback
import zlib

import numpy as np
import torch

from common import HERE, KERNEL, N, PERIOD, SPLIT, load_mesh, surface_points
from spsr import SPSR

TRAIN_DRAWS, TEST_DRAWS = 32, 8


def stencil(pts):
    """Grid cells and weights of a trilinear deposit on the periodic grid."""
    g = (pts * PERIOD) % PERIOD
    base = np.floor(g).astype(int)
    frac = g - base
    for bits in range(8):
        off = np.array([(bits >> i) & 1 for i in range(3)])
        idx = (base + off) % PERIOD
        yield (idx[:, 0], idx[:, 1], idx[:, 2]), np.prod(np.where(off, frac, 1 - frac), axis=1)


def splat(pts, values):
    """Deposit per-point values (n,) or (n, c) on the grid; returns (PERIOD,)*3 or (c, PERIOD, ...)."""
    values = values.reshape(len(pts), -1)
    grid = np.zeros((values.shape[1], PERIOD, PERIOD, PERIOD))
    for idx, w in stencil(pts):
        for c in range(values.shape[1]):
            np.add.at(grid[c], idx, w * values[:, c])
    return grid if values.shape[1] > 1 else grid[0]


def kernel_weight(sp):
    """Squared spectral weights of the GP kernel, placed on the FFT grid."""
    weight = np.zeros((PERIOD, PERIOD, PERIOD))
    idx = sp._freq_idx.copy()
    idx[idx < 0] += PERIOD
    weight[idx[:, 0], idx[:, 1], idx[:, 2]] = sp.v_evals ** 2
    return weight


def source(weight, pts, normals):
    """Divergence of the kernel-smoothed normal field."""
    grid = splat(pts, normals)
    k = np.fft.fftfreq(PERIOD, 1.0 / PERIOD)
    kk = np.stack(np.meshgrid(k, k, k, indexing='ij'))
    hat = weight * sum(1j * kk[d] * np.fft.fftn(grid[d]) for d in range(3))
    return np.fft.ifftn(hat).real


def density(weight, pts):
    return np.fft.ifftn(weight * np.fft.fftn(splat(pts, np.ones(len(pts))))).real


def distance(pts):
    lin = np.arange(PERIOD, dtype=np.float32) / PERIOD
    grid = torch.from_numpy(np.stack(np.meshgrid(lin, lin, lin, indexing='ij'), -1).reshape(-1, 3))
    pts_t = torch.from_numpy(np.asarray(pts, dtype=np.float32))
    d = torch.cat([torch.cdist(c, pts_t).min(dim=1).values for c in torch.split(grid, 8192)])
    return d.numpy().reshape(PERIOD, PERIOD, PERIOD)


def one(job):
    name, mesh_dir, out, n_draws = job
    try:
        torch.set_num_threads(2)
        pts, normals = surface_points(load_mesh(mesh_dir, name))
        pts = np.asarray(pts)
        sp = SPSR(pts, normals, **KERNEL)
        crop = (slice(0, PERIOD),) * 3
        mean = sp.compute_mean_grid(N)[crop]
        draws = sp.sample_grid(N, n_samples=n_draws,
                               rng=np.random.default_rng(zlib.crc32(name.encode())))[(slice(None),) + crop]
        weight = kernel_weight(sp)
        np.savez(os.path.join(out, f'{name}.npz'),
                 source=source(weight, pts, normals).astype(np.float32), dist=distance(pts),
                 density=density(weight, pts).astype(np.float32),
                 mean=mean.astype(np.float32), draws=draws.astype(np.float32))
        return name, None
    except Exception:
        return name, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meshes', default=os.path.join(HERE, 'meshes'))
    parser.add_argument('--out', default=os.path.join(HERE, 'data'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(n, args.meshes, args.out, TRAIN_DRAWS) for n in SPLIT['train']]
    jobs += [(n, args.meshes, args.out, TEST_DRAWS) for n in SPLIT['test'] + SPLIT['graphics']]
    t0 = time.time()
    failed = 0
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for i, (name, err) in enumerate(pool.imap_unordered(one, jobs)):
            if err:
                failed += 1
                print('FAILED', name, err, flush=True)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f'{i + 1}/{len(jobs)} meshes, {time.time() - t0:.0f}s', flush=True)
    if failed:
        raise SystemExit(f'{failed} meshes failed')


if __name__ == '__main__':
    main()
