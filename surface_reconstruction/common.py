"""Settings and helpers shared by the surface reconstruction scripts."""
import json
import os

import numpy as np
import torch
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = json.load(open(os.path.join(HERE, 'split.json')))

N = 64                  # grid points per axis on [0, 1], endpoints included
PERIOD = N - 1          # the periodic grid that is actually stored: 63^3
N_POINTS = 2000         # oriented points sampled on each mesh
KERNEL = dict(lengthscale=0.03, variance=0.1, sigma=0.01, n_trunc=20)
DIST_SCALE = 0.1        # fixed scale of the distance channel


def mesh_path(mesh_dir, name):
    """Thingi10K meshes are stored as .stl, the graphics models as .obj."""
    sub, ext = ('graphics', '.obj') if name in SPLIT['graphics'] else ('thingi10k', '.stl')
    return os.path.join(mesh_dir, sub, name + ext)


def load_mesh(mesh_dir, name, scale=0.8):
    """Load a mesh and fit it inside [0.1, 0.9]^3, centred, keeping its aspect ratio."""
    mesh = trimesh.load(mesh_path(mesh_dir, name), force='mesh')
    v = np.array(mesh.vertices)
    vmin, vmax = v.min(0), v.max(0)
    extent = vmax - vmin
    mx = extent.max()
    mesh.vertices = (v - vmin) / mx * scale + (1.0 - scale * extent / mx) / 2.0
    return mesh


def surface_points(mesh, count=N_POINTS, seed=0):
    """Uniform surface samples and unit face normals."""
    pts, face = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = mesh.face_normals[face].copy()
    norms = np.linalg.norm(normals, axis=-1)
    valid = norms > 1e-10
    return pts[valid], normals[valid] / norms[valid, None]


def load_sample(data_dir, name):
    with np.load(os.path.join(data_dir, f'{name}.npz')) as f:
        d = {k: f[k] for k in f.files}
    d['name'] = name
    d['scale'] = float(d['source'].std())
    return d


def model_input(d, dgain):
    """Three channels: source, distance to the nearest sample, sample density."""
    chans = [d['source'] / d['scale'], d['dist'] / DIST_SCALE, d['density'] / dgain]
    return torch.from_numpy(np.stack(chans).astype(np.float32))


def rel(a, b):
    """Relative L2 error of a with respect to b."""
    return float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))
