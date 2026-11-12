"""Score a trained WS-NO against the reference solution (GeoSPSR posterior mean).

For the 8 held-out Thingi10K meshes and the 9 graphics models, reports the relative L2 error
over the full grid of WS-NO and of a single posterior draw (the first stored draw, the SPSR
baseline), and on the graphics models the symmetric Chamfer distance, in voxels, from each
extracted surface to the reference isosurface.

usage: python evaluate.py [--checkpoint checkpoints/wsno_draw.pt] [--data data] [--meshes meshes]
"""
import argparse
import json
import os

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from common import HERE, PERIOD, SPLIT, load_mesh, load_sample, model_input, rel
from fno3d import FNO3d


def iso_level(vol, mesh):
    """Isovalue: mean of the field at the mesh vertices, as in Poisson surface reconstruction."""
    gi = np.clip(np.round(mesh.vertices * PERIOD).astype(int), 0, PERIOD - 1)
    return float(vol[gi[:, 0], gi[:, 1], gi[:, 2]].mean())


def extract_surface(vol, level):
    verts, faces, _, _ = marching_cubes(vol, level=level)
    mesh = trimesh.Trimesh(verts / PERIOD, faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def chamfer(mesh, ref, n=200000, seed=0):
    """Symmetric Chamfer distance in voxels, from n seeded samples on each surface."""
    ref_pts, pts = ref.sample(n, seed=seed), mesh.sample(n, seed=seed + 1)
    return 0.5 * (cKDTree(ref_pts).query(pts)[0].mean() + cKDTree(pts).query(ref_pts)[0].mean()) * PERIOD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(HERE, 'checkpoints', 'wsno_draw.pt'))
    parser.add_argument('--data', default=os.path.join(HERE, 'data'))
    parser.add_argument('--meshes', default=os.path.join(HERE, 'meshes'))
    parser.add_argument('--out', default=os.path.join(HERE, 'results.json'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    a = ckpt['args']
    model = FNO3d(in_channels=3, out_channels=1, modes=a['modes'], width=a['width'], n_layers=a['layers'])
    model.load_state_dict(ckpt['state_dict'])
    model.to(args.device).eval()

    result = {}
    for split in ('test', 'graphics'):
        rows = []
        for name in SPLIT[split]:
            d = load_sample(args.data, name)
            with torch.no_grad():
                pred = model(model_input(d, ckpt['dgain'])[None].to(args.device)).squeeze().cpu().numpy()
            wsno, draw, ref = pred * ckpt['gain'] * d['scale'], d['draws'][0], d['mean']
            row = dict(mesh=name, wsno_rel_l2=rel(wsno, ref), spsr_rel_l2=rel(draw, ref))
            if split == 'graphics':
                mesh = load_mesh(args.meshes, name)
                ref_surface = extract_surface(ref, iso_level(ref, mesh))
                for key, field in (('wsno', wsno), ('spsr', draw)):
                    row[f'{key}_chamfer_voxels'] = chamfer(extract_surface(field, iso_level(field, mesh)), ref_surface)
            rows.append(row)
            print(json.dumps(row), flush=True)
        result[split] = rows

    for split, rows in result.items():
        w = np.array([r['wsno_rel_l2'] for r in rows])
        s = np.array([r['spsr_rel_l2'] for r in rows])
        print(f'{split}: WS-NO {w.mean():.4f} +- {w.std():.4f}, one draw {s.mean():.4f} +- {s.std():.4f}, '
              f'reduction {s.mean() / w.mean():.2f}x, WS-NO lower on {int((w < s).sum())}/{len(rows)}')
    g = result['graphics']
    w, s = np.array([r['wsno_chamfer_voxels'] for r in g]), np.array([r['spsr_chamfer_voxels'] for r in g])
    print(f'graphics Chamfer: WS-NO {w.mean():.3f}, one draw {s.mean():.3f} voxels, '
          f'WS-NO closer on {int((w < s).sum())}/{len(g)}')
    json.dump(result, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
