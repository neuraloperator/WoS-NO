"""Download the meshes listed in split.json.

Thingi10K meshes (training and held-out) come from the Thingi10K dataset on Hugging Face,
pinned to release v1.5.0, and are saved as STL. The nine graphics models come from
odedstein/meshes, pinned to a commit, with UVs dropped and texture seams merged; their
licence and attribution texts are collected in graphics/ATTRIBUTION.md.

usage: python download_meshes.py [--out meshes]
"""
import argparse
import io
import os
import urllib.request

import numpy as np
import trimesh

from common import HERE, SPLIT

THINGI = 'https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/v1.5.0/npz/{}.npz'
GRAPHICS = 'https://raw.githubusercontent.com/odedstein/meshes/835213277571d625722d25880c9c3010c6f45a5a/objects'
# name -> (folder, file stem) in odedstein/meshes
GRAPHICS_FILES = {
    'bunny': ('bunny', 'bunny'),
    'spot': ('spot', 'spot_low_resolution'),
    'penguin': ('penguin', 'penguin'),
    'koala': ('koala', 'koala_low_resolution'),
    'goathead': ('goathead', 'goathead'),
    'wingnut': ('wingnut', 'wingnut'),
    'skull': ('skull', 'skull_low_resolution'),
    'fish': ('fish', 'fish_low_resolution'),
    'torus': ('torus', 'torus'),
}


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def thingi10k(name, out):
    with np.load(io.BytesIO(fetch(THINGI.format(name)))) as f:
        vertices = np.asarray(f['vertices'], dtype=np.float64)
        faces = np.asarray(f['facets'], dtype=np.int32)
    trimesh.Trimesh(vertices=vertices, faces=faces).export(os.path.join(out, f'{name}.stl'))


def graphics(name, folder, stem, out):
    """Keep positions and faces only, triangulate, and merge vertices across texture seams."""
    verts, faces = [], []
    for line in fetch(f'{GRAPHICS}/{folder}/{stem}.obj').decode('utf-8', errors='replace').splitlines():
        if line.startswith('v '):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(tok.split('/')[0]) for tok in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    assert mesh.is_watertight, name
    mesh.export(os.path.join(out, f'{name}.obj'))
    readme = fetch(f'{GRAPHICS}/{folder}/README.md').decode('utf-8', errors='replace')
    return f'\n## {name}.obj (source: `objects/{folder}/{stem}.obj`)\n\n{readme.strip()}\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(HERE, 'meshes'))
    args = parser.parse_args()
    for sub in ('thingi10k', 'graphics'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    for name in SPLIT['train'] + SPLIT['test']:
        thingi10k(name, os.path.join(args.out, 'thingi10k'))
    print(f"{len(SPLIT['train']) + len(SPLIT['test'])} Thingi10K meshes", flush=True)

    notes = ['# Graphics test models\n',
             'From [odedstein/meshes](https://github.com/odedstein/meshes), with UVs removed and texture '
             'seams merged. Licence texts are copied from each object\'s README in that repository.\n']
    for name in SPLIT['graphics']:
        notes.append(graphics(name, *GRAPHICS_FILES[name], os.path.join(args.out, 'graphics')))
    with open(os.path.join(args.out, 'graphics', 'ATTRIBUTION.md'), 'w') as f:
        f.write('\n'.join(notes))
    print(f"{len(SPLIT['graphics'])} graphics models", flush=True)


if __name__ == '__main__':
    main()
