# Poisson surface reconstruction with WS-NO

This folder reproduces the surface reconstruction experiment of the paper. A 3D Fourier neural operator learns the map from oriented surface points to the Poisson indicator field. It is trained with weak supervision: each epoch, its target is a single posterior draw of stochastic Poisson surface reconstruction in its GeoSPSR form (Holalkere et al., ICML 2025). The posterior mean, the reference solution, is used only for scoring. The network sees only PDE data: the source term, the distance to the nearest sampled point, and the density of the sampled points.

## Setup

Python 3.12 and

```
pip install -r requirements.txt
```

## Steps

Run from this folder. Outputs go to `meshes/`, `data/` and `checkpoints/`, which git ignores.

1. **Meshes.** `python download_meshes.py` fetches the 150 Thingi10K meshes listed in `split.json` from the Thingi10K dataset on Hugging Face (release v1.5.0), and the 9 graphics models from [odedstein/meshes](https://github.com/odedstein/meshes) at a fixed commit. The Thingi10K meshes keep their original Thingiverse licences; the graphics models' licences are copied to `meshes/graphics/ATTRIBUTION.md`.
2. **Inputs and targets.** `python prepare_data.py --workers 4` samples 2000 oriented points on each mesh and computes the three input channels, the GeoSPSR posterior mean and a pool of posterior draws (32 per training mesh, 8 per test mesh) on a 63³ periodic grid. Each worker needs about 2.5 GB of memory, and the output takes about 4.7 GB.
3. **Training.** `python train.py` trains on the 142 training meshes for 500 epochs and writes `checkpoints/wsno_draw.pt`. This took 30 minutes and 7.6 GB of GPU memory on an NVIDIA RTX 5080. `python train.py --target mean --out checkpoints/wsno_mean.pt` trains on the reference solution instead, for comparison.
4. **Evaluation.** `python evaluate.py` scores the 8 held-out Thingi10K meshes and the 9 graphics models, prints a summary and writes `results.json`.

## What is measured

All errors are against the reference solution over the full grid. For each test mesh, `evaluate.py` reports the relative L2 error of WS-NO and of one posterior draw (the first stored draw, the SPSR baseline). On the graphics models, it also extracts each field's isosurface by marching cubes and reports the symmetric Chamfer distance, in voxels, to the reference isosurface. With the weights used in the paper, WS-NO's mean error is 0.1015 on the held-out Thingi10K meshes and 0.0475 on the graphics models, 3.8 and 2.8 times lower than a single draw. Retrained weights may differ slightly from these values.

## Configuration

| Setting | Value |
|---|---|
| Kernel | Periodic Matérn-3/2, length scale 0.03, variance 0.1, noise 0.01, frequencies \|n\| ≤ 20 |
| Grid | 63³, periodic, meshes scaled into [0.1, 0.9]³ |
| Model | `FNO3d`, 4 layers, 21 modes, width 32, 3 input channels |
| Training | Adam, learning rate 10⁻³, weight decay 10⁻⁴, cosine schedule, batch 8, 500 epochs, random axis permutations and flips |
| Seeds | Split fixed in `split.json`; draws seeded from the mesh name; initialization and batch order seeded with 0 |

## Files

- `spsr.py`: GeoSPSR posterior mean and pathwise posterior draws on the periodic grid.
- `fno3d.py`: the 3D Fourier neural operator.
- `common.py`: settings, mesh loading and network inputs shared by the scripts.
- `download_meshes.py`, `prepare_data.py`, `train.py`, `evaluate.py`: the four steps above.
- `split.json`: the 142 training meshes, the 8 held-out Thingi10K meshes and the 9 graphics models.
