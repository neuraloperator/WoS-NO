"""Train WS-NO for Poisson surface reconstruction on the 142 Thingi10K training meshes.

The default target is one GeoSPSR posterior draw per mesh and epoch, picked from the mesh's
stored pool (weak supervision). --target mean trains on the reference solution instead.
Augmentation applies a random axis permutation and flips to input and target together.

usage: python train.py [--data data] [--out checkpoints/wsno_draw.pt] [--target draw|mean]
"""
import argparse
import itertools
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import HERE, SPLIT, load_sample, model_input
from fno3d import FNO3d

PERMS = list(itertools.permutations([0, 1, 2]))


def augment(vol, perm, flips):
    offset = vol.dim() - 3
    v = vol.permute(*(list(range(offset)) + [offset + p for p in perm]))
    for axis in range(3):
        if flips[axis]:
            v = v.flip(offset + axis)
    return v.contiguous()


def target(d, args, rng):
    """Training target in the network's output units."""
    y = d['mean'] if args.target == 'mean' else d['draws'][rng.integers(len(d['draws']))]
    return y / (d['gain'] * d['scale'])


def batches(train, args, gen, rng):
    """One epoch of augmented (input, target) batches."""
    order = torch.randperm(len(train), generator=gen).tolist()
    for start in range(0, len(train), args.batch):
        xs, ys = [], []
        for i in order[start:start + args.batch]:
            d = train[i]
            perm = PERMS[torch.randint(6, (1,), generator=gen).item()]
            flips = [torch.randint(2, (1,), generator=gen).item() for _ in range(3)]
            xs.append(augment(d['x'], perm, flips))
            ys.append(augment(torch.from_numpy(target(d, args, rng).astype(np.float32)), perm, flips))
        yield torch.stack(xs), torch.stack(ys).unsqueeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join(HERE, 'data'))
    parser.add_argument('--out', default=os.path.join(HERE, 'checkpoints', 'wsno_draw.pt'))
    parser.add_argument('--target', default='draw', choices=['draw', 'mean'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--modes', type=int, default=21)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)

    train = [load_sample(args.data, n) for n in SPLIT['train']]
    # One output scale and one density scale for the whole dataset, from the training draws.
    gain = float(np.median([d['draws'].mean(0).std() / d['scale'] for d in train]))
    dgain = float(np.median([d['density'].std() for d in train]))
    for d in train:
        d['gain'] = gain
        d['x'] = model_input(d, dgain)

    torch.manual_seed(0)
    model = FNO3d(in_channels=3, out_channels=1, modes=args.modes, width=args.width,
                  n_layers=args.layers).to(device)
    print(f'{len(train)} training meshes, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters',
          flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse = nn.MSELoss()
    rng = np.random.default_rng(0)
    gen = torch.Generator().manual_seed(0)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total, steps = 0.0, 0
        for x, y in batches(train, args, gen, rng):
            optimizer.zero_grad()
            loss = mse(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
        scheduler.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'epoch {epoch + 1:4d} loss {total / steps:.5f} {time.time() - t0:.0f}s', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), gain=gain, dgain=dgain, args=vars(args),
                    train_time_s=time.time() - t0), args.out)
    print(f'saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
