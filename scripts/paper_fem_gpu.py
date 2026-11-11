"""GPU batched CG for the existing P1 FEM reference discretization.

Identical mass/stiffness matrices, explicit group-FEM reaction, dt=.002,
and 101 float32 spatial snapshots as AllenCahnFEMSolver. Only the linear
system solution changes from CPU LU to float64 Jacobi-preconditioned CG.
"""
import numpy as np
import torch
from skfem import Basis, MeshTet, ElementTetP1
from src.solvers.fem.fem_solver import _mass, _stiff, GridField, SpacetimeField


class BatchedFEM:
    def __init__(self):
        self.n, self.L, self.dt = 41, 4.5, .002
        axis = np.linspace(-self.L, self.L, self.n)
        basis = Basis(MeshTet.init_tensor(axis, axis, axis), ElementTetP1())
        mass = _mass.assemble(basis).tocsr()
        a = (mass + self.dt * _stiff.assemble(basis)).tocsr()
        def gpu(mat):
            return torch.sparse_csr_tensor(torch.tensor(mat.indptr, device="cuda"),
                torch.tensor(mat.indices, device="cuda"), torch.tensor(mat.data, device="cuda"),
                size=mat.shape)
        self.mass, self.a = gpu(mass), gpu(a)
        self.diag = torch.tensor(a.diagonal(), device="cuda")[:, None]
        self.x = torch.tensor(basis.doflocs.T, dtype=torch.float32)
        idx = np.rint((basis.doflocs + self.L) / (2*self.L) * (self.n-1)).astype(np.int64)
        self.indices = tuple(torch.as_tensor(i, device="cuda") for i in idx)

    def field(self, col):
        grid = torch.zeros((self.n,)*3, device="cuda", dtype=torch.float32)
        grid[self.indices] = col.float()
        return GridField(grid, self.L, 3, "cuda")

    def cg(self, b, x):
        r = b - torch.sparse.mm(self.a, x)
        z = r / self.diag
        p = z.clone()
        rz = (r*z).sum(0)
        norm = b.square().sum(0).clamp_min(1e-60)
        for _ in range(100):
            ap = torch.sparse.mm(self.a, p)
            alpha = rz / (p*ap).sum(0).clamp_min(1e-60)
            x = x + p*alpha
            r = r - ap*alpha
            if (r.square().sum(0)/norm).max().item() < 1e-20:
                # Check actual residual, not only the recursively updated one.
                actual = b - torch.sparse.mm(self.a, x)
                if (actual.square().sum(0)/norm).max().item() > 4e-20:
                    raise RuntimeError("FEM CG true relative residual exceeds 2e-10")
                return x
            z = r/self.diag
            next_rz = (r*z).sum(0)
            p = z + p*(next_rz/rz.clamp_min(1e-60))
            rz = next_rz
        raise RuntimeError("FEM CG failed to converge")

    @torch.no_grad()
    def solve(self, instances, times, xqs, horizon):
        u = torch.stack([i.bound_fn(self.x).squeeze(-1) for i in instances], dim=1).double().cuda()
        lams = torch.tensor([i.lam for i in instances], device="cuda", dtype=torch.float64)
        steps = round(horizon/self.dt)
        save_at = {round(j*steps/100) for j in range(101)}
        slices = [[self.field(u[:,i])] for i in range(len(instances))]
        sigmas = [0.]
        for step in range(steps):
            rhs = torch.sparse.mm(self.mass, u) + self.dt*torch.sparse.mm(self.mass, lams*(u-u**3))
            u = self.cg(rhs, u)
            if step+1 in save_at:
                for i in range(len(instances)):
                    slices[i].append(self.field(u[:,i]))
                sigmas.append((step+1)*self.dt/horizon)
        result = []
        for i in range(len(instances)):
            field = SpacetimeField(slices[i], sigmas)
            result.append([field(1-t/horizon, x.cuda()).cpu() for t,x in zip(times[i],xqs[i])])
        return result
