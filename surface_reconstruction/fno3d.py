"""
3D Fourier Neural Operator for Poisson surface reconstruction.

Operates directly on the 64x64x64 grid via spectral convolutions.
No chunking, no point clouds - the full volume is processed at once.

Input:  [B, 2, H, W, D]  (signed_distance + divergence)
Output: [B, 1, H, W, D]  (solution field)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    """3D Fourier layer: learns weights in frequency domain."""

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2))
        self.weights3 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2))
        self.weights4 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2))

    def compl_mul3d(self, a, b):
        # Complex multiplication: a [B, Ci, X, Y, Z], b [Ci, Co, X, Y, Z, 2]
        return torch.einsum("bixyz,ioxyz->boxyz",
                            a,
                            torch.view_as_complex(b.contiguous()))

    def forward(self, x):
        B = x.shape[0]
        # FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft = torch.zeros(B, self.out_channels, x.size(-3), x.size(-2),
                             x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)

        # 4 corners in frequency space
        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(x_ft[:, :, :m1, :m2, :m3], self.weights1)
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, :m2, :m3], self.weights2)
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, :m1, -m2:, :m3], self.weights3)
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], self.weights4)

        # Inverse FFT
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator.

    Args:
        in_channels:  input channels (2 = signed_dist + divergence)
        out_channels: output channels (1 = solution)
        modes:        number of Fourier modes per dimension
        width:        hidden channel width
        n_layers:     number of Fourier layers
    """

    def __init__(self, in_channels=2, out_channels=1, modes=12, width=32, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.width = width

        # Lift input channels to hidden width
        self.lift = nn.Conv3d(in_channels, width, 1)

        # Fourier layers
        self.spectral_convs = nn.ModuleList([
            SpectralConv3d(width, width, modes, modes, modes)
            for _ in range(n_layers)
        ])
        self.conv1x1s = nn.ModuleList([
            nn.Conv3d(width, width, 1)
            for _ in range(n_layers)
        ])

        # Project back to output
        self.proj = nn.Sequential(
            nn.Conv3d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv3d(width * 2, out_channels, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, H, W, D]
        Returns:
            [B, out_channels, H, W, D]
        """
        x = self.lift(x)
        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.conv1x1s[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        return self.proj(x)
