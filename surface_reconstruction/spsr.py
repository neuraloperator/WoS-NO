"""
Stochastic Poisson Surface Reconstruction (SPSR)
Adapted from GeoSPSR (Holalkere, Bindel, Sellán, Terenin, ICML 2025)

Uses a Gaussian Process model where:
  - f ~ GP(0, K_f) is the implicit function
  - v = nabla f are the observed normals
  - Posterior samples are drawn via pathwise conditioning
  - Kernels use periodic Matérn 3/2 on the torus [0, 2π]^3

Grid evaluation uses FFT by placing spectral coefficients on the
DFT grid, giving O(N^3 log N) instead of O(N^3 * M).
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class Matern32Periodic1D:
    """Matérn 3/2 kernel on the circle, periodic on [0, 2π]."""

    def __init__(self, lengthscale, variance=1.0):
        self.ls = lengthscale
        self.variance = variance
        self._norm = self._helper_arr(np.array(0.0))

    def _helper_arr(self, u):
        rt3 = np.sqrt(3.0)
        w = rt3 * (u - 0.5)
        w_star = rt3 / 2.0
        ls = self.ls
        cosh_term = np.exp((w - w_star) / ls) + np.exp((-w - w_star) / ls)
        tanh_term = np.tanh(rt3 / (2.0 * ls))
        first = 2.0 * ls + rt3 / tanh_term
        second = 2.0 * w * np.tanh(w / ls)
        return cosh_term * (np.pi**2 * ls / 3.0) * (first - second)

    def eval_matrix(self, x1, x2):
        """x1: (n1,), x2: (n2,) -> (n1, n2) kernel matrix."""
        u = np.abs(x1[:, None] - x2[None, :]) / (2.0 * np.pi)
        u = np.minimum(u, 1.0 - u)
        return self.variance * self._helper_arr(u) / self._norm

    def spectral_density(self, n):
        """Spectral density at integer frequencies n (array)."""
        rt3 = np.sqrt(3.0)
        ls = self.ls
        n = np.asarray(n, dtype=np.float64)
        second = (3.0 / ls**2 + 4.0 * np.pi**2 * n**2) ** (-2)
        first = (2.0 * rt3 * np.tanh(rt3 / (2.0 * ls))
                 / ((2.0 * np.pi)**(-2) * ls))
        return self.variance * first * second / self._norm


class SPSR:
    """
    Stochastic Poisson Surface Reconstruction on [0,1]^3.

    Uses face centroids and normals as the observation data.
    Evaluates the GP posterior mean and draws samples on a regular N^3 grid
    using FFT for fast spectral evaluation.
    """

    def __init__(self, points, normals, lengthscale=0.5, variance=1.0,
                 sigma=0.01, n_trunc=10):
        """
        Args:
            points:      (n, 3) surface points in [0, 1]^3
            normals:     (n, 3) unit normals at those points
            lengthscale: Matérn 3/2 lengthscale (relative to [0, 2π] period)
            variance:    kernel signal variance
            sigma:       observation noise variance (regularization)
            n_trunc:     spectral truncation per dimension
        """
        self.n_data = len(points)
        self.sigma = sigma
        self.n_trunc = n_trunc

        # Scale points to [0, 2π]^3
        self.x_data = points * 2.0 * np.pi
        self.v_data = normals.copy()

        # 1D kernel (same for all 3 dimensions)
        self.k1d = Matern32Periodic1D(lengthscale, variance)

        # Build 3D frequency lattice [-T, T]^3
        ns = np.arange(-n_trunc, n_trunc + 1)
        n1, n2, n3 = np.meshgrid(ns, ns, ns, indexing='ij')
        self.freq = np.stack([n1.ravel(), n2.ravel(), n3.ravel()],
                             axis=-1).astype(np.float64)
        self.M = len(self.freq)

        # v-kernel eigenvalues: sqrt of product spectral density
        sd = np.stack([
            self.k1d.spectral_density(np.abs(self.freq[:, d]))
            for d in range(3)
        ], axis=-1)
        self.v_evals = np.sqrt(sd[:, 0] * sd[:, 1] * sd[:, 2])

        # f eigenvalues from v eigenvalues (Poisson relationship)
        freq_norm_sq = (self.freq ** 2).sum(-1)
        safe_norm_sq = np.where(freq_norm_sq > 0, freq_norm_sq, 1.0)
        factor = np.sqrt(((self.freq * self.v_evals[:, None]) ** 2).sum(-1))
        self.f_evals = np.where(freq_norm_sq > 0, factor / safe_norm_sq, 0.0)

        # Cross-covariance weight: lambda_v^2 / |n|^2
        self.cov_weight = np.where(
            freq_norm_sq > 0, self.v_evals ** 2 / safe_norm_sq, 0.0
        )

        # Cache trig arrays for data points: (nd, M) each
        _dots = self.x_data @ self.freq.T  # (nd, M)
        self._cos_dots = np.cos(_dots)
        self._sin_dots = np.sin(_dots)

        # Precompute FFT frequency index mapping
        freq_int = self.freq.astype(int)
        self._freq_idx = freq_int.copy()  # will be remapped per P in methods

        # Precompute K_vv and solve for alpha
        self._build_kernel()

    def _build_kernel(self):
        nd = self.n_data
        print(f"  SPSR: Building K_vv ({nd}x{nd}), {self.M} spectral modes ...")
        K = np.ones((nd, nd))
        for d in range(3):
            K *= self.k1d.eval_matrix(self.x_data[:, d], self.x_data[:, d])
        K += np.eye(nd) * self.sigma
        self.cho = cho_factor(K)
        self.alpha = cho_solve(self.cho, self.v_data)  # (nd, 3)
        print(f"  SPSR: K_vv built and factorized.")

    def _precompute_fourier_coeffs(self, weights):
        """
        Precompute data-dependent Fourier coefficients A, B.

        f(x) = sum_m cov_weight[m] * (A[m] sin(n_m.x) - B[m] cos(n_m.x))

        A[m] = sum_j (w_j . n_m) cos(n_m . x_j)
        B[m] = sum_j (w_j . n_m) sin(n_m . x_j)
        """
        wT_cos = weights.T @ self._cos_dots  # (3, M)
        wT_sin = weights.T @ self._sin_dots  # (3, M)
        A = np.einsum('dm,dm->m', self.freq.T, wT_cos)  # (M,)
        B = np.einsum('dm,dm->m', self.freq.T, wT_sin)  # (M,)
        return A, B

    def _fft_eval_crosscov(self, A, B, P):
        """
        Evaluate cross-covariance update on a regular P-periodic grid using FFT.

        f(x_j) = sum_m cov_weight[m] * (A[m] sin(n_m . x_j) - B[m] cos(n_m . x_j))
               = Im[sum_m wA[m] exp(i n_m . x_j)] - Re[sum_m wB[m] exp(i n_m . x_j)]

        On a P-periodic grid x_j = 2pi * j / P, this is a 3D IDFT.

        Args:
            A, B: (M,) Fourier coefficients
            P: FFT size (= N-1 for N grid points in [0,1] inclusive)
        Returns:
            (P, P, P) complex array; caller extracts Im/Re and pads to (N,N,N)
        """
        wA = self.cov_weight * A
        wB = self.cov_weight * B

        coeff_A = np.zeros((P, P, P), dtype=np.complex128)
        coeff_B = np.zeros((P, P, P), dtype=np.complex128)

        # Map frequencies [-T, T] to FFT indices [0, P)
        idx = self._freq_idx.copy()
        idx[idx < 0] += P
        ii, jj, kk = idx[:, 0], idx[:, 1], idx[:, 2]

        # Use np.add.at for duplicate-safe scatter
        np.add.at(coeff_A, (ii, jj, kk), wA)
        np.add.at(coeff_B, (ii, jj, kk), wB)

        # IFFT gives (1/P^3) sum_k c_k exp(i 2pi k.j/P), multiply by P^3
        gA = np.fft.ifftn(coeff_A) * (P ** 3)
        gB = np.fft.ifftn(coeff_B) * (P ** 3)

        return gA.imag - gB.real

    def _fft_eval_prior_correct(self, evals, coeffs_cos, coeffs_sin, P):
        """
        Evaluate KL expansion on P-periodic grid.
        f(x) = sum_m evals[m] * (c_cos[m] cos(n_m.x) + c_sin[m] sin(n_m.x))
        """
        w_cos = evals * coeffs_cos
        w_sin = evals * coeffs_sin

        # f = Re[sum (w_cos - i*w_sin) exp(i n.x)]
        #   = sum w_cos cos(n.x) + w_sin sin(n.x)
        coeff = np.zeros((P, P, P), dtype=np.complex128)
        idx = self._freq_idx.copy()
        idx[idx < 0] += P
        ii, jj, kk = idx[:, 0], idx[:, 1], idx[:, 2]

        np.add.at(coeff, (ii, jj, kk), w_cos - 1j * w_sin)

        g = np.fft.ifftn(coeff) * (P ** 3)
        return g.real

    def _pad_periodic(self, vol_P, N):
        """Pad (P,P,P) periodic volume to (N,N,N) where N = P+1."""
        out = np.zeros((N, N, N))
        out[:N-1, :N-1, :N-1] = vol_P
        # Periodic wrap: last slice = first slice
        out[N-1, :, :] = out[0, :, :]
        out[:, N-1, :] = out[:, 0, :]
        out[:, :, N-1] = out[:, :, 0]
        return out

    def compute_mean_grid(self, N):
        """Compute posterior mean on an N^3 regular grid. Returns (N,N,N)."""
        P = N - 1  # periodic grid size
        assert self.n_trunc <= P // 2, \
            f"n_trunc={self.n_trunc} too large for N={N} (max {P//2})"

        A, B = self._precompute_fourier_coeffs(self.alpha)
        vol = self._fft_eval_crosscov(A, B, P)
        return self._pad_periodic(vol, N)

    def sample_grid(self, N, n_samples=1, rng=None):
        """Draw posterior samples on N^3 grid. Returns (n_samples, N, N, N)."""
        if rng is None:
            rng = np.random.default_rng()

        P = N - 1
        assert self.n_trunc <= P // 2

        samples = np.zeros((n_samples, N, N, N))

        for s in range(n_samples):
            # 1. Draw random KL coefficients for v prior
            xi = rng.standard_normal((3, self.M, 2))

            # 2. Derive f prior coefficients
            gamma = self._f_gamma_from_v_xi(xi)

            # 3. Evaluate v prior at data points (small, no FFT needed)
            v_prior = self._eval_v_prior(xi)

            # 4. Solve for update weights
            delta = cho_solve(self.cho, self.v_data - v_prior)

            # 5a. Cross-covariance update via FFT
            A, B = self._precompute_fourier_coeffs(delta)
            f_update = self._fft_eval_crosscov(A, B, P)

            # 5b. f prior via FFT
            f_prior = self._fft_eval_prior_correct(
                self.f_evals, gamma[:, 0], gamma[:, 1], P)

            samples[s] = self._pad_periodic(f_prior + f_update, N)

        return samples

    def _f_gamma_from_v_xi(self, xi):
        """Transform v-space KL coefficients to f-space."""
        factor = np.sqrt(((self.freq * self.v_evals[:, None]) ** 2).sum(-1))
        safe_factor = np.where(factor > 0, factor, 1.0)

        gamma = np.zeros((self.M, 2))
        sign = np.array([-1.0, 1.0])
        for d in range(3):
            gamma += (self.freq[:, d:d+1] * self.v_evals[:, None]
                      * xi[d, :, ::-1] * sign)

        gamma /= safe_factor[:, None]
        gamma = np.where(factor[:, None] > 0, gamma, 0.0)
        return gamma

    def _eval_v_prior(self, xi):
        """Evaluate v prior at data points. Returns (nd, 3)."""
        # v_prior[:, d] = sum_m v_evals[m] * (xi[d,m,0]*cos + xi[d,m,1]*sin)
        v_prior = np.zeros((self.n_data, 3))
        for d in range(3):
            # (M,) weighted coefficients
            c_cos = self.v_evals * xi[d, :, 0]  # (M,)
            c_sin = self.v_evals * xi[d, :, 1]  # (M,)
            # (nd, M) @ (M,) -> (nd,)
            v_prior[:, d] = self._cos_dots @ c_cos + self._sin_dots @ c_sin
        return v_prior
