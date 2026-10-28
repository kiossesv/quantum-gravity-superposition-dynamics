"""Initial states."""

import numpy as np


def gaussian_wavepacket(x, x0, p0, sigma, hbar=1.0):
    """Normalized Gaussian, Eq. (12):

        psi(x) = (2 pi sigma^2)^(-1/4) exp[-(x-x0)^2/(4 sigma^2) + i p0 (x-x0)/hbar].

    sigma is the position standard deviation. The continuum normalization is used; on a
    grid that resolves the packet and contains it, the discrete norm equals 1 to roundoff.
    """
    x = np.asarray(x, dtype=float)
    return (2.0 * np.pi * sigma**2) ** (-0.25) * np.exp(
        -((x - x0) ** 2) / (4.0 * sigma**2) + 1j * p0 * (x - x0) / hbar
    )


def branch_states(grid, n_branches, x0, p0, sigma):
    """(B, N) array with the same Gaussian in every branch."""
    psi = gaussian_wavepacket(grid.x, x0, p0, sigma, grid.hbar)
    return np.tile(psi, (n_branches, 1)).astype(complex)


def normalize(psi, dx):
    """Normalize each row of psi (1-D or (B, N)) to unit discrete norm."""
    psi = np.asarray(psi, dtype=complex)
    norms = np.sqrt(np.sum(np.abs(psi) ** 2, axis=-1, keepdims=True) * dx)
    return psi / norms
