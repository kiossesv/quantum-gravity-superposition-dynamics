"""Branch-diagonal and branch-coherence observables (Sections 2.2 and 2.4).

All functions accept a single wavefunction (N,) or a branch array (B, N) and return
one value per branch. Expectation values are divided by the branch norm.
"""

import numpy as np


def _as2d(psi):
    psi = np.asarray(psi)
    return psi[None, :] if psi.ndim == 1 else psi


def _squeeze(val, psi):
    return val[0] if np.asarray(psi).ndim == 1 else val


def norms(psi, grid):
    """Discrete norms ||psi_b||_N^2, Eq. (24)."""
    p2 = _as2d(psi)
    return _squeeze(np.sum(np.abs(p2) ** 2, axis=1) * grid.dx, psi)


def expectation_x(psi, grid):
    """<x>_b, Eq. (25). Meaningful only while the packet is away from x = +-L/2."""
    p2 = _as2d(psi)
    rho = np.abs(p2) ** 2
    return _squeeze(np.sum(rho * grid.x, axis=1) / np.sum(rho, axis=1), psi)


def width_x(psi, grid):
    """Position standard deviation sigma_x,b."""
    p2 = _as2d(psi)
    rho = np.abs(p2) ** 2
    n = np.sum(rho, axis=1)
    mx = np.sum(rho * grid.x, axis=1) / n
    mx2 = np.sum(rho * grid.x**2, axis=1) / n
    return _squeeze(np.sqrt(np.maximum(mx2 - mx**2, 0.0)), psi)


def _momentum_density(psi):
    return np.abs(np.fft.fft(_as2d(psi), axis=1)) ** 2


def expectation_p(psi, grid):
    """<p>_b from the Fourier representation, Eq. (26)."""
    w = _momentum_density(psi)
    return _squeeze(np.sum(w * grid.p, axis=1) / np.sum(w, axis=1), psi)


def expectation_p2(psi, grid):
    w = _momentum_density(psi)
    return _squeeze(np.sum(w * grid.p**2, axis=1) / np.sum(w, axis=1), psi)


def width_p(psi, grid):
    return np.sqrt(np.maximum(expectation_p2(psi, grid) - expectation_p(psi, grid) ** 2, 0.0))


def energy(psi, grid, V, mass=1.0):
    """E_b = <p^2>_b / 2m + <V_b>_b."""
    p2 = _as2d(psi)
    V = np.broadcast_to(np.atleast_2d(V), p2.shape)
    rho = np.abs(p2) ** 2
    pot = np.sum(rho * V, axis=1) / np.sum(rho, axis=1)
    kin = np.atleast_1d(expectation_p2(p2, grid)) / (2.0 * mass)
    return _squeeze(kin + pot, psi)


def overlap(psi1, psi2, dx):
    """<psi1|psi2>_N for two single wavefunctions (complex)."""
    return np.sum(np.conj(psi1) * psi2) * dx


def overlap_matrix(psi, grid):
    """O[b, b'] = <psi_b|psi_b'>_N, Eq. (10). Hermitian, diagonal = branch norms."""
    p2 = _as2d(psi)
    return (np.conj(p2) @ p2.T) * grid.dx


def visibility_and_phase(O, b=0, bp=1):
    """Visibility |O_bb'| and relative phase arg O_bb'."""
    return np.abs(O[..., b, bp]), np.angle(O[..., b, bp])


def branch_density_matrix(weights, O):
    """Reduced state of the branch label, Eq. (9): rho_B[b, b'] = w_b w_b'^* O_b'b."""
    w = np.asarray(weights, dtype=complex)
    return np.outer(w, np.conj(w)) * np.swapaxes(O, -1, -2)
