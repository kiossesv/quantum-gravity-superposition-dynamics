"""Periodic Fourier grid (Section 2.4 of the paper)."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Uniform periodic grid on [-L/2, L/2) with its conjugate momentum grid.

    Parameters
    ----------
    L : float
        Length of the periodic domain.
    N : int
        Number of grid points (must be even).
    hbar : float
        Reduced Planck constant used to convert wavenumbers into momenta.
    """

    L: float
    N: int
    hbar: float = 1.0

    def __post_init__(self):
        if self.N < 2 or self.N % 2:
            raise ValueError("N must be an even integer >= 2")
        if self.L <= 0:
            raise ValueError("L must be positive")

    @cached_property
    def dx(self) -> float:
        return self.L / self.N

    @cached_property
    def x(self) -> np.ndarray:
        """Positions x_n = -L/2 + n dx, n = 0..N-1."""
        return -0.5 * self.L + self.dx * np.arange(self.N)

    @cached_property
    def p(self) -> np.ndarray:
        """Momenta p_k = 2 pi hbar k / L in the standard (unshifted) FFT order."""
        return 2.0 * np.pi * self.hbar * np.fft.fftfreq(self.N, d=self.dx)

    @property
    def p_max(self) -> float:
        """Nyquist momentum pi hbar / dx; the grid covers -p_max <= p < p_max."""
        return np.pi * self.hbar / self.dx

    @property
    def phase_space_cells(self) -> float:
        """L * 2 p_max / (2 pi hbar) = N, Eq. (23)."""
        return self.L * 2.0 * self.p_max / (2.0 * np.pi * self.hbar)


def create_grids(N, L, hbar=1.0):
    """Backward-compatible helper returning (x, p, dx)."""
    g = Grid(L=L, N=N, hbar=hbar)
    return g.x, g.p, g.dx
