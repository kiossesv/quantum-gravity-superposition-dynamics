"""Propagation in a freely falling, co-moving frame (Section 3.5 of the revised paper).

The lab-frame branch states are written as

    psi_b(x, t) = exp{i [P(t) xi + S(t)] / hbar} exp{i phi_b(t)} chi_b(xi, t),   xi = x - X(t),

where (X, P) is the classical trajectory in a reference field g_ref starting from
(x0, p0), S its classical action, and

    phi_b(t) = -m (g_b - g_ref) Xi(t) / hbar,   Xi(t) = int_0^t X = x0 t + p0 t^2/2m - g_ref t^3/6.

chi_b then obeys the Schroedinger equation with the residual potential m (g_b - g_ref) xi
and starts from the initial state translated to xi = 0 and brought to rest. The
co-moving grid only has to follow the relative motion of the branches and their
spreading, not the common fall. Overlaps transform as

    <psi_b|psi_b'> = exp{i [phi_b'(t) - phi_b(t)]} <chi_b|chi_b'>.
"""

from dataclasses import dataclass

import numpy as np

from .analytic import classical_action, classical_trajectory
from .propagators import StrangPropagator
from .states import gaussian_wavepacket


def fourier_interpolate(f, grid, xq, chunk=2048):
    """Trigonometric interpolation of grid data f (N,) or (B, N) at points xq.

    Points outside [-L/2, L/2) are set to zero instead of wrapped around.
    """
    f = np.asarray(f)
    squeeze = f.ndim == 1
    f2 = f[None, :] if squeeze else f
    xq = np.atleast_1d(np.asarray(xq, dtype=float))
    c = np.fft.fft(f2, axis=1) / grid.N
    k = grid.p / grid.hbar
    out = np.zeros((f2.shape[0], xq.size), dtype=complex)
    for s in range(0, xq.size, chunk):
        xs = xq[s:s + chunk]
        out[:, s:s + chunk] = c @ np.exp(1j * np.outer(k, xs - grid.x[0]))
    inside = (xq >= -0.5 * grid.L) & (xq < 0.5 * grid.L)
    out[:, ~inside] = 0.0
    return out[0] if squeeze else out


@dataclass(frozen=True)
class FreeFallFrame:
    g_ref: float
    x0: float = 0.0
    p0: float = 0.0
    mass: float = 1.0
    hbar: float = 1.0

    @classmethod
    def for_branches(cls, g_list, x0, p0, mass=1.0, hbar=1.0, g_ref=None):
        """Frame following the initial packet; g_ref defaults to the midrange of g_b,
        which minimizes max_b |g_b - g_ref|."""
        if g_ref is None:
            g_ref = 0.5 * (min(g_list) + max(g_list))
        return cls(float(g_ref), float(x0), float(p0), float(mass), float(hbar))

    # --- reference trajectory -------------------------------------------------
    def X(self, t):
        return classical_trajectory(t, self.g_ref, self.x0, self.p0, self.mass)[0]

    def P(self, t):
        return classical_trajectory(t, self.g_ref, self.x0, self.p0, self.mass)[1]

    def S(self, t):
        return classical_action(t, self.g_ref, self.x0, self.p0, self.mass)

    def Xi(self, t):
        t = np.asarray(t, dtype=float)
        return self.x0 * t + self.p0 * t**2 / (2.0 * self.mass) - self.g_ref * t**3 / 6.0

    # --- branch quantities ----------------------------------------------------
    def residual(self, g_list):
        return np.asarray(g_list, dtype=float) - self.g_ref

    def branch_phases(self, t, g_list):
        """phi_b(t) = -m (g_b - g_ref) Xi(t) / hbar, shape (B,)."""
        return -self.mass * self.residual(g_list) * self.Xi(t) / self.hbar

    def propagator(self, grid, g_list, dt, ordering="VTV", phase_correction=True):
        """Strang propagator for chi_b with the residual fields g_b - g_ref."""
        return StrangPropagator.uniform_field(grid, self.residual(g_list), dt, mass=self.mass,
                                              ordering=ordering, phase_correction=phase_correction)

    def initial_gaussian(self, grid, sigma, n_branches):
        """chi_b(xi, 0) for the Gaussian of Eq. (12): centred at 0 and at rest."""
        chi = gaussian_wavepacket(grid.x, 0.0, 0.0, sigma, grid.hbar)
        return np.tile(chi, (n_branches, 1)).astype(complex)

    def chi_from_lab(self, grid, psi_lab, n_branches=1):
        """chi(xi, 0) = psi_lab(xi + x0) exp(-i p0 xi / hbar) for a callable psi_lab(x)."""
        xi = grid.x
        chi = psi_lab(xi + self.x0) * np.exp(-1j * self.p0 * xi / self.hbar)
        return np.tile(chi, (n_branches, 1)).astype(complex)

    # --- back to the laboratory frame -----------------------------------------
    def to_lab(self, chi, grid, t, g_list, x_lab):
        """Lab-frame wavefunctions psi_b(x_lab, t) from co-moving chi_b on `grid`."""
        xi = np.asarray(x_lab, dtype=float) - self.X(t)
        vals = fourier_interpolate(np.atleast_2d(chi), grid, xi)
        frame_phase = np.exp(1j * (self.P(t) * xi + self.S(t)) / self.hbar)
        branch = np.exp(1j * self.branch_phases(t, g_list))[:, None]
        return branch * frame_phase[None, :] * vals

    def lab_overlaps(self, O_chi, t, g_list):
        """O_lab[b, b'] = exp{i (phi_b' - phi_b)} O_chi[b, b']."""
        e = np.exp(1j * self.branch_phases(t, g_list))
        return np.conj(e)[:, None] * e[None, :] * O_chi

    def lab_x(self, xi_mean, t):
        return np.asarray(xi_mean) + self.X(t)

    def lab_p(self, p_chi, t):
        return np.asarray(p_chi) + self.P(t)
