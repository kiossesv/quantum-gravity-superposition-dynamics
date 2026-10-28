"""Grid conventions, initial states and discrete observables (Section 2.4)."""

import numpy as np

import branchgravity as bg
from branchgravity import observables as ob


def test_grid_layout():
    g = bg.Grid(L=100.0, N=2048)
    assert g.x[0] == -50.0 and np.isclose(g.x[-1], 50.0 - g.dx)
    assert np.isclose(g.dx, 100.0 / 2048)
    assert g.p[0] == 0.0
    assert np.isclose(g.p[g.N // 2], -g.p_max)          # unshifted FFT order
    assert np.isclose(g.p.max(), g.p_max - 2 * np.pi / g.L)
    assert np.isclose(g.p_max, np.pi / g.dx)
    assert np.isclose(g.phase_space_cells, g.N)          # Eq. (23)


def test_grid_rejects_odd_n():
    try:
        bg.Grid(L=1.0, N=101)
    except ValueError:
        return
    raise AssertionError("odd N accepted")


def test_create_grids_compat():
    x, p, dx = bg.create_grids(256, 20.0)
    g = bg.Grid(20.0, 256)
    assert np.array_equal(x, g.x) and np.array_equal(p, g.p) and dx == g.dx


def test_gaussian_normalization_and_moments():
    g = bg.Grid(100.0, 2048)
    x0, p0, s = -10.0, 5.0, 1.3
    psi = bg.gaussian_wavepacket(g.x, x0, p0, s)
    assert abs(ob.norms(psi, g) - 1.0) < 1e-14
    assert abs(ob.expectation_x(psi, g) - x0) < 1e-12
    assert abs(ob.expectation_p(psi, g) - p0) < 1e-12
    assert abs(ob.width_x(psi, g) - s) < 1e-12
    assert abs(ob.width_p(psi, g) - 1.0 / (2.0 * s)) < 1e-12


def test_spectral_momentum_replaces_finite_difference():
    """The original code used np.gradient for <p>; it is ~1% off for this grid."""
    g = bg.Grid(100.0, 2048)
    psi = bg.gaussian_wavepacket(g.x, -10.0, 5.0, 1.0)
    p_spec = ob.expectation_p(psi, g)
    p_fd = np.real(np.sum(np.conj(psi) * (-1j * np.gradient(psi, g.dx))) * g.dx)
    assert abs(p_spec - 5.0) < 1e-12
    assert abs(p_fd - 5.0) > 1e-2


def test_energy_of_gaussian_in_uniform_field():
    g = bg.Grid(100.0, 2048)
    x0, p0, s, m, grav = -10.0, 5.0, 1.0, 1.0, 0.8
    psi = bg.gaussian_wavepacket(g.x, x0, p0, s)
    V = bg.potentials.linear_gravity_potential(g.x, m, grav)
    E = ob.energy(psi, g, V, m)
    E_exact = (p0**2 + 1.0 / (4 * s**2)) / (2 * m) + m * grav * x0
    assert abs(E - E_exact) < 1e-11


def test_normalize_rows():
    g = bg.Grid(20.0, 128)
    psi = np.stack([3.0 * bg.gaussian_wavepacket(g.x, 0, 0, 1), bg.gaussian_wavepacket(g.x, 1, 0, 1)])
    n = ob.norms(bg.normalize(psi, g.dx), g)
    assert np.allclose(n, 1.0, atol=1e-14)
