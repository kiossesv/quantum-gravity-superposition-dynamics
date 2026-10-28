"""The closed-form references of Section 2.3 are exact solutions."""

import numpy as np

import branchgravity as bg
from branchgravity import analytic as an


def _tdse_residual(psi_fn, V, grid, t, h=1e-5, mass=1.0):
    """max |i dpsi/dt - H psi| / max |H psi| using a central difference in t and a
    spectral second derivative in x."""
    dpsi_dt = (psi_fn(t + h) - psi_fn(t - h)) / (2 * h)
    psi = psi_fn(t)
    kin = np.fft.ifft(grid.p**2 / (2 * mass) * np.fft.fft(psi))
    Hpsi = kin + V * psi
    return np.max(np.abs(1j * dpsi_dt - Hpsi)) / np.max(np.abs(Hpsi))


def test_uniform_field_gaussian_solves_tdse():
    g = bg.Grid(100.0, 2048)
    x0, p0, s, grav = -10.0, 5.0, 1.0, 1.2
    V = bg.potentials.linear_gravity_potential(g.x, 1.0, grav)
    for t in (0.5, 2.0, 4.0):
        r = _tdse_residual(lambda tt: an.gaussian_uniform_field(g.x, tt, grav, x0, p0, s), V, g, t)
        assert r < 1e-7, (t, r)


def test_coherent_state_solves_tdse():
    g = bg.Grid(40.0, 512)
    w, x0, p0 = 1.0, 2.0, 1.0
    V = bg.potentials.harmonic_potential(g.x, 1.0, w)
    for t in (0.3, 1.7, 5.0):
        r = _tdse_residual(lambda tt: an.coherent_state_harmonic(g.x, tt, w, x0, p0), V, g, t)
        assert r < 1e-7, (t, r)


def test_overlap_formula_matches_quadrature():
    g = bg.Grid(200.0, 8192)
    x0, p0, s, g1, g2 = -10.0, 5.0, 1.0, 0.8, 1.2
    for t in (0.0, 0.25, 1.0, 2.0, 4.0):
        a = an.gaussian_uniform_field(g.x, t, g1, x0, p0, s)
        b = an.gaussian_uniform_field(g.x, t, g2, x0, p0, s)
        O_num = np.sum(np.conj(a) * b) * g.dx
        O_an = an.overlap_uniform_field(t, g1, g2, x0, p0, s)
        assert abs(O_num - O_an) < 1e-12, (t, O_num, O_an)


def test_visibility_independent_of_x0_p0_and_symmetric():
    t = np.linspace(0, 5, 11)
    v = an.visibility(t, 0.8, 1.2, 1.0)
    O1 = an.overlap_uniform_field(t, 0.8, 1.2, -10.0, 5.0, 1.0)
    O2 = an.overlap_uniform_field(t, 0.8, 1.2, 3.0, -2.0, 1.0)
    assert np.allclose(np.abs(O1), v, atol=1e-15) and np.allclose(np.abs(O2), v, atol=1e-15)
    assert np.allclose(an.overlap_uniform_field(t, 1.2, 0.8, -10, 5, 1),
                       np.conj(O1), atol=1e-15)


def test_relative_phase_is_potential_difference_along_mean_trajectory():
    """Eq. (19): phi = -(m dg/hbar) int_0^t xbar dt'."""
    x0, p0, g1, g2, T = -10.0, 5.0, 0.8, 1.2, 3.0
    t = np.linspace(0, T, 200001)
    xbar = an.classical_trajectory(t, 0.5 * (g1 + g2), x0, p0)[0]
    integral = np.sum(0.5 * (xbar[1:] + xbar[:-1]) * np.diff(t))
    assert abs(an.relative_phase(T, g1, g2, x0, p0) + (g2 - g1) * integral) < 1e-9


def test_action_derivative_is_lagrangian():
    g, x0, p0, m, t, h = 1.3, -2.0, 4.0, 2.0, 1.7, 1e-6
    dS = (an.classical_action(t + h, g, x0, p0, m) - an.classical_action(t - h, g, x0, p0, m)) / (2 * h)
    x, p = an.classical_trajectory(t, g, x0, p0, m)
    assert abs(dS - (p**2 / (2 * m) - m * g * x)) < 1e-7


def test_extended_reference():
    """Extended-precision evaluation of Eq. (16): identical for small phases, and for
    phases ~1e5 rad it removes the error of the double-precision reference itself."""
    g = bg.Grid(80.0, 320)
    a = an.gaussian_uniform_field(g.x, 3.0, 1.2, -10.0, 5.0, 1.0)
    b = an.gaussian_uniform_field(g.x, 3.0, 1.2, -10.0, 5.0, 1.0, extended=True)
    assert np.max(np.abs(a - b)) < 1e-14
    if np.finfo(np.longdouble).eps > 1e-18:          # long double == double here
        return
    x = np.linspace(-560.0, -360.0, 4001)            # the packet is centred at -460 at t = 20
    ref = an.gaussian_uniform_field(x, 20.0, 18.4, -460.0, 184.0, 1.0, extended=True)
    dbl = an.gaussian_uniform_field(x, 20.0, 18.4, -460.0, 184.0, 1.0)
    # both normalized Gaussians agree in modulus; the difference is a phase error ~ eps*|phase|
    assert np.allclose(np.abs(ref), np.abs(dbl), atol=1e-13)
    assert np.max(np.abs(ref - dbl)) > 1e-12
