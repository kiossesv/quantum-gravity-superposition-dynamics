"""Free-fall-frame propagation agrees with the exact solution and with lab-frame runs."""

import numpy as np

import branchgravity as bg
from branchgravity import analytic as an, diagnostics as dg, observables as ob

G, X0, P0, SIG, DT, T = [0.8, 1.2], -10.0, 5.0, 1.0, 0.01, 3.0
LAB = bg.Grid(100.0, 2048)
COM = bg.Grid(40.0, 128)
FRAME = bg.FreeFallFrame.for_branches(G, X0, P0)
N_STEPS = int(round(T / DT))


def _chi():
    prop = FRAME.propagator(COM, G, DT)
    return prop.propagate(FRAME.initial_gaussian(COM, SIG, 2), N_STEPS)


def test_default_reference_is_midrange():
    assert FRAME.g_ref == 1.0
    assert np.allclose(FRAME.residual(G), [-0.2, 0.2])


def test_comoving_state_matches_exact():
    chi = _chi()
    ref = an.branch_states_exact(COM.x, T, FRAME.residual(G), 0.0, 0.0, SIG)
    assert np.max(dg.l2_error(chi, ref, COM)) < 1e-12


def test_lab_reconstruction_matches_exact_and_lab_run():
    psi_ff = FRAME.to_lab(_chi(), COM, T, G, LAB.x)
    ex = an.branch_states_exact(LAB.x, T, G, X0, P0, SIG)
    assert np.max(dg.l2_error(psi_ff, ex, LAB)) < 1e-12
    lab = bg.StrangPropagator.uniform_field(LAB, G, DT, phase_correction=True)
    psi_lab = lab.propagate(bg.branch_states(LAB, 2, X0, P0, SIG), N_STEPS)
    assert np.max(dg.l2_error(psi_ff, psi_lab, LAB)) < 1e-12


def test_lab_overlaps_from_comoving_frame():
    O = FRAME.lab_overlaps(ob.overlap_matrix(_chi(), COM), T, G)
    assert np.max(np.abs(O - an.overlap_matrix_exact(T, G, X0, P0, SIG))) < 1e-13


def test_lab_centroids_from_comoving_frame():
    chi = _chi()
    x = FRAME.lab_x(ob.expectation_x(chi, COM), T)
    p = FRAME.lab_p(ob.expectation_p(chi, COM), T)
    for b, g in enumerate(G):
        xc, pc = an.classical_trajectory(T, g, X0, P0)
        assert abs(x[b] - xc) < 1e-12 and abs(p[b] - pc) < 1e-12


def test_chi_from_lab_equals_initial_gaussian():
    chi = FRAME.chi_from_lab(COM, lambda x: bg.gaussian_wavepacket(x, X0, P0, SIG), 2)
    assert np.max(np.abs(chi - FRAME.initial_gaussian(COM, SIG, 2))) < 1e-14


def test_fourier_interpolation():
    grid = bg.Grid(30.0, 256)
    f = bg.gaussian_wavepacket(grid.x, 1.0, 2.0, 1.0)
    assert np.max(np.abs(bg.fourier_interpolate(f, grid, grid.x) - f)) < 1e-13
    xq = np.linspace(-10, 10, 777)
    assert np.max(np.abs(bg.fourier_interpolate(f, grid, xq)
                         - bg.gaussian_wavepacket(xq, 1.0, 2.0, 1.0))) < 1e-12
    assert np.all(bg.fourier_interpolate(f, grid, np.array([-20.0, 15.0, 40.0])) == 0)
