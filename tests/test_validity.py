"""Phase-space validity criterion (Section 3.4) against observed failure onsets."""

import warnings

import numpy as np

import branchgravity as bg
from branchgravity import analytic as an, diagnostics as dg
from branchgravity.validity import delta_from_kappa


def test_kappa_delta_relation():
    assert abs(bg.kappa_from_delta(1e-9) - 5.998) < 1e-3
    assert abs(bg.kappa_from_delta(1e-12) - 7.034) < 1e-3
    assert abs(delta_from_kappa(8.0) - 6.22e-16) < 1e-18
    for d in (1e-3, 1e-8, 1e-14):
        assert abs(delta_from_kappa(bg.kappa_from_delta(d)) / d - 1) < 1e-6


def test_momentum_limit_closed_form():
    """Eq. (40)."""
    grid = bg.Grid(100.0, 256)
    s, dt, kap = 1.0, 0.01, 7.0
    for g, p0 in ((1.0, 2.0), (-1.5, 2.0), (0.7, -3.0)):
        rep = bg.time_of_validity(grid, [g], 0.0, p0, s, dt, kappa=kap)
        expected = (grid.p_max - kap / (2 * s) - 0.5 * abs(g) * dt + np.sign(g) * p0) / abs(g)
        assert abs(rep.T_p[0] - expected) < 1e-12


def _observed_onset(grid, g, x0, p0, s, dt, tol, t_stop):
    prop = bg.StrangPropagator.uniform_field(grid, [g], dt, phase_correction=True)
    psi = bg.branch_states(grid, 1, x0, p0, s)
    n = 0
    while n * dt < t_stop:
        psi = prop.step(psi)
        n += 1
        ref = an.gaussian_uniform_field(grid.x, n * dt, g, x0, p0, s)[None, :]
        if dg.l2_error(psi, ref, grid)[0] > tol:
            return n * dt
    return np.inf


def test_predicted_onset_matches_observed():
    """Space- and momentum-limited cases; tolerance 1e-6 in L2 <-> delta = 1e-12."""
    tol, dt = 1e-6, 0.005
    cases = [  # L, N, g, x0, p0, sigma
        (100.0, 2048, 6.4, -10.0, 5.0, 1.0),   # original Fig. 4: space-limited
        (100.0, 256, 1.0, 0.0, 0.0, 1.0),      # coarse grid: momentum-limited
        (60.0, 1024, 2.0, 0.0, 12.0, 1.0),     # upward launch: top edge
        (200.0, 1024, 0.5, 40.0, -3.0, 0.4),   # narrow packet: momentum-limited
    ]
    for L, N, g, x0, p0, s in cases:
        grid = bg.Grid(L, N)
        rep = bg.time_of_validity(grid, [g], x0, p0, s, dt, delta=tol**2)
        T_obs = _observed_onset(grid, g, x0, p0, s, dt, tol, 1.5 * rep.T_val)
        ratio = T_obs / rep.T_val
        assert 0.98 <= ratio <= 1.10, (L, N, g, rep.T_val, T_obs)


def test_old_bound_overestimates_space_limited_case():
    grid = bg.Grid(100.0, 2048)
    rep = bg.time_of_validity(grid, [6.4], -10.0, 5.0, 1.0, 0.01)
    assert rep.binding == "space"
    assert (grid.p_max / 6.4) / rep.T_val > 2.5


def test_grid_requirements_scaling():
    """Eq. (43): N_min -> m g^2 T^3 / (4 pi hbar) when the trajectory dominates."""
    g, T = 10.0, 100.0
    req = bg.grid_requirements(T, [g], 0.0, 0.0, 1.0)
    assert abs(req["N_min"] / (g**2 * T**3 / (4 * np.pi)) - 1.0) < 0.03


def test_freefall_frame_gain_when_the_common_fall_dominates():
    """Earth-like g ~ 18 (Rb-87, sigma = 1 um units) with a small branch difference."""
    g, T = [18.0, 18.4], 20.0
    lab = bg.grid_requirements(T, g, 0.0, 0.0, 1.0, dt=0.01)
    frame = bg.FreeFallFrame.for_branches(g, 0.0, 0.0)
    ff = bg.grid_requirements(T, frame.residual(g), 0.0, 0.0, 1.0, dt=0.01)
    assert lab["N_min"] / ff["N_min"] > 300


def test_freefall_frame_gain_is_small_when_widths_dominate():
    """The paper's baseline (T = 5, g ~ 1): spreading dominates and the gain is < 2x."""
    g, T = [0.8, 1.2], 5.0
    lab = bg.grid_requirements(T, g, -10.0, 5.0, 1.0, dt=0.01)
    frame = bg.FreeFallFrame.for_branches(g, -10.0, 5.0)
    ff = bg.grid_requirements(T, frame.residual(g), 0.0, 0.0, 1.0, dt=0.01)
    assert 1.0 < lab["N_min"] / ff["N_min"] < 2.0
    assert lab["N_min"] < 2048 / 10      # the draft's N = 2048 is heavily oversized


def test_check_run_warns_and_raises():
    rep = bg.time_of_validity(bg.Grid(100.0, 2048), [6.4], -10.0, 5.0, 1.0, 0.01)
    assert bg.check_run(rep, 0.5 * rep.T_val)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not bg.check_run(rep, 2 * rep.T_val)
        assert any(issubclass(x.category, RuntimeWarning) for x in w)
    try:
        bg.check_run(rep, 2 * rep.T_val, action="raise")
    except ValueError:
        return
    raise AssertionError("no exception raised")
