"""Strang propagator: unitarity, the exact splitting phase and its correction (Section 3.2)."""

import numpy as np

import branchgravity as bg
from branchgravity import analytic as an, diagnostics as dg, observables as ob

X0, P0, SIG, G = -10.0, 5.0, 1.0, [0.8, 1.2]
GRID = bg.Grid(100.0, 2048)


def _run(dt, T, ordering="VTV", corr=False, g=G):
    prop = bg.StrangPropagator.uniform_field(GRID, g, dt, ordering=ordering, phase_correction=corr)
    n = int(round(T / dt))
    return prop.propagate(bg.branch_states(GRID, len(g), X0, P0, SIG), n), n * dt


def test_norm_conservation():
    """Roundoff only: ~1e-13 after 1000 steps."""
    psi, _ = _run(0.01, 10.0)
    assert np.max(np.abs(ob.norms(psi, GRID) - 1.0)) < 1e-12


def test_splitting_error_is_exactly_the_predicted_global_phase():
    """Eqs. (31)-(32) for both orderings and step sizes far beyond any CFL-type bound."""
    for ordering in ("VTV", "TVT"):
        for dt in (0.5, 0.1, 0.02):
            psi, T = _run(dt, 5.0, ordering)
            ex = an.branch_states_exact(GRID.x, T, G, X0, P0, SIG)
            eps = np.array([an.splitting_phase(T, g, dt, ordering=ordering) for g in G])
            assert np.allclose(dg.phase_error(psi, ex, GRID), eps, rtol=0, atol=1e-12)
            residual = dg.l2_error(psi * np.exp(-1j * eps)[:, None], ex, GRID)
            assert np.max(residual) < 1e-12, (ordering, dt, residual)


def test_phase_correction_is_exact_for_uniform_fields():
    for ordering in ("VTV", "TVT"):
        psi, T = _run(0.5, 5.0, ordering, corr=True)
        ex = an.branch_states_exact(GRID.x, T, G, X0, P0, SIG)
        assert np.max(dg.l2_error(psi, ex, GRID)) < 1e-12


def test_relative_phase_error_and_exact_visibility():
    """Eq. (33): uncorrected runs have exact visibility but a relative phase error."""
    dt = 0.05
    psi, T = _run(dt, 5.0)
    O = ob.overlap_matrix(psi, GRID)
    Oe = an.overlap_matrix_exact(T, G, X0, P0, SIG)
    dphi = (G[1] ** 2 - G[0] ** 2) * dt**2 * T / 24.0
    assert abs(dg.relative_phase_error(O, Oe) - dphi) < 1e-12
    assert abs(abs(O[0, 1]) - abs(Oe[0, 1])) < 1e-13


def test_branch_diagonal_diagnostics_are_blind_to_the_splitting_error():
    psi, T = _run(0.1, 5.0)
    ex = an.branch_states_exact(GRID.x, T, G, X0, P0, SIG)
    assert np.max(dg.l2_error(psi, ex, GRID)) > 1e-4                      # state is wrong ...
    xc = np.array([an.classical_trajectory(T, g, X0, P0)[0] for g in G])
    assert np.max(np.abs(ob.expectation_x(psi, GRID) - xc)) < 1e-11       # ... centroids are not
    assert np.max(np.abs(ob.norms(psi, GRID) - 1.0)) < 1e-13


def test_second_order_convergence_harmonic_oscillator():
    grid = bg.Grid(40.0, 256)
    w, x0, p0, T = 1.0, 2.0, 1.0, 2 * np.pi
    V = bg.potentials.harmonic_potential(grid.x, 1.0, w)
    F = bg.potentials.harmonic_force(grid.x, 1.0, w)
    psi0 = an.coherent_state_harmonic(grid.x, 0.0, w, x0, p0)[None, :]
    for corr in (False, True):
        errs = []
        dts = [0.1, 0.05, 0.025, 0.0125]
        for dt in dts:
            n = int(round(T / dt))
            prop = bg.StrangPropagator(grid, V, dt, force=F, phase_correction=corr)
            psi = prop.propagate(psi0, n)
            ref = an.coherent_state_harmonic(grid.x, n * dt, w, x0, p0)[None, :]
            errs.append(dg.l2_error(psi, ref, grid)[0])
        slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        assert 1.95 < slope < 2.05, (corr, slope)


def test_phase_correction_does_not_help_for_harmonic_potential():
    """Documents the statement in Section 3.2: the counter-term is for uniform fields only."""
    grid = bg.Grid(40.0, 256)
    w, x0, p0, T, dt = 1.0, 2.0, 1.0, 2 * np.pi, 0.05
    V = bg.potentials.harmonic_potential(grid.x, 1.0, w)
    F = bg.potentials.harmonic_force(grid.x, 1.0, w)
    psi0 = an.coherent_state_harmonic(grid.x, 0.0, w, x0, p0)[None, :]
    n = int(round(T / dt))
    ref = an.coherent_state_harmonic(grid.x, n * dt, w, x0, p0)[None, :]
    e = [dg.l2_error(bg.StrangPropagator(grid, V, dt, force=F, phase_correction=c)
                     .propagate(psi0, n), ref, grid)[0] for c in (False, True)]
    assert e[1] > e[0]


def test_evolve_records_samples():
    prop = bg.StrangPropagator.uniform_field(GRID, G, 0.01)
    out = bg.evolve(prop, bg.branch_states(GRID, 2, X0, P0, SIG), 100, sample_every=25)
    assert np.allclose(out["time"], [0, 0.25, 0.5, 0.75, 1.0])
    assert out["x"].shape == (5, 2) and out["norm"].shape == (5, 2)


def test_input_validation():
    prop = bg.StrangPropagator.uniform_field(GRID, G, 0.01)
    for bad in (np.zeros(GRID.N, complex), np.zeros((3, GRID.N), complex)):
        try:
            prop.step(bad)
        except ValueError:
            continue
        raise AssertionError("wrong shape accepted")
    try:
        bg.StrangPropagator(GRID, np.zeros(GRID.N), 0.01, phase_correction=True)
    except ValueError:
        return
    raise AssertionError("phase_correction without force accepted")
