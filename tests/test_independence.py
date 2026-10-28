"""Sharp tests of branch independence (replace the per-branch norm test of the draft).

Norm conservation per branch is implied by unitarity of each block and cannot detect an
implementation that mixes rows of the (B, N) array. These tests can.
"""

import numpy as np

import branchgravity as bg
from branchgravity import observables as ob

GRID = bg.Grid(100.0, 2048)
G3 = [0.8, 1.2, -0.5]


def test_zero_branch_stays_exactly_zero():
    prop = bg.StrangPropagator.uniform_field(GRID, G3, 0.01, phase_correction=True)
    psi = bg.branch_states(GRID, 3, -10.0, 5.0, 1.0)
    psi[1] = 0.0
    psi = prop.propagate(psi, 500)
    assert np.all(psi[1] == 0.0)


def test_branch_alone_is_bitwise_identical_to_branch_with_others():
    psi_all = bg.branch_states(GRID, 3, -10.0, 5.0, 1.0)
    psi_all[2] *= np.exp(0.3j * GRID.x)        # make the rows genuinely different
    multi = bg.StrangPropagator.uniform_field(GRID, G3, 0.01, phase_correction=True)
    out_all = multi.propagate(psi_all, 500)
    for b, g in enumerate(G3):
        single = bg.StrangPropagator.uniform_field(GRID, [g], 0.01, phase_correction=True)
        out_b = single.propagate(psi_all[b:b + 1], 500)
        assert np.array_equal(out_all[b], out_b[0]), b


def test_branch_order_is_irrelevant_bitwise():
    psi = bg.branch_states(GRID, 3, -10.0, 5.0, 1.0)
    a = bg.StrangPropagator.uniform_field(GRID, G3, 0.01).propagate(psi, 200)
    perm = [2, 0, 1]
    b = bg.StrangPropagator.uniform_field(GRID, [G3[i] for i in perm], 0.01).propagate(psi[perm], 200)
    assert np.array_equal(a[perm], b)


def test_overlap_matrix_is_hermitian_with_norms_on_diagonal():
    prop = bg.StrangPropagator.uniform_field(GRID, G3, 0.01)
    psi = prop.propagate(bg.branch_states(GRID, 3, -10.0, 5.0, 1.0), 300)
    O = ob.overlap_matrix(psi, GRID)
    assert np.allclose(O, O.conj().T, atol=1e-15)
    assert np.allclose(np.diag(O).real, ob.norms(psi, GRID), atol=1e-15)


def test_branch_density_matrix_is_a_state():
    prop = bg.StrangPropagator.uniform_field(GRID, G3, 0.01)
    psi = prop.propagate(bg.branch_states(GRID, 3, -10.0, 5.0, 1.0), 300)
    w = np.array([0.6, 0.48j, 0.64])
    rho = ob.branch_density_matrix(w, ob.overlap_matrix(psi, GRID))
    assert abs(np.trace(rho) - 1.0) < 1e-13
    assert np.allclose(rho, rho.conj().T, atol=1e-15)
    assert np.min(np.linalg.eigvalsh(rho)) > -1e-13
