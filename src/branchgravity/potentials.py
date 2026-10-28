"""Potentials and the corresponding forces F = -dV/dx.

Forces are only needed for the splitting-phase correction (Section 3.2, Eq. (34)).
"""

import numpy as np


def free_particle(x):
    return np.zeros_like(np.asarray(x, dtype=float))


def linear_gravity_potential(x, mass, g):
    """V(x) = m g x (g > 0 accelerates towards decreasing x)."""
    return mass * g * np.asarray(x, dtype=float)


def linear_gravity_force(x, mass, g):
    return np.full_like(np.asarray(x, dtype=float), -mass * g)


def branch_linear_potentials(x, mass, g_list):
    """(B, N) potentials and forces for uniform fields g_b."""
    V = np.stack([linear_gravity_potential(x, mass, g) for g in g_list])
    F = np.stack([linear_gravity_force(x, mass, g) for g in g_list])
    return V, F


def harmonic_potential(x, mass, omega, x_c=0.0):
    x = np.asarray(x, dtype=float)
    return 0.5 * mass * omega**2 * (x - x_c) ** 2


def harmonic_force(x, mass, omega, x_c=0.0):
    x = np.asarray(x, dtype=float)
    return -mass * omega**2 * (x - x_c)
