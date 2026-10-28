"""Closed-form reference results (Sections 2.3 and 3.2).

Uniform field V = m g x, Gaussian initial state of width sigma, centre x0, momentum p0.
"""

import numpy as np


def classical_trajectory(t, g, x0, p0, mass=1.0):
    """x(t) = x0 + p0 t/m - g t^2/2,  p(t) = p0 - m g t   (Eq. (13))."""
    t = np.asarray(t, dtype=float)
    return x0 + p0 * t / mass - 0.5 * g * t**2, p0 - mass * g * t


def classical_action(t, g, x0, p0, mass=1.0):
    """S(t) = p0^2 t/2m - p0 g t^2 + m g^2 t^3/3 - m g x0 t   (Eq. (14))."""
    t = np.asarray(t, dtype=float)
    return (p0**2 * t / (2.0 * mass) - p0 * g * t**2
            + mass * g**2 * t**3 / 3.0 - mass * g * x0 * t)


def width(t, sigma, mass=1.0, hbar=1.0):
    """sigma_x(t) = sigma sqrt(1 + tau^2), tau = hbar t / (2 m sigma^2)."""
    tau = hbar * np.asarray(t, dtype=float) / (2.0 * mass * sigma**2)
    return sigma * np.sqrt(1.0 + tau**2)


def gaussian_uniform_field(x, t, g, x0, p0, sigma, mass=1.0, hbar=1.0, extended=False):
    """Exact solution of the TDSE with V = m g x, Eq. (16).

    With extended=True the phase is accumulated and reduced modulo 2 pi in
    np.longdouble before the exponential is taken. Use it when the phase reaches
    ~1e4 rad or more (long falls, large momenta): in double precision the reference
    itself then carries an error of order eps * |phase|. The gain requires a platform
    whose long double has a 64-bit mantissa (x86 Linux/macOS); elsewhere np.longdouble
    is double and the option has no effect.
    """
    if not extended:
        x = np.asarray(x, dtype=float)
        tau = hbar * t / (2.0 * mass * sigma**2)
        xc, pc = classical_trajectory(t, g, x0, p0, mass)
        S = classical_action(t, g, x0, p0, mass)
        pref = (2.0 * np.pi * sigma**2) ** (-0.25) / np.sqrt(1.0 + 1j * tau)
        return pref * np.exp(-(x - xc) ** 2 / (4.0 * sigma**2 * (1.0 + 1j * tau))
                             + 1j * (pc * (x - xc) + S) / hbar)

    ld = np.longdouble
    x = np.asarray(x, dtype=ld)
    t, g, x0, p0, s, m, h = (ld(v) for v in (t, g, x0, p0, sigma, mass, hbar))
    tau = h * t / (2 * m * s * s)
    xc = x0 + p0 * t / m - g * t * t / 2
    pc = p0 - m * g * t
    S = p0 * p0 * t / (2 * m) - p0 * g * t * t + m * g * g * t**3 / 3 - m * g * x0 * t
    d2 = (x - xc) ** 2 / (4 * s * s * (1 + tau * tau))
    envelope = -d2                                   # real part of the exponent
    phase = d2 * tau + (pc * (x - xc) + S) / h       # imaginary part of the exponent
    phase = np.mod(phase, 2 * ld(np.pi))
    pref = (2.0 * np.pi * sigma**2) ** (-0.25) / np.sqrt(1.0 + 1j * float(tau))
    return pref * np.exp(envelope.astype(float)) * np.exp(1j * phase.astype(float))


def branch_states_exact(x, t, g_list, x0, p0, sigma, mass=1.0, hbar=1.0, extended=False):
    return np.stack([gaussian_uniform_field(x, t, g, x0, p0, sigma, mass, hbar, extended)
                     for g in g_list])


def visibility(t, g1, g2, sigma, mass=1.0, hbar=1.0):
    """|O_12(t)|, Eq. (18)."""
    t = np.asarray(t, dtype=float)
    dg = g2 - g1
    return np.exp(-(mass * sigma * dg * t) ** 2 / (2.0 * hbar**2)
                  - (dg * t**2) ** 2 / (32.0 * sigma**2))


def relative_phase(t, g1, g2, x0, p0, mass=1.0, hbar=1.0):
    """arg O_12(t), Eq. (19) (not wrapped to (-pi, pi])."""
    t = np.asarray(t, dtype=float)
    dg, gbar = g2 - g1, 0.5 * (g1 + g2)
    return -mass * dg / hbar * (x0 * t + p0 * t**2 / (2.0 * mass) - gbar * t**3 / 6.0)


def overlap_uniform_field(t, g1, g2, x0, p0, sigma, mass=1.0, hbar=1.0):
    """O_12(t) = <psi_1(t)|psi_2(t)>, Eqs. (18)-(19)."""
    return (visibility(t, g1, g2, sigma, mass, hbar)
            * np.exp(1j * relative_phase(t, g1, g2, x0, p0, mass, hbar)))


def overlap_matrix_exact(t, g_list, x0, p0, sigma, mass=1.0, hbar=1.0):
    B = len(g_list)
    O = np.empty((B, B), dtype=complex)
    for i, gi in enumerate(g_list):
        for j, gj in enumerate(g_list):
            O[i, j] = overlap_uniform_field(t, gi, gj, x0, p0, sigma, mass, hbar)
    return O


def splitting_phase(t, g, dt, mass=1.0, hbar=1.0, ordering="VTV"):
    """Global phase eps_b(t) of the uncorrected Strang scheme, Eq. (32):
    psi_Strang = exp(i eps) psi_exact. VTV: +m g^2 dt^2 t/(24 hbar); TVT: -m g^2 dt^2 t/(12 hbar)."""
    c = 1.0 / 24.0 if ordering == "VTV" else -1.0 / 12.0
    return c * mass * g**2 * dt**2 * np.asarray(t, dtype=float) / hbar


def coherent_state_harmonic(x, t, omega, x0, p0, mass=1.0, hbar=1.0):
    """Exact coherent state of V = m omega^2 x^2 / 2 (width sigma^2 = hbar / (2 m omega)):

        psi = (2 pi sigma^2)^(-1/4) exp[-(x-q)^2/(4 sigma^2) + i(p (x-q) + S)/hbar - i omega t/2],

    with (q, p) the classical trajectory and S = (q p - x0 p0)/2 the classical action.
    """
    x = np.asarray(x, dtype=float)
    sigma2 = hbar / (2.0 * mass * omega)
    c, s = np.cos(omega * t), np.sin(omega * t)
    q = x0 * c + p0 / (mass * omega) * s
    p = p0 * c - mass * omega * x0 * s
    S = 0.5 * (q * p - x0 * p0)
    return (2.0 * np.pi * sigma2) ** (-0.25) * np.exp(
        -(x - q) ** 2 / (4.0 * sigma2) + 1j * (p * (x - q) + S) / hbar - 0.5j * omega * t)
