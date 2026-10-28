"""Phase-space validity criterion (Section 3.4).

A run is faithful while every branch state is negligible near the potential jump at
x = +-L/2 and near the edges of the momentum window |p| < p_max. For the Gaussian
branch states of Section 2.3 this gives Eqs. (37)-(42).
"""

from dataclasses import dataclass, field
from math import erfc, sqrt
from statistics import NormalDist
import warnings

import numpy as np

from .analytic import classical_trajectory, width


def kappa_from_delta(delta):
    """kappa with (1/2) erfc(kappa/sqrt 2) = delta, Eq. (36)."""
    if not 0.0 < delta < 0.5:
        raise ValueError("delta must lie in (0, 0.5)")
    return -NormalDist().inv_cdf(delta)


def delta_from_kappa(kappa):
    return 0.5 * erfc(kappa / sqrt(2.0))


def _first_violation(margin, t0=1.0, n=20001, t_cap=1e12):
    """Smallest t >= 0 with margin(t) < 0 (inf if none below t_cap)."""
    if margin(np.array([0.0]))[0] < 0:
        return 0.0
    H = t0
    while H <= t_cap:
        t = np.linspace(0.0, H, n)
        bad = margin(t) < 0
        if bad.any():
            i = int(np.argmax(bad))
            lo, hi = t[i - 1], t[i]
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if margin(np.array([mid]))[0] < 0:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        H *= 2.0
    return np.inf


def _momentum_limit(p0, g, mass, p_lim):
    """First t >= 0 at which p0 - m g t leaves [-p_lim, p_lim] (Eq. (40))."""
    if p_lim <= 0 or abs(p0) > p_lim:
        return 0.0
    if g == 0:
        return np.inf
    if g > 0:
        return (p_lim + p0) / (mass * g)
    return (p_lim - p0) / (mass * abs(g))


@dataclass
class ValidityReport:
    """Result of the criterion for one grid and set of branches."""
    T_val: float
    T_x: np.ndarray
    T_p: np.ndarray
    kappa: float
    delta: float
    L: float
    N: int
    binding: str
    notes: list = field(default_factory=list)

    def ok(self, T):
        return T <= self.T_val

    def __str__(self):
        lines = [f"validity: kappa = {self.kappa:.2f} (delta = {self.delta:.1e}), L = {self.L}, N = {self.N}",
                 f"  T_x per branch = {np.array2string(self.T_x, precision=4)}",
                 f"  T_p per branch = {np.array2string(self.T_p, precision=4)}",
                 f"  T_val = {self.T_val:.4g} ({self.binding}-limited)"]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def time_of_validity(grid, g_list, x0, p0, sigma, dt=0.0, mass=1.0, delta=1e-12, kappa=None):
    """T_val = min_b min(T_x,b, T_p,b), Eqs. (37)-(40)."""
    hbar = grid.hbar
    kappa = kappa_from_delta(delta) if kappa is None else float(kappa)
    delta = delta_from_kappa(kappa)
    sig_p = hbar / (2.0 * sigma)
    half_L = 0.5 * grid.L
    T_x, T_p = [], []
    for g in g_list:
        def margin(t, g=g):
            xb, _ = classical_trajectory(t, g, x0, p0, mass)
            s = kappa * width(t, sigma, mass, hbar)
            return np.minimum(xb - s + half_L, half_L - s - xb)
        T_x.append(_first_violation(margin))
        p_lim = grid.p_max - kappa * sig_p - 0.5 * mass * abs(g) * dt
        T_p.append(_momentum_limit(p0, g, mass, p_lim))
    T_x, T_p = np.array(T_x), np.array(T_p)
    T_val = float(min(T_x.min(), T_p.min()))
    binding = "space" if T_x.min() <= T_p.min() else "momentum"
    notes = []
    if T_val == 0.0:
        notes.append("the initial state already violates the criterion")
    return ValidityReport(T_val, T_x, T_p, kappa, delta, grid.L, grid.N, binding, notes)


def grid_requirements(T, g_list, x0, p0, sigma, dt=0.0, mass=1.0, hbar=1.0, delta=1e-12,
                      kappa=None, n_samples=10001):
    """Minimum L, momentum window and N_min for a run of duration T, Eqs. (41)-(42).

    Returns a dict with L_min, p_max_min, N_min and the trajectory ranges.
    """
    kappa = kappa_from_delta(delta) if kappa is None else float(kappa)
    t = np.linspace(0.0, T, n_samples)
    xs, ps = [], []
    for g in g_list:
        xb, pb = classical_trajectory(t, g, x0, p0, mass)
        xs.append(xb)
        ps.append(pb)
    xs, ps = np.concatenate(xs), np.concatenate(ps)
    dx_traj, dp_traj = xs.max() - xs.min(), ps.max() - ps.min()
    g_max = max(abs(g) for g in g_list)
    L_min = dx_traj + 2.0 * kappa * width(T, sigma, mass, hbar)
    two_pmax_min = dp_traj + 2.0 * kappa * hbar / (2.0 * sigma) + mass * g_max * dt
    return {
        "kappa": kappa,
        "dx_traj": dx_traj,
        "dp_traj": dp_traj,
        "x_range": (xs.min(), xs.max()),
        "p_range": (ps.min(), ps.max()),
        "L_min": L_min,
        "p_max_min": 0.5 * two_pmax_min,
        "N_min": L_min * two_pmax_min / (2.0 * np.pi * hbar),
    }


def check_run(report, T, action="warn"):
    """Compare a requested duration T with T_val; action in {"warn", "raise", "ignore"}."""
    if report.ok(T) or action == "ignore":
        return True
    msg = (f"requested T = {T:.4g} exceeds the time of validity T_val = {report.T_val:.4g} "
           f"({report.binding}-limited, kappa = {report.kappa:.2f})")
    if action == "raise":
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return False
