"""Strang-Fourier propagator for B uncoupled branches (Section 3.1).

The branch wavefunctions are stored as a (B, N) array. Every operation acts row by
row (elementwise products and FFTs along axis 1), so branches never mix: the block
structure of the evolution operator holds exactly in floating point.
"""

import numpy as np

from .potentials import branch_linear_potentials

ORDERINGS = ("VTV", "TVT")


class StrangPropagator:
    """Second-order symmetric splitting with precomputed phase factors.

    Parameters
    ----------
    grid : Grid
    V : array (N,) or (B, N)
        Branch potentials on the grid.
    dt : float
        Time step.
    mass : float
    ordering : {"VTV", "TVT"}
        "VTV": exp(-iV dt/2h) exp(-iT dt/h) exp(-iV dt/2h)  (default, potential first)
        "TVT": exp(-iT dt/2h) exp(-iV dt/h) exp(-iT dt/2h)
    force : array (N,) or (B, N), optional
        F = -dV/dx, required when phase_correction is True.
    phase_correction : bool
        Add the counter-term of Eq. (34): V -> V + dt^2 F^2/(24 m) for VTV,
        V -> V - dt^2 F^2/(12 m) for TVT. For uniform fields this removes the
        splitting error exactly. For non-uniform potentials it removes only the
        (dV/dx)^2 part of the leading error and is not recommended (for the
        harmonic oscillator it doubles the error).
    """

    def __init__(self, grid, V, dt, mass=1.0, ordering="VTV", force=None,
                 phase_correction=False):
        if ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}")
        V = np.atleast_2d(np.asarray(V, dtype=float))
        if V.shape[1] != grid.N:
            raise ValueError("V must have shape (N,) or (B, N)")

        self.grid = grid
        self.dt = float(dt)
        self.mass = float(mass)
        self.hbar = grid.hbar
        self.ordering = ordering
        self.phase_correction = bool(phase_correction)
        self.V = V

        V_eff = V
        if phase_correction:
            if force is None:
                raise ValueError("phase_correction=True requires the force F = -dV/dx")
            F = np.broadcast_to(np.atleast_2d(np.asarray(force, dtype=float)), V.shape)
            coeff = 1.0 / 24.0 if ordering == "VTV" else -1.0 / 12.0
            V_eff = V + coeff * self.dt**2 * F**2 / self.mass
        self.V_eff = V_eff

        h, m, p = self.hbar, self.mass, grid.p
        if ordering == "VTV":
            self._u = np.exp(-1j * V_eff * self.dt / (2.0 * h))          # (B, N)
            self._eta = np.exp(-1j * p**2 * self.dt / (2.0 * m * h))     # (N,)
        else:
            self._u = np.exp(-1j * V_eff * self.dt / h)
            self._eta = np.exp(-1j * p**2 * self.dt / (4.0 * m * h))

    @property
    def n_branches(self):
        return self._u.shape[0]

    @classmethod
    def uniform_field(cls, grid, g_list, dt, mass=1.0, ordering="VTV",
                      phase_correction=False):
        """Propagator for V_b = m g_b x."""
        V, F = branch_linear_potentials(grid.x, mass, g_list)
        return cls(grid, V, dt, mass=mass, ordering=ordering, force=F,
                   phase_correction=phase_correction)

    def _check(self, psi):
        psi = np.asarray(psi)
        if psi.ndim != 2 or psi.shape != self._u.shape:
            raise ValueError(f"psi must have shape {self._u.shape}")
        return psi

    def step(self, psi):
        """Advance all branches by one time step; returns a new (B, N) array."""
        psi = self._check(psi)
        if self.ordering == "VTV":
            phi = np.fft.fft(self._u * psi, axis=1)
            phi *= self._eta
            out = np.fft.ifft(phi, axis=1)
            out *= self._u
        else:
            phi = np.fft.fft(psi, axis=1)
            phi *= self._eta
            out = np.fft.ifft(phi, axis=1)
            out *= self._u
            phi = np.fft.fft(out, axis=1)
            phi *= self._eta
            out = np.fft.ifft(phi, axis=1)
        return out

    def propagate(self, psi, n_steps):
        psi = np.array(self._check(psi), dtype=complex)
        for _ in range(int(n_steps)):
            psi = self.step(psi)
        return psi


def evolve(propagator, psi0, n_steps, sample_every=1, observables=None,
           store_states=False):
    """Propagate and record observables every `sample_every` steps.

    Parameters
    ----------
    observables : dict name -> callable(psi, t) -> array, optional
        Defaults to branch norms, <x> and <p>.

    Returns
    -------
    dict with "time" (n_samples,), one entry per observable (n_samples, ...),
    and "states" (n_samples, B, N) if store_states.
    """
    from . import observables as obs

    grid = propagator.grid
    if observables is None:
        observables = {
            "norm": lambda psi, t: obs.norms(psi, grid),
            "x": lambda psi, t: obs.expectation_x(psi, grid),
            "p": lambda psi, t: obs.expectation_p(psi, grid),
        }
    n_steps = int(n_steps)
    sample_every = int(sample_every)
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")

    psi = np.array(propagator._check(psi0), dtype=complex)
    times, records, states = [], {k: [] for k in observables}, []

    def record(step):
        t = step * propagator.dt
        times.append(t)
        for k, f in observables.items():
            records[k].append(f(psi, t))
        if store_states:
            states.append(psi.copy())

    record(0)
    for n in range(1, n_steps + 1):
        psi = propagator.step(psi)
        if n % sample_every == 0 or n == n_steps:
            record(n)

    out = {"time": np.array(times)}
    out.update({k: np.array(v) for k, v in records.items()})
    if store_states:
        out["states"] = np.array(states)
    out["final_state"] = psi
    return out
