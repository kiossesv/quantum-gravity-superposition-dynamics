"""BranchGravitySolver: Strang-Fourier propagation of branch-resolved quantum states
in superposed uniform gravitational fields."""

__version__ = "0.2.0.dev0"

from .grid import Grid, create_grids
from .states import gaussian_wavepacket, branch_states, normalize
from .propagators import StrangPropagator, evolve
from .freefall import FreeFallFrame, fourier_interpolate
from .validity import time_of_validity, grid_requirements, check_run, kappa_from_delta
from . import analytic, diagnostics, observables, potentials

__all__ = [
    "Grid", "create_grids", "gaussian_wavepacket", "branch_states", "normalize",
    "StrangPropagator", "evolve", "FreeFallFrame", "fourier_interpolate",
    "time_of_validity", "grid_requirements", "check_run", "kappa_from_delta",
    "analytic", "diagnostics", "observables", "potentials",
]
