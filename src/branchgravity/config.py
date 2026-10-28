"""JSON run configurations: every result in the paper is generated from a stored config.

Schema (all keys optional except grid and time):

{
  "name": "...", "description": "...",
  "physics":    {"hbar": 1, "mass": 1, "sigma": 1, "x0": 0, "p0": 0, "g": [0.8, 1.2], "weights": null},
  "grid":       {"L": 100.0, "N": 2048},
  "time":       {"dt": 0.01, "T": 5.0, "sample_every": 10},
  "propagator": {"frame": "lab" | "freefall", "ordering": "VTV" | "TVT",
                 "phase_correction": true, "g_ref": null},
  "validity":   {"delta": 1e-12, "action": "warn" | "raise" | "ignore"},
  "diagnostics": {"extended_reference": false}
}

"extended_reference" evaluates the exact reference of Eq. (16) with the phase reduced in
long double precision (see analytic.gaussian_uniform_field); use it for runs whose phases
reach ~1e4 rad, where a double-precision reference is itself in error by eps * |phase|.

In the "freefall" frame, "grid" describes the co-moving grid centred on the reference
trajectory. All reported observables are converted to the laboratory frame.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np

from . import analytic, diagnostics, observables as obs
from .freefall import FreeFallFrame
from .grid import Grid
from .propagators import StrangPropagator, evolve
from .states import branch_states
from .validity import check_run, time_of_validity


@dataclass
class PhysicsConfig:
    hbar: float = 1.0
    mass: float = 1.0
    sigma: float = 1.0
    x0: float = 0.0
    p0: float = 0.0
    g: list = field(default_factory=lambda: [1.0])
    weights: list = None


@dataclass
class GridConfig:
    L: float
    N: int


@dataclass
class TimeConfig:
    dt: float
    T: float
    sample_every: int = 1


@dataclass
class PropagatorConfig:
    frame: str = "lab"
    ordering: str = "VTV"
    phase_correction: bool = True
    g_ref: float = None


@dataclass
class ValidityConfig:
    delta: float = 1e-12
    action: str = "warn"


@dataclass
class DiagnosticsConfig:
    extended_reference: bool = False


@dataclass
class SimulationConfig:
    grid: GridConfig
    time: TimeConfig
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    propagator: PropagatorConfig = field(default_factory=PropagatorConfig)
    validity: ValidityConfig = field(default_factory=ValidityConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    name: str = "unnamed"
    description: str = ""

    @classmethod
    def from_dict(cls, d):
        known = {"grid", "time", "physics", "propagator", "validity", "diagnostics", "name",
                 "description"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(
            grid=GridConfig(**d["grid"]),
            time=TimeConfig(**d["time"]),
            physics=PhysicsConfig(**d.get("physics", {})),
            propagator=PropagatorConfig(**d.get("propagator", {})),
            validity=ValidityConfig(**d.get("validity", {})),
            diagnostics=DiagnosticsConfig(**d.get("diagnostics", {})),
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
        )

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @property
    def n_steps(self):
        n = self.time.T / self.time.dt
        if abs(n - round(n)) > 1e-9 * max(1.0, n):
            raise ValueError("T must be an integer multiple of dt")
        return int(round(n))


@dataclass
class Simulation:
    config: SimulationConfig
    grid: Grid
    propagator: StrangPropagator
    psi0: np.ndarray
    frame: FreeFallFrame = None


def build(cfg):
    """Construct grid, propagator and initial branch states from a config."""
    ph, pc = cfg.physics, cfg.propagator
    grid = Grid(L=cfg.grid.L, N=cfg.grid.N, hbar=ph.hbar)
    B = len(ph.g)
    if pc.frame == "lab":
        prop = StrangPropagator.uniform_field(grid, ph.g, cfg.time.dt, mass=ph.mass,
                                              ordering=pc.ordering,
                                              phase_correction=pc.phase_correction)
        psi0 = branch_states(grid, B, ph.x0, ph.p0, ph.sigma)
        return Simulation(cfg, grid, prop, psi0)
    if pc.frame == "freefall":
        frame = FreeFallFrame.for_branches(ph.g, ph.x0, ph.p0, ph.mass, ph.hbar, pc.g_ref)
        prop = frame.propagator(grid, ph.g, cfg.time.dt, ordering=pc.ordering,
                                phase_correction=pc.phase_correction)
        psi0 = frame.initial_gaussian(grid, ph.sigma, B)
        return Simulation(cfg, grid, prop, psi0, frame)
    raise ValueError("propagator.frame must be 'lab' or 'freefall'")


def validity_report(sim):
    cfg, ph = sim.config, sim.config.physics
    if sim.frame is None:
        return time_of_validity(sim.grid, ph.g, ph.x0, ph.p0, ph.sigma, cfg.time.dt,
                                ph.mass, delta=cfg.validity.delta)
    return time_of_validity(sim.grid, sim.frame.residual(ph.g), 0.0, 0.0, ph.sigma,
                            cfg.time.dt, ph.mass, delta=cfg.validity.delta)


def run(cfg, store_states=False):
    """Run a configuration; observables are returned in the laboratory frame.

    Recorded per sample: branch norms, <x>, <p>, the overlap matrix and, since all
    configs use uniform fields, the L2 error of every branch against Eq. (16).
    """
    if not isinstance(cfg, SimulationConfig):
        cfg = SimulationConfig.from_dict(cfg)
    sim = build(cfg)
    ph = cfg.physics
    report = validity_report(sim)
    check_run(report, cfg.time.T, action=cfg.validity.action)

    grid, frame = sim.grid, sim.frame
    ext = cfg.diagnostics.extended_reference
    if frame is None:
        def exact(t):
            return analytic.branch_states_exact(grid.x, t, ph.g, ph.x0, ph.p0, ph.sigma,
                                                ph.mass, ph.hbar, extended=ext)
        to_lab_x = lambda v, t: v
        to_lab_p = lambda v, t: v
        to_lab_O = lambda O, t: O
    else:
        res = frame.residual(ph.g)

        def exact(t):
            return analytic.branch_states_exact(grid.x, t, res, 0.0, 0.0, ph.sigma,
                                                ph.mass, ph.hbar, extended=ext)
        to_lab_x = frame.lab_x
        to_lab_p = frame.lab_p
        to_lab_O = lambda O, t: frame.lab_overlaps(O, t, ph.g)

    observables = {
        "norm": lambda psi, t: obs.norms(psi, grid),
        "x": lambda psi, t: to_lab_x(obs.expectation_x(psi, grid), t),
        "p": lambda psi, t: to_lab_p(obs.expectation_p(psi, grid), t),
        "overlap": lambda psi, t: to_lab_O(obs.overlap_matrix(psi, grid), t),
        "l2_error": lambda psi, t: diagnostics.l2_error(psi, exact(t), grid),
    }
    out = evolve(sim.propagator, sim.psi0, cfg.n_steps, cfg.time.sample_every,
                 observables, store_states=store_states)

    t = out["time"]
    out["x_exact"] = np.stack([analytic.classical_trajectory(t, g, ph.x0, ph.p0, ph.mass)[0]
                               for g in ph.g], axis=1)
    out["p_exact"] = np.stack([analytic.classical_trajectory(t, g, ph.x0, ph.p0, ph.mass)[1]
                               for g in ph.g], axis=1)
    out["overlap_exact"] = np.stack([analytic.overlap_matrix_exact(tt, ph.g, ph.x0, ph.p0,
                                                                   ph.sigma, ph.mass, ph.hbar)
                                     for tt in t])
    w = ph.weights if ph.weights is not None else [1.0 / np.sqrt(len(ph.g))] * len(ph.g)
    out["rho_branch"] = np.stack([obs.branch_density_matrix(w, O) for O in out["overlap"]])
    out["validity"] = report
    out["simulation"] = sim
    return out


def summarize(out, visibility_floor=1e-8):
    """Headline numbers for a run (used by scripts/run.py).

    Relative-phase errors are evaluated only where |O_01| > visibility_floor."""
    norm_dev = np.max(np.abs(out["norm"] - out["norm"][0]))
    O, Oe = out["overlap"], out["overlap_exact"]
    B = O.shape[1]
    s = {
        "max_norm_drift": float(norm_dev),
        "max_l2_error": float(np.max(out["l2_error"])),
        "max_x_error": float(np.max(np.abs(out["x"] - out["x_exact"]))),
        "max_p_error": float(np.max(np.abs(out["p"] - out["p_exact"]))),
    }
    if B > 1:
        s["max_visibility_error"] = float(np.max(np.abs(np.abs(O[:, 0, 1]) - np.abs(Oe[:, 0, 1]))))
        # The phase of O_01 is only defined while the visibility is well above roundoff.
        resolved = np.abs(Oe[:, 0, 1]) > visibility_floor
        dphi = np.abs(diagnostics.relative_phase_error(O, Oe, 0, 1))[resolved]
        s["max_relative_phase_error"] = float(dphi.max()) if dphi.size else float("nan")
        s["phase_resolved_until"] = float(out["time"][resolved].max()) if resolved.any() else 0.0
    return s
