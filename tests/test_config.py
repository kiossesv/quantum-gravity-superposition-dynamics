"""Shipped configurations load, build and reproduce their documented behaviour."""

import json
import tempfile
from pathlib import Path

import numpy as np

from branchgravity import diagnostics
from branchgravity.config import SimulationConfig, build, run, summarize

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_all_configs_load_and_build():
    paths = sorted(CONFIGS.glob("*.json"))
    assert paths
    for p in paths:
        cfg = SimulationConfig.load(p)
        sim = build(cfg)
        assert sim.psi0.shape == (len(cfg.physics.g), cfg.grid.N)
        assert cfg.name == p.stem


def test_roundtrip():
    cfg = SimulationConfig.load(CONFIGS / "two_branch_lab.json")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "c.json"
        cfg.save(path)
        assert SimulationConfig.load(path) == cfg
        assert json.loads(path.read_text())["grid"]["N"] == 2048


def test_unknown_section_rejected():
    try:
        SimulationConfig.from_dict({"grid": {"L": 1, "N": 8}, "time": {"dt": 1, "T": 1}, "bogus": {}})
    except ValueError:
        return
    raise AssertionError("unknown section accepted")


def test_lab_and_freefall_configs_agree():
    lab = run(SimulationConfig.load(CONFIGS / "two_branch_lab.json"))
    ff = run(SimulationConfig.load(CONFIGS / "two_branch_freefall.json"))
    assert np.allclose(lab["time"], ff["time"])
    assert np.max(np.abs(lab["overlap"] - ff["overlap"])) < 1e-12
    assert np.max(np.abs(lab["x"] - ff["x"])) < 1e-12
    assert summarize(lab)["max_l2_error"] < 1e-12
    assert summarize(ff)["max_l2_error"] < 1e-9


def test_uncorrected_config_shows_eq33_phase_error():
    out = run(SimulationConfig.load(CONFIGS / "two_branch_lab_uncorrected.json"))
    s = summarize(out)
    predicted = (1.2**2 - 0.8**2) * 0.01**2 * 5.0 / 24.0
    assert abs(s["max_relative_phase_error"] - predicted) < 1e-12
    assert s["max_visibility_error"] < 1e-13


def test_earth_gravity_freefall_config():
    """Common fall dominated: co-moving grid with 1024 points reproduces the exact branch
    states and overlaps; the phase is checked while the visibility is resolvable."""
    out = run(SimulationConfig.load(CONFIGS / "earth_gravity_freefall.json"))
    s = summarize(out)
    assert s["max_l2_error"] < 1e-8
    assert s["phase_resolved_until"] >= 6.0
    # The overlap itself is accurate to ~roundoff, so the phase error scales as eps/V.
    O, Oe = out["overlap"], out["overlap_exact"]
    assert np.max(np.abs(O[:, 0, 1] - Oe[:, 0, 1])) < 1e-13
    V = np.abs(Oe[:, 0, 1])
    dphi = np.abs(diagnostics.relative_phase_error(O, Oe))
    mask = V > 1e-12
    assert np.max(dphi[mask] * V[mask]) < 1e-13
