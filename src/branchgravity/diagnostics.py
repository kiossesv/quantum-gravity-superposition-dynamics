"""Wavefunction-level comparison with a reference state (one value per branch)."""

import numpy as np


def _as2d(a):
    a = np.asarray(a)
    return a[None, :] if a.ndim == 1 else a


def l2_error(psi, ref, grid):
    """||psi_b - ref_b||_N."""
    d = _as2d(psi) - _as2d(ref)
    out = np.sqrt(np.sum(np.abs(d) ** 2, axis=1) * grid.dx)
    return out[0] if np.asarray(psi).ndim == 1 else out


def inner(ref, psi, grid):
    """<ref_b|psi_b>_N per branch."""
    out = np.sum(np.conj(_as2d(ref)) * _as2d(psi), axis=1) * grid.dx
    return out[0] if np.asarray(psi).ndim == 1 else out


def fidelity(psi, ref, grid):
    """|<ref_b|psi_b>| / (||ref_b|| ||psi_b||)."""
    ov = np.atleast_1d(inner(ref, psi, grid))
    nr = np.sqrt(np.sum(np.abs(_as2d(ref)) ** 2, axis=1) * grid.dx)
    npsi = np.sqrt(np.sum(np.abs(_as2d(psi)) ** 2, axis=1) * grid.dx)
    out = np.abs(ov) / (nr * npsi)
    return out[0] if np.asarray(psi).ndim == 1 else out


def phase_error(psi, ref, grid):
    """arg <ref_b|psi_b>: the global phase of psi relative to the reference."""
    return np.angle(inner(ref, psi, grid))


def relative_phase_error(O_num, O_ref, b=0, bp=1):
    """Error in arg O_bb', wrapped to (-pi, pi]."""
    return np.angle(O_num[..., b, bp] * np.conj(O_ref[..., b, bp]))
