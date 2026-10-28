"""Figure 3: convergence beyond uniform fields (Section 4.4).

(a) Temporal convergence: harmonic coherent state against its exact solution, with and
    without the counter-term of Eq. (34), and a quartic oscillator against a fine-step
    reference. Dashed: slope 2.
(b) Spatial convergence at fixed L: uniform-field Gaussian against Eq. (16) at t = T as
    a function of N. Vertical lines: N at which the momentum window satisfies Eq. (38)
    for delta = 1e-12 and 1e-28 (L2 tolerances ~1e-6 and ~1e-14).
"""

import numpy as np

from common import (AQUA, BLUE, FLOOR, ORANGE, REF, WIDE, floor, load_figure_config,
                    panel_label, plt, save, setup_style, sim_config)
import branchgravity as bg
from branchgravity import analytic, diagnostics, potentials
from branchgravity.config import run

NAME = "fig3_convergence"


def temporal(cfg):
    T, dts = cfg["T"], cfg["dts"]
    h = cfg["harmonic"]
    grid = bg.Grid(h["L"], h["N"])
    V = potentials.harmonic_potential(grid.x, 1.0, h["omega"])
    F = potentials.harmonic_force(grid.x, 1.0, h["omega"])
    psi0 = analytic.coherent_state_harmonic(grid.x, 0.0, h["omega"], h["x0"], h["p0"])[None]
    ref = analytic.coherent_state_harmonic(grid.x, T, h["omega"], h["x0"], h["p0"])[None]
    harm = {}
    for corr in (False, True):
        harm[corr] = np.array([
            diagnostics.l2_error(bg.StrangPropagator(grid, V, dt, force=F, phase_correction=corr)
                                 .propagate(psi0, int(round(T / dt))), ref, grid)[0]
            for dt in dts])

    q = cfg["quartic"]
    gq = bg.Grid(q["L"], q["N"])
    Vq = q["lam"] * gq.x**4
    psi0q = bg.gaussian_wavepacket(gq.x, q["x0"], q["p0"], q["sigma"])[None].astype(complex)
    refq = bg.StrangPropagator(gq, Vq, q["dt_ref"]).propagate(psi0q, int(round(T / q["dt_ref"])))
    quart = np.array([
        diagnostics.l2_error(bg.StrangPropagator(gq, Vq, dt).propagate(psi0q, int(round(T / dt))),
                             refq, gq)[0] for dt in dts])
    return np.array(dts), harm, quart


def required_N(L, p_extreme, sigma, g_max, dt, delta, mass=1.0, hbar=1.0):
    """Smallest N at fixed L for which the momentum window satisfies Eq. (38) at t = 0..T."""
    kappa = bg.kappa_from_delta(delta)
    p_needed = p_extreme + kappa * hbar / (2 * sigma) + 0.5 * mass * g_max * dt
    return L * p_needed / (np.pi * hbar)


def spatial(cfg):
    errs = []
    for N in cfg["N"]:
        out = run(sim_config(cfg["base"], {
            "grid": {"N": N},
            "time": {"dt": cfg["dt"], "T": cfg["T"], "sample_every": int(round(cfg["T"] / cfg["dt"]))},
            "validity": {"action": "ignore"}}))
        errs.append(out["l2_error"][-1].max())
    ph = out["simulation"].config.physics
    L = out["simulation"].config.grid.L
    p_ext = max(abs(analytic.classical_trajectory(t, g, ph.x0, ph.p0)[1])
                for g in ph.g for t in (0.0, cfg["T"]))
    n_req = {d: required_N(L, p_ext, ph.sigma, max(map(abs, ph.g)), cfg["dt"], d)
             for d in (1e-12, 1e-28)}
    return np.array(cfg["N"]), np.array(errs), n_req, L


def main():
    cfg = load_figure_config(NAME)
    dts, harm, quart = temporal(cfg["temporal"])
    Ns, errs, n_req, L = spatial(cfg["spatial"])

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE)
    ax1.plot(dts, harm[False], color=BLUE, marker="o", label="harmonic, exact reference")
    ax1.plot(dts, harm[True], color=BLUE, marker="o", mfc="white", ls=":",
             label="harmonic, with Eq. (34) counter-term")
    ax1.plot(dts, quart, color=ORANGE, marker="s", label="quartic, fine-step reference")
    ax1.plot(dts, 0.35 * harm[False][-1] * (dts / dts[-1]) ** 2, color=REF, lw=0.8, ls="--",
             label="slope 2", zorder=0)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\Delta t$")
    ax1.set_ylabel(r"$\|\psi(T) - \psi^{\rm ref}(T)\|$")
    ax1.legend(loc="lower right")
    panel_label(ax1, "(a)")

    ax2.plot(Ns, floor(errs), color=AQUA, marker="D", ms=3.8, label=r"error at $t = T$")
    styles = {1e-12: ("--", r"Eq. (38), $\delta = 10^{-12}$"),
              1e-28: (":", r"Eq. (38), $\delta = 10^{-28}$")}
    for d, n in n_req.items():
        ls, lab = styles[d]
        ax2.axvline(n, color=REF, lw=0.9, ls=ls, label=lab)
    ax2.axhline(1e-6, color=FLOOR, lw=0.6, ls="--", zorder=0)
    ax2.axhline(1e-14, color=FLOOR, lw=0.6, ls=":", zorder=0)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-15, 1e3)
    ax2.set_xlabel(rf"$N$  ($L = {L:g}$)")
    ax2.set_ylabel(r"$\max_b \|\psi_b(T) - \psi_b^{\rm exact}(T)\|$")
    ax2.legend(loc="upper right")
    panel_label(ax2, "(b)")
    fig.tight_layout(w_pad=2.0)

    asym = dts <= 0.05 + 1e-12      # asymptotic range used for the fitted orders

    def slope(e):
        return float(np.polyfit(np.log(dts[asym]), np.log(e[asym]), 1)[0])
    derived = {
        "slope_harmonic": slope(harm[False]),
        "slope_harmonic_counterterm": slope(harm[True]),
        "slope_quartic": slope(quart),
        "slope_fit_range": [float(dts[asym].min()), float(dts[asym].max())],
        "ratio_counterterm_over_plain": (harm[True] / harm[False]).tolist(),
        "N_required": {str(k): v for k, v in n_req.items()},
        "spatial_errors": dict(zip(map(str, Ns.tolist()), errs.tolist())),
    }
    data = {"dts": dts, "harmonic": harm[False], "harmonic_counterterm": harm[True],
            "quartic": quart, "N": Ns, "spatial_error": errs}
    save(fig, NAME, data, cfg, derived)


if __name__ == "__main__":
    main()
