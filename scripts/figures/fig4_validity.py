"""Figure 4: the phase-space validity criterion predicts the onset of failure (Section 4.5).

(a) L2 error against Eq. (16) versus t / T_val for four representative cases
    (delta = tol^2 with tol = 1e-6, i.e. kappa = 7.03).
(b) Observed onset (first time the L2 error exceeds tol) versus the predicted T_val
    for randomly drawn configurations, split by the binding window; for comparison,
    the drift-only estimate t_max = p_max/(m|g|) of the original draft.
"""

import numpy as np

from common import (AQUA, BLUE, FLOOR, ORANGE, REF, SERIES, WIDE, floor, load_figure_config,
                    panel_label, plt, save, setup_style)
import branchgravity as bg
from branchgravity import analytic, diagnostics

NAME = "fig4_validity"


def error_history(grid, g, x0, p0, sigma, dt, t_stop, tol=None):
    """L2 error after every step until t_stop (or until it first exceeds tol)."""
    prop = bg.StrangPropagator.uniform_field(grid, [g], dt, phase_correction=True)
    psi = bg.branch_states(grid, 1, x0, p0, sigma)
    ts, es = [0.0], [0.0]
    n = 0
    while n * dt < t_stop:
        psi = prop.step(psi)
        n += 1
        ref = analytic.gaussian_uniform_field(grid.x, n * dt, g, x0, p0, sigma)[None]
        e = diagnostics.l2_error(psi, ref, grid)[0]
        ts.append(n * dt)
        es.append(e)
        if tol is not None and e > tol:
            break
    return np.array(ts), np.array(es)


def draw_cases(rcfg):
    rng = np.random.default_rng(rcfg["seed"])
    tol = None
    cases = []
    while len(cases) < rcfg["n_cases"]:
        L = rng.uniform(*rcfg["L"])
        N = int(rng.choice(rcfg["N"]))
        grid = bg.Grid(L, N)
        sigma = rng.uniform(*rcfg["sigma"])
        g = rng.uniform(*rcfg["abs_g"]) * rng.choice([-1.0, 1.0])
        x0 = rng.uniform(*rcfg["x0_fraction_of_L"]) * L
        p0 = rng.uniform(*rcfg["p0_fraction_of_pmax"]) * grid.p_max
        cases.append(dict(L=L, N=N, sigma=sigma, g=g, x0=x0, p0=p0))
    return cases


def main():
    cfg = load_figure_config(NAME)
    tol = cfg["tolerance"]
    delta = tol**2

    # (a) representative cases
    curves = []
    for c in cfg["cases"]:
        grid = bg.Grid(c["L"], c["N"])
        rep = bg.time_of_validity(grid, [c["g"]], c["x0"], c["p0"], c["sigma"], 0.005, delta=delta)
        t, e = error_history(grid, c["g"], c["x0"], c["p0"], c["sigma"], 0.005, 1.3 * rep.T_val)
        onset = t[np.argmax(e > tol)] if np.any(e > tol) else np.inf
        curves.append(dict(c, T_val=rep.T_val, binding=rep.binding, t=t, e=e, onset=onset,
                           t_max_old=grid.p_max / abs(c["g"])))

    # (b) random cases
    rcfg = cfg["random"]
    results = []
    for c in draw_cases(rcfg):
        grid = bg.Grid(c["L"], c["N"])
        dt_probe = 0.0
        rep = bg.time_of_validity(grid, [c["g"]], c["x0"], c["p0"], c["sigma"], dt_probe, delta=delta)
        if rep.T_val <= 0:
            continue
        dt = rep.T_val / rcfg["steps_per_T_val"]
        rep = bg.time_of_validity(grid, [c["g"]], c["x0"], c["p0"], c["sigma"], dt, delta=delta)
        if rep.T_val <= 0:
            continue
        t, e = error_history(grid, c["g"], c["x0"], c["p0"], c["sigma"], dt, 2.0 * rep.T_val, tol)
        onset = t[-1] if e[-1] > tol else np.inf
        results.append(dict(c, dt=dt, T_val=rep.T_val, binding=rep.binding, onset=onset,
                            t_max_old=grid.p_max / abs(c["g"])))

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE)
    for i, cv in enumerate(curves):
        ax1.plot(cv["t"] / cv["T_val"], floor(cv["e"]), color=SERIES[i], lw=1.2,
                 label=cv["label"])
    ax1.axvline(1.0, color=REF, lw=0.9, ls="--", label=r"$t = T_{\rm val}$, Eq. (39)")
    ax1.axhline(tol, color=FLOOR, lw=0.6, ls="--", zorder=0)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-16, 10)
    ax1.set_xlim(0, 1.3)
    ax1.set_xlabel(r"$t / T_{\rm val}$")
    ax1.set_ylabel(r"$\|\psi(t) - \psi^{\rm exact}(t)\|$")
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    panel_label(ax1, "(a)")

    finite = [r for r in results if np.isfinite(r["onset"])]
    lo = 0.5 * min(min(r["T_val"] for r in finite), min(r["onset"] for r in finite))
    hi = 2.0 * max(max(r["t_max_old"] for r in finite), max(r["onset"] for r in finite))
    ax2.plot([lo, hi], [lo, hi], color=REF, lw=0.8, ls="--", zorder=0, label="observed = predicted")
    old = np.array([[r["t_max_old"], r["onset"]] for r in finite])
    ax2.plot(old[:, 0], old[:, 1], ls="none", marker="^", color=AQUA, mfc="white", mew=1.0,
             ms=4.5, label=r"$p_{\max}/(m|g|)$ (original draft)")
    for binding, color, marker in (("space", BLUE, "o"), ("momentum", ORANGE, "s")):
        pts = np.array([[r["T_val"], r["onset"]] for r in finite if r["binding"] == binding])
        if len(pts):
            ax2.plot(pts[:, 0], pts[:, 1], ls="none", marker=marker, color=color, ms=4.5,
                     label=rf"$T_{{\rm val}}$, {binding}-limited")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel("predicted time of validity")
    ax2.set_ylabel(r"observed onset ($\|\psi - \psi^{\rm exact}\| > 10^{-6}$)")
    h, l = ax2.get_legend_handles_labels()
    order = [3, 2, 1, 0] if len(h) == 4 else list(range(len(h)))[::-1]
    ax2.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", bbox_to_anchor=(0.0, 0.93))
    panel_label(ax2, "(b)")
    fig.tight_layout(w_pad=2.0)

    r_new = np.array([r["onset"] / r["T_val"] for r in finite])
    r_old = np.array([r["onset"] / r["t_max_old"] for r in finite])
    derived = {
        "n_cases": len(results),
        "n_no_onset_within_2Tval": len(results) - len(finite),
        "n_space_limited": sum(r["binding"] == "space" for r in results),
        "n_momentum_limited": sum(r["binding"] == "momentum" for r in results),
        "ratio_observed_over_Tval": {"min": float(r_new.min()), "median": float(np.median(r_new)),
                                     "max": float(r_new.max())},
        "ratio_observed_over_tmax_old": {"min": float(r_old.min()), "median": float(np.median(r_old)),
                                         "max": float(r_old.max())},
        "fraction_tmax_old_overestimates": float(np.mean(r_old < 1.0)),
        "fraction_tmax_old_overestimates_by_2x": float(np.mean(r_old < 0.5)),
        "cases": [{k: v for k, v in cv.items() if k not in ("t", "e")} for cv in curves],
    }
    data = {
        "random_T_val": np.array([r["T_val"] for r in results]),
        "random_onset": np.array([r["onset"] for r in results]),
        "random_t_max_old": np.array([r["t_max_old"] for r in results]),
        "random_space_limited": np.array([r["binding"] == "space" for r in results]),
        "random_params": np.array([[r["L"], r["N"], r["sigma"], r["g"], r["x0"], r["p0"], r["dt"]]
                                   for r in results]),
    }
    for i, cv in enumerate(curves):
        data[f"case{i}_t"] = cv["t"]
        data[f"case{i}_error"] = cv["e"]
    save(fig, NAME, data, cfg, derived)


if __name__ == "__main__":
    main()
