"""Figure 2: the Strang error for uniform fields is a global phase (Section 4.3).

(a) |eps_b(T)| = |arg <psi_exact|psi_Strang>| versus dt for both orderings and both
    branches; dashed lines: Eq. (32).
(b) Largest L2 error over branches at t = T, uncorrected and corrected (Eq. (34)).
"""

import numpy as np

from common import (BLUE, ORANGE, REF, WIDE, floor, load_figure_config, panel_label, plt,
                    save, setup_style, sim_config)
from branchgravity import analytic, diagnostics
from branchgravity.config import run

NAME = "fig2_splitting_phase"


def final_state_errors(base, dt, T, ordering, corr):
    n = int(round(T / dt))
    out = run(sim_config(base, {
        "time": {"dt": dt, "T": T, "sample_every": n},
        "propagator": {"ordering": ordering, "phase_correction": corr},
        "validity": {"action": "ignore"}}))
    sim = out["simulation"]
    ph = sim.config.physics
    ex = analytic.branch_states_exact(sim.grid.x, T, ph.g, ph.x0, ph.p0, ph.sigma, ph.mass, ph.hbar)
    psi = out["final_state"]
    return diagnostics.phase_error(psi, ex, sim.grid), diagnostics.l2_error(psi, ex, sim.grid), ph


def main():
    cfg = load_figure_config(NAME)
    dts, T = np.array(cfg["dts"]), cfg["T"]
    res = {}
    for ordering in ("VTV", "TVT"):
        for corr in (False, True):
            ph_err, l2 = [], []
            for dt in dts:
                p, e, phys = final_state_errors(cfg["base"], dt, T, ordering, corr)
                ph_err.append(p)
                l2.append(e)
            res[(ordering, corr)] = (np.array(ph_err), np.array(l2))
    g, m, hbar = phys.g, phys.mass, phys.hbar
    dt_fine = np.geomspace(dts.min() * 0.8, dts.max() * 1.25, 100)

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE)
    colors, markers = [BLUE, ORANGE], {"VTV": "o", "TVT": "s"}
    for ordering in ("VTV", "TVT"):
        phase = res[(ordering, False)][0]
        for b, gb in enumerate(g):
            filled = ordering == "VTV"
            ax1.plot(dts, np.abs(phase[:, b]), ls="none", marker=markers[ordering],
                     color=colors[b], mfc=colors[b] if filled else "white", mew=1.0,
                     label=rf"{ordering}, $g_{b + 1} = {gb}$")
            pred = np.abs(analytic.splitting_phase(T, gb, dt_fine, m, hbar, ordering))
            ax1.plot(dt_fine, pred, color=REF, lw=0.8, ls="--",
                     label="Eq. (32)" if (ordering, b) == ("VTV", 0) else None, zorder=0)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\Delta t$")
    ax1.set_ylabel(r"$|\epsilon_b(T)|$")
    h, l = ax1.get_legend_handles_labels()
    order = [l.index(x) for x in l if x != "Eq. (32)"] + [l.index("Eq. (32)")]
    ax1.legend([h[i] for i in order], [l[i] for i in order], loc="lower right")
    panel_label(ax1, "(a)")

    for ordering, color in (("VTV", BLUE), ("TVT", ORANGE)):
        l2u = res[(ordering, False)][1].max(axis=1)
        l2c = res[(ordering, True)][1].max(axis=1)
        ax2.plot(dts, l2u, color=color, marker=markers[ordering], lw=1.3,
                 label=f"{ordering}, uncorrected")
        ax2.plot(dts, floor(l2c), color=color, marker=markers[ordering], mfc="white",
                 lw=1.0, ls=":", label=f"{ordering}, corrected")
    anchor = res[("VTV", False)][1].max(axis=1)[-1]
    ax2.plot(dt_fine, anchor * (dt_fine / dts[-1]) ** 2, color=REF, lw=0.8, ls="--",
             label=r"$\propto \Delta t^2$", zorder=0)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-16, 1)
    ax2.set_xlabel(r"$\Delta t$")
    ax2.set_ylabel(r"$\max_b \|\psi_b(T) - \psi_b^{\rm exact}(T)\|$")
    ax2.legend(loc="center right", bbox_to_anchor=(1.0, 0.52))
    panel_label(ax2, "(b)")
    fig.tight_layout(w_pad=2.0)

    derived = {}
    for (ordering, corr), (p, e) in res.items():
        key = f"{ordering}_{'corrected' if corr else 'uncorrected'}"
        pred = np.array([[analytic.splitting_phase(T, gb, dt, m, hbar, ordering) for gb in g]
                         for dt in dts])
        derived[key] = {
            "max_l2": float(e.max()),
            "max_abs_phase_minus_prediction": float(np.max(np.abs(p - pred))) if not corr else None,
            "max_abs_phase": float(np.max(np.abs(p))),
        }
    data = {"dts": dts}
    for (ordering, corr), (p, e) in res.items():
        tag = f"{ordering}_{'c' if corr else 'u'}"
        data[f"phase_{tag}"] = p
        data[f"l2_{tag}"] = e
    save(fig, NAME, data, cfg, derived)


if __name__ == "__main__":
    main()
