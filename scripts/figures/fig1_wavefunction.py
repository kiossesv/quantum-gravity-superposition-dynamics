"""Figure 1: wavefunction-level verification for uniform fields (Section 4.2).

(a) L2 error of each branch against the exact solution, Eq. (16), for the uncorrected
    and the phase-corrected Strang scheme; dashed: |1 - exp(i eps_b)| from Eq. (32).
(b) Branch-diagonal diagnostics of the uncorrected run: norm drift and centroid error.
"""

import numpy as np

from common import (BLUE, ORANGE, REF, WIDE, load_figure_config, panel_label, plt, save,
                    setup_style, sim_config, floor)
from branchgravity import analytic
from branchgravity.config import run, summarize

NAME = "fig1_wavefunction"


def main():
    cfg = load_figure_config(NAME)
    ov = {"time": cfg["time"]}
    out_c = run(sim_config(cfg["base"], {**ov, "propagator": {"phase_correction": True}}))
    out_u = run(sim_config(cfg["base"], {**ov, "propagator": {"phase_correction": False}}))
    sim = out_u["simulation"].config
    g, dt, m, hbar = sim.physics.g, sim.time.dt, sim.physics.mass, sim.physics.hbar
    t = out_u["time"]
    pred = np.stack([np.abs(1 - np.exp(1j * analytic.splitting_phase(t, gb, dt, m, hbar)))
                     for gb in g], axis=1)

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE)
    colors = [BLUE, ORANGE]
    markers = ["o", "s"]
    for b in range(len(g)):
        ax1.plot(t, floor(out_u["l2_error"][:, b]), color=colors[b], lw=1.6,
                 label=rf"uncorrected, $g_{b + 1} = {g[b]}$")
    ax1.plot(t, floor(pred[:, 0]), color=REF, lw=0.9, ls="--", label="Eq. (32)")
    ax1.plot(t, floor(pred[:, 1]), color=REF, lw=0.9, ls="--")
    for b in range(len(g)):
        ax1.plot(t, floor(out_c["l2_error"][:, b]), color=colors[b], lw=1.0, ls=":",
                 marker=markers[b], markevery=10, mfc="white", ms=3.5,
                 label=rf"corrected, $g_{b + 1} = {g[b]}$")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-16, 1e-3)
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\|\psi_b - \psi_b^{\rm exact}\|$")
    ax1.legend(loc="center right", bbox_to_anchor=(1.0, 0.45))
    panel_label(ax1, "(a)")

    norm_dev = np.abs(out_u["norm"] - 1.0)
    x_err = np.abs(out_u["x"] - out_u["x_exact"])
    p_err = np.abs(out_u["p"] - out_u["p_exact"])
    for b in range(len(g)):
        ax2.plot(t, floor(norm_dev[:, b]), color=colors[b], lw=1.2,
                 label=rf"$|\,\|\psi_b\|^2 - 1|$, $b={b + 1}$")
        ax2.plot(t, floor(x_err[:, b]), color=colors[b], lw=1.0, ls="-.",
                 label=rf"$|\langle x\rangle_b - x_b(t)|$, $b={b + 1}$")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-16, 1e-3)
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel("deviation (uncorrected scheme)")
    ax2.legend(loc="center", ncol=2, columnspacing=1.0, bbox_to_anchor=(0.5, 0.6))
    panel_label(ax2, "(b)")
    fig.tight_layout(w_pad=2.0)

    s_u, s_c = summarize(out_u), summarize(out_c)
    derived = {
        "uncorrected": s_u, "corrected": s_c,
        "max_p_error_uncorrected": float(p_err.max()),
        "predicted_final_l2_uncorrected": pred[-1].tolist(),
        "observed_final_l2_uncorrected": out_u["l2_error"][-1].tolist(),
        "T_val": out_u["validity"].T_val, "kappa": out_u["validity"].kappa,
    }
    data = {"t": t, "l2_uncorrected": out_u["l2_error"], "l2_corrected": out_c["l2_error"],
            "l2_predicted": pred, "norm_dev": norm_dev, "x_err": x_err, "p_err": p_err}
    save(fig, NAME, data, cfg, derived)


if __name__ == "__main__":
    main()
