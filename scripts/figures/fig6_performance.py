"""Figure 6: cost of one time step (Section 5).

(a) Wall time per step versus N for B = 1, 2, 4, 8 branches; dashed: N log2 N scaling.
(b) Wall time per step and per branch, t_step / (B N log2 N), showing the approach to
    FFT-dominated cost and the per-branch cost of the batched (B, N) layout.
Timings are machine dependent; the hardware is recorded in figures/data/fig6_performance.json.
"""

import time

import numpy as np

from common import (REF, SERIES, WIDE, load_figure_config, panel_label, plt, save, setup_style)
import branchgravity as bg

NAME = "fig6_performance"


def time_per_step(N, B, L_per_point, min_seconds, repeats):
    grid = bg.Grid(L_per_point * N, N)
    g = list(np.linspace(0.8, 1.2, B))
    prop = bg.StrangPropagator.uniform_field(grid, g, 0.01, phase_correction=True)
    psi = bg.branch_states(grid, B, 0.0, 0.0, 1.0)
    for _ in range(5):
        psi = prop.step(psi)
    n = 1
    while True:                                   # calibrate the number of steps
        t0 = time.perf_counter()
        for _ in range(n):
            psi = prop.step(psi)
        dt = time.perf_counter() - t0
        if dt > min_seconds / repeats:
            break
        n *= 2
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            psi = prop.step(psi)
        samples.append((time.perf_counter() - t0) / n)
    return float(np.median(samples))


def main():
    cfg = load_figure_config(NAME)
    Ns, Bs = np.array(cfg["N"]), cfg["B"]
    t = np.array([[time_per_step(N, B, cfg["L_per_point"], cfg["min_seconds_per_measurement"],
                                 cfg["repeats"]) for N in Ns] for B in Bs])

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE)
    nlogn = Ns * np.log2(Ns)
    markers = ["o", "s", "D", "^"]
    for i, B in enumerate(Bs):
        ax1.plot(Ns, t[i] * 1e3, color=SERIES[i], marker=markers[i], label=rf"$B = {B}$")
        ax2.plot(Ns, t[i] / (B * nlogn) * 1e9, color=SERIES[i], marker=markers[i],
                 label=rf"$B = {B}$")
    ax1.plot(Ns, t[-1, -1] * 1e3 * nlogn / nlogn[-1], color=REF, lw=0.8, ls="--",
             label=r"$\propto N\log_2 N$", zorder=0)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$N$")
    ax1.set_ylabel("wall time per step (ms)")
    ax1.legend(loc="upper left")
    panel_label(ax1, "(a)")
    ax2.set_xscale("log", base=2)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"$t_{\rm step}/(B\,N\log_2 N)$ (ns)")
    ax2.legend(loc="upper right")
    panel_label(ax2, "(b)")
    fig.tight_layout(w_pad=2.0)

    derived = {
        "ms_per_step": {f"B={B}": dict(zip(map(str, Ns.tolist()), (t[i] * 1e3).tolist()))
                        for i, B in enumerate(Bs)},
        "ratio_B8_over_B1": dict(zip(map(str, Ns.tolist()), (t[-1] / t[0]).tolist())),
        "ns_per_point_per_branch_largest_N": {f"B={B}": float(t[i, -1] / (B * nlogn[-1]) * 1e9)
                                              for i, B in enumerate(Bs)},
    }
    save(fig, NAME, {"N": Ns, "B": np.array(Bs), "seconds_per_step": t}, cfg, derived)


if __name__ == "__main__":
    main()
