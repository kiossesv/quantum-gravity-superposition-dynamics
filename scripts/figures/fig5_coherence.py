"""Figure 5 and Table 3: branch coherence and the free-fall frame (Section 4.7).

(a) Visibility |O_12(t)| for the baseline and the fountain case, each propagated in the
    lab frame and in the free-fall frame; dashed: Eq. (18) (identical for both cases,
    which share Delta g and sigma).
(b) Error of the relative phase arg O_12 against Eq. (19), shown while |O_12| > 1e-8;
    dashed: Eq. (33); dotted: the roundoff floor eps_mach / |O_12|.
Table: lab versus free-fall propagation of the fountain case.
"""

import time

import numpy as np

from common import (AQUA, BLUE, FIG_DIR, FLOOR, ORANGE, REF, WIDE, floor, load_figure_config,
                    panel_label, plt, save, setup_style, sim_config)
from branchgravity import analytic, diagnostics
from branchgravity.config import run, summarize
from branchgravity.validity import grid_requirements

NAME = "fig5_coherence"
EPS = np.finfo(float).eps


def timed_run(name, time_cfg):
    """Run with diagnostics, then time the bare propagation (no observables) separately."""
    cfg = sim_config(name, {"time": time_cfg})
    out = run(cfg)
    sim = out["simulation"]
    t0 = time.perf_counter()
    sim.propagator.propagate(sim.psi0, cfg.n_steps)
    out["wall"] = time.perf_counter() - t0
    return out


def main():
    cfg = load_figure_config(NAME)
    vf = cfg["visibility_floor"]
    b, f = cfg["baseline"], cfg["fountain"]
    runs = {
        "base_lab": timed_run(b["lab"], b["time"]),
        "base_lab_unc": timed_run(b["lab_uncorrected"], b["time"]),
        "base_ff": timed_run(b["freefall"], b["time"]),
        "fount_lab": timed_run(f["lab"], f["time"]),
        "fount_ff": timed_run(f["freefall"], f["time"]),
    }
    ph = runs["base_lab"]["simulation"].config.physics
    g1, g2 = ph.g

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE)
    t_line = np.linspace(0, f["time"]["T"], 400)
    ax1.plot(t_line, floor(analytic.visibility(t_line, g1, g2, ph.sigma), 1e-300), color=REF,
             lw=0.9, ls="--", label="Eq. (18)", zorder=0)
    style = {
        "base_lab": dict(color=BLUE, marker="o", label="baseline, lab frame"),
        "base_ff": dict(color=BLUE, marker="o", mfc="white", mew=0.9,
                        label="baseline, free-fall frame"),
        "fount_lab": dict(color=ORANGE, marker="s", label="fountain, lab frame"),
        "fount_ff": dict(color=ORANGE, marker="s", mfc="white", mew=0.9,
                         label="fountain, free-fall frame"),
    }
    every = {"base_lab": (0, 10), "base_ff": (5, 10), "fount_lab": (0, 6), "fount_ff": (3, 6)}
    lines_a = {"base_lab": "-", "base_ff": ":", "fount_lab": "-", "fount_ff": ":"}
    for key, st in style.items():
        out = runs[key]
        ax1.plot(out["time"], floor(np.abs(out["overlap"][:, 0, 1]), 1e-300), ls=lines_a[key],
                 lw=1.0, ms=4.0, markevery=every[key], **st)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-18, 3)
    ax1.set_xlim(0, 12)
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\mathcal{V}_{12}(t) = |O_{12}(t)|$")
    ax1.legend(loc="upper right")
    panel_label(ax1, "(a)")

    def phase_err(out):
        O, Oe = out["overlap"], out["overlap_exact"]
        V = np.abs(Oe[:, 0, 1])
        m = V > vf
        return out["time"][m], np.abs(diagnostics.relative_phase_error(O, Oe))[m], V[m]

    t, e, _ = phase_err(runs["base_lab_unc"])
    dt = b["time"]["dt"]
    ax2.plot(t, floor(e), color=AQUA, lw=1.5, label="baseline, uncorrected")
    ax2.plot(t, (g2**2 - g1**2) * dt**2 * t / 24.0, color=REF, lw=0.9, ls="--",
             label="Eq. (33)", zorder=3)
    for key in ("base_lab", "base_ff", "fount_lab", "fount_ff"):
        t, e, _ = phase_err(runs[key])
        st = dict(style[key])
        st.update(ls=lines_a[key], lw=1.0, ms=4.0,
                  markevery=every[key] if key.startswith("base") else (every[key][0] // 3, 2))
        ax2.plot(t, floor(e), **st)
    t_v = np.linspace(0, f["time"]["T"], 400)
    V = analytic.visibility(t_v, g1, g2, ph.sigma)
    m = V > vf
    ax2.plot(t_v[m], EPS / V[m], color=FLOOR, lw=0.9, ls=(0, (1, 1.5)),
             label=r"$\epsilon_{\rm mach}/\mathcal{V}_{12}$")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-17, 1e2)
    ax2.set_xlim(0, 7.2)
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$|\arg O_{12} - \phi_{12}^{\rm exact}|$")
    ax2.legend(loc="upper right", ncol=2, columnspacing=0.8, fontsize=7)
    panel_label(ax2, "(b)")
    fig.tight_layout(w_pad=2.0)

    # --- fountain table ---------------------------------------------------------
    rows = []
    for key, label in (("fount_lab", "lab frame"), ("fount_ff", "free-fall frame")):
        out = runs[key]
        sim = out["simulation"]
        phs = sim.config.physics
        if sim.frame is None:
            req = grid_requirements(f["time"]["T"], phs.g, phs.x0, phs.p0, phs.sigma,
                                    dt=f["time"]["dt"], delta=sim.config.validity.delta)
        else:
            req = grid_requirements(f["time"]["T"], sim.frame.residual(phs.g), 0.0, 0.0,
                                    phs.sigma, dt=f["time"]["dt"], delta=sim.config.validity.delta)
        s = summarize(out, vf)
        rows.append(dict(label=label, L=sim.grid.L, N=sim.grid.N, N_min=req["N_min"],
                         T_val=out["validity"].T_val, steps=sim.config.n_steps, wall=out["wall"],
                         l2=s["max_l2_error"], dO=float(np.max(np.abs(out["overlap"][:, 0, 1]
                                                                      - out["overlap_exact"][:, 0, 1])))))
    phs = runs["fount_lab"]["simulation"].config.physics
    rest = grid_requirements(f["time"]["T"], phs.g, 0.0, 0.0, phs.sigma, dt=f["time"]["dt"],
                             delta=runs["fount_lab"]["simulation"].config.validity.delta)
    def sci(v):
        mant, exp = "{:.1e}".format(v).split("e")
        return rf"{mant}\times10^{{{int(exp)}}}"
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\hline",
             r"Frame & $L$ & $N$ & $N_{min}$ & $T_{\rm val}$ & Propagation time (s) & "
             r"$\max_b\|\psi_b-\psi_b^{\rm exact}\|$ & $\max_t|O_{12}-O_{12}^{\rm exact}|$ \\", r"\hline"]
    for r in rows:
        lines.append(f"{r['label']} & {r['L']:g} & {r['N']} & {r['N_min']:.0f} & {r['T_val']:.1f} & "
                     f"{r['wall']:.2f} & ${sci(r['l2'])}$ & ${sci(r['dO'])}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (FIG_DIR / "table_fountain.tex").write_text("\n".join(lines) + "\n")

    derived = {k: summarize(v, vf) | {"wall_s": v["wall"], "T_val": v["validity"].T_val}
               for k, v in runs.items()}
    derived["fountain_table"] = rows

    # Roundoff floor of lab-frame propagation: final error against the extended-precision
    # reference for two time steps, and the error of the double-precision reference itself.
    fl = runs["fount_lab"]["simulation"]
    T = f["time"]["T"]
    ref_ext = analytic.branch_states_exact(fl.grid.x, T, phs.g, phs.x0, phs.p0, phs.sigma,
                                           phs.mass, phs.hbar, extended=True)
    ref_dbl = analytic.branch_states_exact(fl.grid.x, T, phs.g, phs.x0, phs.p0, phs.sigma,
                                           phs.mass, phs.hbar)
    floor_check = {"double_reference_error": diagnostics.l2_error(ref_dbl, ref_ext, fl.grid).tolist()}
    for dt_chk in (0.01, 0.04):
        c = sim_config(f["lab"], {"time": {"dt": dt_chk, "T": T, "sample_every": int(round(T / dt_chk))}})
        psi_T = run(c)["final_state"]
        floor_check[f"lab_final_error_dt={dt_chk}"] = diagnostics.l2_error(psi_T, ref_ext, fl.grid).tolist()
    floor_check["estimate_eps_p2T_over_2m_hbar"] = float(EPS * phs.p0**2 * T / (2 * phs.mass * phs.hbar))
    derived["lab_roundoff_floor"] = floor_check
    derived["fountain_N_min_release_from_rest"] = rest["N_min"]
    derived["speedup_wall"] = rows[0]["wall"] / rows[1]["wall"]
    derived["relative_phase_at_T_fountain_rad"] = float(analytic.relative_phase(
        f["time"]["T"], g1, g2, phs.x0, phs.p0))
    derived["relative_phase_at_T_baseline_rad"] = float(analytic.relative_phase(
        b["time"]["T"], g1, g2, ph.x0, ph.p0))
    data = {}
    for k, v in runs.items():
        data[f"{k}_t"] = v["time"]
        data[f"{k}_O12"] = v["overlap"][:, 0, 1]
        data[f"{k}_O12_exact"] = v["overlap_exact"][:, 0, 1]
        data[f"{k}_l2"] = v["l2_error"]
    save(fig, NAME, data, cfg, derived)


if __name__ == "__main__":
    main()
