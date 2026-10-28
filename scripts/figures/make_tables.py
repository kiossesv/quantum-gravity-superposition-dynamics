"""Generate figures/table_parameters.tex (Table 2 of the paper) from the stored configs,
so that the parameters printed in the paper are the ones the figure scripts used."""

import json

from common import DATA_DIR, FIG_DIR, load_figure_config, load_run_config


def fmt(v):
    if isinstance(v, (list, tuple)):
        return ", ".join(fmt(x) for x in v)
    txt = f"{int(v)}" if float(v).is_integer() else f"{v:g}"
    return f"${txt}$" if txt.startswith("-") else txt


def rng(a, b):
    return f"{fmt(a)}--{fmt(b)}"


def run_row(test, fig, name, frame_note="", **over):
    c = load_run_config(name)
    ph, gr, tm = c["physics"], c["grid"], c["time"]
    row = dict(test=test + frame_note, fig=fig, V=r"$m g_b x$", g=fmt(ph["g"]), x0=fmt(ph["x0"]),
               p0=fmt(ph["p0"]), sigma=fmt(ph["sigma"]), L=fmt(gr["L"]), N=fmt(gr["N"]),
               dt=fmt(tm["dt"]), T=fmt(tm["T"]))
    row.update(over)
    return row


def main():
    f2 = load_figure_config("fig2_splitting_phase")
    f3 = load_figure_config("fig3_convergence")
    f4 = load_figure_config("fig4_validity")
    rows = [
        run_row("Two-branch baseline", "1, 2, 5", "two_branch_lab",
                dt=f"{fmt(min(f2['dts']))}--{fmt(max(f2['dts']))}$^a$"),
        run_row("Two-branch baseline", "5", "two_branch_freefall", frame_note=" (free fall$^b$)"),
    ]
    h, q = f3["temporal"]["harmonic"], f3["temporal"]["quartic"]
    dts = f3["temporal"]["dts"]
    rows += [
        dict(test="Coherent state", fig="3(a)", V=rf"$\tfrac12 m\omega^2x^2$, $\omega={fmt(h['omega'])}$",
             g="--", x0=fmt(h["x0"]), p0=fmt(h["p0"]), sigma=r"$1/\sqrt{2}$", L=fmt(h["L"]),
             N=fmt(h["N"]), dt=rng(min(dts), max(dts)), T=fmt(f3["temporal"]["T"])),
        dict(test="Quartic oscillator", fig="3(a)", V=rf"${fmt(q['lam'])}\,x^4$", g="--",
             x0=fmt(q["x0"]), p0=fmt(q["p0"]), sigma=fmt(q["sigma"]), L=fmt(q["L"]), N=fmt(q["N"]),
             dt=rng(min(dts), max(dts)) + f"$^c$", T=fmt(f3["temporal"]["T"])),
    ]
    sp = f3["spatial"]
    rows.append(run_row("Spatial convergence", "3(b)", sp["base"],
                        N=rng(min(sp["N"]), max(sp["N"])), dt=fmt(sp["dt"]), T=fmt(sp["T"])))
    for c in f4["cases"]:
        rows.append(dict(test=c["label"][0].upper() + c["label"][1:], fig="4(a)", V=r"$m g x$",
                         g=fmt(c["g"]), x0=fmt(c["x0"]), p0=fmt(c["p0"]), sigma=fmt(c["sigma"]),
                         L=fmt(c["L"]), N=fmt(c["N"]), dt="0.005", T=r"$1.3\,T_{\rm val}$"))
    r = f4["random"]
    n_valid = json.loads((DATA_DIR / "fig4_validity.json").read_text())["derived"]["n_cases"]
    rows.append(dict(test=f"{n_valid} of {r['n_cases']} random draws$^d$", fig="4(b)", V=r"$m g x$",
                     g=r"$\pm$" + rng(*r["abs_g"]), x0=rf"$[{r['x0_fraction_of_L'][0]:g},{r['x0_fraction_of_L'][1]:g}]\,L$",
                     p0=rf"$[{r['p0_fraction_of_pmax'][0]:g},{r['p0_fraction_of_pmax'][1]:g}]\,p_{{max}}$",
                     sigma=rng(*r["sigma"]), L=rng(*r["L"]), N=rng(min(r["N"]), max(r["N"])),
                     dt=rf"$T_{{\rm val}}/{r['steps_per_T_val']}$", T=r"$2\,T_{\rm val}$"))
    rows.append(run_row("Fountain", "5, Tab. 3", "fountain_lab"))
    rows.append(run_row("Fountain", "5, Tab. 3", "fountain_freefall", frame_note=" (free fall$^b$)"))

    cols = ["test", "fig", "V", "g", "x0", "p0", "sigma", "L", "N", "dt", "T"]
    head = (r"Test & Fig. & $V$ & $g_b$ & $x_0$ & $p_0$ & $\sigma$ & $L$ & $N$ & $\Delta t$ & $T$ \\")
    lines = [r"\begin{tabular}{lllllllllll}", r"\hline", head, r"\hline"]
    lines += [" & ".join(row[c] for c in cols) + r" \\" for row in rows]
    lines += [r"\hline", r"\end{tabular}"]
    (FIG_DIR / "table_parameters.tex").write_text("\n".join(lines) + "\n")
    print("wrote figures/table_parameters.tex")


if __name__ == "__main__":
    main()
