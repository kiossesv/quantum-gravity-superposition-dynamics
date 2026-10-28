"""Shared helpers for the figure scripts: paths, configs, plot style, output.

Every figure script reads configs/figures/<name>.json, writes
figures/<name>.pdf and .png, and stores the plotted data in figures/data/<name>.npz
together with figures/data/<name>.json (config, derived numbers, software versions).
"""

import copy
import datetime
import json
import platform
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import branchgravity as bg  # noqa: E402
from branchgravity.config import SimulationConfig  # noqa: E402

FIG_DIR = ROOT / "figures"
DATA_DIR = FIG_DIR / "data"
CFG_DIR = ROOT / "configs"
FIG_CFG_DIR = CFG_DIR / "figures"

# Categorical slots in fixed order (validated palette, light mode); reference and
# floor lines use neutral inks so that color always means "series".
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
INK, INK2, REF, FLOOR, GRID = "#0b0b0b", "#52514e", "#52514e", "#9a9893", "#e4e3df"

MM = 1.0 / 25.4
WIDE = (180 * MM, 68 * MM)      # two panels, full text width
SINGLE = (88 * MM, 68 * MM)


def setup_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def panel_label(ax, text):
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=INK)


def load_figure_config(name):
    with open(FIG_CFG_DIR / f"{name}.json") as fh:
        return json.load(fh)


def load_run_config(name):
    with open(CFG_DIR / f"{name}.json") as fh:
        return json.load(fh)


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def sim_config(base, overrides=None):
    """SimulationConfig from a run-config name or dict, with nested overrides."""
    d = load_run_config(base) if isinstance(base, str) else base
    return SimulationConfig.from_dict(_merge(d, overrides or {}))


def floor(a, value=1e-17):
    """Replace exact zeros (and smaller) for log-scale plotting."""
    return np.maximum(np.asarray(a, dtype=float), value)


def save(fig, name, data, config, derived=None):
    FIG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    np.savez_compressed(DATA_DIR / f"{name}.npz", **data)
    meta = {
        "figure": name,
        "config": config,
        "derived": derived or {},
        "branchgravity": bg.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    (DATA_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")
    print(f"wrote figures/{name}.pdf/.png and figures/data/{name}.npz/.json")
