"""Run one or more JSON configurations and store the results.

    python scripts/run.py configs/two_branch_lab.json [more.json ...] [--out results]

For each config this prints the validity report and the headline errors against the
exact solution, and writes <out>/<name>.npz plus <out>/<name>.json (config + metadata).
"""

import argparse
import datetime
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import branchgravity  # noqa: E402
from branchgravity.config import SimulationConfig, run, summarize  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    for path in args.configs:
        cfg = SimulationConfig.load(path)
        print(f"=== {cfg.name}: {cfg.description}")
        t0 = time.perf_counter()
        out = run(cfg)
        wall = time.perf_counter() - t0
        print(out["validity"])
        summary = summarize(out)
        for k, v in summary.items():
            print(f"  {k:28s} {v:.3e}")
        print(f"  wall time                    {wall:.2f} s")

        np.savez_compressed(
            args.out / f"{cfg.name}.npz",
            **{k: out[k] for k in ("time", "norm", "x", "p", "overlap", "l2_error",
                                   "x_exact", "p_exact", "overlap_exact", "rho_branch")},
            final_state=out["final_state"],
        )
        meta = {
            "config": cfg.to_dict(),
            "summary": summary,
            "T_val": out["validity"].T_val,
            "validity_binding": out["validity"].binding,
            "wall_time_s": wall,
            "branchgravity": branchgravity.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        (args.out / f"{cfg.name}.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
