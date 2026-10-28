"""Regenerate every figure and table of the paper:  python scripts/figures/make_all.py

Takes about a minute on a laptop; fig5 includes a 2^17-point lab-frame reference run
and fig6 is a timing benchmark (results are machine dependent).
"""
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCRIPTS = ["fig1_wavefunction", "fig2_splitting_phase", "fig3_convergence", "fig4_validity",
           "fig5_coherence", "fig6_performance", "make_tables"]

if __name__ == "__main__":
    only = sys.argv[1:] or SCRIPTS
    for name in only:
        t0 = time.perf_counter()
        importlib.import_module(name).main()
        print(f"  {name}: {time.perf_counter() - t0:.1f} s")
