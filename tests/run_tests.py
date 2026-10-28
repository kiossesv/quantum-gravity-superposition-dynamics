"""Run the test suite without pytest:  python tests/run_tests.py

Every function named test_* in tests/test_*.py is executed. With pytest installed,
`pytest` collects the same tests.
"""
import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))


def main():
    passed, failed = 0, []
    t_start = time.perf_counter()
    for path in sorted(HERE.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t0 = time.perf_counter()
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS {path.stem}::{name} ({time.perf_counter() - t0:.2f} s)")
            except Exception:
                failed.append(f"{path.stem}::{name}")
                print(f"FAIL {path.stem}::{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {len(failed)} failed in {time.perf_counter() - t_start:.1f} s")
    for f in failed:
        print("  failed:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
