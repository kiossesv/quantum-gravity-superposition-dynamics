# BranchGravitySolver

### Branch-resolved quantum dynamics in superposed uniform gravitational fields

`branchgravity` propagates a quantum test particle whose source can be in a superposition
of configurations, each producing a uniform gravitational field $g_b$. The state is
$\sum_b w_b\,|b\rangle\otimes\psi_b$, and the branches evolve independently under
$H_b = p^2/2m + m g_b x$. The physically relevant output is not only each branch's
trajectory but the **overlaps** $\langle\psi_b|\psi_{b'}\rangle$, whose modulus (visibility)
and phase (relative phase) determine any measurement that reads out the source.

The solver uses second-order Strang splitting with a Fourier spectral discretization.
For uniform fields it is **exact up to a known branch-dependent phase**, which is removed
at no cost, and it comes with a **phase-space validity criterion** that predicts when a
periodic grid stops representing the dynamics.

This repository accompanies the paper *Branch-Resolved Spectral Dynamics for Quantum
Gravitational Fields: A Split-Operator Approach* (V. I. Kiosses, in preparation).

## Features

- Strang–Fourier propagation of $B$ branches stored as one $(B, N)$ array; branches are
  provably and bitwise independent.
- Splitting-phase correction: Strang splitting for $V = mgx$ equals the exact propagator times
  $e^{i m g^2 \Delta t^3/24\hbar}$ per step; the counter-term $m g_b^2\Delta t^2/24$ restores
  the exact solution, including relative phases, for any $\Delta t$.
- Phase-space validity criterion: time of validity $T_{\rm val}$ (spatial and momentum
  windows) and minimum grid size $N_{\min}$, evaluated before a run.
- Free-fall (co-moving) frame: removes the common fall so the grid only has to resolve the
  relative motion of the branches. For an Earth-like fall over 27 ms this reduces the
  required grid from $N_{\min}\approx 2.3\times 10^5$ to about $10^3$ points.
- Exact references: Gaussian wavepacket in a uniform field (with phase), closed-form branch
  overlaps, harmonic-oscillator coherent states.
- Observables: norms, $\langle x\rangle$, $\langle p\rangle$ (spectral), widths, energy,
  overlap matrix, reduced density matrix of the branch label.
- JSON run configurations; every result is generated from a stored config.

## Installation

```bash
git clone https://github.com/kiossesv/quantum-gravity-superposition-dynamics
cd quantum-gravity-superposition-dynamics
pip install -e .            # or: pip install -r requirements.txt  (pinned versions)
```

The only runtime dependency is NumPy; Matplotlib is used for plotting scripts.

## Quick start

```python
import branchgravity as bg
from branchgravity import analytic, diagnostics, observables

g = [0.8, 1.2]                                   # two branches
grid = bg.Grid(L=100.0, N=2048)
prop = bg.StrangPropagator.uniform_field(grid, g, dt=0.01, phase_correction=True)
psi0 = bg.branch_states(grid, len(g), x0=-10.0, p0=5.0, sigma=1.0)

print(bg.time_of_validity(grid, g, -10.0, 5.0, 1.0, dt=0.01))   # check before running

psi = prop.propagate(psi0, n_steps=500)                         # t = 5
O = observables.overlap_matrix(psi, grid)                       # <psi_b|psi_b'>
O_exact = analytic.overlap_matrix_exact(5.0, g, -10.0, 5.0, 1.0)
print(abs(O - O_exact).max())                                   # ~1e-13
```

Free-fall frame:

```python
frame = bg.FreeFallFrame.for_branches(g, x0=-10.0, p0=5.0)      # g_ref = midrange of g
grid_c = bg.Grid(L=56.0, N=128)                                  # co-moving grid
chi = frame.propagator(grid_c, g, dt=0.01).propagate(frame.initial_gaussian(grid_c, 1.0, 2), 500)
O_lab = frame.lab_overlaps(observables.overlap_matrix(chi, grid_c), 5.0, g)
psi_lab = frame.to_lab(chi, grid_c, 5.0, g, grid.x)             # back to lab coordinates
```

## Running configurations

```bash
python scripts/run.py configs/two_branch_lab.json configs/two_branch_freefall.json --out results
```

| Config | Purpose |
|---|---|
| `two_branch_lab` | Baseline two-branch run (lab frame, phase-corrected) |
| `two_branch_lab_uncorrected` | Same without correction: relative-phase error of Eq. (33) |
| `two_branch_freefall` | Same physics in the free-fall frame on 128 points |
| `space_limited_single_branch` | Original Fig. 4 parameters; space-limited $T_{\rm val}=3.695$ |
| `momentum_limited_single_branch` | Coarse grid; momentum-limited $T_{\rm val}=4.52$ |
| `earth_gravity_freefall` | Earth-like fall, 2% branch difference, $T = 27$ ms |

Each run writes `results/<name>.npz` (time series of norms, centroids, overlaps, errors
against the exact solution) and `results/<name>.json` (config, summary, software versions).

## Tests

```bash
pytest                       # or, without pytest:  python tests/run_tests.py
```

The suite (48 tests, about 2 s) checks, among other things:

- the exact references solve the Schrödinger equation and the overlap formula matches quadrature;
- the Strang error is exactly the predicted global phase for both orderings and $\Delta t$ up to 0.5;
- the phase correction is exact for uniform fields (and is not beneficial for the harmonic oscillator);
- second-order convergence for a harmonic-oscillator coherent state;
- a zero branch stays bitwise zero and a branch run alone is bitwise identical to the same branch run with others;
- the predicted $T_{\rm val}$ matches the observed onset of failure within 10% in space- and momentum-limited cases;
- free-fall and lab-frame propagation agree to $10^{-12}$.

## Package layout

```
src/branchgravity/
  grid.py          periodic grid and momentum grid
  states.py        Gaussian initial states
  potentials.py    potentials and forces
  propagators.py   StrangPropagator (VTV/TVT, phase correction), evolve()
  observables.py   norms, moments, energy, overlap matrix, branch density matrix
  analytic.py      exact solutions, overlaps, splitting phase, coherent states
  diagnostics.py   L2 error, fidelity, phase errors against a reference
  validity.py      time of validity, grid requirements, pre-run check
  freefall.py      free-fall (co-moving) frame and Fourier interpolation
  config.py        JSON configurations and run()
configs/           stored run configurations
scripts/run.py     command-line runner
tests/             test suite (pytest or tests/run_tests.py)
```

## Units

Simulations use $\hbar = m = 1$ and $\sigma = 1$. For $^{87}$Rb with $\sigma = 1\,\mu$m the time
unit is 1.37 ms and the acceleration unit 0.53 m s$^{-2}$, so Earth's gravity is $g \approx 18.4$.

## Research program

This project is part of a three-repository program on quantum superpositions of
gravitational fields:

- [quantum-gravity-superposition-dynamics](https://github.com/kiossesv/quantum-gravity-superposition-dynamics) (this repository): branch-resolved quantum dynamics.
- [quantum-gravity-superposition-stability](https://github.com/kiossesv/quantum-gravity-superposition-stability): statistical stability analysis of effective gravitational models.
- [quantum-gravity-superposition-sciml](https://github.com/kiossesv/quantum-gravity-superposition-sciml) (planned): physics-informed machine learning for stability regimes.

## License

MIT License.
