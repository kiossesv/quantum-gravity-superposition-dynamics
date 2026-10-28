# Changelog

## 0.2.0 (unreleased)

Rewrite accompanying the revised manuscript. The flat `src/*.py` modules are replaced by
the installable package `src/branchgravity/`.

### Fixed
- `<p>` was computed with `np.gradient` (second-order finite differences), about 1% off
  for the paper's grid (4.949 instead of 5.000). It is now computed in the Fourier
  representation, exact for band-limited states.
- The potential phase factor was recomputed at every time step although the paper states
  it is precomputed. Both phase arrays are now built once in `StrangPropagator`.
- `observables.overlap()` raised `NotImplementedError`; overlaps are now implemented,
  together with the full overlap matrix and the reduced density matrix of the branch label.
- Modules imported each other by bare name (`from observables import ...`), which breaks
  when used as a package. All imports are now package-relative.
- Initial Gaussians are normalized analytically (continuum normalization); the discrete
  norm equals 1 to roundoff whenever the packet is resolved and inside the domain.

### Added
- Splitting-phase correction (`phase_correction=True`), exact for uniform fields; both
  VTV and TVT orderings.
- `analytic`: exact Gaussian in a uniform field including its phase, closed-form branch
  overlaps (visibility and relative phase), splitting phase, harmonic coherent states.
- `diagnostics`: wavefunction L2 error, fidelity, global and relative phase errors.
- `validity`: phase-space validity criterion (time of validity, minimum grid), pre-run check.
- `freefall`: propagation in a freely falling, co-moving frame, with exact reconstruction
  of lab-frame states, centroids and overlaps.
- JSON configurations (`configs/`), command-line runner (`scripts/run.py`) that stores
  results with software versions.
- Test suite (48 tests), runnable with pytest or `python tests/run_tests.py`.

### Changed
- The `t_max = p_max/(m|g|)` bound is superseded by the validity criterion: the original
  Fig. 4 failure is spatial wrap-around at t = 3.70, predicted as T_val = 3.695.
- `evolve()` observable callbacks receive `(psi, t)`.
- `scipy` is no longer a dependency.

### Removed
- `src/solvers.py`, `src/observables.py`, `src/initial_states.py`, `src/potentials.py`,
  `src/utils.py`, `src/benchmark_*.py` (superseded; available in the git history).
  Figure scripts for the revised paper are added with the Section 4 revision.
