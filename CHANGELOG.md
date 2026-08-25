# Changelog

All notable changes to Kronecker.jl are documented in this file.

## Version 0.6.0

### Breaking

- The minimum supported Julia version is now 1.10 (was 1.0).
- NamedDims.jl support is now a package extension: NamedDims is a weak
  dependency, so users must load NamedDims themselves (`using NamedDims`) to
  activate the `NamedDimsArray` methods. Behaviour once loaded is unchanged.
- StatsBase.jl is no longer a dependency; weighted sampling for Kronecker
  graphs is implemented internally. The hard dependencies of the package are
  now LinearAlgebra, Random and SparseArrays only.

### Added

- `naivesample`, `fastsample` and `sampleindices` accept an optional leading
  `rng::AbstractRNG` argument (default `Random.default_rng()`), making
  Kronecker graph sampling reproducible.

### Changed

- Documentation migrated to Documenter 1.x, building in strict mode with
  doctests enabled; several missing docstrings added.
- Removed the pre-Julia-1.3 compatibility branch for five-argument `mul!` in `src/vectrick.jl`.
- Continuous integration modernised: current action versions (`actions/checkout@v4`, `julia-actions/setup-julia@v2`, `codecov/codecov-action@v5`), package caching via `julia-actions/cache@v2`, test matrix now covers Julia 1.10 (LTS), the latest stable release, and pre-releases. Coverage upload now requires the `CODECOV_TOKEN` repository secret.
- Added Dependabot configuration to keep GitHub Actions up to date.
- Aqua.jl quality checks updated to Aqua 0.8 with method-ambiguity checking re-enabled.
- `sampleindices` on a Kronecker product is faster (single flat walk over the
  factors with preallocated buffers instead of a pairwise recursion; weights of
  a repeated `KroneckerPower` factor are computed once). The sampling
  distribution is unchanged and pinned by tests against the dense `kron`
  distribution.
- `isprob` no longer allocates (predicate-based `all` instead of broadcasting).

### Fixed

- `fastsample` systematically produced fewer edges than the expected count
  `sum(P)`: indices were drawn with replacement and duplicates collapsed into a
  single edge. Collisions are now re-sampled, as in Leskovec et al. (2008), so
  the graph contains exactly `round(Int, sum(P))` edges. Note that graphs
  sampled with a given seed differ from those of version 0.5.x.
- `_sample_weighted` and `sampleindices` now throw an `ArgumentError` for
  negative weights, which would previously corrupt the sampling silently.
- `IndexedKroneckerProduct` multiplication (`genvectrick!`) could return garbage or `NaN`: its scratch array was accumulated into without zero-initialisation, so results depended on heap state. The second ("S = NV") branch additionally used wrong scratch dimensions and a wrong index, writing out of bounds under `@inbounds`. Both branches are fixed and covered by seeded regression tests.
- Renamed `src/indexedkroncker.jl` to `src/indexedkronecker.jl` and `scrips/` to `scripts/` (typos).
- README fixes: "comparision" → "comparison", updated benchmark script link, added a "Citing" section for the JuliaCon proceedings paper.
