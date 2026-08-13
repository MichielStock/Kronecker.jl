# Changelog

All notable changes to Kronecker.jl are documented in this file.

## Unreleased

### Changed

- The minimum supported Julia version is now 1.10 (was 1.0).
- Removed the pre-Julia-1.3 compatibility branch for five-argument `mul!` in `src/vectrick.jl`.
- Continuous integration modernised: current action versions (`actions/checkout@v4`, `julia-actions/setup-julia@v2`, `codecov/codecov-action@v5`), package caching via `julia-actions/cache@v2`, test matrix now covers Julia 1.10 (LTS), the latest stable release, and pre-releases. Coverage upload now requires the `CODECOV_TOKEN` repository secret.
- Added Dependabot configuration to keep GitHub Actions up to date.
- Aqua.jl quality checks updated to Aqua 0.8 with method-ambiguity checking re-enabled.

### Fixed

- Renamed `src/indexedkroncker.jl` to `src/indexedkronecker.jl` and `scrips/` to `scripts/` (typos).
- README fixes: "comparision" → "comparison", updated benchmark script link, added a "Citing" section for the JuliaCon proceedings paper.
