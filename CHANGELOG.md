# Changelog

All notable changes to `metal2d` are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-08-22

### Added

- Monometallic T-REX-Full input with public parsing, topology classification,
  molecule construction, depiction and drawing APIs.
- Unified `mol_from_input()` and `depict_input()` entry points for SMILES or
  T-REX strings.
- CLI support for direct T-REX strings and `.trex` files.
- Dedicated layout for metal-metal bonded cores, atom-bridged dimers,
  whole-ligand bridges and rigid multidentate bridges.
- Macrocycle cavity placement and guarded whole-molecule candidates for closed
  high-denticity ligands.
- Symmetry handling for equivalent metal halves, eta-bound ligands, terminal
  carbonyls and related dinuclear motifs.
- Octahedral perspective bonds and topology-aware fac/mer classification for
  suitable T-REX inputs.
- Reproducible T-REX examples and SVG figures for cisplatin, transplatin and
  fac/mer `Ir(ppy)3`.
- Reproducible three-engine comparison figures for representative dinuclear
  iridium, iron and ruthenium complexes.
- Regression coverage for the reported mono- and polynuclear failure cases.

### Changed

- Multi-metal drawing now processes every metal centre instead of only the
  first one.
- Eta-bound Cp, Cp*, arene and allyl groups are oriented and rendered through
  centroid interactions more consistently.
- Candidate ranking now accounts for bond crossings, close bonds, stretch,
  atoms printed on metal labels and collapsed eta-centroid geometry.
- Chelate reflection, macrocycle placement, terminal-sector assignment and
  ligand collision relaxation were substantially improved.
- CSV input can select a named SMILES column with `--column`.
- Polynuclear candidate searches were reduced and guarded to avoid slowing the
  common mononuclear path.

### Fixed

- Numerous crossed, overlapping, asymmetric and excessively stretched
  depictions collected from MetalLipoDB and tmQM.
- Incorrect orientation of equivalent Cp/arene and terminal CO pairs.
- Metals placed on top of macrocycles, eta-bound rings or organic ligand atoms.
- Order-dependent layouts of equivalent ligands and symmetric dinuclear
  complexes.

### Limitations

- T-REX conversion is currently monometallic and supports `SMILES:` ligand
  payloads only.
- Dense cores containing three or more metals and intrinsically non-planar cage
  ligands can still require manual depiction.

## [0.2.0] - 2026-08-04

### Added

- Initial support for polynuclear complexes, recursive metal-containing ligand
  layout, metal-metal cluster mode and eta-ring orientation.

## [0.1.0]

- First packaged release.

[Unreleased]: https://github.com/levakrasnovs/metal2d/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/levakrasnovs/metal2d/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/levakrasnovs/metal2d/compare/v0.1.0...v0.2.0
