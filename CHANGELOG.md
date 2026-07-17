# Changelog

All notable changes to Senbonzakura are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] "Pilot" — 2026-07-17

The first public release: on PyPI, on GitHub, AGPL-3.0.

### Added
- `kageyoshi`, a one-shot balanced preset: supply only the paths and it scales
  the search budget to the model's size and turns on every quality lever. The KL
  ceiling and coherence penalty hold the result at the most uncensored config
  that stays coherent.
- A resource governor that paces GPU work under live VRAM pressure and offloads a
  model larger than VRAM, so the tool runs on constrained cards. Includes a
  background mode that yields the GPU to a foreground game and resumes on close.
- Shell completion via shtab: `--print-completion` emits a script for bash, zsh,
  or tcsh, the one-time install pattern pip and gh use.
- A packaged coherence probe (`senbonzakura.coherence`), the neutral-passage
  perplexity check, reusing the shared model loader so its flags match the scorer.

### Changed
- Relicensed to AGPL-3.0-or-later. The keyword metric copies Heretic verbatim and
  Heretic is AGPL, so copyleft reaches the whole work; Apache was never available
  to us. Upstream notice restored on the copied region, THIRD-PARTY-NOTICES added.
- README rebuilt around reproducible Qwen3-4B numbers, with an honest note on how
  the single-versus-multi gap widens with model size.

### Removed
- The superseded Qwen3-1.7B Heretic comparison table. Leading a first-time reader
  with a comparison we have publicly disowned is worse than none; replaced with a
  note that a matched re-run under the corrected code is pending.

### Fixed
- The coherence scorer rejected `--load-in-4bit`, so a bench chain that passed the
  flag logged a false coherence failure and stalled the 4B numbers. The packaged
  module accepts the scorer's full flag set and a parser test guards the regression.

## [0.2.0] — 2026-07-14

### Added
- Installable package: `src/` layout, a `senbonzakura` console command,
  `python -m senbonzakura`, `[quant]` and `[dev]` extras, a man page, `--version`.
- Multi-direction refusal search grounded in the head-to-head data: the keyword
  gap versus Heretic was a search and contrast problem, not weak ablation.
  - Three-objective NSGA-II over strict non-compliance, the keyword rate, and KL,
    so the keyword and hedging axes are optimised, not just reported.
  - `--hedge-ds` folds a hedged-versus-clean contrast direction into the basis.
  - `--mlp-off` pins the MLP ablation to zero to test where refusal lives.
  - `--patience` stops the search once the frontier stalls.
  - `--eval-refusal-final` re-scores the top frontier on a larger eval before the
    knee is chosen, so the pick is not overfit to the small search eval.

## [0.1.0] — earlier

### Added
- The initial abliteration pipeline: activation capture, difference-of-means
  refusal direction, weight orthogonalisation, and refusal/coherence scoring.

### Other
- Bug fixes and improvements.

[0.3.0]: https://github.com/elementmerc/senbonzakura/releases/tag/v0.3.0
