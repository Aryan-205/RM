# HANDOFF — completed project, 12 September 2026

## Current status

The original project scope is complete. All stages 0–13 have their implemented experiments
and delivered results. The research paper has Sections 1–16 and Appendices A–D; the former
missing Sections 11, 12 and 14 are finished. No experiment needs to remain running.

Start with [docs/RESEARCH_PAPER.md](docs/RESEARCH_PAPER.md), or use [README.md](README.md)
for the repository overview and reproduction commands.

## What the completion pass finished

- Wrote the orbital ML results, controlled system comparison, general discussion and conclusions
  in the existing physics-first style, with design tables, measured results and limitations.
- Replaced the incomplete candidate bibliography with the existing verified list of 23 sources.
  No new citations were introduced. Corrected earlier references to 24 sources.
- Replaced obsolete embedded source listings with a source-file index, updated reproduction
  instructions, and indexed all 61 result artifacts and 56 report tables in Appendix D.
- Removed the stale `docs/RESEARCH_PAPER.tex`; its historical version remains in git. The current
  deliverable is Markdown, not a newly typeset PDF or LaTeX paper.
- Confirmed completion of the original Stage 11 and Stage 12 runs. Stage 12 produces five CSVs
  and five PNGs; Stage 11 now includes `stage11_representative_rollout.csv`.
- Checked the escaped-orbit label and readable log axis. Corrected the representative plot's
  duration from “5 orbits” to “5 years”. Saved predicted and reference state coordinates in the
  representative CSV so that the plot can be reproduced from saved numerical data.
- Added `requirements.txt` with the recorded dependency versions, `.venv/` to the existing ignore
  file, and `validate_project.py` for fast artifact and numerical checks.
- Aligned the methodology, presentation and viva notes with the completed report.

## Verification

Run from the repository root:

```bash
python validate_project.py
python run_all.py --list
```

The validation checks all 61 expected PNG/CSV artifacts, report section order and relative links,
Python syntax, the orbital reference against Kepler, trajectory-pair boundaries, projectile
physics features, metre-scale rollout and escape handling, and all parameters of a small PINN
for each of its three loss gradients. These checks pass. The orbital reference sanity check
has maximum position discrepancy 9.47 × 10⁻¹¹ AU over its one-orbit test.

The original Stage 11 and Stage 12 logs at completion were `/tmp/stage11b.log` and
`/tmp/stage12.log`. Temporary logs may disappear after reboot. A first process check missed
macOS's capitalised `Python` executable and two redundant runs were started; those redundant
processes were stopped after the originals were identified. No incomplete redundant run is
required to reproduce the delivered results.

## Findings and qualifications to preserve

- Stage 11 MLP held-out one-step position error: 6.448 × 10⁻⁵ AU; median horizon at 0.01 AU:
  0.372 years. Polynomial regression has the smaller one-step position error, but 45% of its
  rollouts escape. A position-only one-step ranking is not a simulator ranking.
- Direct MLP at two years: 0.006403 AU versus recursive MLP 0.639648 AU. Dense training time
  coverage ends around 2.2 years for short-period orbits; the global maximum training time
  does not describe local coverage.
- Stage 12 one-timescale relative errors: projectile/planetary forest 0.002878/0.2738;
  MLP 0.001472/0.4303. The ratios are about 95× and 292× respectively.
- Stage 12 fixes formulation, model families and data volume, but **does not match steps per
  natural timescale**. One-step scores use training pairs, and learning-curve evaluation overlaps
  fitted subsets. Separate rollout trajectories provide its independent predictive comparison.
- The Stage 12 perturbations change launch speed and semi-major axis respectively. Equal
  fractional parameter offsets are not identical state-space perturbations. Sensitivity scores
  apply to those specified directions.
- Escape thresholds must use the system's own length scale. Stage 11's 100 AU replacement is
  a failure penalty, not the actual post-escape trajectory or an exact error lower bound.
- A projectile flight duration is not a periodic return. The figure duration is five years;
  the sampled semi-major axes imply different orbital periods.
- The physics-feature projectile fit is exact to round-off; the PINN remains approximate. Its
  zero-label result still uses the known equation, initial conditions and collocation points.
  Do not claim the PINN recovered exact physics everywhere or discovered a law.
- Both systems have independent reference solutions. Kepler's parametrisation requires a root
  solve but does not require time integration. These are integrable idealised systems.
- Seeds reproduce scientific metrics within numerical tolerance, not timing columns byte for
  byte. Timings compare different workloads and are not a matched-accuracy speedup benchmark.
- Comparison with literature is thematic; no Hamiltonian network, graph simulator or chaotic
  three-body study was replicated here.

## Repository state and optional future work

Completion changes are saved locally and remain uncommitted; nothing was pushed. Existing
experimental results were preserved and completed. Review `git diff` before making a commit.

No further experiment is required for the current scope. Report Section 14 proposes optional
extensions: noisy data, matched dimensionless time steps, strictly held-out Stage 12 learning
curves, repeated training seeds and conservation-aware architectures. These are future work,
not unfinished claims of this project.
