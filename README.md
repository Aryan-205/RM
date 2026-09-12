# Machine Learning-Based Prediction of Physical Motion

### A Comparative Study of Projectile and Planetary Dynamics

**Aryan Bola** · Research Methodology (RM), 5th Semester B.Sc. Physics

---

## What this project asks

> **How effectively can machine learning approximate and predict physical motion, and how does its
> predictive performance differ between projectile motion and planetary dynamics?**

The obvious objection comes first, so here is the answer first.

**"Why use AI to predict something the equations already give us exactly?"**

Because the object of study is *the algorithm*, not the projectile. This is a benchmark, in the
same sense that one tests a new numerical integrator on a problem with a known analytic solution —
precisely so that any discrepancy is attributable. Choosing systems whose exact answers are known
is not a weakness of the design; it is the condition that makes every measurement in it possible.

**This study does not claim that machine learning beats physics.** It loses, here, by roughly ten
orders of magnitude, and the report says so in those words.

---

## What was found

| | Result |
|---|---|
| **Interpolation looks excellent** | Random Forest: R² = 0.9924 (x), 0.9487 (y) on held-out projectile data |
| **...and it was measuring the wrong thing** | The same models degrade by up to **286×** when tested outside their training envelope |
| **A Random Forest cannot extrapolate at all** | Its prediction goes exactly **flat** beyond the last training split — see `stage8_response_slice.png` |
| **Stepped prediction compounds error** | One-step accuracy **6.4 × 10⁻⁵ AU**, yet past 0.01 AU within **0.37 years** (≈ ⅓ orbit) |
| **It is not chaos** | The two-body problem is integrable; perturbations grow **linearly**. The divergence is model error |
| **Learned rollouts break conservation laws** | Relative energy error climbs by orders of magnitude — nothing in the loss ever mentioned energy |
| **The two systems differ structurally** | The projectile flow map is **affine** (linear regression recovers it to 10⁻¹⁴ m); orbital acceleration contains 1/r³ and the fitted unconstrained maps remain approximate |
| **Physics in the loss beats more data** | A PINN trained on **zero labelled points** matches a plain network given ~200; **14.7×** better where no data exists |
| **Physics in the features beats everything** | Linear regression in a physics basis: exact to floating-point round-off, everywhere, including far outside the training range |

The last two rows are the constructive half of the study. The middle rows are the diagnostic half.

---

## Start here

| If you want... | Read |
|---|---|
| The full report | [`docs/RESEARCH_PAPER.md`](docs/RESEARCH_PAPER.md) |
| The formal methodology chapter | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) |
| The literature review (23 verified sources) | [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) |
| A slide-by-slide talk plan | [`docs/PRESENTATION_OUTLINE.md`](docs/PRESENTATION_OUTLINE.md) |
| Viva questions and answers | [`docs/VIVA_PREP.md`](docs/VIVA_PREP.md) |
| Just the figures | [`results/`](results/) |
| The physics and the derivations | the module docstrings in [`src/`](src/) |

---

## The three figures that carry the argument

**1. `results/stage8_response_slice.png` — why "R² = 0.99" was not the whole story.**
A one-dimensional slice through input space along which the true answer is *exactly* a straight
line. Every deviation you see is therefore the model's own inductive bias, with nothing else mixed
in. The Random Forest's prediction is literally constant outside the training range.

**2. `results/stage11_error_growth.png` — the central result of Part II.**
Position error against time during autoregressive rollout, on a log axis, averaged over 20 unseen
orbits, with the RK4 reference error stated in an annotation below the displayed ML range.

**3. `results/stage13_extrapolation.png` — what actually fixes it.**
Two networks with identical architecture and training, given labels only from the first 40% of
each flight. The one that also has `d²y/dt² = −g` in its loss tracks the parabola through the
region where it has no data at all. The one without flies off it.

---

## Repository layout

```
RM/
├── run_all.py                 Reproduce the entire study (~49 min)
│
├── data/
│   ├── generation/
│   │   ├── projectileDataGeneration.py    Projectile datasets, incl. state-space form
│   │   └── planetaryDataGeneration.py     Orbital trajectories and the three ML tables
│   └── *.csv                              Generated datasets
│
├── src/
│   ├── projectile_ml.py       Stages 3-6: train, evaluate, visualise
│   ├── planetary_physics.py   Four integrators + the analytic Kepler solution
│   ├── planetary_ml.py        Flow-map / history / direct models, rollout machinery
│   ├── pinn.py                A physics-informed neural network, from scratch in NumPy
│   ├── models.py              Model factory (fixed hyperparameters, so comparisons are controlled)
│   └── evaluation.py          Metrics, defined once and used everywhere
│
├── experiments/
│   ├── stage7_data_size.py          Effect of training-set size
│   ├── stage8_generalization.py     Interpolation vs extrapolation
│   ├── stage10_integrators.py       Numerical validation against Kepler
│   ├── stage11_planetary_ml.py      ML for orbital motion
│   ├── stage12_comparison.py        Projectile vs planetary, head to head
│   └── stage13_physics_informed.py  Hard constraints, soft constraints, none
│
├── plots/                     One module per stage; shared palette in style.py
├── results/                   Every figure (.png) and every number (.csv)
└── docs/                      Report, methodology, literature review, talk, viva prep
```

Every script's docstring opens with its experimental design — independent variable, dependent
variable, and what was held constant.

---

## Reproducing everything

```bash
pip install -r requirements.txt
python run_all.py                  # everything, about 49 minutes
python run_all.py --list           # stages and their runtimes
python run_all.py --only 8 13      # just those two
python run_all.py --skip 11 12     # everything except the slow ones
python validate_project.py        # fast artifact and numerical checks, no retraining
```

Every random step is seeded — dataset generation, train/test splits, model initialisation, subset
draws — so seeded scientific metrics should reproduce within numerical tolerance; timing columns vary.

**No figure in this repository was drawn or adjusted by hand.** Every one is generated by a script from recorded results or directly from the defining equations.
Appendix D of the report maps the result files to their producing stages.

Tested on Python 3.12.8 with NumPy 2.2.1, pandas 2.2.3, scikit-learn 1.6.1, matplotlib 3.10.0.
No GPU, no deep-learning framework. The neural network in `src/pinn.py` — forward pass,
backpropagation and Adam — is ~120 lines of NumPy, written by hand because the physics loss needs
derivatives of the network's output with respect to its own input, which scikit-learn cannot
express and which PyTorch would have hidden.

---

## The stages

| Stage | What it does | Key output |
|---|---|---|
| 0–1 | Classical projectile motion; the physics baseline | — |
| 2 | Generate the dataset; document the sampling design | `data/projectile_dataset.csv` |
| 3–4 | First models; train/test split | — |
| 5 | Evaluation metrics, interpreted rather than printed | `actual_vs_predicted.png` |
| 6 | Visual comparison against the classical trajectory | `trajectory_comparison.png` |
| 7 | How training-set size affects accuracy | `stage7_learning_curve.png` |
| 8 | **Interpolation vs extrapolation** | `stage8_response_slice.png` |
| 9–10 | Two-body physics; four integrators validated against Kepler | `stage10_convergence.png` |
| 11 | **ML for orbital motion; error accumulation** | `stage11_error_growth.png` |
| 12 | **Projectile vs planetary, formulation held fixed** | `stage12_flow_map_comparison.png` |
| 13 | **Physics-informed ML: hard, soft and no constraints** | `stage13_extrapolation.png` |

---

## Validation

A study that measures error must first show its reference is right. The orbital ground truth is
numerical, so it was checked four independent ways:

| Check | Result | Theory |
|---|---|---|
| RK4 vs the analytic Kepler solution (2 orbits, e = 0.6) | 8.1 × 10⁻¹³ AU | 0 |
| Convergence orders (quarter orbit) | 0.97, 0.99, 1.99, 3.92 | 1, 1, 2, 4 |
| Velocity Verlet energy drift over 200 orbits | ≤ 1.5 × 10⁻⁸, no trend | bounded |
| Kepler's third law, fitted log–log slope | **1.500000** | 1.5 |

The reference error across the whole 5-year horizon is ≈ 3 × 10⁻¹¹ AU. The smallest Stage 11 held-out one-step position error
is 3.2 × 10⁻⁵ AU — about six orders of magnitude larger — so the reference can be
treated as exact, and that statement is quantitative rather than rhetorical.

The hand-written network's gradients were verified against central finite differences: 6 × 10⁻¹⁰
(data loss), 5 × 10⁻⁹ (initial-condition loss), 8.6 × 10⁻⁷ relative (physics loss, which carries a
1/h² = 10⁴ factor that amplifies finite-difference noise).

---

## Scientific integrity

Stated plainly, because a study of this kind is easy to oversell.

**What is claimed:**
1. Standard regression models interpolate accurately and extrapolate poorly on these systems, with
   the failure mode set by each model's inductive bias.
2. Autoregressive rollout accumulates error rapidly; accumulation, not per-step accuracy,
   determines the usable horizon.
3. The difference between the two systems is structural — one flow map is affine, the other is not.
4. Embedding physical law cuts the data requirement substantially and improves physical
   consistency, with the benefit largest where data is scarce.

**What is *not* claimed:**
1. **Not** that ML outperforms classical physics here. RK4 holds ~10⁻¹¹ AU over five years; the
   best learned rollout exceeds 0.01 AU within about a third of one orbit.
2. **Not** that any model discovered a physical law. Every model was trained on data generated
   *from* the law or constrained by it. The physics-basis fit is exact to round-off; the PINN
   remains approximate. Neither constitutes discovery of a law.
3. **Not** that these results transfer to real experimental data, to chaotic systems, or to model
   scales far larger than those tested.
4. **Not** that the Stage 11 rollout numbers are state of the art. Noise injection during training
   (Sanchez-Gonzalez et al. 2020), multi-step rollout losses (Lam et al. 2023), Hamiltonian
   architectures (Greydanus et al. 2019) and Neural ODEs (Chen et al. 2018) all target exactly this
   failure mode and were deliberately left out so the baseline phenomenon could be measured
   cleanly.

**Two inconvenient results were kept and investigated rather than smoothed over**, and both became
findings: Euler–Cromer measuring as second order (the known conjugacy of symplectic Euler to
Störmer–Verlet, confirmed by measuring at a quarter period as well as a full one), and the
polynomial model's rollout diverging to infinity (now recorded as an explicit outcome, because
dropping an unstable model's worst runs would flatter it).

All 23 literature citations were checked against publisher records. Nothing in
`docs/LITERATURE_REVIEW.md` is invented.

---

## Limitations, and the experiment that would address each

| Limitation | Next experiment |
|---|---|
| Noise-free synthetic data | Repeat the study with Gaussian noise at several levels |
| Neither system is chaotic | Extend to the three-body problem (cf. Breen et al. 2020) |
| Rollout baseline is unoptimised | Add noise injection during training |
| Nothing enforces conservation | Hamiltonian Neural Networks |
| Flow map locked to a single Δt | Neural ODEs with an adaptive solver |
| 2-D, single central body | N-body, orbital resonances |

---

## Acknowledgement

This project was developed with Claude (Anthropic) acting as tutor and research supervisor
throughout, under a working agreement recorded in
[`Role_ AI-ML Tutor + Physics Research Mentor + Python Project Supervisor.md`](Role_%20AI-ML%20Tutor%20+%20Physics%20Research%20Mentor%20+%20Python%20Project%20Supervisor.md):
concepts taught before code, minimum code per step, and every result interpreted rather than
merely printed. The physics, the experimental design and the interpretation are the point of the
exercise; the code is downstream of them.

The completion pass used OpenAI Codex to finish the report, rerun interrupted experiments,
check numerical invariants and gradients, and align the documentation with the implemented design.
