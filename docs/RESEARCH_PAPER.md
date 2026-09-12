# Machine Learning-Based Prediction of Physical Motion: A Comparative Study of Projectile and Planetary Dynamics

**Author:** Aryan Bola
**Course:** Research Methodology (RM), 5th Semester, B.Sc. Physics
**Document status:** Complete — all stages 0–13 finished
**Date:** 12 September 2026
**Code repository:** `RM/` (git, branch `main`)

---

## Abstract

This study asks whether machine learning can learn and predict physical motion, and how its performance
depends on the dynamical character of the system it is applied to. Two systems are used: **projectile
motion**, which has an elementary closed-form solution, and **two-body orbital motion**, whose analytic Kepler
solution requires a numerical root solve. Analytic projectile trajectories and independently validated
numerical orbital trajectories provide references with quantified errors. Four standard model families are compared — linear regression, degree-2 polynomial ridge, random forest, a
multilayer perceptron — alongside two physics-informed approaches — under a design in which one factor is manipulated
at a time.

**Part I (projectile motion).** On held-out data drawn from the training distribution, a Random Forest
reaches RMSE = 3.47 m, R² = 0.9924 for x and RMSE = 4.95 m, R² = 0.9487 for y. Linear regression fails
(R² = 0.745 / 0.584) for a reason the physics predicts exactly: the true mapping contains the products
v₀cos(θ)t and v₀sin(θ)t and the term ½gt², none of which a weighted sum can represent. A learning-curve
experiment over 100–10,000 training rows, repeated five times at each size, shows whole-trajectory error
falling from 14.85 m to 2.50 m and still decreasing at the largest size tested.

**The central negative result.** Those scores measure *interpolation*. When the same trained models are
evaluated outside their training envelope, accuracy collapses: the MLP, the best model in-distribution at
0.11 m, degrades by a factor of **286** to 32.7 m. A one-dimensional slice through input space along which
the true response is exactly linear shows the mechanism directly — a Random Forest's prediction becomes
**exactly constant** beyond its last training split, because a forest can only ever output an average of
training targets.

**Part II (orbital motion).** Four integrators are implemented and validated against the analytic Kepler
solution, recovering convergence orders 0.97, 0.99, 1.99 and 3.92 against theoretical 1, 1, 2 and 4, and
reproducing Kepler's third law with a fitted log–log slope of 1.500000. Against this reference, accurate to
≈ 3 × 10⁻¹¹ AU, three ML formulations are compared. A learned one-step flow map achieves a one-step position
error of 6.4 × 10⁻⁵ AU — yet under autoregressive rollout it exceeds 0.01 AU within 0.37 years, about a third
of an orbit. Perturbing the exact dynamics shows that separation grows only **linearly**: the two-body
problem is integrable, so the divergence is compounding model error, not chaos.

**Comparison.** With the problem formulation, model family and data volume all held fixed, and errors
normalised by each system's own characteristic length, the comparison exposes an important
structural difference, subject to the sampling qualifications in Section 12. The exact projectile flow map is **affine**, and ordinary linear
regression recovers it to ~10⁻¹⁴ m; the orbital acceleration contains 1/r³, and none of these fitted unconstrained models reproduces
the orbital flow map exactly.

**Physics-informed models.** A physics-informed neural network, implemented from scratch in NumPy with every
gradient verified against finite differences, reaches 1.39 m trained on **zero labelled points** — a level a
plain network of identical architecture needs roughly 200 labelled points to match — and is 14.7× more
accurate than that plain network in the region of the flight where neither has data. A hard-constraint model
that fits in a physics basis is exact to floating-point round-off everywhere, including far outside the
training range, and recovers −g/2 = −4.905000.

**Conclusion.** Machine learning is not equally effective for the two systems, and the study locates the
difference precisely. For long orbital rollouts, the classical reference remains orders of magnitude more accurate. No
model discovered a physical law; exact projectile fits arise where the correct function is already
contained in the chosen representation.

**Keywords:** computational physics, supervised regression, surrogate models, learned simulators, error
accumulation, symplectic integration, physics-informed neural networks, extrapolation, synthetic data.

---

## How To Read This Document

This report serves two purposes at once.

1. **As a research paper.** Sections 1, 2, 4, 6–14 follow the standard RM structure (Introduction, Theory,
   Methodology, Results, Discussion, Conclusion) and read as a conventional academic report.

2. **As a complete build guide for a reader with no AI/ML background.** Section 3 teaches every
   machine-learning concept used, from nothing. Section 5 contains the code in the order it was written.
   Section 15 explains how to reproduce the study from the repository.

If you know no machine learning at all, read in this order: **Section 3 → Section 2 → Section 5 →
Section 15**, then return to the rest.

**Companion documents.** The formal methodology chapter is `docs/METHODOLOGY.md`; the literature review, with
23 verified sources, is `docs/LITERATURE_REVIEW.md`; the talk plan is `docs/PRESENTATION_OUTLINE.md`; viva
questions and answers are in `docs/VIVA_PREP.md`.

**Provenance of every number.** Figures were produced by the repository scripts; numerical result tables are stored under `results/`.
Some illustrative curves are recomputed directly from the equations rather than saved as separate CSVs. No result has been estimated,
rounded from memory, or invented, and no figure was drawn or adjusted by hand. `python run_all.py` regenerates
all of it in about 49 minutes.

---

## Table of Contents

**Part I — Projectile motion**
1. Introduction
2. Theoretical Background: The Physics
3. Machine Learning From Zero (for readers with no ML background)
4. Methodology
5. Implementation, Stage by Stage
6. Results
7. Discussion

**Part I (continued) — The experiments that matter**
8. Stage 7 — How Much Data Do We Actually Need?
9. Stage 8 — Interpolation, Extrapolation, and What R² = 0.99 Was Really Measuring

**Part II — Planetary motion**
10. Stages 9–10 — The Physics and the Numerical Baseline
11. Stage 11 — Machine Learning for Orbital Motion
12. Stage 12 — The Controlled Comparison

**Part III — Synthesis**
13. Stage 13 — Physics-Informed Machine Learning
14. General Discussion, Limitations and Conclusions
15. How To Reproduce This Project From Scratch
16. References

- Appendix A — Source File Index
- Appendix B — Glossary
- Appendix C — Viva Preparation
- Appendix D — Index of Every Figure and Table

---

# 1. Introduction

## 1.1 Background

Classical mechanics gives us closed-form equations for a large class of motions. If we know the initial speed,
launch angle and height of a thrown object, we can compute its position at any later time exactly, in a single
line of arithmetic. Machine learning, by contrast, learns an input–output relationship from examples, without
being told the rule.

Over the last decade ML has become a standard tool inside computational physics, not as a replacement for
physical law but as a **surrogate**: a fast approximate stand-in for an expensive calculation. Surrogate models
are used in fluid dynamics, molecular simulation, climate modelling and orbital mechanics, where the governing
equations are known but solving them repeatedly is computationally costly.

## 1.2 Motivation and the Obvious Objection

The first objection any physicist will raise — and the objection this project must answer before anything else —
is:

> **"Why use AI to predict something that physics equations can already calculate exactly?"**

If we only wanted the answer, ML would be absurd here. Projectile motion has an exact analytical solution that
runs in microseconds. Using a 100-tree Random Forest to approximate it is strictly worse in every practical sense.

The defensible answer has three parts.

**(a) Projectile motion is a control, not the object of study.** We are not proposing ML as a tool for projectile
motion. We are using a system whose exact answer is known so that we can *measure how well ML learns physics at
all*. When the ground truth is exact, every unit of error is unambiguously the model's fault. This is the same
logic that makes a physicist calibrate an instrument on a known standard before measuring an unknown.

**(b) The interesting question is comparative.** The research question concerns how ML performance *degrades*
as the physics becomes harder. Projectile motion is a non-chaotic, closed-form, single-step mapping. Planetary
motion (Part II) is a coupled differential system integrated over time, where errors compound. Comparing the
two tells us something about ML that neither system tells us alone. To make that comparison, we need the easy
case measured properly first.

**(c) Surrogate modelling is a genuine application domain.** In cases where the equations are known but
evaluation is expensive — many-body gravitational systems, N-body cosmological simulations, trajectory
optimisation loops that need millions of evaluations — a trained model that is accurate enough and thousands of
times faster has real value. Establishing *how* accurate such a surrogate can be, and what fails when it is
inaccurate, is a legitimate research contribution.

## 1.3 Research Gap

Introductory ML-for-physics demonstrations typically report a high R² on a held-out test set and stop there.
What is far less commonly examined at this level is the distinction between **point-wise statistical accuracy**
and **trajectory-level physical fidelity** — whether a model that scores well on scattered test points actually
produces a physically valid curve when asked to trace one out. Section 6.4 of this report shows those two things
diverging sharply, which is the gap this study addresses at its own scale.

## 1.4 Research Question

**Main question:**
> How effectively can machine learning approximate and predict physical motion, and how does its predictive
> performance differ between projectile motion and planetary dynamics?

**Sub-questions addressed in Part I (this document):**

| # | Sub-question | Status |
|---|---|---|
| 1 | Can an ML model accurately predict projectile trajectories? | Answered — Section 6 |
| 2 | How does ML prediction compare with classical physics calculation? | Answered — Sections 6.4, 7.2 |
| 3 | Does a good test score mean the model learned the physics? | Answered — Section 7.3 |
| 4 | How does the amount of training data affect accuracy? | **Stage 7 — in progress** |
| 5 | Does the model generalise outside its training range? | Stage 8 — planned |

**Sub-questions addressed in Part II:** applicability to planetary motion, error accumulation over time,
and how predictive performance differs between physical systems.

## 1.5 Objectives

1. Implement projectile motion from first principles in Python as a physics baseline.
2. Generate a controlled synthetic dataset from that baseline.
3. Train at least two structurally different regression models on the dataset.
4. Evaluate them with standard regression metrics on data never seen during training.
5. Compare ML predictions with the classical solution both statistically and visually.
6. Quantify the effect of training-set size on accuracy (Stage 7).
7. Carry the validated methodology forward to the planetary-motion system.

## 1.6 Scope and Delimitations

**In scope for the complete study:** ideal two-dimensional projectile and restricted two-body
orbital motion; noise-free synthetic data; linear, polynomial, forest and neural models; direct
and recursive prediction; physical constraints and numerical-reference validation.

**Out of scope:** air resistance, spin/Magnus effects, wind, three-dimensional motion and real
experimental measurements. The introductory Stages 0–6 omit neural networks and recursive
prediction; the later stages add them. Section 7.5 records the initial scope, while Section 14
states the limitations of the complete study.

---

# 2. Theoretical Background: The Physics

Because this is a physics research project and not merely an AI demonstration, the equations used to generate
our data are derived rather than asserted, and their assumptions are stated explicitly.

## 2.1 Derivation From Newton's Second Law

Consider a point mass launched from height y₀ with initial speed v₀ at angle θ above the horizontal. Once the
projectile is in flight, and if we neglect air resistance, the **only** force acting is gravity, which acts
vertically downward. Newton's second law, F = ma, therefore gives:

```
Horizontal:   Fx = 0        ⟹   ax = 0
Vertical:     Fy = −mg      ⟹   ay = −g
```

The mass cancels in the vertical equation — this is why a heavy and a light object fall identically in vacuum,
and why our dataset contains no mass column.

The initial velocity resolves into components by simple trigonometry:

```
vx(0) = v₀ cos(θ)
vy(0) = v₀ sin(θ)
```

Integrating the accelerations once with respect to time:

```
vx(t) = v₀ cos(θ)              (constant — no horizontal force)
vy(t) = v₀ sin(θ) − g t
```

Integrating a second time, with x(0) = 0 and y(0) = y₀:

```
x(t) = v₀ cos(θ) · t                                   ... (2.1)
y(t) = y₀ + v₀ sin(θ) · t − ½ g t²                     ... (2.2)
```

These two equations are the **entire ground truth** of this project. Everything the ML model is asked to learn
is contained in them.

## 2.2 Time of Flight

We need to know when the projectile lands, so that we never generate data for times after impact (a projectile
at t > t_flight would have negative y, which is physically meaningless for our setup). Setting y(t) = 0 in
(2.2) and solving the quadratic ½g t² − v₀sin(θ)t − y₀ = 0 gives:

```
              v₀ sin(θ) + √( (v₀ sin(θ))² + 2 g y₀ )
t_flight  =  ─────────────────────────────────────────      ... (2.3)
                              g
```

The positive root is taken because the negative root corresponds to a time before launch. Note that when
y₀ = 0 this correctly reduces to the familiar t_flight = 2v₀sin(θ)/g.

## 2.3 Units and Physical Meaning

| Symbol | Meaning | Unit | Range used in this study |
|---|---|---|---|
| v₀ | initial speed | m·s⁻¹ | 5 – 50 |
| θ | launch angle above horizontal | rad (5° – 85°) | 0.087 – 1.484 |
| y₀ | launch height | m | 0 – 20 |
| t | time since launch | s | 0 – t_flight (per sample) |
| g | gravitational acceleration | m·s⁻² | 9.81 (fixed) |
| x, y | position coordinates | m | outputs |

The ranges were chosen to be physically reasonable for human-scale throwing and launching: 5 m/s is a gentle
toss, 50 m/s is roughly a fast-bowled cricket ball; heights up to 20 m correspond to a launch from a building
of a few storeys. Angles are clipped away from 0° and 90° because both extremes are degenerate (at exactly 90°
the horizontal range is identically zero, which would give the model a large cluster of x = 0 rows that carry
no information about the v₀cos(θ)t relationship).

## 2.4 Assumptions and Their Limitations

Every one of the following assumptions is built into equations (2.1)–(2.3), and therefore into every row of our
dataset. A model trained on this data inherits all of them.

| Assumption | Why it is made | When it fails |
|---|---|---|
| **No air resistance** | Makes the equations exactly solvable | Fails badly for light objects, high speeds, or large cross-sections. A real cricket ball at 50 m/s can lose >30% of its vacuum range |
| **Constant g = 9.81 m/s²** | g varies by <0.1% over our height range | Fails for ballistic/orbital altitudes where g ∝ 1/r² |
| **Flat Earth** | Over ranges of ~200 m, Earth's curvature is negligible | Fails for long-range ballistics (>10 km) |
| **Point mass** | No rotation, no spin, no extended body | Fails for spinning balls (Magnus effect), non-rigid bodies |
| **No wind / still air** | Keeps the system deterministic | Any crosswind adds an unmodelled acceleration |
| **2D motion** | Motion stays in the vertical launch plane | Valid given the above; breaks with spin or crosswind |

**This matters for the research conclusion.** Section 7.4 argues that because our data is generated from these
equations, our ML model can at best learn *this idealised physics*, not *real physics*. It has never seen air
resistance and cannot possibly account for it.

---

# 3. Machine Learning From Zero

*This section assumes no prior knowledge whatsoever. A reader who already knows supervised regression can skip
to Section 4.*

## 3.1 What Machine Learning Actually Is

A normal computer program is a rule you write, applied to data:

```
rule + input  →  output          (ordinary programming)
```

Machine learning inverts this:

```
input + output  →  rule          (machine learning)
```

You show the computer many examples of inputs paired with their correct outputs, and an algorithm searches for
a rule that reproduces those pairs. The hope is that the discovered rule also works on inputs it has never seen.

In our project:
- **Input** = [initial speed, launch angle, initial height, time] — four numbers.
- **Output** = [x position, y position] — two numbers.
- The rule the model is searching for is, unknown to it, equations (2.1) and (2.2).

## 3.2 Supervised Learning, Features and Targets

Because we give the model both the questions *and* the correct answers during training, this is **supervised
learning**. (Unsupervised learning, where you only give inputs and ask the machine to find structure, is not
used in this project.)

Vocabulary you will meet everywhere:

- **Feature** — one input column. We have four features: `v0`, `theta`, `y0`, `t`.
- **Target** (or *label*) — one output column we want predicted. We have two: `x`, `y`.
- **Sample** (or *row*, *instance*, *example*) — one complete input–output pair.
- **Feature matrix, `X`** — the table of all features for all samples. Shape here: 2000 × 4.
- **Target matrix, `y`** — the table of all targets. Shape here: 2000 × 2.

> **Why is `t` a feature and not something else?** This is a genuine design decision, discussed in Section 4.3.
> Because `t` is an *input to the physics equations*, it belongs on the input side. The model is being asked:
> "given these launch conditions and this instant in time, where is the object?"

## 3.3 Regression vs Classification

There are two basic supervised tasks:

- **Classification** — predict a category ("cat" or "dog", "spam" or "not spam").
- **Regression** — predict a continuous number (a position in metres).

Ours is **regression**. Specifically it is *multi-output regression*, because we predict two numbers (x and y)
from the same inputs. Scikit-learn handles this automatically for both models we use: it effectively fits one
model per output and packages them together.

## 3.4 What `model.fit()` and `model.predict()` Really Do

These are the two operations at the heart of every scikit-learn model.

**`model.fit(X_train, y_train)` — the learning step.**
The model looks at the training samples and adjusts its internal parameters to reduce its error on them. What
"parameters" means depends on the model:

- For **Linear Regression**, the parameters are the coefficients w₁…w₄ and intercept b in the equation
  `prediction = w₁·v₀ + w₂·θ + w₃·y₀ + w₄·t + b`. `fit` computes the coefficient values that minimise the sum
  of squared errors. There is a closed-form linear-algebra solution for this — no iteration is needed.
- For a **Random Forest**, `fit` grows 100 decision trees. Each tree repeatedly asks yes/no questions about the
  features ("is v₀ > 27.4?") and splits the data, until each final group (*leaf*) is small. The value stored in
  a leaf is the average target value of the training samples that landed there.

**`model.predict(X_new)` — the inference step.**
The model applies its learned rule to new inputs. It does not learn anything here and does not change. Linear
Regression plugs the numbers into its learned equation. The Random Forest drops each input down all 100 trees,
reads the leaf value from each, and averages the 100 answers.

**The crucial consequence for our project:** a Random Forest can only ever output values it has seen averages of
during training. It cannot extrapolate beyond the range of its training targets, and *between* training points
it produces a staircase of constant values, not a smooth curve. This single fact explains the most important
result in Section 6.4.

## 3.5 Why We Must Split the Data

If you evaluate a model on the same data you trained it on, you are measuring **memorisation**, not learning.
A sufficiently flexible model can achieve near-zero error on training data by essentially storing the answers,
while being useless on anything new. This failure is called **overfitting**.

The standard defence is a **train/test split**: randomly set aside a fraction of the data (we use 20%), never
let the model see it during `fit`, and evaluate only on that held-out portion. A score on held-out data is an
honest estimate of performance on genuinely new inputs.

**`random_state`** — the split is random, so a different run would produce a different split and slightly
different numbers. Setting `random_state=42` fixes the random number generator's seed so that the "random"
choices are identical every time the code runs. This is what makes our results **reproducible**: anyone running
our code gets exactly the metrics reported in Section 6. (The value 42 is a convention, not a magic number;
any fixed integer works.)

## 3.6 The Two Models We Chose, and Why

The project roadmap explicitly warns against reaching for a neural network merely because the project is about
AI. We deliberately chose two simple, structurally opposite models:

**Linear Regression** — the simplest possible regression model. It assumes the output is a weighted sum of the
inputs. We include it as a **deliberate baseline that we expect to fail**, because the physics tells us in
advance that the true relationship is *not* linear: equation (2.1) contains the product v₀·cos(θ)·t (three
features multiplied together) and equation (2.2) contains t². A linear model has no mechanism to represent
either. Its failure is therefore *diagnostic*: it confirms our dataset genuinely encodes non-linear physics.

**Random Forest Regressor** — an ensemble of 100 decision trees, each trained on a random resample of the data
with randomised feature choices at each split, whose predictions are averaged. It captures non-linear
interactions between features automatically, requires no feature scaling, and is robust with modest amounts of
data. It is a reasonable first serious model for a tabular regression problem of this size.

**Why not a neural network (yet)?** With 2,000 samples and four features, a neural network would offer no clear
advantage while adding architecture choices, learning-rate tuning and training instability — all of which would
obscure the physics question we are actually asking. A neural network becomes a defensible addition in Part II,
where smoothness of the learned function matters much more (see Section 7.6).

## 3.7 The Evaluation Metrics, Explained

Let yᵢ be the true value of a sample, ŷᵢ the model's prediction, and n the number of test samples.

**MAE — Mean Absolute Error**
```
MAE = (1/n) Σ |yᵢ − ŷᵢ|
```
The average size of the error, ignoring sign. Same units as the target (metres). *Intuition:* "on average, the
prediction is off by this many metres." Easy to explain to a non-specialist; treats all errors proportionally.

**MSE — Mean Squared Error**
```
MSE = (1/n) Σ (yᵢ − ŷᵢ)²
```
The average of the *squared* errors. Units are metres². Because errors are squared, a single large error
contributes disproportionately, so MSE punishes big mistakes much harder than many small ones. Its awkward
units make it hard to interpret directly, which is why we report its square root instead.

**RMSE — Root Mean Squared Error**
```
RMSE = √MSE
```
Back in metres, but still weighted toward large errors. **RMSE ≥ MAE always**, and the gap between them tells
you something: if RMSE ≫ MAE, the error distribution has a few large outliers. *Concrete interpretation for this
project:* an RMSE of 0.5 m would mean the predicted landing point is typically half a metre from the true one —
excellent for a projectile with a 100 m range, poor for a dart board.

**R² — Coefficient of Determination**
```
R² = 1 − ( Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)² )
```
where ȳ is the mean of the true values. This compares your model against the dumbest possible model — one that
always predicts the average, ignoring the inputs entirely.

- R² = 1.0 → perfect prediction.
- R² = 0.0 → your model is no better than always guessing the mean.
- R² < 0 → your model is *worse* than guessing the mean.

R² is dimensionless, which makes it useful for comparing across the x and y coordinates even though they have
different spreads. It is **not** a substitute for RMSE: R² tells you the *fraction of variance explained*, RMSE
tells you *how many metres you are wrong by*. A paper should report both, and this one does.

> **Warning that matters for Section 7.3:** all four metrics are computed on *scattered independent test points*.
> None of them measures whether the model produces a physically coherent *trajectory*. This is exactly the gap
> our Stage 6 visualisation exposed.

---

# 4. Methodology

## 4.1 Research Design

This study is best characterised as **quantitative, computational, simulation-based comparative experimental
research**. Each descriptor is chosen deliberately:

- **Quantitative** — all conclusions rest on numerical error metrics, not on qualitative judgement.
- **Computational / simulation-based** — no physical apparatus is used. The "experiment" is executed entirely in
  software, and the data is generated by a simulation rather than measured. This is a recognised and legitimate
  research mode in physics; it is how most of modern computational physics operates.
- **Comparative** — the design compares (i) two ML models against each other, (ii) ML against the classical
  analytical solution, and ultimately (iii) projectile dynamics against planetary dynamics.
- **Experimental** — we manipulate independent variables under controlled conditions and measure the effect on
  dependent variables.

## 4.2 Variables

| Type | Variable | Role in this study |
|---|---|---|
| **Independent** | Model type (Linear vs Random Forest) | Manipulated in Stages 3–6 |
| | Training-set size (100 → 2,000 samples) | Manipulated in Stage 7 (in progress) |
| | Physical-parameter range (train vs test) | To be manipulated in Stage 8 |
| **Dependent** | MAE, RMSE, R² for x and for y | Measured on held-out test data |
| | Trajectory deviation from classical curve | Measured in Stage 6 |
| **Controlled** | g = 9.81 m·s⁻² | Fixed for every sample |
| | Dataset generation seed (42) | Fixed — same dataset every run |
| | Train/test split (80/20, `random_state=42`) | Identical split for both models |
| | Feature ranges (v₀, θ, y₀) | Identical for all models compared |
| | Random Forest seed (`random_state=42`) | Fixed — same forest every run |
| | Evaluation procedure and metrics | Identical for both models |

Holding the split and both seeds fixed is what makes the Linear-vs-Forest comparison **fair**: the two models
see exactly the same 1,600 training rows and are scored on exactly the same 400 test rows.

## 4.3 Dataset Design — The Key Methodological Decision

Our first dataset attempt (commit `a8613b7`, file `data/projectile.csv`, 100 rows) was generated by sweeping `t`
along **one single trajectory** with fixed v₀, θ and y₀. This was discarded, and understanding why is important
methodologically.

If every row shares the same launch conditions, then the columns `v0`, `theta` and `y0` are *constant*. A model
trained on that data sees no variation in three of its four features and therefore cannot possibly learn how x
and y depend on them. It would learn only x(t) and y(t) for one specific throw — a curve-fit, not a physical
relationship.

The corrected design (commit `18a64af`, file `data/projectile_dataset.csv`, 2,000 rows) makes **each row an
independent random experiment**:

1. Draw v₀ ~ Uniform(5, 50) m/s
2. Draw θ ~ Uniform(5°, 85°), stored in radians
3. Draw y₀ ~ Uniform(0, 20) m
4. Compute that sample's own t_flight from equation (2.3)
5. Draw t ~ Uniform(0, t_flight) — so time is sampled *within the valid flight window of that specific launch*
6. Compute x and y from equations (2.1) and (2.2)

Step 5 is the subtle one. Drawing `t` from a fixed global range would produce many physically invalid rows
(times after the projectile has already landed, giving negative y). Drawing it from each sample's own flight
window guarantees every row is physically valid, with y ≥ 0 by construction. The realised minimum y in the
dataset is 0.0008 m, confirming this.

**Sample size justification.** 2,000 samples for a 4-feature problem gives 500 samples per feature dimension,
which is comfortable for a Random Forest on a smooth, noise-free function. It is small enough to train in under
two seconds on a laptop, which matters because Stage 7 requires retraining many times. Whether 2,000 is more
than necessary is precisely the question Stage 7 answers empirically rather than by assertion.

**Realised dataset statistics** (computed from `data/projectile_dataset.csv`):

| Column | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| v0 (m/s) | 2000 | 27.628 | 13.070 | 5.023 | 15.983 | 27.321 | 39.255 | 49.974 |
| theta (rad) | 2000 | 0.776 | 0.405 | 0.088 | 0.431 | 0.778 | 1.128 | 1.483 |
| y0 (m) | 2000 | 9.764 | 5.689 | 0.006 | 4.829 | 9.666 | 14.624 | 19.994 |
| t (s) | 2000 | 2.089 | 1.741 | 0.001 | 0.789 | 1.611 | 2.898 | 9.210 |
| **x (m)** | 2000 | 38.193 | 42.884 | 0.027 | 7.838 | 21.833 | 53.561 | 241.924 |
| **y (m)** | 2000 | 23.302 | 22.781 | 0.001 | 8.202 | 16.093 | 29.465 | 139.746 |

Two features of this table matter for interpreting the results. First, the target ranges are wide — x spans
0 to 242 m — so an RMSE of 3.5 m is roughly 1.4% of the range, not a large error. Second, both targets are
strongly right-skewed (mean well below the midpoint of the range), because most random launch configurations
produce modest ranges and only a few produce very long ones. Section 6.3 shows this skew directly in the
residual structure.

**Is synthetic data scientifically valid?** Yes, for this purpose, with an explicit caveat. It is valid because
the research question is about *the learning algorithm's ability to approximate a known mapping*, and synthetic
data gives us that mapping exactly, with no measurement noise, no missing values and unlimited samples. It would
**not** be valid if the question were "can ML predict real thrown objects" — for that, the absence of air
resistance would be a fatal flaw. This limitation is carried forward explicitly in Section 7.5.

## 4.4 Training Procedure

```
Full dataset (2,000 rows)
        │
        ├─ train_test_split(test_size=0.2, random_state=42)
        │
        ├──→ Training set: 1,600 rows  ──→ model.fit()
        │
        └──→ Test set:       400 rows  ──→ model.predict() ──→ metrics
```

Both models are trained on the identical 1,600-row training set and evaluated on the identical 400-row test set.
No hyperparameter tuning was performed at this stage: the Random Forest uses scikit-learn defaults apart from
`n_estimators=100` and a fixed seed. This is a deliberate choice — introducing tuning before establishing a
baseline would make it impossible to attribute performance differences to model *type* rather than to tuning
effort.

## 4.5 Software Environment

| Component | Role |
|---|---|
| Python 3.12 | Language runtime |
| NumPy | Array mathematics, random sampling |
| pandas | Tabular data handling, CSV I/O |
| scikit-learn | `LinearRegression`, `RandomForestRegressor`, `train_test_split`, metrics |
| Matplotlib | All figures |

## 4.6 Repository Structure

```
RM/
├── data/
│   ├── generation/
│   │   └── projectileDataGeneration.py   # Stages 1–2: physics + dataset generation
│   └── projectile_dataset.csv            # 2,000-row generated dataset
├── src/
│   └── projectile_ml.py                  # Stages 3–6: training, evaluation, orchestration
├── plots/
│   └── projectile_plots.py               # Stage 6: all plotting functions
├── results/
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   └── trajectory_comparison.png
├── docs/
│   └── RESEARCH_PAPER.md                 # this document
└── README.md
```

The separation of `plots/` from `src/` was introduced in commit `4702cf9`. The rationale: plotting code is long,
repetitive and conceptually separate from the scientific logic. Keeping it out of `projectile_ml.py` reduced
that file from 148 to 82 lines and made the actual experiment readable at a glance. This is the "use functions
rather than one giant script" principle applied concretely.

---

# 5. Implementation, Stage by Stage

This section walks through every stage in the order it was actually built, with the code, an explanation of the
important lines, the expected output, and what to verify before moving on. A reader following along can build the
entire project from this section alone.

## Stage 0 — Understand the Research Problem

No code. This stage produced the justification presented in Section 1.2 and the research question in Section 1.4.

**Checkpoint before proceeding:** you should be able to answer, in your own words, "why use AI when we already
have the equation?" If you cannot, the rest of the project is not defensible in a viva. The answer is in
Section 1.2.

## Stage 1 — Classical Projectile Motion (The Physics Baseline)

**Goal:** implement equations (2.1)–(2.3) in Python and confirm they behave physically. This code becomes the
**ground truth** for everything after it.

**File:** `data/generation/projectileDataGeneration.py`

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------- CONSTANTS --------------------

g = 9.81  # acceleration due to gravity (m/s^2)


def generateProjectileData(numSamples, v0, theta, y0):
    """
    Generate projectile motion data for ONE trajectory.

    Parameters:
        numSamples : Number of points in the trajectory
        v0         : Initial velocity (m/s)
        theta      : Launch angle (radians)
        y0         : Initial height (m)

    Returns:
        pandas DataFrame containing time, x and y
    """

    # Calculate time of flight  -- equation (2.3)
    timeOfFlight = (
        v0 * np.sin(theta)
        + np.sqrt((v0 * np.sin(theta)) ** 2 + 2 * g * y0)
    ) / g

    # Generate evenly spaced time values from launch to landing
    t = np.linspace(0, timeOfFlight, numSamples)

    # Generate x(t)  -- equation (2.1)
    x = v0 * np.cos(theta) * t

    # Generate y(t)  -- equation (2.2)
    y = y0 + v0 * np.sin(theta) * t - 0.5 * g * t**2

    return pd.DataFrame({"time": t, "x": x, "y": y})
```

**Important lines explained:**

- `np.linspace(0, timeOfFlight, numSamples)` creates `numSamples` evenly spaced time values from launch to
  landing. Using `linspace` rather than a hand-written loop means the arithmetic below operates on whole arrays
  at once (*vectorisation*) — this is both faster and closer to how the equations are written on paper.
- `np.sin` and `np.cos` expect **radians**, not degrees. Passing degrees is the single most common bug in
  projectile code, and it fails silently — you get a plausible-looking wrong parabola. Use `np.deg2rad()` to
  convert.
- `t**2` in the y-equation is what makes the trajectory a parabola. It is also, as Section 6.2 shows, precisely
  what Linear Regression cannot represent.

**Expected output and what to check:** plot x against y and confirm (a) the curve is a smooth downward parabola,
(b) it starts at y = y₀ and ends at y ≈ 0, (c) increasing v₀ increases the range, and (d) with y₀ = 0 the maximum
range occurs at θ = 45°. That last check is the strongest physical validation available — if 45° is not optimal,
something is wrong with your angle handling.

## Stage 2 — Generate the ML Dataset

**Goal:** produce many independent samples spanning a range of launch conditions, as argued in Section 4.3.

```python
def generateProjectileDataset(numSamples, seed=42):
    """
    Generate an ML-ready projectile dataset.

    Each row = one random (v0, theta, y0, t) input combination and
    the resulting (x, y) position. Rows are independent samples, not
    points along a single trajectory, so the model sees how x and y
    vary WITH v0, theta, y0 -- not just with t.
    """

    rng = np.random.default_rng(seed)

    # Feature ranges -- keep them physically reasonable
    v0    = rng.uniform(5, 50, numSamples)                            # m/s
    theta = rng.uniform(np.deg2rad(5), np.deg2rad(85), numSamples)    # rad
    y0    = rng.uniform(0, 20, numSamples)                            # m

    # Time of flight for each sample (when y returns to 0)
    timeOfFlight = (
        v0 * np.sin(theta)
        + np.sqrt((v0 * np.sin(theta)) ** 2 + 2 * g * y0)
    ) / g

    # Pick t randomly within [0, timeOfFlight] for each sample
    # so t is a feature too, not the sweep variable.
    t = rng.uniform(0, timeOfFlight)

    x = v0 * np.cos(theta) * t
    y = y0 + v0 * np.sin(theta) * t - 0.5 * g * t**2

    return pd.DataFrame({
        "v0": v0, "theta": theta, "y0": y0, "t": t, "x": x, "y": y
    })
```

**Important lines explained:**

- `np.random.default_rng(seed)` creates a seeded random-number generator. Because the seed is fixed at 42, this
  function produces **byte-identical output every time it runs**. Anyone who runs this code gets our exact
  dataset, and therefore our exact metrics. This is reproducibility in practice, not just in principle.
- `rng.uniform(0, timeOfFlight)` — note that `timeOfFlight` here is an *array* of 2,000 different values, one per
  sample. NumPy broadcasts this, drawing each sample's `t` from its own individual `[0, t_flightᵢ]` interval.
  This one line is what guarantees every row is physically valid. Writing `rng.uniform(0, 10, numSamples)`
  instead would silently poison roughly half the dataset with post-landing times.
- The dataset stores **theta in radians**, not degrees. This is intentional: the model consumes the same units
  the physics equations use, so there is no hidden conversion anywhere in the pipeline.

**Run it:**
```bash
cd data/generation
python3 projectileDataGeneration.py
```

**Expected output:**
```
          v0     theta         y0         t          x          y
0  39.828022  1.263942  14.321999  3.689825  44.390433  87.635154
1  24.749530  0.710544   7.142312  0.438125   8.219359  13.273331
...
Dataset generated successfully.
Saved to: ../projectile_dataset.csv
Rows: 2000, Columns: ['v0', 'theta', 'y0', 't', 'x', 'y']
```

**What to check:** open the CSV and confirm (a) 2,001 lines including the header, (b) no y value is negative,
(c) v₀, θ and y₀ genuinely vary from row to row — if they are constant, you have made the Stage-2 mistake
described in Section 4.3.

## Stages 3 & 4 — Train the Models on a Train/Test Split

**Goal:** load the dataset, separate features from targets, split it, and train both models.

**File:** `src/projectile_ml.py`

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# allow "from plots.projectile_plots import ..." when run from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from plots.projectile_plots import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_trajectory_comparison,
)

g = 9.81
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# -------------------- LOAD DATA --------------------

dataset = pd.read_csv("data/projectile_dataset.csv")

features = dataset[["v0", "theta", "y0", "t"]]
targets  = dataset[["x", "y"]]

X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=42
)

# -------------------- TRAIN MODELS --------------------

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)
```

**Important lines explained:**

- `dataset[["v0", "theta", "y0", "t"]]` — the double brackets select multiple columns and return a DataFrame
  (a table). Single brackets would return a Series (one column), which scikit-learn would reject. This selection
  is where the feature/target decision from Section 3.2 becomes concrete code.
- `train_test_split(..., test_size=0.2, random_state=42)` shuffles the rows and splits off 20% (400 rows) as the
  test set. It returns four objects in a fixed order: train features, test features, train targets, test targets.
  Getting that order wrong is a classic beginner bug that produces nonsense metrics.
- `RandomForestRegressor(n_estimators=100, random_state=42)` — `n_estimators` is the number of trees. More trees
  generally means slightly better and more stable predictions with diminishing returns and longer training.
  `random_state` fixes both the bootstrap resampling and the random feature selection at each split, so the same
  forest is grown every run.
- Both models accept a 2-column target and handle multi-output regression internally, so we never write separate
  x and y models.

**What to check:** `X_train.shape` should be `(1600, 4)` and `X_test.shape` should be `(400, 4)`. If the second
number is not 4, your feature selection is wrong.

## Stage 5 — Evaluation Metrics

**Goal:** measure both models on the held-out test set with MAE, RMSE and R², reported separately for x and y.

```python
def report_metrics(model_name, y_true, y_pred):
    """
    Print MAE, RMSE, R^2 for x and y separately.
    y_true, y_pred: arrays/dataframes with columns [x, y] in that order.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    print(f"\n--- {model_name} ---")
    for i, label in enumerate(["x", "y"]):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{label}: MAE={mae:.3f} m, RMSE={rmse:.3f} m, R2={r2:.4f}")


report_metrics("Linear Regression", y_test, linear_predictions)
report_metrics("Random Forest",     y_test, forest_predictions)
```

**Important lines explained:**

- `np.asarray(...)` converts the pandas DataFrame to a plain NumPy array so that `[:, i]` selects column `i` by
  position. Without this, pandas would try to interpret `i` as a column *name* and fail.
- The loop over `["x", "y"]` reports each coordinate separately rather than averaging them. This is a deliberate
  scientific choice: x and y are governed by different equations — x is linear in t, y is quadratic — so a
  combined metric would hide exactly the asymmetry we most want to see. Section 6.2 shows that asymmetry clearly.
- `np.sqrt(mean_squared_error(...))` computes RMSE. Scikit-learn's `mean_squared_error` returns MSE; taking the
  square root manually converts it to the interpretable metres-scale metric of Section 3.7.

## Stage 6 — Visual Comparison

**Goal:** three diagnostic figures that show what the numbers cannot. All plotting functions live in
`plots/projectile_plots.py`; `src/projectile_ml.py` calls them.

### 6a. Actual vs Predicted scatter

```python
plot_actual_vs_predicted(
    y_test_arr, forest_predictions,
    save_path=RESULTS_DIR / "actual_vs_predicted.png"
)
```

Plots true value on the horizontal axis against predicted value on the vertical, with a red dashed 45° line
marking perfect prediction. Points on the line are perfect; vertical scatter about the line is error. This plot
answers "how good is the model overall, and does its accuracy depend on the magnitude of the value?"

### 6b. Residual plot

```python
residuals_linear = y_test_arr[:, 0] - linear_predictions[:, 0]
residuals_forest = y_test_arr[:, 0] - forest_predictions[:, 0]

plot_residuals(
    y_test_arr[:, 0], residuals_linear, residuals_forest,
    save_path=RESULTS_DIR / "residuals.png"
)
```

A **residual** is `actual − predicted`, i.e. the signed error of a single prediction. Plotting residuals against
the true value reveals *structure* in the errors. Random scatter tightly around zero means the model has captured
everything systematic and only noise remains. Any visible pattern — a curve, a fan, a tilt — means the model is
making predictable mistakes, which means there is signal it has not learned. This is a far more informative
diagnostic than a single summary number.

### 6c. Trajectory comparison — the decisive test

```python
def classical_trajectory(v0, theta, y0, t_array):
    x = v0 * np.cos(theta) * t_array
    y = y0 + v0 * np.sin(theta) * t_array - 0.5 * g * t_array**2
    return x, y

# pick one condition from the test set to compare against
sample = X_test.iloc[0]
v0_s, theta_s, y0_s = sample["v0"], sample["theta"], sample["y0"]

time_of_flight = (
    v0_s * np.sin(theta_s)
    + np.sqrt((v0_s * np.sin(theta_s)) ** 2 + 2 * g * y0_s)
) / g
t_sweep = np.linspace(0, time_of_flight, 50)

x_classical, y_classical = classical_trajectory(v0_s, theta_s, y0_s, t_sweep)

sweep_features = pd.DataFrame({
    "v0": v0_s, "theta": theta_s, "y0": y0_s, "t": t_sweep
})
forest_sweep_pred = forest_model.predict(sweep_features)
linear_sweep_pred = linear_model.predict(sweep_features)

plot_trajectory_comparison(
    x_classical, y_classical,
    forest_sweep_pred, linear_sweep_pred,
    v0_s, theta_s, y0_s,
    save_path=RESULTS_DIR / "trajectory_comparison.png"
)
```

**Why this is the most important experiment in Stage 6.** The metrics in Stage 5 evaluate the models on 400
*scattered, unrelated* points. This code does something fundamentally different: it fixes one launch condition
and sweeps time continuously, asking each model to draw a **whole trajectory**. It then overlays the exact
classical parabola on top.

This converts an abstract question ("is the error small?") into a physical one ("does the model produce a curve
that could actually be the path of an object?"). Section 6.4 shows that the two questions have different answers.

The specific condition used is `X_test.iloc[0]`: **v₀ = 31.21 m/s, θ = 0.7435 rad (42.6°), y₀ = 12.03 m**, giving
t_flight = 4.816 s and a true range of 110.65 m. It comes from the *test* set, so neither model has ever trained
on this exact launch condition.

**Run the whole pipeline:**
```bash
python3 src/projectile_ml.py      # must be run from the project root
```

Note the working-directory requirement: `pd.read_csv("data/projectile_dataset.csv")` uses a path relative to
where you invoke Python, so running the script from inside `src/` will fail with a `FileNotFoundError`.

---

# 6. Results

All results below were produced by a single execution of `python3 src/projectile_ml.py` from the project root on
10 September 2026, on the fixed 400-row held-out test set.

## 6.1 Quantitative Model Performance

**Table 1 — Test-set performance (n = 400 held-out samples)**

| Model | Target | MAE (m) | RMSE (m) | R² |
|---|---|---:|---:|---:|
| Linear Regression | x | 13.917 | 20.085 | 0.7448 |
| Linear Regression | y | 10.295 | 14.089 | 0.5840 |
| **Random Forest** | **x** | **2.233** | **3.465** | **0.9924** |
| **Random Forest** | **y** | **3.299** | **4.949** | **0.9487** |

**Table 2 — Relative improvement of Random Forest over Linear Regression**

| Target | RMSE reduction | Unexplained variance reduction (1−R²) |
|---|---:|---:|
| x | 82.7% | from 0.255 → 0.008 (a 33× reduction) |
| y | 64.9% | from 0.416 → 0.051 (an 8× reduction) |

**Putting the errors in physical context.** The horizontal coordinate x in this dataset spans 0.03–241.9 m with
a standard deviation of 42.9 m. A Random Forest RMSE of 3.47 m is therefore about **1.4% of the full range** and
**8% of one standard deviation**. For a projectile with a typical range of ~100 m, the model places the object
within a few metres of its true position. That is a genuinely good result for a model that was told nothing about
gravity, trigonometry, or Newton's laws.

The Linear Regression RMSE of 20.09 m, by contrast, is nearly half a standard deviation — the model is
frequently wrong by an amount comparable to the spread of the data itself.

## 6.2 Why Linear Regression Fails — and What It Nevertheless Got Right

This is not a case of an underpowered model failing mysteriously. Its failure is precisely predicted by
equations (2.1) and (2.2), and inspecting its learned coefficients confirms the diagnosis.

**Table 3 — Learned Linear Regression coefficients** (computed for this report)

| Output | v₀ | θ | y₀ | t | intercept |
|---|---:|---:|---:|---:|---:|
| x | 1.056 | −38.799 | −0.093 | 18.268 | 1.786 |
| y | 0.972 | 27.514 | 0.824 | −0.337 | −32.274 |

Read these against the true physics, and four things stand out.

**(a) The model correctly discovered that x does not depend on y₀.** The true equation x = v₀cos(θ)t contains no
y₀ term at all, and the fitted coefficient is −0.093 — essentially zero relative to the others. The model was
never told this; it inferred it from the data. This is a small but real demonstration that regression coefficients
carry physical information.

**(b) The model correctly recovered the coefficient of y₀ in the y-equation.** The true equation
y = y₀ + v₀sin(θ)t − ½gt² has a coefficient of **exactly 1** on y₀. The model learned **0.824**. This is the one
term in the physics that genuinely *is* linear, and it is the one term the linear model gets nearly right.

**(c) The t-coefficient in the y-equation collapses to nearly zero (−0.337).** This is the signature of the failure.
The true relationship between y and t is a downward parabola: y *rises* for the first half of the flight and
*falls* for the second half. A straight line fitted through a symmetric rise-and-fall has almost zero slope. The
model has averaged the two halves of the trajectory into nothing. It has not learned "gravity"; it has learned
"on average, time doesn't tell me much about height."

**(d) The θ-coefficient in the x-equation (−38.8) is a crude linear stand-in for cos(θ).** Over the range
5°–85°, cos(θ) does decrease monotonically, so a negative linear coefficient captures the general trend — but a
straight line through a cosine is a poor approximation, and worse, the true relationship is *multiplicative*
(v₀·cos(θ)·t), not additive. A linear model can add its features but can never multiply them.

**Conclusion of this sub-analysis:** Linear Regression's poor score is *informative*, not merely disappointing.
It confirms that our dataset genuinely encodes non-linear, multiplicative physics — which is exactly what makes
it a meaningful test for a more capable model.

## 6.3 Figure Analysis

### Figure 1 — Actual vs Predicted (Random Forest)
**File:** `results/actual_vs_predicted.png`

Two panels: predicted x against actual x (left, blue), predicted y against actual y (right, green). The red
dashed line marks perfect prediction.

**What it shows.** Both panels cluster tightly along the diagonal, confirming the high R² values numerically
reported in Table 1. Three further observations:

1. **The x panel is visibly tighter than the y panel.** This matches R² = 0.9924 vs 0.9487 and has a physical
   cause: x = v₀cos(θ)t is a product of three inputs, whereas y additionally contains the quadratic −½gt². The
   vertical motion is the harder of the two functions to approximate.
2. **Point density is heavily concentrated near the origin.** Most random launch configurations produce short
   ranges; only a few produce very long ones. This reflects the right-skew visible in the Section 4.3 statistics
   table, and it has a direct methodological consequence — the model has far more training examples in the
   low-range regime than the high-range regime.
3. **Scatter widens at large values.** The largest deviations from the diagonal occur at the highest x and y
   values, exactly where training data is sparsest. This is the visual signature of the data imbalance in (2).

### Figure 2 — Residual Plot
**File:** `results/residuals.png`

Residual (actual x − predicted x) plotted against actual x, for both models on the same axes.

**What it shows.** The two models produce visibly different error structures:

- **Random Forest (orange):** residuals form a tight, roughly symmetric band around zero across the whole range,
  with modest widening at large x. There is no strong systematic curve or tilt — the error is close to being
  unstructured, which is what a well-fitted model looks like.
- **Linear Regression (blue):** a clear, unmistakable **diagonal pattern**. At small actual x the residuals are
  systematically *positive* (the model over-predicts short throws); at large actual x they become strongly
  *negative*, reaching +85 m at the extreme. This upward-tilting fan is the classic textbook signature of a model
  whose functional form is wrong — it is systematically compressing the range, pulling every prediction toward
  the middle because a plane cannot bend to follow a curved surface.

The Linear residuals also spread far wider (roughly −85 m to +85 m) than the Random Forest residuals (roughly
−15 m to +15 m), consistent with the RMSE ratio in Table 1.

**Why residual plots matter more than a single metric.** A summary number tells you *how much* error there is.
A residual plot tells you *whether the error is structured*. Structured error means recoverable signal was left
on the table. Table 1 alone would tell us Linear Regression is worse; Figure 2 tells us *why*, and that no amount
of extra data would fix it.

### Figure 3 — Trajectory Comparison
**File:** `results/trajectory_comparison.png`
**Launch condition:** v₀ = 31.2 m/s, θ = 0.74 rad (42.6°), y₀ = 12.0 m

This is the most important figure in the report. Three curves are overlaid:

- **Black solid** — the exact classical parabola from equations (2.1)–(2.2).
- **Green dashed** — the Random Forest's predicted trajectory.
- **Blue dotted** — the Linear Regression's predicted trajectory.

**What it shows.**

The **classical curve** behaves exactly as physics requires: it starts at y = 12 m, rises to an apex of ≈ 34.7 m
near x ≈ 50 m, and descends to y = 0 at x ≈ 110.7 m.

The **Linear Regression curve** is a nearly flat, gently declining line hovering around y ≈ 27 m for the entire
flight. It does not rise. It does not fall. It never lands. It is not a trajectory at all — it is the
straight-line best compromise discussed in Section 6.2(c), rendered visually.

The **Random Forest curve** is far more interesting, and its failure mode is subtle:

1. It **broadly follows the parabola's shape** — rising, peaking around the right region, then descending.
2. But it is visibly **jagged and staircase-like**, made of short flat segments rather than a smooth arc. This is
   the direct visual consequence of the mechanism described in Section 3.4: a Random Forest's output is an
   *average of leaf constants*, so its prediction function is piecewise constant and cannot be smooth.
3. It **systematically under-predicts the apex**, topping out around y ≈ 33 m against the true ≈ 34.7 m, and
   flattens across the peak region rather than curving.
4. Most seriously, it **fails to land**. The curve terminates at approximately (99 m, 7 m) instead of
   (110.7 m, 0 m). At the moment the real projectile hits the ground, the model believes it is still 7 m in the
   air and 12 m short.

## 6.4 Trajectory-Level Error vs Point-Wise Error — The Central Result

To make the visual finding quantitative, we measured the deviation between each model's swept trajectory and the
exact classical curve over the 50 sweep points (computed specifically for this report):

**Table 4 — Trajectory-level deviation from the classical solution**

| Model | RMSE in x (m) | RMSE in y (m) | Mean point-wise distance (m) | Max point-wise distance (m) |
|---|---:|---:|---:|---:|
| Random Forest | 4.39 | 2.91 | **4.74** | **13.38** |
| Linear Regression | 9.36 | 9.82 | 11.45 | 32.25 |

Now compare Table 4 against Table 1 for the Random Forest:

| Measurement mode | x error (RMSE) |
|---|---:|
| Scattered independent test points (Table 1) | 3.47 m |
| Continuous trajectory sweep (Table 4) | 4.39 m |

The trajectory-level error is **27% higher** than the point-wise test error, and the maximum single-point
deviation reaches 13.38 m — nearly four times the reported RMSE. The headline metric of R² = 0.9924 does not
describe what happens when the model is asked to trace a coherent path.

**This is the central scientific finding of Part I.** A model can score extremely well on the standard
statistical evaluation while producing output that is *physically invalid*: not smooth, not closing, not
conserving the shape that the governing equations demand. Standard ML metrics are blind to this, because they
evaluate points in isolation and never ask whether the collection of points forms a physically admissible curve.

---

# 7. Discussion

## 7.1 Answering Sub-Question 1: Can ML Predict Projectile Trajectories?

**Yes, approximately, with an important qualification.** A Random Forest trained on 1,600 samples predicts
positions to within ~3.5 m in x and ~4.9 m in y (R² = 0.99 and 0.95) for launch conditions inside its training
range. For applications that need an approximate position — a rough targeting estimate, a fast surrogate inside
an optimisation loop — this is usable.

The qualification is Section 6.4: the model is accurate *statistically* but not *physically*. It does not produce
smooth trajectories and does not respect the landing condition. Any application that requires the *shape* of the
path, rather than isolated positions, would be poorly served by this model as it stands.

## 7.2 Answering Sub-Question 2: ML vs Classical Physics

There is no contest, and it is important to say so plainly rather than let the high R² imply otherwise.

| Criterion | Classical equations | Random Forest |
|---|---|---|
| Accuracy | Exact (to floating-point precision) | ~3–5 m typical error |
| Speed | Microseconds, one arithmetic expression | Millisecond-scale; must traverse 100 trees |
| Memory | Three lines of code | ~1,600 stored training patterns across 100 trees |
| Extrapolation | Valid wherever the assumptions hold | Fails outside the training range (Stage 8) |
| Interpretability | Every term has physical meaning | Effectively a black box |
| Requires data? | No | Yes — thousands of samples |

For projectile motion, classical physics wins on every axis. **This must be stated explicitly in the final report.**
The value of the ML experiment is not that it competes with the equations; it is that it *measures how well
learning-from-data can recover a known law*, which is the calibration needed before applying the same method
where no closed-form law is available.

## 7.3 Answering Sub-Question 3: Did the Model "Learn Physics"?

**No.** This is the most important conceptual point in the entire project, and Figure 3 is the evidence.

If the Random Forest had learned the *law* y = y₀ + v₀sin(θ)t − ½gt², it would produce a perfect smooth parabola
for any launch condition, and it would land exactly at y = 0. It does neither. What it has actually learned is:
*"inputs that look like these produced outputs like those, so for a new input I will average the outputs of the
nearest training examples I remember."* That is sophisticated **interpolation over memorised examples**, not
discovery of a law.

Three specific pieces of evidence support this:

1. **The staircase structure of Figure 3** is a direct fingerprint of leaf-averaging. A learned equation would be
   smooth; a lookup-and-average is piecewise constant.
2. **The failure to land** shows the model has no concept of the constraint y(t_flight) = 0. The physics enforces
   this identically for every launch; the model has no mechanism to represent a constraint.
3. **The coefficient analysis in Section 6.2** shows what "learning a term of the physics" actually looks like —
   the Linear model recovered the coefficient of y₀ as 0.824 ≈ 1, the true value. Nothing comparable can be
   extracted from the Random Forest, because it has no coefficients, only stored averages.

A useful way to state this in a viva: **the model learned the data, and the data came from the physics — but the
model did not learn the physics.** Fitting the outputs of a law is not the same as discovering the law.

## 7.4 On Synthetic Data — Scientific Integrity

Our dataset was generated *from the very equations the model is being asked to approximate*. Several consequences
follow, and honesty requires stating all of them.

- The data is **noise-free**. Real experimental measurements would contain instrument error, and every reported
  metric would degrade. Our numbers are therefore an *upper bound* on achievable accuracy, not a realistic
  estimate.
- The model **inherits every assumption in Section 2.4**. It has never seen air resistance, so it cannot model
  it. Applied to a real thrown ball, it would fail in exactly the way the idealised equations fail — and it has
  no way to signal that it is out of its depth.
- **A low error does not mean ML is "better than physics."** It means ML has successfully copied physics.
  The ground truth *is* the physics; the model can at best approach it and can never exceed it.
- The correct framing of ML's value is not accuracy but **cost and applicability**: ML becomes interesting when
  simulations are expensive, when data is noisy, when the governing equations are unknown or intractable, or when
  a fast surrogate is needed inside a loop. None of those conditions holds for projectile motion — which is
  precisely why projectile motion is the right *control*, not the right *application*.

## 7.5 Limitations of Part I

1. **Idealised physics.** No air resistance, no spin, no wind, constant g, flat Earth, point mass. Section 2.4.
2. **Synthetic, noise-free data.** Section 7.4.
3. **Single-step prediction only.** The model predicts position at an arbitrary requested time. It does not
   propagate a state forward step by step, so error cannot accumulate — the very phenomenon that will dominate
   Part II. The two systems are therefore not yet compared on equal footing, and Part II's design must address
   this explicitly.
4. **One train/test split.** All metrics come from a single split with `random_state=42`. Cross-validation
   (repeating over several splits and reporting mean ± standard deviation) would give error bars on Table 1.
   Currently we have point estimates with no uncertainty quantification — a genuine methodological weakness that
   should be corrected before the final report.
5. **No hyperparameter tuning.** `n_estimators=100` and otherwise-default settings were used without
   justification beyond convention.
6. **Interpolation only.** Every test sample lies inside the training range. Whether the model generalises
   outside it is untested — this is Stage 8.
7. **A single trajectory was used for Figure 3.** The trajectory-level analysis in Section 6.4 rests on one
   launch condition. It should be repeated over many conditions before being stated as a general result.

## 7.6 Implications for Part II (Planetary Motion)

The Part I findings shape the Part II design in three concrete ways.

- **The smoothness problem will get worse.** A Random Forest's piecewise-constant output already visibly damages
  a simple parabola. An orbit is a closed curve that must return to its starting point; a staircase approximation
  will not close, and the error will be obvious immediately. This argues for including a smooth-function model
  (polynomial features, or a small neural network) in Part II rather than relying on tree ensembles alone.
- **Error accumulation must be measured, not assumed.** Planetary motion is naturally posed as step-by-step
  propagation, where each prediction feeds the next and errors compound. This is a structurally different and
  harder task than Part I's single-step mapping, and the comparison must acknowledge that rather than presenting
  a simple "projectile error vs planetary error" table as though the tasks were equivalent.
- **The evaluation protocol needs a physical component.** Section 6.4 showed that R² alone is misleading.
  Part II should report physics-aware diagnostics alongside the statistical metrics — for a two-body orbit, the
  natural candidates are conservation of total energy and of angular momentum, both of which the true dynamics
  preserve exactly and which a purely statistical model has no reason to respect.

---
# 8. Stage 7 — How Much Data Do We Actually Need?

**Sub-question 2: How does the amount of training data affect prediction accuracy?**

This is the first stage that is research rather than implementation. Stages 3–6 established *that* a Random
Forest works; Stage 7 manipulates an independent variable systematically and measures the response. It converts
"we used 2,000 samples" from an arbitrary choice into an empirically justified one — or, as it turns out,
into a demonstrably insufficient one.

## 8.1 Design

| Element | Specification |
|---|---|
| Independent variable | Training-set size: 100, 500, 1,000, 5,000, 10,000 rows |
| Dependent variables | Test RMSE for x and y; whole-trajectory radial error |
| Controlled | **One fixed 4,000-row test set, split off once before the loop**; g; model settings; feature ranges |
| Repeats | 5 independent random draws at each size, reported as mean ± SD |
| Models | Random Forest (100 trees) and Linear Regression as the flat-line control |

Stages 3–6 used a 2,000-row dataset, which cannot supply a 10,000-row training set *and* an untouched test
set. Stage 7 therefore generates its own 20,000-row dataset from the same simulator with a different seed.

**Why the test set is split off once, before anything else.** If each training size were evaluated on its own
freshly drawn test set, a change in RMSE could be caused by the test set rather than by the training size, and
the experiment would answer no question at all. This is the controlled-comparison principle in its simplest
form, and it recurs in every stage that follows.

**Why five repeats.** A single 100-row training set is one lucky or unlucky sample. Without repeats there is no
way to tell a real effect from a draw.

## 8.2 Results

**Table 8.1 — Test RMSE (m), mean ± SD over five random training draws, on one fixed 4,000-row test set**

| n_train | Forest RMSE x | Forest RMSE y | Linear RMSE x | Linear RMSE y |
|---|---|---|---|---|
| 100 | 16.37 ± 2.34 | 13.27 ± 1.11 | 21.14 ± 0.24 | 15.35 ± 1.01 |
| 500 | 8.95 ± 1.23 | 7.82 ± 0.39 | 20.87 ± 0.16 | 14.55 ± 0.09 |
| 1,000 | 5.33 ± 0.38 | 6.28 ± 0.27 | 20.76 ± 0.07 | 14.52 ± 0.06 |
| 5,000 | 2.85 ± 0.09 | 4.09 ± 0.11 | 20.69 ± 0.01 | 14.45 ± 0.01 |
| 10,000 | **2.05 ± 0.02** | **3.38 ± 0.04** | 20.69 ± 0.01 | 14.44 ± 0.00 |

**Table 8.2 — Whole-trajectory reconstruction error, averaged over 40 unseen launch conditions**

| n_train | Mean radial error (m) | Interquartile range (m) |
|---|---|---|
| 100 | 14.85 | 6.73 – 20.31 |
| 500 | 6.28 | 3.59 – 8.53 |
| 1,000 | 5.10 | 3.07 – 6.70 |
| 5,000 | 3.00 | 1.71 – 4.23 |
| 10,000 | **2.50** | 1.46 – 3.45 |

**Figures:** `stage7_learning_curve.png`, `stage7_stability.png`, `stage7_trajectory_by_datasize.png`.

## 8.3 Interpretation

**The Random Forest's curve is still falling at 10,000 rows.** RMSE x improves from 16.37 m to 2.05 m — a
factor of eight — and has not flattened. The honest conclusion is that **2,000 samples was insufficient**, and
that a five-fold increase would still buy accuracy. This is a legitimate finding rather than a failure: the
experiment was run to answer this question, and it answered it.

**Linear Regression's curve is flat, and that is the most instructive part of the figure.** Its RMSE moves from
21.14 m to 20.69 m — a 2% improvement across a hundredfold increase in data. Infinite data would not help,
because its limitation is not a shortage of examples but its functional form: it cannot represent
v₀cos(θ)t at any sample size. **More data fixes a data problem, not a wrong-model problem**, and having both
curves on one axis makes that distinction visible rather than asserted.

**The spread shrinks by two orders of magnitude.** The Random Forest's SD falls from ±2.34 m at n = 100 to
±0.02 m at n = 10,000. This answers a different question from the learning curve — not "how accurate?" but "how
much does the answer depend on *which* rows we happened to get?" At n = 100 a single run is not reproducible
and should not be quoted; by n = 10,000 the result is stable to the second decimal place.

**Trajectory-level error tracks RMSE but stays above it.** At n = 10,000 the RMSE is ~2.0 m while the mean
whole-trajectory error is 2.50 m. The gap is the staircase artefact of Section 6.4 — a forest predicts in
piecewise-constant steps, so a swept trajectory is jagged even when scattered points are accurate. Crucially,
**the gap narrows but does not close**, exactly as the leaf-averaging mechanism predicts: more data means more,
finer steps, never a smooth curve.

---

# 9. Stage 8 — Interpolation, Extrapolation, and What R² = 0.99 Was Really Measuring

**Sub-question 3: How far can the model be trusted outside the conditions it was trained on?**

This is the most important experiment in Part I, and it reframes everything before it.

## 9.1 The problem with every number reported so far

Section 6 reported R² = 0.9924. Read carelessly, that is "the model learned projectile motion". It did not.
R² was computed on test rows drawn from **the same distribution** as the training rows — same speed range, same
angle range, same height range. That measures **interpolation**: filling gaps between things the model has seen.

A physical law is not a statement about a region. It is a statement about everywhere. If the model had genuinely
induced x = v₀cos(θ)t, it would work at v₀ = 60 m/s having only ever seen v₀ ≤ 40 m/s. Stage 8 tests exactly
that.

**Definitions used throughout.**
- **Interpolation** — the query lies inside the convex hull of the training inputs. The model is filling a gap.
- **Extrapolation** — the query lies outside it. The model is being asked to invent behaviour, and its answer is
  decided entirely by its built-in assumptions — its **inductive bias** — not by data.

## 9.2 Design

| Element | Specification |
|---|---|
| Training envelope | v₀ ∈ [15, 40] m/s, θ ∈ [20°, 70°], y₀ ∈ [0, 15] m — a strict subset of what is physical |
| Training set | **One fixed set of 8,000 rows**, used by all five models in all seven regimes |
| Independent variable | Which region the **test** data comes from |
| Test regimes | 1 interpolation control + 6 extrapolation regimes, 2,000 rows each |
| Models | Linear, polynomial ridge (deg 2), random forest, MLP, physics-features |

A fifth model is added here: **physics-informed features** — ordinary linear regression applied to the basis
[v₀cos(θ)t, v₀sin(θ)t, y₀, t²], in which the true solution is exactly linear. It is not a general model. It is
the ceiling: what a model that has been *given* the physics can do.

## 9.3 Results

**Table 9.1 — Mean radial error (m) by test regime. Every model saw the same training set.**

| Test regime | Linear | Poly (2) | Forest | MLP | Physics features |
|---|---|---|---|---|---|
| **interpolation** (control) | 11.18 | 1.97 | 2.14 | **0.11** | 5.4 × 10⁻¹⁴ |
| extrapolate v₀ high (40–55 m/s) | 36.26 | 8.15 | 31.25 | 18.53 | 5.9 × 10⁻¹⁴ |
| extrapolate v₀ low (5–15 m/s) | 14.84 | 3.81 | 6.20 | 1.32 | 6.6 × 10⁻¹⁴ |
| extrapolate θ high (70°–85°) | 28.27 | 8.38 | 15.28 | 2.87 | 6.5 × 10⁻¹⁴ |
| extrapolate θ low (5°–20°) | 16.52 | 6.15 | 5.17 | 1.61 | 5.7 × 10⁻¹⁴ |
| extrapolate y₀ high (15–30 m) | 13.37 | 2.28 | 10.24 | 0.80 | 7.9 × 10⁻¹⁴ |
| **all variables outside at once** | 66.24 | 34.04 | 55.59 | 32.70 | 1.2 × 10⁻¹³ |
| **Degradation factor** | 5.9× | 17.2× | 26.0× | **286.4×** | none |

**Table 9.2 — R² for y. Negative means worse than always predicting the mean.**

| Test regime | Linear | Poly (2) | Forest | MLP | Physics |
|---|---|---|---|---|---|
| interpolation | 0.548 | 0.988 | 0.973 | 1.000 | 1.000 |
| extrapolate v₀ low | **−1.703** | 0.569 | **−0.229** | 0.959 | 1.000 |
| extrapolate θ low | **−1.356** | −0.012 | **−0.684** | 0.891 | 1.000 |
| all outside at once | **−0.402** | 0.975 | **−1.073** | 0.170 | 1.000 |

**Figures:** `stage8_regime_bars.png`, `stage8_extrapolation_distance.png`, `stage8_response_slice.png`,
`stage8_trajectories.png`, `stage8_time_extrapolation.png`.

## 9.4 The mechanism, made visible

Table 9.1 says extrapolation is bad. `stage8_response_slice.png` says *why*, and it is the single most
informative figure in Part I.

The construction: hold θ = 45°, y₀ = 5 m and t = 1.5 s fixed, and sweep v₀ from 0 to 80 m/s while the models
were trained only on [15, 40]. Along this slice the true answer is x = v₀·cos(45°)·1.5 — **exactly a straight
line through the origin**. Therefore every deviation in the figure is purely the model's own inductive bias,
with no other effect mixed in.

What each model does:

- **Random Forest** — follows the line inside the training range, then goes **exactly flat** in both directions.
  This is not a failure to converge; it is the definition of the model. A forest outputs the average of training
  targets in the leaf a point falls into, and beyond the last split threshold there are no further leaves, so
  every input past that boundary lands in the same leaf and receives the same number. **A tree ensemble's
  prediction is bounded by the range of its training targets, always.**
- **Linear Regression** — a straight line, but the wrong one. It extrapolates smoothly and confidently in the
  wrong direction, which is arguably more dangerous than the forest's obvious flat-lining.
- **MLP** — tracks well inside the range, then wanders: it turns over and comes back down around v₀ ≈ 55 m/s.
  Its smoothness prior decides what happens outside, and that prior knows nothing about projectile motion.
- **Physics features** — exactly on the true line across the whole sweep, because in that basis the problem is
  linear and the fit is exact.

## 9.5 The best model in-distribution is the worst out of it

The MLP achieves 0.11 m on the control — twenty times better than the forest — and degrades by **286×**. Linear
regression degrades by only 5.9×, but from a starting point of 11 m, which is faint praise.

This is not a coincidence, and it is a general property worth stating: **flexibility and extrapolation pull in
opposite directions.** A flexible model has many ways to fit the training region well while behaving arbitrarily
outside it. The data does not constrain it there; only its own inductive bias does. A model selected purely on
in-distribution accuracy is therefore selected, in part, for the very property that makes it untrustworthy
outside.

## 9.6 Extrapolating forward in time

A second, physically different kind of extrapolation: models were retrained using only times in the **first
60%** of each flight, then asked about the whole flight.

**Table 9.3 — Mean radial error (m) by position within the flight. The last four rows are extrapolation.**

| t / t_flight | Linear | Poly (2) | Forest | MLP | Physics |
|---|---|---|---|---|---|
| 0.25 | 4.32 | 0.61 | 1.22 | 0.08 | 0 |
| 0.45 | 6.21 | 1.33 | 1.65 | 0.09 | 0 |
| 0.55 | 9.06 | 1.87 | 2.11 | 0.12 | 0 |
| **0.65** | 15.08 | 2.59 | 7.62 | 1.02 | 0 |
| **0.75** | 22.62 | 3.35 | 15.97 | 4.04 | 0 |
| **0.85** | 33.43 | 4.17 | 26.76 | 10.80 | 0 |
| **0.95** | 45.83 | 5.04 | 38.59 | 20.12 | 0 |

The MLP's error rises by a factor of 170 — from 0.12 m at the edge of its data to 20.12 m at the end of the
flight. This is the closest thing in Part I to the question that dominates Part II: *how far into the future can
a learned model be trusted?* The answer here is "to the edge of its data, and no further".

The polynomial model does best under time extrapolation (5.04 m at t/t_flight = 0.95), which is not an accident:
it can represent t² exactly, and t² is precisely the term that governs the late part of the flight. A model
whose functional form matches the physics degrades gracefully; one whose form does not, does not.

## 9.7 What this stage establishes

1. **R² = 0.99 measured interpolation, and interpolation only.** Any report quoting an in-distribution score
   without an extrapolation test is quoting an incomplete result.
2. **The failure mode is set by the model family, not by the amount of data.** The forest flat-lines; the linear
   model is confidently wrong in a straight line; the network wanders. None of these is fixable by training on
   more of the same region.
3. **No model that was not given the physics generalises outside its training envelope.** The one that was is
   exact everywhere — five orders of magnitude better than anything else even on the control, and unaffected by
   extrapolation at all.
4. **Therefore, no model learned the law.** A model that had induced x = v₀cos(θ)t would work at v₀ = 45 m/s.
   Not one of them did. This is the quantitative form of the claim that Section 7.3 made qualitatively, and it
   is the finding Part II builds on.

---# PART II — PLANETARY MOTION

---

# 10. Stages 9–10 — The Physics and the Numerical Baseline

Part I studied a system with a closed-form solution. Part II studies one without, and that single change
turns out to drive every difference between the two halves of this study.

## 10.1 The physics

Newton's law of universal gravitation, written as an acceleration on the small body with the large body at
the origin:

> **a** = −(GM / r³) **r**    (10.1)

The r³ is not a different force law — **r**/r is the unit vector, so (1/r³)**r** = (1/r²)**r̂**. Writing it
this way avoids taking a square root twice.

**Assumptions, all of which the study inherits:**

| Assumption | Meaning | When it breaks |
|---|---|---|
| Two-body | No other planets | Real systems have perturbations; Jupiter measurably perturbs Mars |
| Restricted | The star is fixed at the origin | Valid when m ≪ M. For Sun–Earth, m/M ≈ 3 × 10⁻⁶ |
| Point masses | No tidal distortion, no oblateness (no J₂) | Matters for close orbits and for Earth satellites |
| Newtonian | No general relativity | Mercury precesses 43″/century; Newtonian gravity cannot account for it |
| No dissipation | No drag, no radiation pressure, no mass loss | Matters for low satellites and for comets |

**Units.** We work in AU, years and solar masses, in which Kepler's third law forces

> GM = 4π² AU³ yr⁻²    (10.2)

so a 1 AU circular orbit takes exactly 1 year. This is not cosmetic: in SI the same quantities are ~10¹¹ m
and ~10³⁰ kg, and squaring them inside an energy calculation discards precision for nothing. In AU-year units
every quantity in the simulation is of order 1, and "0.01 AU of error" is immediately readable as 1% of the
orbit.

**Conserved quantities.** Two, and both are used as diagnostics:

> E = v²/2 − GM/r   (specific energy, conserved because gravity is conservative)
> L = x·v_y − y·v_x   (specific angular momentum, conserved because gravity is a *central* force and so
> exerts no torque about the origin — equivalently, Kepler's second law, since dA/dt = L/2)

Their value as diagnostics is that **they require no reference solution**. The true system conserves them
exactly, so any drift is numerical error, measurable even where no exact answer is available.

## 10.2 Two independent routes to position at a given time

Unlike projectile motion, the generic eccentric orbit has no elementary explicit r(t). Numerical
time integration is one route; solving the analytic Kepler parametrisation is another. The orbit *shape* is solvable — it is a conic section, Kepler's first law — but
position as a function of **time** requires solving Kepler's transcendental equation

> E − e·sin E = M    (10.3)

for the eccentric anomaly E, and (10.3) has no algebraic solution. We solve it by Newton–Raphson to a
tolerance of 10⁻¹⁴, which gives us an **analytic reference** — a route to the exact answer that involves no
time stepping at all, and therefore an independent check on every integrator.

## 10.3 Four integrators

| Method | Update | Order | Symplectic? | Force evals/step |
|---|---|---|---|---|
| Explicit Euler | r ← r + v Δt; v ← v + a(r) Δt | 1 | No | 1 |
| Euler–Cromer | v ← v + a(r) Δt; r ← r + **v_new** Δt | 1 | **Yes** | 1 |
| Velocity Verlet | r ← r + vΔt + ½aΔt²; v ← v + ½(a + a_new)Δt | 2 | **Yes** | 2 |
| Runge–Kutta 4 | weighted average of four probes inside the step | 4 | No | 4 |

Euler and Euler–Cromer differ by **one character** — which velocity the position update uses — and that
single change is the difference between an orbit that spirals away for ever and one that stays closed.

## 10.4 Results: the orbit picture

Same initial condition (a = 1 AU, e = 0.3), same step size (200 steps per orbit), 40 orbits, four methods.
`stage10_integrator_orbits.png`.

| Method | Final radius / initial radius |
|---|---|
| Explicit Euler | **7.640** |
| Euler–Cromer | 1.000 |
| Velocity Verlet | 1.000 |
| Runge–Kutta 4 | 1.000 |

Explicit Euler ends at more than seven times its starting radius. The failure is not random error but a
**systematic bias**: Euler evaluates everything at the start of the interval, so on a curving path it
consistently overshoots outward and adds energy every step. Halving Δt halves the rate of the spiral and never
removes it.

## 10.5 Results: conservation over 200 orbits

**Table 10.1 — Relative drift of E and L over 200 orbits at 400 steps/orbit**

| Orbit | Method | max ǀΔE/E₀ǀ | final ǀΔE/E₀ǀ | max ǀΔL/L₀ǀ |
|---|---|---|---|---|
| circular (e = 0) | Euler | 7.98 × 10⁻¹ | 7.98 × 10⁻¹ | 1.22 |
| | Euler–Cromer | 2.47 × 10⁻⁴ | 1.24 × 10⁻⁶ | 1.09 × 10⁻¹⁴ |
| | **Velocity Verlet** | 1.52 × 10⁻⁸ | **2.52 × 10⁻¹²** | 1.89 × 10⁻¹⁴ |
| | RK4 | 3.34 × 10⁻⁸ | **3.34 × 10⁻⁸** | 1.67 × 10⁻⁸ |
| eccentric (e = 0.6) | Euler | 9.84 × 10⁻¹ | 9.81 × 10⁻¹ | 3.98 × 10⁻¹ |
| | Euler–Cromer | 4.34 × 10⁻² | 1.24 × 10⁻⁵ | 1.31 × 10⁻¹⁴ |
| | **Velocity Verlet** | 1.76 × 10⁻³ | **7.68 × 10⁻¹⁰** | 2.67 × 10⁻¹⁴ |
| | RK4 | 4.16 × 10⁻⁵ | **4.16 × 10⁻⁵** | 4.62 × 10⁻⁶ |

**Read the shape of these numbers, not just their size.** For Verlet, the *maximum* error is far larger than
the *final* error — the error oscillates within a bounded envelope and returns. For RK4 the maximum and final
values are identical, which is the signature of a one-way **drift**. This is the practical meaning of
symplecticity: by backward error analysis a symplectic method solves a nearby *modified* Hamiltonian exactly,
so its energy error is bounded for exponentially long times. RK4 is thousands of times more accurate per step
and still loses energy secularly.

Angular momentum is conserved to ~10⁻¹⁴ — machine precision — by both symplectic methods, because they
preserve the geometric structure that the central-force symmetry implies. RK4 does not.

**Practical conclusion.** Use Verlet for long-term dynamics; use RK4 when maximum accuracy over a moderate
number of orbits matters. This study needed the latter, so RK4 generates all Part II training data.

## 10.6 Results: convergence order — verifying the implementation

A method of order p has global error ≈ CΔt^p, so log(error) = p·log(Δt) + const — a straight line of **slope
p** on log-log axes. Measuring that slope is how an integrator implementation is verified; a coding error
almost always shows up as the wrong slope.

**Table 10.2 — Measured convergence order against the analytic Kepler solution**

| Method | Theory | Measured at ¼ orbit | Measured at 1 full orbit |
|---|---|---|---|
| Explicit Euler | 1 | **0.974** | 0.623 |
| Euler–Cromer | 1 | **0.994** | **1.998** |
| Velocity Verlet | 2 | **1.994** | 1.997 |
| Runge–Kutta 4 | 4 | **3.917** | 4.119 |

The quarter-orbit column recovers theory exactly. Two entries in the last column need explaining, and both
explanations are results rather than excuses.

**Euler's 0.623 at a full orbit** is the asymptotic expansion failing. At coarse Δt its error is ~1–3 AU on a
1 AU orbit; error ≈ CΔt^p only holds while the error is small, and Euler's is not.

**Euler–Cromer's 1.998 at a full orbit** is more interesting, and it was investigated rather than smoothed
over. Symplectic Euler is **conjugate** to Störmer–Verlet: the two produce the same trajectory up to a fixed
O(Δt) change of coordinates (Hairer, Lubich & Wanner 2006). That coordinate shift is periodic with the orbit,
so sampling after a whole period cancels it and exposes the underlying second-order behaviour. This was
confirmed by measuring at a quarter period as well — where the order comes out at 0.994, exactly as theory
says. Both measurements are in `results/stage10_convergence.csv` and both panels are in the figure.

## 10.7 Results: Kepler's third law emerges

The simulator was told only Newton's inverse-square law. Kepler's third law T² ∝ a³ was never imposed.

**Table 10.3 — Orbital period measured from simulation**

| a (AU) | T theory (yr) | T measured (yr) | Relative error |
|---|---|---|---|
| 0.4 | 0.252982 | 0.252982 | 1.4 × 10⁻¹³ |
| 1.0 | 1.000000 | 1.000000 | 1.4 × 10⁻¹³ |
| 2.0 | 2.828427 | 2.828427 | 1.5 × 10⁻¹³ |
| 5.0 | 11.180340 | 11.180340 | 1.4 × 10⁻¹³ |

**Fitted log–log slope: 1.500000** against Kepler's 3/2.

This is an end-to-end validation that the simulator reproduces celestial mechanics rather than merely being
self-consistent, and it runs in the historically correct direction: Kepler's law is *derived from* Newton's,
not assumed alongside it.

## 10.8 Computational cost

| Method | µs per step (pure Python) | Force evaluations |
|---|---|---|
| Explicit Euler | 5.64 | 1 |
| Euler–Cromer | 5.49 | 1 |
| Velocity Verlet | 10.02 | 2 |
| Runge–Kutta 4 | 27.87 | 4 |

Cost tracks force evaluations almost exactly, as it should. The right-hand panel of
`stage10_convergence.png` replaces Δt with force evaluations on the x axis — the fair comparison — and RK4
still wins by many orders of magnitude at equal cost. Higher order is not merely more accurate per step here;
it is more accurate per unit of work.

## 10.9 The error budget for Part II

This is the number that licenses everything in Section 11.

The reference used for all Stage 11 experiments is RK4 with Δt_store = 0.004 yr and **ten internal substeps**.
Separating those two intervals is deliberate: Δt_store is what the ML model steps over, so it must be large
enough that a five-orbit rollout is ~1,250 model calls rather than tens of thousands; Δt_integrate must be
small enough that the ground truth carries no visible error of its own. One value for both would force a bad
compromise.

**Measured reference accuracy:**

| Check | Result |
|---|---|
| RK4 vs analytic Kepler, 2 orbits, e = 0.0 | 4.6 × 10⁻¹⁴ AU |
| RK4 vs analytic Kepler, 2 orbits, e = 0.6 | 8.1 × 10⁻¹³ AU |
| Reference trajectory over the full 5-year horizon | ≈ 3 × 10⁻¹¹ AU |

The smallest ML error measured anywhere in Section 11 is 6.4 × 10⁻⁵ AU — **six orders of magnitude larger**.
The reference is therefore exact for present purposes, and that statement is quantitative rather than
rhetorical.

## 10.10 The dataset

60 training and 20 held-out trajectories, a ∈ [0.8, 1.2] AU, e ∈ [0.0, 0.4], three orbits each, sampled every
0.004 yr: **44,285 one-step training pairs**. Worst relative energy drift across the whole training set:
5.4 × 10⁻¹². Trajectory 0 differs from the analytic Kepler solution by at most 1.2 × 10⁻¹⁰ AU.

Varying a and e is essential rather than decorative. A model trained on one single orbit has no way to
distinguish "the law of motion" from "this particular curve", and would be a lookup table in disguise. Varying
the orbit forces any one-step model to learn a mapping that genuinely depends on the **state**.


# 11. Stage 11 — Machine Learning for Orbital Motion

A simulator must do more than predict the next point when the correct current state is supplied.
After the first step it must operate on its **own previous prediction**. This stage measures the
consequences of that change, using the independently validated reference from Section 10.

## 11.1 Design

| Design element | Implementation |
|---|---|
| Independent variables | Model family; absolute versus delta targets; flow-map, direct-map or history formulation; training-set size |
| Dependent variables | One-step position error, rollout error, tolerance-crossing time, conservation error and escape fraction |
| Training data | 60 trajectories; 44,285 adjacent state pairs; three orbits per trajectory |
| One-step test data | Separate trajectories generated with a different seed; pairs never cross a trajectory boundary |
| Rollout evaluation | 20 unseen initial conditions; 5 years; Δt = 0.004 yr, giving 1,250 recursive steps |
| Controlled | Model hyperparameters within each family, physical law, reference solver and evaluation conditions |
| Numerical baseline | RK4 with substeps, checked against the analytic Kepler solution; representative maximum discrepancy ≈ 3 × 10⁻¹¹ AU |

The state is s = (x, y, vₓ, vᵧ). **Approach A** learns s(t) → s(t + Δt).
In its delta form it instead learns the change Δs, and adds that change to the current state.
**Approach B** learns (a, e, t) → s directly, so a prediction at a late time does not depend
on earlier predictions. **Approach C** supplies three consecutive states rather than one.
These alternatives change the prediction task; they are not simply three names for the same model.

## 11.2 One-step accuracy: predicting the change helps some models

**Table 11.1 — Mean held-out one-step position error (AU).**
Source: `results/stage11_one_step_accuracy.csv`.

| Model | Absolute target | Delta target | Absolute / delta error |
|---|---:|---:|---:|
| Linear | 5.258 × 10⁻⁵ | **5.258 × 10⁻⁵** | 1.00× |
| Polynomial, degree 2 | 3.203 × 10⁻⁵ | **3.203 × 10⁻⁵** | 1.00× |
| Random Forest | 2.499 × 10⁻² | **8.691 × 10⁻⁴** | 28.8× |
| MLP | 1.800 × 10⁻³ | **6.448 × 10⁻⁵** | 27.9× |

**Figure 11.1 — `results/stage11_one_step_accuracy.png`.** Compare the two bars within each
model before comparing models. The target reparametrisation substantially helps the forest and
network, while the linear and polynomial fits are effectively unchanged.

For a small step, the next state is close to the present state. Learning the small correction
reduces the burden of representing the identity map. Linear regression already includes that map:
subtracting the input from the target changes its coefficients, without enriching its function
class. The polynomial model contains linear terms as well. Its ridge regularisation means exact
invariance is not guaranteed, but the measured difference is negligible here.

The polynomial model has the smallest delta **position** error in this table. That does not make
it the best simulator: velocity error and repeated application also matter.

## 11.3 Rollout: local accuracy is not a usable forecast horizon

**Table 11.2 — Median time to cross a position-error tolerance (years).**
Source: `results/stage11_horizons.csv`; all 20 orbits cross each listed tolerance within the run.

| Model | 0.001 AU | 0.01 AU | 0.1 AU |
|---|---:|---:|---:|
| Linear | 0.020 | **0.062** | 0.192 |
| Polynomial, degree 2 | 0.028 | **0.091** | 0.312 |
| Random Forest | 0.007 | **0.076** | 0.386 |
| MLP | 0.039 | **0.372** | 0.973 |

The MLP's 6.448 × 10⁻⁵ AU one-step error becomes a **0.01 AU error within 0.372 years**,
approximately 93 model steps and roughly one-third of a year-scale orbit. A tolerance must always
accompany a horizon: accepting 0.1 AU extends the same model's median horizon to 0.973 years.

**Figure 11.2 — `results/stage11_error_growth.png`.** The left panel shows mean position error
and the middle 50% of the unseen-orbit distribution; the right panel shows Table 11.2. The y-axis
starts at 10⁻⁵ AU so the learned curves remain readable. The much smaller reference error is
stated in an annotation, rather than setting the bottom of the axis.

Let F be the true step and F̂ the learned step. If eₙ is the current state error, then locally

> eₙ₊₁ ≈ DF(sₙ)eₙ + [F̂(sₙ) − F(sₙ)].

There are two contributions: propagation of an existing error, and a fresh approximation error.
Training on correct states constrains the second term only where the training trajectories go.
A rollout gradually supplies displaced states, so it also tests inputs unlike those used during
fitting. Correlated errors can systematically change orbital phase, radius and energy.

## 11.4 Failure modes and physical consistency

**Figure 11.3 — `results/stage11_rollout_orbits.png`.** Read the learned path against the same
reference ellipse in each panel. The representative linear rollout spirals inward; the polynomial
rollout escapes; the forest gives the wrong radial envelope; and the MLP's orbit precesses.
These are different manifestations of model bias, not four measurements of the same random noise.

Across the ensemble, **45% of polynomial rollouts escape**. Stage 11 marks states beyond its
100 AU radius threshold as non-finite and retains an explicit escape outcome in
`results/stage11_rollout_per_orbit.csv`. For aggregate error curves the implementation substitutes
100 AU after escape. This is a **failure penalty**, not a measured continuation of the trajectory
or a mathematically exact lower bound on position error. The reported late-time polynomial mean
therefore depends on that convention. The escape fraction and early tolerance crossings are the
more interpretable measures of failure.

**Figure 11.4 — `results/stage11_conservation_ml.png`.** Energy and angular-momentum errors
provide checks independent of visual resemblance. A nearly elliptical path can still have the
wrong energy or accumulate phase error. The learned models have no conservation constraint in
their loss. The representative diagnostics are saved in
`results/stage11_representative_rollout.csv`, so these checks remain inspectable without refitting.

## 11.5 Direct prediction avoids recursion but still needs time coverage

**Figure 11.5 — `results/stage11_direct_extrapolation.png`.** Use the full time curve, not only
the summary average in `results/stage11_direct_map.csv`. The MLP's mean error is approximately
0.004–0.006 AU over the densely covered interval from about 0.1 to 2.2 years. At t = 2 years it is
**0.0064 AU**, compared with **0.6396 AU** for the recursive MLP, about a hundredfold difference.
Sources: `results/stage11_direct_map_curve.csv` and `results/stage11_rollout_error.csv`.

The training trajectories each contain three **orbits**, not three years. Short-period orbits
near a = 0.8 AU stop near 2.15 years. Consequently, time coverage becomes uneven beyond about
2.2 years even though the longest training trajectory continues later. The direct-map error rises
to approximately 0.64 AU near 2.9 years and about 1.3 AU beyond 3.5 years. Its nominal
“in-window” mean of 0.301 AU mixes dense coverage with this thinning region.

Removing recursion therefore helps while the query is well supported by training data; it does
not remove extrapolation failure. This is the orbital counterpart of Section 9.

## 11.6 History, more data, and sensitivity

**Figure 11.6 — `results/stage11_approach_comparison.png`.** The three-state history model
finishes at 1.458 AU mean error for the forest and 1.885 AU for the MLP
(`results/stage11_history.csv`). History provides no consistent improvement: the forest's final
error is slightly smaller than its one-state result, but the MLP's is worse. Because the full
position-and-velocity state already determines the future, extra history supplies redundancy,
not a missing physical variable. These tests do not rule out other sequence architectures.

**Figure 11.7 — `results/stage11_learning_curve.png`.** Increasing MLP training pairs from
500 to about 44,000 lowers held-out one-step error from about 2.9 × 10⁻⁴ to 4.3 × 10⁻⁵ AU,
but extends the 0.01 AU horizon only from about 0.08 to 0.25 years. The separate subset-training
experiment need not reproduce the full-data horizon in Table 11.2. Linear and polynomial fits
show little improvement, consistent with a limitation of their function classes.
Source: `results/stage11_learning_curve.csv`.

**Figure 11.8 — `results/stage11_sensitivity.png`.** Small perturbations of the exact orbital
solution amplify by about 45–47 times over five years for the tested perturbations. The curves
show phase separation consistent with linear secular growth, rather than a chaotic exponential
instability. An exponential fit to a learned error curve is not a measurement of a physical
Lyapunov exponent. This two-body benchmark supplies no basis for blaming chaos for model failure.

## 11.7 What this stage establishes — and what it does not

One-step position accuracy is insufficient to judge a learned simulator. Rollout horizons,
escapes and conservation errors reveal failures that the one-step ranking conceals. Direct
prediction can greatly reduce accumulated error within a well-covered domain; more data and
short history alone do not reliably solve long-horizon prediction here.

These are fixed, modest baseline models, not the best possible learned orbital simulators.
No conservation architecture, rollout-aware loss or training-noise intervention was tested.
The conclusions concern this implementation and these distributions; they do not prove that
machine learning cannot approximate two-body dynamics accurately.

---

# 12. Stage 12 — The Controlled Comparison

Comparing a projectile error in metres with an orbital error in AU would say little. Comparing
a direct projectile prediction with a recursive orbital prediction would also mix two questions.
This stage first holds the **flow-map formulation** fixed and then normalises the errors.

## 12.1 Design and the controls actually implemented

| Design element | Implementation |
|---|---|
| Independent variables | Physical system, model family, training rows and normalised prediction time |
| Dependent variables | Relative position error, training and inference cost, sensitivity response |
| Common formulation | Current four-component state → next state, trained on delta targets |
| Common data volume | 20,000 training pairs per system; 20 separate rollout test trajectories |
| Common models | Linear, polynomial degree 2, Random Forest and MLP with the existing fixed hyperparameters |
| Position normalisation | Error / horizontal range for a flight; error / semi-major axis for an orbit |
| Time normalisation | Time / flight duration or orbital period |
| Sampling | Fixed 0.02 s projectile steps and 0.004 yr orbital steps; steps per natural timescale are **not matched** |
| Horizon | One flight for projectiles; three orbital periods for planets; compare both at normalised time 1 |

A flight duration is a natural timescale, not a periodic return. The finite projectile time grid
may end slightly before landing; interpolation to normalised time 1 then uses the last sampled
error. The comparison is therefore at approximately one complete flight.

The one-step columns in `results/stage12_summary.csv` are evaluated on **training pairs**.
Likewise, `results/stage12_learning_curves.csv` evaluates on the full training pool, which overlaps
each fitted subset. Those are fit diagnostics, not independent generalisation measurements.
The separate rollout trajectories provide the predictive comparison below. Stage 11 supplies
held-out one-step measurements for the orbital problem.

## 12.2 Results at one natural timescale

**Table 12.1 — Mean dimensionless position error at one flight/orbit.**
Source: `results/stage12_summary.csv`, column `relative_error_at_one_period`.

| Model | Projectile | Planetary | Planetary / projectile |
|---|---:|---:|---:|
| Linear | **3.000 × 10⁻¹⁵** | 0.8641 | 2.88 × 10¹⁴ |
| Polynomial, degree 2 | **1.159 × 10⁻¹⁰** | 0.8343 | 7.20 × 10⁹ |
| Random Forest | **0.002878** | 0.2738 | 95.1 |
| MLP | **0.001472** | 0.4303 | 292.3 |

The forest reaches about **0.29% of a flight range**, versus **27.4% of an orbital semi-major
axis**. Its orbital error is about 95 times larger. The MLP ratio is about 292. Ratios involving
a denominator near round-off should not be interpreted as meaningful counts of extra accuracy
digits; the important observation is exact representability of the projectile map.

**Figure 12.1 — `results/stage12_flow_map_comparison.png`.** Compare the two panels at
normalised time 1. The planet panel continues to three periods, while the flight panel stops
at one. The logarithmic y-axis exposes the near-round-off linear projectile result as well as
the much larger accumulated orbital errors.

## 12.3 Why the difference follows from the equations

For a fixed Δt the exact projectile state update is

> x′ = x + vₓΔt;  y′ = y + vᵧΔt − ½gΔt²;
> vₓ′ = vₓ;  vᵧ′ = vᵧ − gΔt.

Every output is an affine function of the input state. Ordinary linear regression includes this
function, so it can recover the map to numerical precision. This does not contradict Section 6:
that section supplied launch speed, angle, height and time, whose mapping to position contains
products and trigonometric functions. **The input representation changes what a linear model can
express.** The polynomial fit also contains affine functions, but regularisation and numerical
conditioning prevent identical round-off behaviour.

Orbital acceleration instead depends on −GM r / |r|³. The finite-time flow map is nonlinear
and is not supplied explicitly as a physical basis in these models. A low-degree polynomial
can approximate it locally without enforcing the correct global geometry. A network can
approximate nonlinear functions, but its fitted accuracy and stability are empirical questions.
The result establishes a representational advantage for projectile motion; unmatched step counts,
state distributions and different physical scales prevent attributing the entire numerical gap
to one isolated factor.

## 12.4 More data, sensitivity and cost

**Figure 12.2 — `results/stage12_learning_curves.png`.** Training-set sizes are 500, 2,000,
8,000 and 20,000 pairs. Read these as approximation-to-training-pool curves, with the overlap
qualification in Section 12.1. They do not establish held-out sample efficiency. For independent
learning-curve evidence use Stages 7 and 11.

**Figure 12.3 — `results/stage12_sensitivity.png`.** The exact projectile is perturbed through
launch speed, while the exact orbit is perturbed through semi-major axis. Their fractional
parameter offsets are equal, but they are different perturbation directions, not identical
state-space displacements. The figure compares those specified responses and is not a universal
condition-number ranking. At one natural timescale the measured amplification is **1.414×**
for the speed-perturbed projectile and **7.407×** for the semi-major-axis-perturbed orbit. Neither
experiment supplies evidence of chaotic dynamics.
Source: `results/stage12_sensitivity.csv`.

**Figure 12.4 — `results/stage12_cost.png`.** Separate training time from recurring prediction
cost. Learned inference is timed in batches, projectile physics is a vectorised analytic
calculation, and the orbital baseline is a sequential RK4 calculation with reference substeps.
The recorded classical costs are approximately **0.032 μs** per vectorised projectile position
and **453 μs** per stored orbital step with reference substeps. Those workloads and accuracies differ. A smaller microseconds-per-prediction number alone does
not establish a useful end-to-end speedup at a matched error tolerance. Timings in
`results/stage12_summary.csv` are machine- and load-dependent measurements.

**Figure 12.5 — `results/stage12_scorecard.png`.** This is a visual summary, not a statistical
composite score. Criteria choose their own best model and normalisation. The representability
row uses illustrative scores (1 and 0.05), not measured probabilities or accuracies. Consult
Table 12.1 and the underlying CSVs for quantitative claims.

Stage 12 declares escape relative to 50 times the mean test-system length, rather than using
an AU threshold on metre-valued trajectories. Non-finite predictions receive that scale-based
failure penalty. This preserves failed runs without confusing position units.

## 12.5 What this stage establishes — and what it does not

Under the implemented common model families, data volume and flow-map formulation, projectile
rollouts are much more accurate in relative position than orbital rollouts. The affine projectile
map explains why even the simplest fitted model can be effectively exact. Normalising the axes
does not by itself equalise sampling or perturbation directions, and the training-pool diagnostics
must not be described as test performance. A stricter follow-up would match Δt / natural timescale,
use disjoint one-step evaluation trajectories for both arms, and compare computational cost at
an equal accuracy target.

---

# PART III — SYNTHESIS

---

# 13. Stage 13 — Physics-Informed Machine Learning

Sections 9 and 11 are diagnostic: they establish *where* a standard ML pipeline fails and why. This section
is constructive. It asks whether putting the physics **into the model** rather than only into the data changes
the failure modes — and it tests three levels of doing so under identical conditions.

| Level | What is injected | Guarantee | What it needs to know |
|---|---|---|---|
| **13a — hard** | A basis in which the solution is linear | Exact by construction | The **solution** |
| **13b — soft** | The differential equation, as a loss penalty | Approximate | Only the **equation** |
| control | Nothing | None | Only data |

The three-level structure follows the taxonomy in Karniadakis et al. (2021): inductive bias (built into the
architecture), learning bias (built into the loss), and observational bias (built into the data alone).

## 13.1 Level (a) — hard constraints: fitting in a physics basis

The exact solution is x = (v₀cos θ)·t and y = y₀ + (v₀sin θ)·t − (g/2)·t². So in the basis

> [ v₀cos(θ)t , v₀sin(θ)t , y₀ , t² ]

both targets are **linear**, with coefficients that are known in advance. Ordinary linear regression in that
basis should therefore be exact.

**Learned coefficients:**

| | v₀cos(θ)t | v₀sin(θ)t | y₀ | t² |
|---|---|---|---|---|
| **x** | 1.000000 | −0.000000 | 0.000000 | −0.000000 |
| **y** | −0.000000 | 1.000000 | 1.000000 | **−4.905000** |

with intercepts of −0.000000. The t² coefficient for y is **−4.905000**, and −g/2 = −4.905000.

**The fit recovered the gravitational acceleration of the simulation to six decimal places.** This is the
closest anything in this study comes to "extracting physics from data" — and the qualification matters: we
supplied the functional form and the fit supplied only the constant. That is parameter estimation, not
discovery. It is a hand-specified, four-term version of what SINDy (Brunton et al. 2016) does by searching a
large library with a sparsity penalty.

This model's error is ~10⁻¹³ m **everywhere**, including in all six extrapolation regimes of Stage 8 (Table
9.1), because a correct functional form has no inside or outside.

## 13.2 Level (b) — the PINN: soft constraints from the equation alone

A PINN's loss has three terms instead of one:

> L = w_d·L_data + w_p·L_physics + w_i·L_IC

where L_data is the ordinary supervised term, and the other two need **no labels at all**. They are evaluated
at **collocation points** — arbitrary inputs where we have no measurement but do know the answer must obey

> d²x/dt² = 0   and   d²y/dt² = −g    (13.1)

subject to x(0) = 0, y(0) = y₀, x′(0) = v₀cos θ, y′(0) = v₀sin θ.    (13.2)

Together (13.1) and (13.2) determine the solution **uniquely**, which means a PINN can in principle be trained
with zero labelled examples. Section 13.4 tests exactly that.

### Implementation notes (and why they are in the report)

**Written from scratch in NumPy.** Every other model in this study came from scikit-learn, which was the right
tool. A PINN is not standard supervised regression: its loss contains derivatives of the model's own output
with respect to its own input, which no scikit-learn estimator exposes. Using PyTorch would have hidden the
one mechanism this section exists to explain. So `src/pinn.py` implements the forward pass, backpropagation
and Adam explicitly, in about 120 lines.

**Derivatives by central difference, not autodiff.**

> d²f/dt² ≈ [ f(t+h) − 2f(t) + f(t−h) ] / h²    (13.3)

Automatic differentiation would require differentiating *twice* through the network. With (13.3), the residual
is built from three ordinary **forward** passes, so the gradient with respect to the weights is plain
backpropagation — everything stays first-order and hand-checkable. Truncation error is O(h²) ≈ 10⁻⁴ at
h = 0.01 s, thousands of times below the model error we are trying to reduce. The real trade-off is scaling:
finite differences cost an extra pair of forward passes per derivative direction, so they would be a poor
choice for a PDE in many dimensions. For one direction — time — they are strictly simpler.

**tanh, not ReLU.** A ReLU network is piecewise linear, so its second derivative is zero almost everywhere: it
is *structurally incapable* of satisfying (13.1), which demands a specific non-zero curvature. tanh is smooth
and infinitely differentiable.

**The initial-condition term is mandatory.** (13.1) alone is satisfied by *every* parabola with the right
curvature, regardless of where it starts. Omitting (13.2) is the most common way a hand-written PINN silently
fails: it converges to a perfect physics loss and completely wrong positions.

**All three loss terms are normalised to be dimensionless and O(1)** before weighting — the physics residual by
g, the data term by the target standard deviation, the IC terms by the spreads of y₀ and v₀. Without this the
data term (measured in m², typical value ~10²) would outweigh the physics term (dimensionless, ~1) by two
orders of magnitude and `physics_weight` would not mean what it says. Karniadakis et al. (2021) flag loss
balancing as an open problem; this is a defensible convention, not a solution.

**Gradient verification.** Because the gradients are hand-derived, all three were checked against central
finite differences of the loss:

| Loss term | Max gradient error | Relative to largest gradient |
|---|---|---|
| Data | 6.2 × 10⁻¹⁰ | — |
| Physics | 6.2 × 10⁻⁶ | **8.6 × 10⁻⁷** |
| Initial conditions | 5.0 × 10⁻⁹ | — |

The physics figure is larger in absolute terms only because that loss carries a factor 1/h² = 10⁴ which
amplifies both the true gradient and the finite-difference noise; 10⁻⁶ relative is the expected floor.

## 13.3 Design of the comparison

The control is not a strawman. **The "plain network" is the same class with `physics_weight = 0`** — same
architecture (32-32 tanh), same initialisation scheme, same Adam optimiser and learning rate, same 4,000
epochs, same collocation points. The only difference between the two arms is the loss. Any gap is therefore
attributable to the physics term and to nothing else. Three seeds per configuration; every number below is a
mean over them.

## 13.4 Results: data efficiency

**Table 13.1 — Mean radial test error (m), mean ± SD over three seeds**

| Labelled points | PINN | Plain NN | Ratio |
|---|---|---|---|
| **0** | **1.392 ± 0.174** | *(cannot be trained)* | — |
| 10 | 1.279 ± 0.086 | 21.399 ± 1.238 | **16.7×** |
| 20 | 0.893 ± 0.104 | 14.995 ± 0.536 | **16.8×** |
| 50 | 0.742 ± 0.045 | 7.925 ± 0.736 | **10.7×** |
| 100 | 0.546 ± 0.071 | 3.739 ± 0.131 | 6.9× |
| 200 | 0.548 ± 0.086 | 0.811 ± 0.033 | 1.5× |
| 500 | 0.466 ± 0.053 | 0.567 ± 0.107 | 1.2× |
| 2,000 | 0.429 ± 0.047 | 0.364 ± 0.060 | 0.85× |

**Figure:** `stage13_data_efficiency.png`.

**The headline.** A PINN trained on **zero labelled points** reaches 1.39 m. The plain network needs roughly
**200 labelled points** to match that. The physics term is worth about two orders of magnitude of data in the
scarce-data regime.

**The honest right-hand side of the table.** At 2,000 labels the two are equivalent within the spread of the
seeds (0.429 ± 0.047 vs 0.364 ± 0.060) — and if anything the plain network is marginally ahead. **Physics-
informed learning is a low-data technique.** A report that showed only the left-hand side would be
overselling it, and the figure deliberately includes the crossover.

**Why the PINN plateaus around 0.4–0.5 m rather than reaching machine precision.** The physics constraint is a
*penalty*, not a guarantee: the optimiser trades a small residual against the other terms, and the finite
network capacity and finite training length set a floor. That is precisely the difference between level (a)
and level (b) — the hard-constraint model of Section 13.1 cannot violate the physics at all and is exact at
10⁻¹³ m.

## 13.5 Results: physics fills in where data runs out

This is the most persuasive experiment in the section, and `stage13_extrapolation.png` is the figure to show.

Both networks receive labelled points from **only the first 40% of each flight**. The PINN additionally
receives collocation points — inputs with no labels attached — covering the whole flight, where the only thing
it is told is that (13.1) must hold.

**Table 13.2 — Mean radial error (m) inside and beyond the labelled time window**

| Region | PINN | Plain NN | Ratio |
|---|---|---|---|
| Inside the labelled window (t/t_flight < 0.4) | 0.292 | 0.469 | 1.6× |
| **Beyond it (t/t_flight > 0.4)** | **1.343** | **19.737** | **14.7×** |

Inside the data, both are fine. Beyond it, the plain network is extrapolating in exactly the sense of Section
9 and it wanders — visibly flying upward off the parabola. The PINN has no data there either, but it is not
unconstrained: it still has to satisfy d²y/dt² = −g, and that single requirement very nearly pins the
trajectory down.

**This is the practical content of the phrase "physics-informed".** A physical law is a statement about
everywhere, so it can constrain a model in regions where no measurement exists. Nothing a purely data-driven
model can be given has that property — which is the deep reason Section 9's extrapolation failures were not
fixable with more data of the same kind.

## 13.6 Results: is the output physically possible?

Position error asks *is it in the right place*. The residual of (13.1) asks a different question: *could
anything obeying Newton's laws move like this at all?* The same finite-difference measurement was applied to
every model in the study, so the numbers are directly comparable.

**Table 13.3 — Physical-consistency audit. For scale, g = 9.81 m/s².**

| Model | Mean radial error (m) | ǀd²x/dt²ǀ (m/s²) | ǀd²y/dt² + gǀ (m/s²) |
|---|---|---|---|
| Linear Regression | 10.691 | **0.000** | **9.810** |
| Polynomial Ridge (deg 2) | 1.857 | 0.239 | 0.332 |
| **Random Forest** | **2.652** | **2015.79** | **1494.89** |
| Neural Network (MLP) | 0.146 | 1.648 | 1.699 |
| Physics-Informed Features (hard) | **0.000** | **0.000** | **0.000** |
| PINN (data + physics, soft) | 0.858 | 0.207 | 0.210 |
| Plain NN (same net, data only) | 12.415 | 5.426 | 9.655 |

**Figure:** `stage13_physics_residual.png`.

Three rows deserve comment, and each is a different lesson.

**Linear Regression: residual_x exactly 0, residual_y exactly 9.81.** Its prediction is a straight line in t,
so d²x/dt² = 0 — which happens to be *correct*. And d²y/dt² = 0 too, so ǀ0 + gǀ = 9.81 — the maximum possible
error, because the model has no curvature at all. Both numbers are exactly what the model's functional form
implies, which is a satisfying check that the measurement is doing what it claims.

**Random Forest: position error 2.65 m, residual ~2,000 m/s².** This is the important row. By RMSE the forest
is one of the better models; by physical consistency it is by far the worst, off by a factor of 200 relative
to g. The reason is the staircase structure of Sections 6.4 and 9.4: a forest's output is piecewise constant
in t, jumping between leaf values, so its numerical second derivative is enormous even where its positions are
good. **The two evaluation questions come apart completely, and only one of them is usually asked.**

**The two physics-informed models are the only ones with small residuals**, at 0.000 and ~0.21 m/s². That is
the point of the whole section: they are the only models that were told what the equation is.

## 13.7 What this stage establishes — and what it does not

**Establishes:**
1. The differential equation can substitute for roughly two orders of magnitude of labelled data.
2. Physics in the loss constrains a model where no data exists, which no amount of data of the same kind can do.
3. Physical consistency is a distinct evaluation axis from position error, and models can score well on one
   while failing badly on the other.
4. Hard constraints (a correct basis) beat soft constraints (a penalty) by ten orders of magnitude when the
   solution is known — but they require knowing the solution, which is exactly the case where one would not
   need machine learning.

**Does not establish:**
1. **That PINNs beat plain networks in general.** At 2,000 labels they are equivalent here.
2. **That the PINN discovered physics.** It was *told* d²y/dt² = −g. It learned a function consistent with
   what it was told — a different and much weaker claim than discovery.
3. **That PINNs are fast.** The PINN takes ~6 s to train against ~0.6 s for the plain network, and both are
   infinitely slower than evaluating the closed-form solution. Raissi et al. (2019) position PINNs for problems
   where data and physics are *both* partial; that is not this problem, and this section is a controlled
   demonstration of the mechanism, not a recommendation to use a PINN for projectile motion.

---

# 14. General Discussion, Limitations and Conclusions

## 14.1 Answer to the research question

Machine learning approximates these motions with very different success depending on the input
representation, training domain and prediction procedure. It performs well on interpolated
projectile positions; its learned orbital flow maps lose accuracy under repeated application.
Classical equations remain the accuracy baseline. The hard-constraint projectile fit matches
that baseline because its features already encode the correct physical structure.

| Question | Evidence | What follows |
|---|---|---|
| Can ordinary regression predict motion? | Forest projectile R² = **0.9924** for x (Section 6) | Yes, within the evaluated distribution; a score alone is not evidence of a physical law |
| Does more data help? | Forest x RMSE falls to **2.05 m** at 10,000 rows (Section 8) | It helps a flexible model; it cannot repair an inadequate linear feature representation |
| Does interpolation imply extrapolation? | MLP error rises **286×** with all variables outside the range (Section 9) | The training envelope is part of the result |
| Does a small step error imply a long useful forecast? | MLP passes 0.01 AU in **0.372 yr** (Section 11) | Evaluate recursive rollouts and conservation explicitly |
| Are the physical systems equally easy for these models? | Forest relative error differs by **95×** at one natural timescale (Section 12) | No, under these controls; affine representability explains an important part of the gap |
| Can known physics reduce the data requirement? | Zero-label PINN: **1.392 ± 0.174 m**; extrapolation advantage **14.7×** (Section 13) | Physical constraints help most where labelled coverage is scarce |

## 14.2 Representation, propagation and constraints

The study separates three mechanisms. First, representation determines whether the target map
can be expressed at all: linear regression fails on raw projectile launch parameters but succeeds
on a state-to-state update or a suitable physics basis. Second, recursive prediction propagates
both state error and fresh model error. Third, physical constraints restrict the functions a model
can fit. A hard constraint gives an algebraic guarantee in the idealised system; a soft residual
penalty encourages physical consistency without guaranteeing it everywhere.

These mechanisms explain why a single “best model” ranking would be misleading. The polynomial
orbital model wins one-step position accuracy yet often escapes during rollout. The direct MLP
is strong in its densely covered time interval and poor beyond it. At 2,000 labels the PINN
(0.429 ± 0.047 m) and plain network (0.364 ± 0.060 m) are comparable; physics regularisation is
not an unconditional improvement at every data volume.

## 14.3 Relation to the literature

The supplied literature review provides the context for these results. Its discussions of
Breiman (2001) and Hairer, Lubich and Wanner (2006) motivate tree behaviour and numerical
conservation diagnostics. Raissi, Perdikaris and Karniadakis (2019) provides the physics-residual
approach examined in Stage 13. The sections on learned simulators, Hamiltonian networks and
Neural ODEs identify possible responses to the failure modes measured here.

This project is a small controlled benchmark, not a replication of those large studies. In
particular, an unconstrained model's conservation failure is not an experimental evaluation of
a Hamiltonian network, and integrable two-body dynamics do not reproduce a chaotic three-body
experiment. Bibliographic details and the scope of each comparison appear in Section 16 and
[LITERATURE_REVIEW.md](LITERATURE_REVIEW.md).

## 14.4 Limitations and the next experiment for each

| Limitation | Consequence | Specific next experiment |
|---|---|---|
| Ideal noise-free synthetic data | No estimate of performance on measurement noise or unmodelled forces | Add controlled position noise and drag, then repeat independent tests |
| Fixed hyperparameters and modest models | Results are baseline performance, not an architecture-wide limit | Tune on a validation set, retaining untouched test trajectories |
| Integrable two-body system | No conclusion about chaotic forecasting | Introduce a defined three-body benchmark with independent reference-error checks |
| Unequal normalised step sizes in Stage 12 | System gap includes sampling differences | Match Δt / natural timescale and test both arms on disjoint trajectories |
| Overlapping Stage 12 learning-curve evaluation pool | Curves cannot measure independent generalisation | Replace that pool with held-out trajectory pairs |
| Uneven direct-map time coverage | A single global time boundary hides local extrapolation | Train on a common time horizon or phase coordinate; report coverage by a and t |
| Escape penalties and finite horizons | Late means depend on reporting conventions | Report survival fractions and censored horizons alongside multiple penalty choices |
| Limited repetitions | Across-orbit spread is not uncertainty over all training seeds | Repeat Stage 11 and 12 training with several seeds and report seed-level variation |
| Different timing workloads | Inference speed is not a matched-accuracy solver comparison | Benchmark end-to-end time at specified error tolerances |
| Physics constraints use the correct ideal law | Advantage may shrink with a misspecified equation | Perturb g or introduce drag while retaining the old residual, then quantify bias |

## 14.5 Conclusions

1. Ordinary regression can fit motion data accurately without recovering a physical law.
2. More data improves some approximations, while input representation sets fundamental limits
   for the tested linear and polynomial models.
3. Extrapolation, rollout error and conservation must be evaluated separately from an
   interpolation score or one-step position error.
4. The projectile state update is affine; its fitted linear accuracy is an expected consequence
   of the equations. Orbital rollouts remain much harder for the tested unconstrained models.
5. Known physics improves the low-data and uncovered-domain results here. The zero-label PINN
   still receives the equation, initial-condition information and collocation points; it is not
   learning without information.

The contribution is an interpretable measurement of when the approximations work, when they fail,
and which physical and statistical mechanisms account for those outcomes. It is not a claim
that a black-box model discovered Newton's laws or superseded classical mechanics.

---

# 15. How To Reproduce This Project From Scratch

Use the repository source files, rather than copying the historical teaching snippets in Section 5.
From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py --list
python run_all.py
```

On Windows activate with `.venv\Scripts\activate`. A full run takes approximately 49 minutes
on the original machine; actual runtime varies. The driver runs datasets and experiments in
dependency order, stops on failure and prints a final stage summary.

```bash
python run_all.py --only 8 13
python run_all.py --skip 11 12
python validate_project.py
```

The final command performs fast structural and numerical sanity checks without retraining.
Individual stage selection does not automatically run all prerequisites; use the full run for
a fresh checkout. `data/` contains generated datasets and `results/` contains experiment tables
and figures. Appendix D maps every delivered result artifact to its producing stage.

The dependencies are pinned to the versions recorded for the original study. Random seeds are
fixed, but floating-point libraries and execution environments can cause small numerical changes.
**Timing columns are measurements and will vary even on the same machine.** Byte-identical CSVs
are therefore not a reproduction requirement. Compare scientific metrics at their reported
precision and review any larger discrepancies.

---

# 16. References

The bibliography below is drawn from the project's existing verified reference list. The
companion [literature review](LITERATURE_REVIEW.md) explains each source's role and distinguishes
its published findings from this project's experiments. No new citations were added during
completion.


Verified bibliographic details. Sorted alphabetically by first author.

1. **Breen, P. G., Foley, C. N., Boekholt, T. & Portegies Zwart, S.** (2020). Newton versus the
   machine: solving the chaotic three-body problem using deep neural networks. *Monthly Notices
   of the Royal Astronomical Society*, **494**(2), 2465–2470.
   DOI: [10.1093/mnras/staa713](https://doi.org/10.1093/mnras/staa713) · arXiv:1910.07291

2. **Breiman, L.** (2001). Random Forests. *Machine Learning*, **45**(1), 5–32.
   DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

3. **Brunton, S. L., Proctor, J. L. & Kutz, J. N.** (2016). Discovering governing equations from
   data by sparse identification of nonlinear dynamical systems. *Proceedings of the National
   Academy of Sciences*, **113**(15), 3932–3937.
   DOI: [10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113)

4. **Chen, R. T. Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. K.** (2018). Neural Ordinary
   Differential Equations. *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*,
   6572–6583. arXiv:[1806.07366](https://arxiv.org/abs/1806.07366)

5. **Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D. & Ho, S.** (2020).
   Lagrangian Neural Networks. *ICLR 2020 Workshop on Integration of Deep Neural Models and
   Differential Equations*. arXiv:[2003.04630](https://arxiv.org/abs/2003.04630)

6. **Greydanus, S., Dzamba, M. & Yosinski, J.** (2019). Hamiltonian Neural Networks. *Advances in
   Neural Information Processing Systems 32 (NeurIPS 2019)*, 15353–15363.
   arXiv:[1906.01563](https://arxiv.org/abs/1906.01563)

7. **Hairer, E., Lubich, C. & Wanner, G.** (2006). *Geometric Numerical Integration:
   Structure-Preserving Algorithms for Ordinary Differential Equations* (2nd ed.). Springer
   Series in Computational Mathematics, Vol. 31. Springer.
   DOI: [10.1007/3-540-30666-8](https://doi.org/10.1007/3-540-30666-8)

8. **Hornik, K., Stinchcombe, M. & White, H.** (1989). Multilayer feedforward networks are
   universal approximators. *Neural Networks*, **2**(5), 359–366.
   DOI: [10.1016/0893-6080(89)90020-8](https://doi.org/10.1016/0893-6080(89)90020-8)

9. **Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S. & Yang, L.** (2021).
   Physics-informed machine learning. *Nature Reviews Physics*, **3**(6), 422–440.
   DOI: [10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5)

10. **Kingma, D. P. & Ba, J.** (2015). Adam: A Method for Stochastic Optimization.
    *3rd International Conference on Learning Representations (ICLR 2015)*.
    arXiv:[1412.6980](https://arxiv.org/abs/1412.6980)

11. **Lagaris, I. E., Likas, A. & Fotiadis, D. I.** (1998). Artificial neural networks for solving
    ordinary and partial differential equations. *IEEE Transactions on Neural Networks*, **9**(5),
    987–1000. DOI: [10.1109/72.712178](https://doi.org/10.1109/72.712178)

12. **Lam, R., Sanchez-Gonzalez, A., Willson, M., et al.** (2023). Learning skillful medium-range
    global weather forecasting. *Science*, **382**(6677), 1416–1421.
    DOI: [10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336) · arXiv:2212.12794

13. **Pedregosa, F., Varoquaux, G., Gramfort, A., et al.** (2011). Scikit-learn: Machine Learning
    in Python. *Journal of Machine Learning Research*, **12**, 2825–2830.

14. **Raissi, M., Perdikaris, P. & Karniadakis, G. E.** (2019). Physics-informed neural networks:
    A deep learning framework for solving forward and inverse problems involving nonlinear partial
    differential equations. *Journal of Computational Physics*, **378**, 686–707.
    DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

15. **Rein, H. & Liu, S.-F.** (2012). REBOUND: an open-source multi-purpose N-body code for
    collisional dynamics. *Astronomy & Astrophysics*, **537**, A128.
    DOI: [10.1051/0004-6361/201118085](https://doi.org/10.1051/0004-6361/201118085)

16. **Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J. & Battaglia, P. W.**
    (2020). Learning to Simulate Complex Physics with Graph Networks. *Proceedings of the 37th
    International Conference on Machine Learning*, PMLR **119**, 8459–8468.
    arXiv:[2002.09405](https://arxiv.org/abs/2002.09405)

17. **Schmidt, M. & Lipson, H.** (2009). Distilling Free-Form Natural Laws from Experimental Data.
    *Science*, **324**(5923), 81–85.
    DOI: [10.1126/science.1165893](https://doi.org/10.1126/science.1165893)

18. **Verlet, L.** (1967). Computer "Experiments" on Classical Fluids. I. Thermodynamical
    Properties of Lennard-Jones Molecules. *Physical Review*, **159**(1), 98–103.
    DOI: [10.1103/PhysRev.159.98](https://doi.org/10.1103/PhysRev.159.98)

### Textbooks used for the physics and the statistics

19. **Goldstein, H., Poole, C. & Safko, J.** (2002). *Classical Mechanics* (3rd ed.).
    Addison-Wesley. — Lagrangian/Hamiltonian mechanics, the two-body reduction, orbital elements.

20. **Hastie, T., Tibshirani, R. & Friedman, J.** (2009). *The Elements of Statistical Learning*
    (2nd ed.). Springer. — bias–variance decomposition, model selection, ensemble methods.

21. **Murray, C. D. & Dermott, S. F.** (1999). *Solar System Dynamics*. Cambridge University
    Press. — Kepler's equation, orbital elements, the two-body problem.

22. **Press, W. H., Teukolsky, S. A., Vetterling, W. T. & Flannery, B. P.** (2007).
    *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.
    — Runge–Kutta methods, root finding (Newton–Raphson for Kepler's equation), finite differences.

23. **Taylor, J. R.** (2005). *Classical Mechanics*. University Science Books. — projectile
    motion, the assumptions behind constant-g kinematics, central-force motion.


---

# Appendix A — Source File Index

The repository is the authoritative executable source. Section 5 retains the original teaching
walkthrough; those historical snippets are not a complete listing of the extended project.

| Source | Purpose |
|---|---|
| [run_all.py](../run_all.py) | Reproduction driver in dependency order |
| [validate_project.py](../validate_project.py) | Fast artifact and numerical sanity checks |
| [requirements.txt](../requirements.txt) | Recorded dependency versions |
| [data/generation/](../data/generation/) | Projectile and planetary dataset generators |
| [src/projectile_ml.py](../src/projectile_ml.py) | Original projectile models and evaluation |
| [src/models.py](../src/models.py) | Shared model factory and physics-basis model |
| [src/evaluation.py](../src/evaluation.py) | Common error metrics |
| [src/planetary_physics.py](../src/planetary_physics.py) | Integrators and independent Kepler reference |
| [src/planetary_ml.py](../src/planetary_ml.py) | Flow-map, direct and history models; rollout diagnostics |
| [src/pinn.py](../src/pinn.py) | NumPy network, physics residual, gradients and optimiser |
| [experiments/](../experiments/) | Stage 7, 8, 10, 11, 12 and 13 experimental designs and runs |
| [plots/](../plots/) | Shared figure style and per-stage plotting functions |

---

# Appendix B — Glossary

Every technical term used in this document, defined for a reader with no ML background.

| Term | Definition |
|---|---|
| **Coefficient** | A learned weight multiplying one feature in a linear model. See Table 3. |
| **Cross-validation** | Repeating the train/test split several different ways and averaging the results, to obtain error bars rather than a single point estimate. |
| **Decision tree** | A model that makes predictions by asking a sequence of yes/no questions about the features until it reaches a final group (leaf) holding a stored answer. |
| **Ensemble** | A collection of models whose predictions are combined (here, averaged). A Random Forest is an ensemble of decision trees. |
| **Extrapolation** | Prediction *outside* the range of the training data. Dangerous for ML models — a Random Forest literally cannot output a value beyond the range of its training targets. |
| **Feature** | One input column. Ours: `v0`, `theta`, `y0`, `t`. |
| **`fit()`** | The learning step: the model adjusts its internal parameters to reduce error on the training data. |
| **Ground truth** | The known-correct answer. Here, the output of the classical equations (2.1)–(2.2). |
| **Held-out data** | Samples deliberately excluded from training, used only for evaluation. Our 400-row test set. |
| **Hyperparameter** | A setting chosen *before* training, not learned from data (e.g. `n_estimators=100`). |
| **Interpolation** | Prediction *inside* the range of the training data. All of our current testing is interpolation. |
| **Leaf** | A terminal node of a decision tree, storing the average target value of the training samples that reached it. |
| **MAE** | Mean Absolute Error — the average size of the error, in metres. Section 3.7. |
| **Multi-output regression** | Predicting several numbers at once from the same inputs (here, x and y). |
| **Overfitting** | When a model memorises the training data instead of learning a generalisable rule — good training scores, poor test scores. |
| **`predict()`** | The inference step: applying the already-learned rule to new inputs. Changes nothing about the model. |
| **R²** | Coefficient of determination — the fraction of variance explained, relative to always guessing the mean. Section 3.7. |
| **`random_state` / seed** | A fixed number that makes "random" operations produce identical results on every run, ensuring reproducibility. |
| **Regression** | Predicting a continuous number (as opposed to classification, which predicts a category). |
| **Residual** | `actual − predicted` for one sample — the signed error. Plotting residuals reveals *structure* in the errors. |
| **RMSE** | Root Mean Squared Error — like MAE but penalises large errors more heavily; same units as the target. Section 3.7. |
| **Sample / row / instance** | One complete input–output pair. |
| **Supervised learning** | Learning from examples where both the inputs and the correct outputs are provided. |
| **Surrogate model** | A fast approximate stand-in for an expensive exact calculation. The main legitimate use case for ML in physics. |
| **Synthetic data** | Data generated by simulation rather than measured experimentally. Section 7.4. |
| **Target / label** | One output column we want to predict. Ours: `x`, `y`. |
| **Test set** | Held-out data used only for final evaluation. |
| **Training set** | The data the model learns from. Ours: 1,600 rows. |
| **Train/test split** | Dividing the data so that evaluation happens on samples the model has never seen. Section 3.5. |
| **Vectorisation** | Performing arithmetic on whole arrays at once rather than element-by-element in a loop. Faster, and closer to how equations are written. |

---

# Appendix C — Viva Preparation

Questions likely to be asked, with the answer and where in this document the supporting evidence lives.

**Q1. Why use AI to predict something physics already calculates exactly?**
Projectile motion is a *control*, not the application. With an exact ground truth, every unit of error is
unambiguously the model's fault, which lets us measure how well ML learns physics at all. That measurement is
what makes the comparison with planetary dynamics meaningful. → §1.2

**Q2. What exactly is the model learning?**
A mapping from [v₀, θ, y₀, t] to [x, y]. It is *not* learning the equations. It is learning to interpolate among
memorised training examples. → §3.4, §7.3

**Q3. Why can't you evaluate on the training data?**
That measures memorisation, not learning. A flexible model can score near-perfectly on data it has stored while
failing on anything new — overfitting. → §3.5

**Q4. What does an RMSE of 3.47 m actually mean here?**
The horizontal position is typically wrong by about 3.5 m — roughly 1.4% of the 242 m range of x in the dataset,
or 8% of one standard deviation. → §3.7, §6.1

**Q5. Why did Linear Regression fail?**
Because the true relationships are multiplicative (v₀·cos θ·t) and quadratic (−½gt²), and a linear model can only
add its features, never multiply them or bend. Its fitted t-coefficient in the y-equation collapsed to −0.337,
because a straight line through a symmetric rise-and-fall has nearly zero slope. → §6.2

**Q6. Did Linear Regression get anything right?**
Yes, two real physical facts. It learned that x does not depend on y₀ (coefficient −0.093 ≈ 0, and the true
equation has no y₀ term), and it recovered the coefficient of y₀ in the y-equation as 0.824, against a true value
of exactly 1. → §6.2

**Q7. Does a good R² mean the AI discovered Newton's laws?**
No — and this is the project's central finding. The same model that scores R² = 0.9924 produces a jagged
trajectory that under-shoots the apex and never lands, with trajectory-level error 27% higher than its reported
RMSE. Statistical accuracy and physical validity are different things. → §6.4, §7.3

**Q8. Why is the Random Forest trajectory jagged?**
Because a Random Forest predicts by averaging constant values stored in tree leaves. Its output function is
piecewise constant by construction and can never be smooth, no matter how much data it is given. → §3.4, §6.3

**Q9. Is your data real or synthetic?**
Synthetic — generated from the classical equations. This is valid for measuring learnability, but it means the
model inherits every idealisation (no air resistance, constant g, flat Earth, point mass) and that our metrics are
an upper bound on real-world accuracy. → §2.4, §7.4

**Q10. What would happen with measurement noise?**
All metrics would degrade. Noise sets a floor on achievable error that no model can beat, and separating "the
model is wrong" from "the data is noisy" becomes a genuine difficulty. Our noise-free setting removes that
confound deliberately. → §7.4

**Q11. Why not use a neural network?**
With 2,000 samples and four features it offers no clear advantage, while adding architecture, learning-rate and
convergence choices that would obscure the physics question. It becomes defensible in Part II, where smoothness
of the learned function matters much more. → §3.6, §7.6

**Q12. What is the difference between interpolation and extrapolation, and why does it matter?**
Interpolation is prediction inside the training range; extrapolation is outside it. Our results are entirely
interpolation. A Random Forest cannot extrapolate at all — its predictions are averages of stored training
targets, so it can never output a value beyond the range it has seen. Testing this is Stage 8. → §7.5, §8.1

**Q13. What happens if you train with too little data?**
This is exactly what Stage 7 measures. The expected result is a learning curve that falls steeply and then
flattens, with the flattening point identifying how much data is actually needed. Linear Regression should stay
flat throughout, because its limitation is functional form, not data supply. → §8.2

**Q14. Why is planetary prediction expected to be harder?**
Three reasons: the dynamics are governed by coupled differential equations rather than a closed-form solution;
prediction is naturally step-by-step, so errors compound over time; and an orbit is a closed curve that must
return to its start, which a piecewise-constant model cannot produce. → §7.6

**Q15. What is the biggest weakness of your study so far?**
Honestly, three: all metrics come from a single train/test split with no error bars; the trajectory-level result
in §6.4 rests on one launch condition; and no generalisation testing has been done yet. All three are identified
in §7.5 with a plan to address them.

---

*End of Part I. Part II (planetary motion, Stages 9–12) will be appended to this document as those stages are
completed.*

*All numerical results in this report were produced by executing the code in this repository on 10 September 2026.
Figures referenced are the files in `results/` at commit `4702cf9`.*


---

# Appendix D — Index of Every Figure and Table

## D.1 Result artifacts

Every PNG figure and CSV result table is listed below. The stage determines its producing
experiment in `experiments/`; the original three figures come from `src/projectile_ml.py`.
Some schematic curves are calculated directly from the equations during plotting. CSVs containing
rollout diagnostics retain non-finite escaped states deliberately.

| Artifact | Type | Producing stage |
|---|---|---|
| [actual_vs_predicted.png](../results/actual_vs_predicted.png) | Figure | 3–6 |
| [residuals.png](../results/residuals.png) | Figure | 3–6 |
| [stage10_conservation.csv](../results/stage10_conservation.csv) | Data table | 10 |
| [stage10_conservation.png](../results/stage10_conservation.png) | Figure | 10 |
| [stage10_convergence.csv](../results/stage10_convergence.csv) | Data table | 10 |
| [stage10_convergence.png](../results/stage10_convergence.png) | Figure | 10 |
| [stage10_integrator_orbits.png](../results/stage10_integrator_orbits.png) | Figure | 10 |
| [stage10_kepler_third_law.csv](../results/stage10_kepler_third_law.csv) | Data table | 10 |
| [stage10_kepler_third_law.png](../results/stage10_kepler_third_law.png) | Figure | 10 |
| [stage10_orbit_family.png](../results/stage10_orbit_family.png) | Figure | 10 |
| [stage11_approach_comparison.csv](../results/stage11_approach_comparison.csv) | Data table | 11 |
| [stage11_approach_comparison.png](../results/stage11_approach_comparison.png) | Figure | 11 |
| [stage11_conservation_ml.png](../results/stage11_conservation_ml.png) | Figure | 11 |
| [stage11_direct_extrapolation.png](../results/stage11_direct_extrapolation.png) | Figure | 11 |
| [stage11_direct_map.csv](../results/stage11_direct_map.csv) | Data table | 11 |
| [stage11_direct_map_curve.csv](../results/stage11_direct_map_curve.csv) | Data table | 11 |
| [stage11_error_growth.png](../results/stage11_error_growth.png) | Figure | 11 |
| [stage11_history.csv](../results/stage11_history.csv) | Data table | 11 |
| [stage11_horizons.csv](../results/stage11_horizons.csv) | Data table | 11 |
| [stage11_learning_curve.csv](../results/stage11_learning_curve.csv) | Data table | 11 |
| [stage11_learning_curve.png](../results/stage11_learning_curve.png) | Figure | 11 |
| [stage11_one_step_accuracy.csv](../results/stage11_one_step_accuracy.csv) | Data table | 11 |
| [stage11_one_step_accuracy.png](../results/stage11_one_step_accuracy.png) | Figure | 11 |
| [stage11_representative_rollout.csv](../results/stage11_representative_rollout.csv) | Data table | 11 |
| [stage11_rollout_error.csv](../results/stage11_rollout_error.csv) | Data table | 11 |
| [stage11_rollout_orbits.png](../results/stage11_rollout_orbits.png) | Figure | 11 |
| [stage11_rollout_per_orbit.csv](../results/stage11_rollout_per_orbit.csv) | Data table | 11 |
| [stage11_sensitivity.png](../results/stage11_sensitivity.png) | Figure | 11 |
| [stage12_cost.png](../results/stage12_cost.png) | Figure | 12 |
| [stage12_flow_map_comparison.png](../results/stage12_flow_map_comparison.png) | Figure | 12 |
| [stage12_flow_map_curves.csv](../results/stage12_flow_map_curves.csv) | Data table | 12 |
| [stage12_learning_curves.csv](../results/stage12_learning_curves.csv) | Data table | 12 |
| [stage12_learning_curves.png](../results/stage12_learning_curves.png) | Figure | 12 |
| [stage12_scorecard.csv](../results/stage12_scorecard.csv) | Data table | 12 |
| [stage12_scorecard.png](../results/stage12_scorecard.png) | Figure | 12 |
| [stage12_sensitivity.csv](../results/stage12_sensitivity.csv) | Data table | 12 |
| [stage12_sensitivity.png](../results/stage12_sensitivity.png) | Figure | 12 |
| [stage12_summary.csv](../results/stage12_summary.csv) | Data table | 12 |
| [stage13_consistency_audit.csv](../results/stage13_consistency_audit.csv) | Data table | 13 |
| [stage13_data_efficiency.csv](../results/stage13_data_efficiency.csv) | Data table | 13 |
| [stage13_data_efficiency.png](../results/stage13_data_efficiency.png) | Figure | 13 |
| [stage13_extrapolation.csv](../results/stage13_extrapolation.csv) | Data table | 13 |
| [stage13_extrapolation.png](../results/stage13_extrapolation.png) | Figure | 13 |
| [stage13_physics_residual.png](../results/stage13_physics_residual.png) | Figure | 13 |
| [stage13_training_history.png](../results/stage13_training_history.png) | Figure | 13 |
| [stage7_learning_curve.csv](../results/stage7_learning_curve.csv) | Data table | 7 |
| [stage7_learning_curve.png](../results/stage7_learning_curve.png) | Figure | 7 |
| [stage7_learning_curve_summary.csv](../results/stage7_learning_curve_summary.csv) | Data table | 7 |
| [stage7_stability.png](../results/stage7_stability.png) | Figure | 7 |
| [stage7_trajectory_by_datasize.png](../results/stage7_trajectory_by_datasize.png) | Figure | 7 |
| [stage7_trajectory_error.csv](../results/stage7_trajectory_error.csv) | Data table | 7 |
| [stage8_extrapolation_distance.csv](../results/stage8_extrapolation_distance.csv) | Data table | 8 |
| [stage8_extrapolation_distance.png](../results/stage8_extrapolation_distance.png) | Figure | 8 |
| [stage8_generalization.csv](../results/stage8_generalization.csv) | Data table | 8 |
| [stage8_regime_bars.png](../results/stage8_regime_bars.png) | Figure | 8 |
| [stage8_response_slice.csv](../results/stage8_response_slice.csv) | Data table | 8 |
| [stage8_response_slice.png](../results/stage8_response_slice.png) | Figure | 8 |
| [stage8_time_extrapolation.csv](../results/stage8_time_extrapolation.csv) | Data table | 8 |
| [stage8_time_extrapolation.png](../results/stage8_time_extrapolation.png) | Figure | 8 |
| [stage8_trajectories.png](../results/stage8_trajectories.png) | Figure | 8 |
| [trajectory_comparison.png](../results/trajectory_comparison.png) | Figure | 3–6 |

## D.2 Tables within this report

This index includes design, results, interpretation and glossary tables in reading order.
The section and first column identify each table; the artifact index above is not repeated here.

| Table in reading order | Section | First column |
|---|---|---|
| D.1 | 1.4 Research Question | # |
| D.2 | 2.3 Units and Physical Meaning | Symbol |
| D.3 | 2.4 Assumptions and Their Limitations | Assumption |
| D.4 | 4.2 Variables | Type |
| D.5 | 4.2 Variables | **Independent** |
| D.6 | 4.2 Variables | (unlabelled) |
| D.7 | 4.2 Variables | **Dependent** |
| D.8 | 4.2 Variables | **Controlled** |
| D.9 | 4.2 Variables | (unlabelled) |
| D.10 | 4.2 Variables | (unlabelled) |
| D.11 | 4.2 Variables | (unlabelled) |
| D.12 | 4.2 Variables | (unlabelled) |
| D.13 | 4.3 Dataset Design — The Key Methodological Decision | Column |
| D.14 | 4.5 Software Environment | Component |
| D.15 | 6.1 Quantitative Model Performance | Model |
| D.16 | 6.1 Quantitative Model Performance | Target |
| D.17 | 6.2 Why Linear Regression Fails — and What It Nevertheless Got Right | Output |
| D.18 | 6.4 Trajectory-Level Error vs Point-Wise Error — The Central Result | Model |
| D.19 | 6.4 Trajectory-Level Error vs Point-Wise Error — The Central Result | Measurement mode |
| D.20 | 7.2 Answering Sub-Question 2: ML vs Classical Physics | Criterion |
| D.21 | 8.1 Design | Element |
| D.22 | 8.2 Results | n_train |
| D.23 | 8.2 Results | n_train |
| D.24 | 9.2 Design | Element |
| D.25 | 9.3 Results | Test regime |
| D.26 | 9.3 Results | Test regime |
| D.27 | 9.6 Extrapolating forward in time | t / t_flight |
| D.28 | 10.1 The physics | Assumption |
| D.29 | 10.3 Four integrators | Method |
| D.30 | 10.4 Results: the orbit picture | Method |
| D.31 | 10.5 Results: conservation over 200 orbits | Orbit |
| D.32 | 10.5 Results: conservation over 200 orbits | circular (e = 0) |
| D.33 | 10.5 Results: conservation over 200 orbits | (unlabelled) |
| D.34 | 10.5 Results: conservation over 200 orbits | (unlabelled) |
| D.35 | 10.5 Results: conservation over 200 orbits | eccentric (e = 0.6) |
| D.36 | 10.5 Results: conservation over 200 orbits | (unlabelled) |
| D.37 | 10.5 Results: conservation over 200 orbits | (unlabelled) |
| D.38 | 10.6 Results: convergence order — verifying the implementation | Method |
| D.39 | 10.7 Results: Kepler's third law emerges | a (AU) |
| D.40 | 10.8 Computational cost | Method |
| D.41 | 10.9 The error budget for Part II | Check |
| D.42 | 11.1 Design | Design element |
| D.43 | 11.2 One-step accuracy: predicting the change helps some models | Model |
| D.44 | 11.3 Rollout: local accuracy is not a usable forecast horizon | Model |
| D.45 | 12.1 Design and the controls actually implemented | Design element |
| D.46 | 12.2 Results at one natural timescale | Model |
| D.47 | 13. Stage 13 — Physics-Informed Machine Learning | Level |
| D.48 | 13.1 Level (a) — hard constraints: fitting in a physics basis | (unlabelled) |
| D.49 | Implementation notes (and why they are in the report) | Loss term |
| D.50 | 13.4 Results: data efficiency | Labelled points |
| D.51 | 13.5 Results: physics fills in where data runs out | Region |
| D.52 | 13.6 Results: is the output physically possible? | Model |
| D.53 | 14.1 Answer to the research question | Question |
| D.54 | 14.4 Limitations and the next experiment for each | Limitation |
| D.55 | Appendix A — Source File Index | Source |
| D.56 | Appendix B — Glossary | Term |
