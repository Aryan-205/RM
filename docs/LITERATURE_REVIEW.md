# Literature Review

**Machine Learning-Based Prediction of Physical Motion: A Comparative Study of Projectile and Planetary Dynamics**

Aryan Bola — Research Methodology, 5th Semester B.Sc. Physics

---

## How to use this document

Every source below is a real, locatable publication. Each entry follows the same five-question
structure the project set out to use:

1. What problem did they study?
2. What method did they use?
3. What did they find?
4. How is it relevant to this project?
5. What limitation or gap remains?

Citation details (journal, volume, pages, DOI/arXiv id) were checked against the publishers'
own records. **Nothing in this list is invented.** Where a claim is a summary rather than a
direct quotation it is written as a summary.

The review is organised as a funnel: from the broad question of whether machine learning can
do physics at all, down to the specific gap this project occupies.

---

## Contents

- [1. Foundational machine learning and numerical methods](#1-foundational-machine-learning-and-numerical-methods)
- [2. Learning physical laws from data](#2-learning-physical-laws-from-data)
- [3. Physics-informed machine learning](#3-physics-informed-machine-learning)
- [4. Structure-preserving and inductive-bias approaches](#4-structure-preserving-and-inductive-bias-approaches)
- [5. Learned simulators and error accumulation](#5-learned-simulators-and-error-accumulation)
- [6. Machine learning in orbital and gravitational dynamics](#6-machine-learning-in-orbital-and-gravitational-dynamics)
- [7. Synthesis: what is established, what is contested](#7-synthesis-what-is-established-what-is-contested)
- [8. The research gap this project occupies](#8-the-research-gap-this-project-occupies)
- [9. Full reference list](#9-full-reference-list)

---

# 1. Foundational machine learning and numerical methods

These are not about physics. They are the sources for the methods this project uses, and are
cited so that the methodology section can point to a primary description of each technique
rather than to a tutorial.

### 1.1 Breiman (2001) — Random Forests

**Problem.** How can the variance of a decision-tree predictor be reduced without introducing
the bias that comes from pruning it?

**Method.** Grow many trees, each on a bootstrap resample of the data and each choosing its
splits from a random subset of the features; average their outputs.

**Findings.** The ensemble's generalisation error converges as the number of trees grows, and
is controlled by the strength of the individual trees and the correlation between them.

**Relevance.** The Random Forest is one of the two headline models in Part I of this project.
Breiman's construction is also the direct explanation of the Stage 8 result: a forest's
prediction is an average of training targets within a region of input space, so outside the
region covered by training data there are no new regions to average over and the prediction
necessarily becomes constant.

**Limitation / gap.** The paper is about generalisation *within* the training distribution.
It makes no claim about extrapolation, and this project's Stage 8 documents how severe the
failure is in a physical setting.

---

### 1.2 Hornik, Stinchcombe & White (1989) — Universal approximation

**Problem.** What class of functions can a feedforward network with one hidden layer represent?

**Method.** Functional analysis, via the Stone–Weierstrass theorem.

**Findings.** Multilayer feedforward networks with an arbitrary squashing activation are
universal approximators: they can approximate any Borel-measurable function to arbitrary
accuracy **on a compact set**, given enough hidden units.

**Relevance.** This is the theoretical licence for using a neural network at all in Part I and
Part II. It is also routinely over-quoted, which matters here.

**Limitation / gap.** The words "on a compact set" are the entire story of Stage 8. The theorem
guarantees approximation on a *bounded region*, says nothing about behaviour outside it, and
says nothing about how many units or how much data are needed. Citing it as "neural networks
can learn anything" is a misreading this report explicitly avoids.

---

### 1.3 Kingma & Ba (2015) — Adam

**Problem.** First-order stochastic optimisation with a single learning rate performs badly
when different parameters have gradients of very different magnitude.

**Method.** Maintain exponential moving averages of the gradient and of its square, bias-correct
both, and step by their ratio.

**Findings.** The resulting step size is approximately invariant to gradient rescaling, making
one learning rate workable across layers.

**Relevance.** Adam is the optimiser used both by scikit-learn's `MLPRegressor` and by the
from-scratch PINN in `src/pinn.py`, where it is implemented explicitly so the mechanism is
visible rather than hidden.

**Limitation / gap.** Adam gives no convergence guarantee for non-convex problems; the PINN's
training curves in Stage 13 are reported precisely so that the optimisation can be audited
rather than assumed.

---

### 1.4 Hairer, Lubich & Wanner (2006) — Geometric Numerical Integration

**Problem.** Which numerical integrators preserve the geometric structure (symplecticity,
reversibility, conserved quantities) of a Hamiltonian system, and why does that matter more
than their formal order of accuracy over long times?

**Method.** Backward error analysis: a symplectic integrator is shown to solve a nearby
*modified* Hamiltonian exactly, rather than the true one approximately.

**Findings.** Symplectic methods (Euler–Cromer, Störmer–Verlet) exhibit **bounded** energy error
over exponentially long times, while non-symplectic methods of higher order (RK4) show a slow
secular drift. The text also establishes that symplectic Euler is *conjugate* to Störmer–Verlet.

**Relevance.** This is the theoretical backing for the entire Stage 10 comparison. It also
explains an initially puzzling measurement in this project: Euler–Cromer measures as a
**second-order** method when its error is sampled at whole orbital periods, even though it is
formally first order — the conjugacy result says its leading O(Δt) error is a periodic change
of coordinates that cancels over a full period. Stage 10 verifies this directly by measuring at
a quarter period (order 1.0) and at a full period (order 2.0).

**Limitation / gap.** It is a numerical-analysis text, not an ML one. No equivalent theory
exists for learned integrators, which is exactly the space Hamiltonian/Lagrangian networks
(Section 4) are trying to fill.

---

# 2. Learning physical laws from data

### 2.1 Schmidt & Lipson (2009) — Distilling free-form natural laws from experimental data

**Problem.** Can a machine derive the *analytic form* of a conservation law from raw
experimental measurements, rather than merely fitting them?

**Method.** Symbolic regression by genetic programming, searching over expression trees. The
critical idea is the fitness function: candidate expressions are scored on how well the
**partial derivatives** they imply match the partial derivatives observed in the data, which
selects for invariants rather than for curve fits.

**Findings.** From motion-tracking data of a double pendulum and other oscillators, the method
recovered Hamiltonians, Lagrangians and momentum conservation laws in closed form.

**Relevance.** This is the strongest existing answer to the central objection facing this
project — "if you trained on data generated from an equation, the model has not discovered
anything". Schmidt and Lipson show what genuine discovery looks like, and the contrast is
instructive: their output is an *equation a physicist can read*, whereas a Random Forest's
output is a set of split thresholds. The report uses this contrast in the discussion of
scientific integrity.

**Limitation / gap.** The search is combinatorially expensive and scales poorly with the number
of variables. It also requires the law to be expressible in a small symbolic vocabulary that
must be chosen in advance.

---

### 2.2 Brunton, Proctor & Kutz (2016) — SINDy

**Problem.** Recover the governing differential equations of a dynamical system from time-series
measurements of its state.

**Method.** Build a large library of candidate nonlinear terms of the state, then solve for a
**sparse** coefficient vector mapping that library to the measured time derivatives, on the
argument that physical laws have few terms.

**Findings.** The method recovers the correct equations for the Lorenz system, fluid vortex
shedding and others, from noisy data, at a small fraction of the cost of symbolic regression.

**Relevance.** SINDy occupies exactly the conceptual position between this project's two
extremes. The "physics-informed features" model in `src/models.py` is a deliberately simplified
version of the same idea: hand-build a basis in which the solution is linear, then fit linear
coefficients. Stage 13 shows that this recovers the coefficient −g/2 = −4.905 to six decimal
places — a miniature, pre-specified SINDy.

**Limitation / gap.** SINDy requires the correct terms to be present in the library and needs
accurate time derivatives, which are difficult to estimate from noisy data. It also assumes the
dynamics are sparse in the chosen basis, which is a strong assumption and not always true.

---

# 3. Physics-informed machine learning

### 3.1 Lagaris, Likas & Fotiadis (1998) — Neural networks for solving ODEs and PDEs

**Problem.** Solve initial- and boundary-value problems with a neural network instead of a mesh.

**Method.** Write the trial solution as a fixed term that satisfies the boundary conditions
*exactly* plus a neural network term multiplied by a function vanishing on the boundary. Train
by minimising the residual of the differential equation at collocation points.

**Findings.** Accurate, closed-form, differentiable solutions; better interpolation between
collocation points than finite differences, with far fewer parameters.

**Relevance.** This is the original physics-informed neural network, twenty years before the
term existed, and it is the direct ancestor of `src/pinn.py`. The one design difference is
instructive: Lagaris et al. impose the boundary conditions as a **hard** constraint built into
the trial function, whereas this project's PINN imposes them as a **soft** penalty term. The
soft version is easier to write and generalises to more complicated conditions; the hard version
cannot violate them at all. Stage 13 reports the initial-condition loss separately precisely
because, in the soft formulation, it *can* be violated.

**Limitation / gap.** The 1998 paper handles one equation at a time, on fixed domains, with
small networks. Scaling to parameterised families of problems — which is what this project's
PINN does, since it takes (v₀, θ, y₀) as inputs — came much later.

---

### 3.2 Raissi, Perdikaris & Karniadakis (2019) — Physics-informed neural networks

**Problem.** Solve forward and inverse problems for nonlinear PDEs in the small-data regime,
where a purely data-driven model has too little to learn from.

**Method.** A neural network u(x,t) trained on a composite loss: a data term on whatever
measurements exist, plus a residual term f = u_t + N[u] evaluated at collocation points, with
derivatives obtained by automatic differentiation. Inverse problems are handled by making the
PDE's unknown coefficients trainable parameters.

**Findings.** Accurate solutions to Burgers', Schrödinger and Navier–Stokes equations from very
few data points; in the inverse setting, PDE parameters recovered to within a fraction of a
percent from scattered, noisy observations.

**Relevance.** This is the reference implementation of the idea Stage 13 reproduces from
scratch. The specific claim this project tests independently is Raissi et al.'s central one —
that the physics term substitutes for data. Stage 13 measures the exchange rate on projectile
motion: a PINN trained on **zero** labelled points reaches a mean radial error that an
identical network without the physics term needs on the order of a hundred labelled points to
match, and in the region of the flight where no labels exist at all, the PINN is several times
more accurate than the plain network.

**Limitation / gap.** Two, both directly relevant here. First, PINNs are much more expensive to
train than to evaluate and are usually *slower* than a classical solver when the equations are
fully known — Raissi et al. position them for problems where data and physics are both partial.
Second, the balance between the loss terms is a hyperparameter with no principled default; this
project addresses it by normalising all three loss terms to be dimensionless and O(1) before
weighting, and says so explicitly in `src/pinn.py`.

---

### 3.3 Karniadakis, Kevrekidis, Lu, Perdikaris, Wang & Yang (2021) — Physics-informed machine learning (review)

**Problem.** Survey how physical knowledge can be embedded into machine learning: through the
loss (soft), the architecture (hard), or the data (augmentation).

**Method.** Review across fluid mechanics, materials, biomedicine.

**Findings.** Three categories of bias — observational, inductive and learning bias — with
different trade-offs. Hard architectural constraints guarantee satisfaction but are difficult
to design; soft penalty constraints are general but only approximate.

**Relevance.** This taxonomy is the organising principle of this project's Stage 13, which
implements one example of each: the physics-feature model is an *inductive* bias (exact by
construction, error at floating-point round-off, and — uniquely among all models tested —
perfect extrapolation), while the PINN is a *learning* bias (approximate, but needs only the
differential equation rather than its solution).

**Limitation / gap.** The review notes that PINN convergence and error bounds remain largely
open theoretically. Nothing in this project contradicts that; the results here are empirical.

---

# 4. Structure-preserving and inductive-bias approaches

### 4.1 Greydanus, Dzamba & Yosinski (2019) — Hamiltonian Neural Networks

**Problem.** A network trained to predict the next state has no reason to conserve energy, and
in practice does not — its rollouts drift.

**Method.** Do not predict the dynamics. Predict a scalar **Hamiltonian** H(q,p), and obtain the
dynamics from it via Hamilton's equations dq/dt = ∂H/∂p, dp/dt = −∂H/∂q, differentiating the
network to get them. Conservation of H is then structural, not learned.

**Findings.** On the two-body problem, a mass-spring system and a pendulum, HNNs train faster,
generalise better, and conserve an energy-like quantity, whereas a baseline network's total
energy drifts. The learned dynamics are exactly time-reversible.

**Relevance.** This is the single most directly relevant paper to Part II of this project, and
it is relevant as a **diagnosis of the exact failure documented here**. Stage 11 measures the
relative energy error of an autoregressive rollout and finds it climbing by many orders of
magnitude while the RK4 reference holds ~10⁻¹¹, purely because nothing in the training
objective ever mentioned energy. Greydanus et al. is the principled fix, and is named in the
report's future-work section for that reason.

**Limitation / gap.** HNNs need the system to be Hamiltonian and need canonical coordinates
(q, p). They also still integrate the learned vector field numerically, so rollout error still
accumulates — conserving energy bounds one failure mode, it does not eliminate error growth.

---

### 4.2 Cranmer, Greydanus, Hoyer, Battaglia, Spergel & Ho (2020) — Lagrangian Neural Networks

**Problem.** HNNs require canonical momenta, which are awkward or impossible to obtain for many
systems.

**Method.** Parameterise an arbitrary **Lagrangian** with a network and obtain the dynamics from
the Euler–Lagrange equations, which requires inverting a Hessian of the network output.

**Findings.** Works in arbitrary generalised coordinates; conserves energy on a double pendulum
and a relativistic particle where HNNs struggle.

**Relevance.** Establishes that the choice of *what the network represents* (a scalar potential
versus a vector field) matters more than its size — the same lesson this project reaches
empirically in Stage 11, where the choice between Approach A, B and C changes the error by more
than the choice of model family does.

**Limitation / gap.** Computationally heavy (a Hessian inverse per evaluation) and, like HNNs,
restricted to systems admitting the relevant variational structure.

---

### 4.3 Chen, Rubanova, Bettencourt & Duvenaud (2018) — Neural Ordinary Differential Equations

**Problem.** Deep networks apply a discrete sequence of transformations; what happens in the
continuous limit?

**Method.** Parameterise the *derivative* of the hidden state with a network and integrate it
with an off-the-shelf adaptive ODE solver, training via the adjoint sensitivity method so that
memory cost is constant in depth.

**Findings.** Constant-memory backpropagation, adaptive accuracy/speed trade-off at evaluation
time, and a natural formulation for irregularly-sampled time series.

**Relevance.** Neural ODEs are the principled version of this project's Approach A. Instead of
learning the map s_t → s_{t+Δt} for one fixed Δt and iterating it — which is what Stage 11 does,
and which locks the model to a single step size — a Neural ODE learns ds/dt and hands the
stepping to a proper adaptive integrator. Stage 12's discussion names this as the main structural
improvement available to the flow-map formulation.

**Limitation / gap.** Training through a solver is expensive, and stiff systems remain hard.
Accuracy of the *learned vector field* still bounds everything downstream — a better integrator
cannot repair a wrong derivative.

---

# 5. Learned simulators and error accumulation

### 5.1 Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec & Battaglia (2020) — Graph Network Simulators

**Problem.** Can a learned model simulate complex particle physics (fluids, sand, deformables)
over long rollouts?

**Method.** Represent the state as a graph of particles, learn message passing to compute
per-particle accelerations, integrate with a fixed semi-implicit Euler step. Crucially, the
model is trained on **single-step** prediction, with small random noise injected into the inputs
during training.

**Findings.** Trained on one-step prediction with thousands of particles, the model generalises
to thousands of timesteps and to an order of magnitude more particles. The authors identify
**error accumulation during rollout** as the central difficulty, and the input-noise
augmentation as the key mitigation: it teaches the model to correct its own mistakes, because
it has seen slightly-wrong inputs during training.

**Relevance.** This is the most important methodological reference for Stage 11. It confirms
that the accumulation phenomenon this project measures is the known, central obstacle in learned
simulation and not a beginner's implementation error. It also supplies the specific,
well-evidenced remedy — noise injection — that this project did **not** apply, which makes the
Stage 11 rollout numbers an honest baseline rather than a state-of-the-art result. The report
says so, and names it as the highest-value next experiment.

**Limitation / gap.** Even with noise injection, rollouts are stable rather than accurate: the
simulations look physically plausible but do not track the ground truth trajectory point by
point. For a two-body orbit, where pointwise accuracy is the whole question, "plausible" is not
enough.

---

### 5.2 Lam et al. (2023) — GraphCast

**Problem.** Medium-range global weather forecasting, the largest-scale real test of learned
simulation.

**Method.** A graph neural network trained on ~40 years of ERA5 reanalysis, predicting the
global atmospheric state 6 hours ahead and rolled out autoregressively to 10 days; trained with
multi-step rollout in the loss rather than single-step only.

**Findings.** Outperforms the operational deterministic HRES system on ~90% of 1,380 verification
targets, producing a 10-day forecast in under a minute on a single TPU against roughly an hour
on a supercomputer.

**Relevance.** This is the counter-example that stops this project's conclusions from being
over-general, and it is cited for exactly that purpose. Stage 11 finds that autoregressive
rollout degrades badly; GraphCast shows the same formulation succeeding at enormous scale. The
differences are instructive and are set out in the discussion: (i) GraphCast trains on
multi-step rollouts, so accumulated error is inside the objective; (ii) the atmosphere has no
cheap exact solver to compete against, whereas a two-body orbit does; (iii) the useful forecast
horizon is a few hundred steps, not thousands of orbits.

**Limitation / gap.** GraphCast's skill degrades at longer lead times like every forecast system,
and it is trained on reanalysis rather than raw observations. It also does not conserve mass or
energy, which remains an open criticism of learned weather models.

---

# 6. Machine learning in orbital and gravitational dynamics

### 6.1 Breen, Foley, Boekholt & Portegies Zwart (2020) — Newton versus the machine

**Problem.** The three-body problem is chaotic and has no general closed-form solution;
converged numerical solutions require arbitrary-precision arithmetic and are extremely
expensive. Can a network replace them?

**Method.** Generate a training set of three-body solutions with the Brutus arbitrary-precision
integrator, then train a deep feedforward network to map an initial configuration plus a time
directly to the positions at that time — a **direct map**, not a stepped simulation.

**Findings.** The network reproduced the trajectories accurately over the trained time interval
and did so up to ~10⁵ times faster than the numerical integrator. The authors are explicit that
it works over a restricted time window and is a surrogate, not a replacement.

**Relevance.** The most directly comparable published study to Part II of this project, and the
comparison is a close one. Their formulation is precisely this project's **Approach B**
(initial conditions + time → position), and this project independently finds Approach B's
characteristic signature: because there is no feedback loop, its error does **not** grow with
time — it is as accurate at t = 4 yr as at t = 0.1 yr — but it collapses the moment it is asked
for a time beyond its training window, which is the same extrapolation failure documented in
Stage 8. Their headline speed-up also motivates Stage 12's computational-cost comparison, which
finds the same qualitative result (learned inference is much cheaper per query than stepping the
integrator) while noting that the two-body case is not expensive enough for the trade to be
worth making.

**Limitation / gap.** Trained on equal-mass, zero-velocity initial conditions in a plane, over
a fixed time window; no claim of extrapolation beyond it. The paper also does not report whether
the network's output conserves energy — a question Stage 11 of this project measures directly
for its own models, and answers in the negative.

---

### 6.2 Rein & Liu (2012) — REBOUND

**Problem.** Provide an open, well-tested N-body integrator for collisional dynamics and
planetary systems.

**Method.** A modular C code with multiple integrators (leapfrog, WHFast, IAS15) and collision
detection, with a Python interface.

**Relevance.** Cited as the professional standard that this project's ~200-line NumPy
integrator (`src/planetary_physics.py`) is a teaching-scale version of. The report is explicit
that the hand-written simulator is validated against the analytic Kepler solution rather than
claimed to be production software — but the validation is quantitative: it reproduces the
analytic solution to ~10⁻¹³ AU, recovers the theoretical convergence orders 1, 1, 2 and 4 for
the four integrators, and reproduces Kepler's third law with a fitted log–log slope of 1.500000
against a theoretical 1.5.

**Limitation / gap.** REBOUND solves a far more general problem than this project needs. The
relevance is methodological: it is the reason the report claims "validated educational
implementation" rather than "new integrator".

---

# 7. Synthesis: what is established, what is contested

**Established by the literature, and reproduced in this project:**

| Claim | Source | Where this project reproduces it |
|---|---|---|
| Networks approximate well inside the training region | Hornik et al. (1989) | Stage 5: R² = 0.99 for the Random Forest; Stage 8 interpolation control |
| ...and give no guarantee outside it | Breiman (2001); implicit in Hornik | Stage 8: up to ~290× degradation for the MLP |
| Physics in the loss substitutes for data | Lagaris et al. (1998); Raissi et al. (2019) | Stage 13: a PINN with zero labels beats a plain network with many |
| Learned rollouts accumulate error | Sanchez-Gonzalez et al. (2020) | Stage 11: ~10⁻⁵ AU per step becomes ~1 AU within one orbit |
| Learned models do not conserve energy unless made to | Greydanus et al. (2019) | Stage 11: relative energy error climbs many orders of magnitude |
| Integrator structure beats integrator order for long runs | Hairer et al. (2006) | Stage 10: Euler–Cromer bounded, RK4 secular drift |
| Direct maps are fast and do not accumulate error | Breen et al. (2020) | Stage 11 Approach B: flat error curve, ~10⁵× faster than stepping |

**Contested or unresolved in the literature:**

1. **How to weight the terms in a PINN loss.** No principled method; Karniadakis et al. (2021)
   flag it as open. This project normalises each term to be dimensionless and O(1), which is a
   defensible convention, not a solution.
2. **Whether learned simulators can be made stable over very long horizons.** GraphCast succeeds
   at ~40 steps; Sanchez-Gonzalez et al. at thousands, but with "plausible" rather than accurate
   trajectories. Nobody claims indefinite accuracy.
3. **Whether a learned model can be said to have "discovered" a law.** Schmidt & Lipson (2009)
   and Brunton et al. (2016) produce readable equations and have a genuine claim. A Random
   Forest does not. This project sides firmly with the narrow reading, and Stage 8 provides its
   own evidence for doing so: a model that had induced x = v₀cos(θ)t would not fail at
   v₀ = 45 m/s, and every model that was not handed the physics did fail there.

---

# 8. The research gap this project occupies

The literature above is dominated by two kinds of study: **large-scale demonstrations** that a
learned model can handle a system too hard to simulate cheaply (GraphCast, GNS, Breen et al.),
and **methodological proposals** for building physics into models (PINNs, HNNs, LNNs, Neural
ODEs). Both kinds are conducted at the research frontier, on systems where the ground truth is
expensive, partial, or unavailable.

Very little of it does the opposite: take two systems whose exact answers are **completely
known**, hold the methodology fixed, and measure precisely where the learned approach stops
working and why. That is a less glamorous question, but it is the one a beginner must be able
to answer before any of the above can be read critically, and it is a legitimate piece of
comparative experimental work in its own right.

**The gap this study occupies** is therefore:

> A controlled, like-for-like comparison of the same machine-learning pipeline applied to two
> physical systems of different dynamical character — one with a closed-form solution
> (projectile motion) and one without (two-body orbital motion) — with every result measured
> against an exactly-known ground truth, and with the failure modes (extrapolation, error
> accumulation, violation of conservation laws) quantified rather than avoided.

Five specific contributions follow from framing it this way, each of which is an experiment
rather than a claim:

1. **An interpolation/extrapolation decomposition with a mechanism.** Stage 8 does not merely
   report that extrapolation is worse; it plots a one-dimensional slice through input space
   along which the true response is exactly linear, so each model's deviation *is* its inductive
   bias made visible — the Random Forest's prediction is literally flat beyond the last training
   split.
2. **An error-accumulation budget.** Stage 11 relates one-step accuracy to usable horizon and
   shows the exchange rate is logarithmic: reducing per-step error by an order of magnitude
   buys an additive, not multiplicative, extension of the forecast.
3. **A separation of model error from physical sensitivity.** Because the two-body problem is
   integrable, perturbed initial conditions diverge only *linearly*. Stage 11 measures this
   directly and thereby rules out the convenient excuse that the ML rollout diverges "because
   the dynamics are chaotic" — they are not.
4. **Conservation laws used as an evaluation metric.** Energy and angular-momentum drift are
   reported for every learned rollout alongside position error, following the diagnosis in
   Greydanus et al. (2019) but applied as a measurement rather than as a fix.
5. **A three-way comparison of physics integration.** Stage 13 implements a hard-constraint
   model, a soft-constraint PINN (hand-written in NumPy, with every gradient verified against
   finite differences) and an unconstrained network, under identical conditions.

**What this project explicitly does not claim.** It does not claim that machine learning
outperforms classical physics on these systems — it does not, by roughly ten orders of
magnitude, and the report says so in those words. It does not claim any model discovered a
physical law. And it does not claim its Stage 11 rollout numbers are state of the art: the
noise-injection technique from Sanchez-Gonzalez et al. (2020) and the structural approaches of
Greydanus et al. (2019) and Chen et al. (2018) are all known improvements that were deliberately
left for future work so that the baseline phenomenon could be measured cleanly.

---

# 9. Full reference list

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

## A note on how these sources were used

Sources 1–18 were checked against publisher records for author list, year, journal, volume and
page range before being included. Where this document summarises a paper's findings, the summary
is of its abstract and headline results, not of a full replication — with the exception of Raissi
et al. (2019), Greydanus et al. (2019), Sanchez-Gonzalez et al. (2020) and Breen et al. (2020),
whose central phenomena are independently reproduced at small scale in Stages 11 and 13 of this
project and can be compared directly against the numbers in `results/`.
