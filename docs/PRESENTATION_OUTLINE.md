# Presentation Outline

**Machine Learning-Based Prediction of Physical Motion: A Comparative Study of Projectile and Planetary Dynamics**

Aryan Bola — Research Methodology, 5th Semester B.Sc. Physics

---

## Design notes before you build the slides

**Target: 18 slides, 15 minutes, ~50 seconds per slide.** Two slides will take two minutes each
(slides 7 and 12) and several will take twenty seconds. Budget accordingly.

**One rule for every slide: one claim, one figure, one number.** If a slide has two claims, it is
two slides. If a figure needs more than one sentence of explanation, it is the wrong figure.

**Do not put code on a slide.** Nobody reads it. If asked, open the file.

**Every figure already exists** in `results/`, at 150 dpi. The filename is given on each slide
below. Do not redraw anything by hand — if a figure needs changing, change the script and re-run,
so the figure and the numbers can never drift apart.

**The story arc.** The presentation is built around one reversal. Slides 1–7 build the impression
that machine learning works well. Slide 8 breaks it. Slides 9–15 explain why, and show what fixes
it. Do not spoil slide 8 early — the audience must believe the R² = 0.99 first, or the reversal
carries no weight.

---

# Section 1 — Framing (slides 1–3, ~2 min)

## Slide 1 — Title

- Title, name, course, date.
- One line underneath: *"Can a machine learn physics we already know — and what happens when we
  check carefully?"*

**Say:** Nothing beyond the title. Do not start explaining yet.

---

## Slide 2 — The question a physicist should ask first

Large, centred, one line:

> **"Why use AI to predict something the equations already give us exactly?"**

Underneath, three bullets:
- We are not trying to beat physics. Physics wins, and the report says by how much.
- We are studying **the algorithm**, using a system where we know the exact answer.
- Known ground truth is not a weakness of the design — it is the whole point of it.

**Say (this is the most important 30 seconds of the talk):** "If I can't answer this, nothing else
matters. So: this is a benchmark, in the same sense that you test a new integrator on a problem
whose answer you already know — precisely so that any discrepancy is attributable. The object of
study is the learning algorithm, not the projectile."

**Anticipate:** an examiner may interrupt here. Good — it means they are engaged, and you have the
answer ready.

---

## Slide 3 — What was built

A simple diagram, four boxes left to right:

```
 Classical physics  →  Synthetic dataset  →  ML models  →  Compare against truth
  (exact solution)      (documented         (5 families)    (error, generalisation,
                         sampling design)                    conservation laws)
```

Below, the two systems side by side:

| | Projectile | Planetary |
|---|---|---|
| Governing law | ÿ = −g | **a** = −(GM/r³)**r** |
| Solution | Elementary closed form | Analytic Kepler parametrisation with numerical root solve |
| Ground truth | Exact algebra | RK4, validated to 10⁻¹³ AU |

**Say:** "These two systems have independently checkable answers. Their state updates differ
structurally: the projectile update is affine, while the orbital update is nonlinear."

---

# Section 2 — Part I: projectile motion (slides 4–8, ~4 min)

## Slide 4 — The physics baseline

`results/trajectory_comparison.png` — or a clean plot of a single parabola.

- x = v₀cos(θ)t, y = y₀ + v₀sin(θ)t − ½gt²
- Assumptions stated out loud: constant g, no air resistance, flat ground, point mass.

**Say:** "Everything downstream inherits these four assumptions. A model trained on this data has
learned to reproduce a simulation, not nature."

---

## Slide 5 — The dataset, and the one decision that mattered

A five-row table of the actual CSV.

| v₀ | θ | y₀ | t | x | y |
|---|---|---|---|---|---|

- **Features** = what you could know beforehand, plus the time asked about.
- **Targets** = the position.
- **The decision:** t is drawn within *each sample's own* flight time.

**Say:** "If I'd drawn t from a fixed global range, a slow low launch would be sampled mostly
after it had already landed — rows describing a projectile continuing underground. Sampling as a
fraction of each flight keeps every row physical."

**This slide signals methodological care. Do not skip it.**

---

## Slide 6 — First results

`results/actual_vs_predicted.png`

| Model | RMSE x | R² x | RMSE y | R² y |
|---|---|---|---|---|
| Linear Regression | 20.09 m | 0.745 | 14.09 m | 0.584 |
| **Random Forest** | **3.47 m** | **0.9924** | **4.95 m** | **0.9487** |

**Say:** "R² of 0.99. Looks like a success." — *and let it sit for a beat.*

---

## Slide 7 — Why linear regression fails (≈2 min, take your time)

Side by side: the trajectory comparison figure, and the equation

> x = **v₀ · cos(θ) · t**

- A linear model computes w₁v₀ + w₂θ + w₃y₀ + w₄t + b — a **weighted sum**.
- The true answer is a **product of three inputs**, and contains cos(θ) and t².
- No choice of weights can produce a product. The failure is structural, not a tuning problem.

**Say:** "This is the most useful negative result in Part I. It tells us *what kind* of function
we are trying to learn, and that is what justifies everything that follows."

---

## Slide 8 — **The turn**: R² = 0.99 was measured on the wrong thing

`results/stage8_response_slice.png` — the money figure.

- Models trained on v₀ ∈ [15, 40] m/s. Test at v₀ up to 80.
- The true response along this slice is a **straight line**, so every deviation is the model's own
  inductive bias, made visible.
- The Random Forest's prediction goes **exactly flat** beyond the training range.

| Test region | Linear | Poly2 | Forest | MLP | Physics features |
|---|---|---|---|---|---|
| Same region as training | 11.2 m | 1.97 m | 2.14 m | **0.11 m** | 5×10⁻¹⁴ m |
| All variables outside | 66.2 m | 34.0 m | 55.6 m | **32.7 m** | 1×10⁻¹³ m |
| **Degradation** | 5.9× | 17× | 26× | **286×** | none |

**Say:** "The best model in-distribution is the worst-degrading out of it. And R² = 0.99 was
measured on test data drawn from the *same distribution* as training. That measures interpolation.
A physical law is a claim about everywhere."

**Pause here.** This is the pivot of the whole talk.

---

# Section 3 — Part II: orbital motion (slides 9–13, ~5 min)

## Slide 9 — A harder system, and why the integrator matters first

`results/stage10_integrator_orbits.png`

- Same orbit, same step size, four integrators, 40 orbits.
- Explicit Euler ends at **7.6×** its starting radius. It is adding energy every step.
- Verlet and RK4 trace the exact ellipse.

**Say:** "Before I can say a model has an error of X, I have to prove my reference doesn't."

---

## Slide 10 — Validating the ground truth

`results/stage10_convergence.png`

- Measured convergence orders: **0.97, 0.99, 1.99, 3.92** vs theory 1, 1, 2, 4.
- Kepler's third law recovered with fitted log–log slope **1.500000** — from a simulator that was
  only told Newton's inverse-square law.
- Reference error over the whole 5-year horizon: **3 × 10⁻¹¹ AU**.

**Say:** "The smallest ML error I measure anywhere is 6 × 10⁻⁵ AU — six orders of magnitude
larger. So the reference is exact for my purposes, and I can say that quantitatively."

*(If time is short, this slide can be cut to a single line on slide 9. It is insurance against the
question "how do you know your ground truth is right?" — so know it even if you don't show it.)*

---

## Slide 11 — Three ways to pose the problem

```
A.  s_t → s_{t+Δt}        step, then feed the output back in     ← the general one
B.  (a, e, t) → (x, y)    ask for a time, get a position         ← needs an integrable system
C.  [s_{t-2}, s_{t-1}, s_t] → s_{t+Δt}    a window of history
```

- A is the only one that would work for a system with no analytic solution.
- B has no feedback loop — and therefore **no error accumulation at all**.

**Say:** "The formulation turns out to matter more than the model. That's the finding of this
section."

---

## Slide 12 — **The central result** (≈2 min)

`results/stage11_error_growth.png`

Two numbers, side by side, in large type:

> **One step: 6.4 × 10⁻⁵ AU**
> **After one-third of an orbit: 0.01 AU**

- Every prediction becomes the next input, so errors **compound**.
- A model that is excellent at one step is untrustworthy after ~90 of them.
- RK4 holds 10⁻¹¹ AU across the entire five years (dotted line at the bottom).

**Say:** "This is the gap between 'fits the data' and 'can simulate'. And it's not a small gap —
it's ten orders of magnitude."

---

## Slide 13 — And it isn't chaos

`results/stage11_conservation_ml.png` (top panel) or `results/stage11_sensitivity.png`

- Two-body motion is **integrable**, not chaotic. Perturbed initial conditions separate
  *linearly*, and I measured it.
- So the divergence cannot be blamed on the physics. It is compounding model error.
- Separately: the learned rollout's energy drifts by orders of magnitude, because **nothing in
  the loss function ever mentioned energy**.

**Say:** "I want to be careful not to let myself off the hook here. The convenient explanation
would be 'chaotic dynamics'. I checked, and it isn't — this system has no chaos to blame."

---

# Section 4 — Comparison and the fix (slides 14–16, ~3 min)

## Slide 14 — The controlled comparison

`results/stage12_flow_map_comparison.png`

- Same formulation, same models, same amount of data. **Only the physics changes.**
- The exact projectile flow map is **affine**:
  x′ = x + vₓΔt, y′ = y + v_yΔt − ½gΔt², vₓ′ = vₓ, v_y′ = v_y − gΔt
- So plain linear regression recovers it to **10⁻¹⁴ m** — it lies exactly inside the hypothesis
  space of the simplest model tested.
- Orbital acceleration contains 1/r³; the tested fitted maps remain approximate.
- Qualification: Stage 12 does not match steps per natural timescale, and its one-step
  scores use training pairs. The separate rollout trajectories provide the prediction test.

**Say:** "That's the answer to the comparative question, and it's structural rather than vague.
It's not that orbits are 'more complicated' — it's that one flow map is inside the model class and
the other isn't."

---

## Slide 15 — What actually helps: put the physics in the model

`results/stage13_extrapolation.png` — the second money figure.

- Both networks: identical architecture, identical training. Labels only from the **first 40%** of
  each flight.
- The PINN also gets the equation d²y/dt² = −g, at points where there is **no data**.
- Red (data only) flies off the parabola. Brown (data + physics) tracks it.
- Beyond the labelled window, the PINN is **14.7× more accurate**.

**Say:** "A physical law is a statement about everywhere. That's why it can constrain a model in
regions where no measurement exists — and that is the practical content of the phrase
'physics-informed'."

---

## Slide 16 — And it is dramatically more data-efficient

`results/stage13_data_efficiency.png`

- A PINN trained on **zero labelled points** reaches 1.39 m.
- The plain network needs **~200 labelled points** to match that.
- With 2,000 labels the two are equivalent — **physics-informed learning is a low-data technique,
  and I'm not going to oversell it.**

**Say:** "The honest version of this result includes the right-hand side of the graph, where the
advantage disappears."

---

# Section 5 — Closing (slides 17–18, ~2 min)

## Slide 17 — Conclusions

**Answering the research question:**

1. **No** — machine learning is not equally effective for the two systems, and the reason is
   structural: the projectile flow map is affine and lies inside the model class; the orbital one
   does not.
2. Standard models **interpolate** well and **extrapolate** badly, with a failure mode determined
   by each model's inductive bias.
3. In autoregressive prediction, **error accumulation**, not per-step accuracy, sets the usable
   horizon.
4. Embedding physical law improves low-data accuracy and the tested extrapolation; the soft
   constraint does not guarantee exact motion or improve every high-data result.

**What this study does not claim** (leave this on screen — it earns credibility):

- ML did **not** beat classical physics here. It lost by ~10 orders of magnitude.
- **No model discovered a law.** The physics-basis fit is exact here; the PINN is approximate and receives the equation.
- These are simulated, noise-free, non-chaotic systems.
- The Stage 11 rollout numbers are a **baseline**, not state of the art — known improvements
  (noise injection, multi-step losses, Hamiltonian networks) were deliberately left out so the
  baseline phenomenon could be measured cleanly.

---

## Slide 18 — Limitations and future work

| Limitation | The experiment that would address it |
|---|---|
| Noise-free synthetic data | Repeat with Gaussian noise at several levels |
| Neither system is chaotic | Extend to the three-body problem (cf. Breen et al. 2020) |
| Rollout baseline is unoptimised | Add noise injection during training (Sanchez-Gonzalez et al. 2020) |
| Nothing enforces conservation | Hamiltonian Neural Networks (Greydanus et al. 2019) |
| Flow map locked to one Δt | Neural ODEs (Chen et al. 2018) |

**Closing line:** *"The useful result of this project isn't that machine learning worked. It's
that I can say exactly where it stopped working, and why."*

---

# Appendix slides (hold in reserve, do not present)

Have these ready to jump to if questioned. Know the slide numbers.

| # | Content | Answers the question |
|---|---|---|
| A1 | `stage8_regime_bars.png` | "How bad is extrapolation, exactly?" |
| A2 | `stage10_conservation.png` | "Why Verlet rather than RK4?" |
| A3 | `stage11_approach_comparison.png` | "Why not just use the direct map?" |
| A4 | `stage11_learning_curve.png` | "Wouldn't more data fix it?" |
| A5 | `stage13_physics_residual.png` | "Is the output physically possible?" |
| A6 | `stage7_learning_curve.png` | "How much data did you need?" |
| A7 | `stage12_cost.png` | "Isn't ML supposed to be faster?" |
| A8 | Euler–Cromer conjugacy | "Why does a 1st-order method measure as 2nd order?" |

---

# Delivery checklist

**The evening before**
- [ ] Run `python run_all.py --list` and be able to say roughly how long the study takes to
      reproduce (~49 minutes).
- [ ] Re-read `docs/VIVA_PREP.md` start to finish.
- [ ] Know these six numbers cold: **0.9924** (R² forest), **286×** (MLP degradation),
      **6.4 × 10⁻⁵ AU** (best one-step), **0.37 yr** (usable horizon at 1%), **1.500000**
      (Kepler slope), **14.7×** (PINN advantage beyond the data).
- [ ] Open `results/` in a file browser so any figure is two clicks away.

**On the day**
- [ ] Slide 2 is the make-or-break slide. Rehearse it until it's automatic.
- [ ] Slide 8 is the pivot. Pause after it.
- [ ] When you don't know something, say "I didn't test that" — and then say what experiment
      would settle it. That answer is worth more than a guess, every time.
- [ ] Never claim the model "learned physics". Say "learned a function consistent with the
      physics" and be ready to explain the difference.
