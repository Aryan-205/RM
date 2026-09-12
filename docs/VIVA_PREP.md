# Viva Preparation

**Machine Learning-Based Prediction of Physical Motion: A Comparative Study of Projectile and Planetary Dynamics**

Aryan Bola — Research Methodology, 5th Semester B.Sc. Physics

---

## How to use this document

Every answer below is written as *you* would say it out loud, not as a textbook paragraph. Read it
once end to end, then practise saying the **bolded** first sentence of each answer from memory —
that sentence is the answer, and everything after it is supporting detail you deploy only if the
examiner wants more.

Three rules that will serve you better than any memorised answer:

1. **Answer the question asked, then stop.** Volunteering extra detail invites follow-ups on
   ground you haven't prepared.
2. **"I didn't test that" is a strong answer** when followed by "— the experiment that would
   settle it is…". It shows you know where your evidence ends. Guessing shows the opposite.
3. **Never say the model "learned physics".** Say "learned a function consistent with the
   physics", and be ready to explain the difference. This distinction is the intellectual core of
   the whole project.

---

## Contents

- [Part 1 — The numbers you must know cold](#part-1--the-numbers-you-must-know-cold)
- [Part 2 — The question that decides the viva](#part-2--the-question-that-decides-the-viva)
- [Part 3 — Machine learning fundamentals](#part-3--machine-learning-fundamentals)
- [Part 4 — Physics questions](#part-4--physics-questions)
- [Part 5 — Methodology and research design](#part-5--methodology-and-research-design)
- [Part 6 — Results and interpretation](#part-6--results-and-interpretation)
- [Part 7 — Hostile questions](#part-7--hostile-questions)
- [Part 8 — Questions about your own code](#part-8--questions-about-your-own-code)
- [Part 9 — If you get stuck](#part-9--if-you-get-stuck)

---

# Part 1 — The numbers you must know cold

If you remember nothing else, remember these. An examiner testing whether you actually did the
work will probe for a number.

| Quantity | Value | Where from |
|---|---|---|
| Random Forest R² (x, y), projectile | 0.9924 / 0.9487 | Stage 5 |
| Random Forest RMSE (x, y) | 3.47 m / 4.95 m | Stage 5 |
| Linear Regression RMSE x | 20.09 m | Stage 5 |
| MLP degradation, interpolation → full extrapolation | **286×** (0.11 m → 32.7 m) | Stage 8 |
| Physics-feature model error, everywhere | ~10⁻¹³ m (machine precision) | Stage 8, 13 |
| Measured integrator orders (quarter orbit) | 0.97, 0.99, 1.99, 3.92 | Stage 10 |
| Kepler's third law, fitted log–log slope | **1.500000** (theory 1.5) | Stage 10 |
| RK4 reference error over 5 years | ~3 × 10⁻¹¹ AU | Stage 10 |
| Best one-step orbital accuracy (MLP, delta) | **6.4 × 10⁻⁵ AU** | Stage 11 |
| ...its usable horizon at 1% of 1 AU | **0.37 years** (~1/3 orbit) | Stage 11 |
| Delta vs absolute targets, error reduction | ~28–29× (forest, MLP); **exactly 1×** (linear) | Stage 11 |
| Projectile flow map, linear regression error | ~3 × 10⁻¹⁴ m | Stage 12 |
| PINN with **zero** labels | 1.39 m | Stage 13 |
| Plain NN needed to match that | ~200 labels | Stage 13 |
| PINN advantage beyond the labelled window | **14.7×** | Stage 13 |
| Random Forest physics residual | ~2,000 m/s² (vs g = 9.81) | Stage 13 |

---

# Part 2 — The question that decides the viva

### ❓ "Why use AI to predict something physics can already calculate exactly?"

**This will be asked. Possibly first. Your answer to it sets the tone for everything else.**

> **"Because the object of study is the algorithm, not the projectile. I'm not trying to beat
> physics — physics wins here by about ten orders of magnitude, and my report says so in those
> words. I chose systems with exactly-known answers precisely so that every error I measure is
> attributable to the model and to nothing else. It's the same logic as testing a new numerical
> integrator on a problem you can solve analytically: you use the known answer as a ruler."**

If they push — *"but then what's the point?"*:

> "Three things. First, a benchmark: I can state exactly where a standard ML pipeline stops
> working and why, which you cannot do on a problem where you don't know the truth. Second, the
> failure modes I measured — extrapolation collapse, error accumulation, violation of conservation
> laws — are the same failure modes reported in the published literature on problems where ML
> *is* genuinely needed. Third, the situations where ML actually earns its place are ones where
> you have the differential equation but the solve is expensive, or you have data but no equation.
> Stage 13 tests the first of those directly: a network given only the ODE and no labelled data at
> all reconstructs the trajectory to about a metre."

**Do not** say "to learn machine learning" or "because it's interesting". Those are true and they
will not survive a follow-up.

---

# Part 3 — Machine learning fundamentals

### ❓ What is machine learning, in one sentence?

> **"Fitting a function with adjustable parameters to examples, by minimising a measure of how
> wrong it is."** Traditional programming is: rules + input → output. Machine learning is:
> input + output → rules. In my project, "rules" means the parameters of a model that maps
> [v₀, θ, y₀, t] to [x, y].

### ❓ What is supervised learning? Regression vs classification?

> **Supervised means every training example comes with the right answer attached.** Regression
> predicts a continuous number; classification predicts a category. This is regression — positions
> are real numbers on a continuum.

### ❓ What exactly does `model.fit()` do?

**Know the answer for each model separately — a generic answer here is a red flag.**

> **"It's different for each model, and that difference is the whole story of Stage 8."**
>
> - **Linear regression**: solves the normal equations (least squares) in closed form — there's no
>   iteration at all. It finds the weights minimising Σ(y − ŷ)².
> - **Random forest**: grows 100 decision trees. Each tree gets a bootstrap resample of the rows
>   and, at each split, a random subset of the features; it picks the split that most reduces
>   variance. Prediction is the average over the trees.
> - **Neural network**: repeatedly computes the loss, backpropagates the gradient, and takes an
>   Adam step. Thousands of iterations.

### ❓ What does `model.predict()` do?

> Runs the fitted function forward. For linear regression it's one dot product. For the forest,
> each tree routes the input down to a leaf and returns that leaf's stored average; the forest
> averages those. **That last detail matters:** a forest can only ever output an average of
> training targets, which is precisely why it cannot extrapolate.

### ❓ Why can't we evaluate on the training data?

> **"Because a model that memorised the training set would score perfectly and be useless."** A
> Random Forest grown to full depth can get near-zero training error by construction. Test error
> on data it has never seen is the only measure of whether it learned the *pattern* rather than
> the *points*. My project pushes this one step further: even test error isn't enough, because my
> test data came from the same distribution as training. That measures interpolation. Stage 8
> exists to measure the other thing.

### ❓ Overfitting vs underfitting — and which did you see?

> **Underfitting** = the model is too simple to represent the pattern; training and test error are
> both high. **Overfitting** = it fit noise; training error is low, test error is high.
>
> **"I saw textbook underfitting, and essentially no overfitting."** Linear regression on
> projectile motion is underfitting: it cannot represent v₀cos(θ)t no matter how much data you
> give it, so it fails on training and test alike. I saw no real overfitting because my data is
> noise-free — there is no noise to fit. That's a direct consequence of synthetic data, and I say
> so in the limitations.

### ❓ MAE, MSE, RMSE, R² — define them, and tell me what RMSE = 3.47 m means here.

> MAE is the mean absolute error — the typical error. MSE is the mean squared error, which
> punishes large errors much harder. RMSE is its square root, back in metres but still dominated
> by the worst cases. R² is 1 − SS_res/SS_tot: the fraction of the variance explained.
>
> **"RMSE = 3.47 m means that on trajectories up to about 200 m long, the model is typically
> wrong by around three and a half metres — a couple of percent."** For a video game that's
> invisible; for artillery it's a miss. The number only means something once you name the
> tolerance you care about, which is why Stage 11 reports a *usable horizon* at a stated tolerance
> rather than a bare error.

### ❓ What does a negative R² mean? You have several.

> **"It means the model is worse than a constant predictor — worse than just always guessing the
> mean."** R² = 1 − SS_res/SS_tot, and nothing stops SS_res exceeding SS_tot. Every negative R² in
> my results is in an extrapolation regime in Stage 8, and it's the metric working correctly: it
> is telling me the model is producing actively misleading output, not merely inaccurate output.

### ❓ Why did you not use deep learning from the start?

> **"Because starting simple produced the two most useful results in the project."** Linear
> regression's *failure* in Part I told me the mapping contains a product of three inputs — that's
> what motivated everything after. And in Stage 12, linear regression isn't just adequate for the
> projectile flow map, it's **exact**, at 10⁻¹⁴ m, because that flow map is affine. A project that
> opened with a neural network would have missed both.

---

# Part 4 — Physics questions

### ❓ Derive the projectile equations.

> Newton's second law with the only force being weight: **F** = −mg ĵ, so **a** = −g ĵ, independent
> of mass. Then aₓ = 0 and a_y = −g. Integrate once: vₓ = v₀cos θ (constant), v_y = v₀sin θ − gt.
> Integrate again: x = v₀cos(θ)t, y = y₀ + v₀sin(θ)t − ½gt².
>
> The mass cancelling is Galileo's result and it's worth saying out loud: it's why the trajectory
> doesn't depend on what you throw.

### ❓ What assumptions are built into that?

> **Four, and every conclusion in Part I inherits them.** Constant g (true to about 0.03% over
> 1 km, so fine here). No air resistance — the big one; a real projectile at 50 m/s experiences
> drag comparable to its weight, and the true path is noticeably non-parabolic. Flat ground and a
> non-rotating Earth (no Coriolis). Point mass (no spin, no Magnus effect).

### ❓ Why does the two-body problem need numerical integration?

> **"Because the acceleration depends on the position, which is what you're solving for."** The
> orbit *shape* is solvable — it's a conic section, that's Kepler's first law. But position as a
> function of *time* requires solving Kepler's equation E − e sin E = M, which is transcendental
> and has no algebraic solution. I solve it by Newton–Raphson in `kepler_solution()` and use that
> as my analytic reference.

### ❓ Why is Euler bad for orbits? Verlet is only second order — why is it better than RK4 over long times?

> **"Because for oscillatory motion Euler is systematically biased: it adds energy every step."**
> It evaluates everything at the start of the interval, so on a curving path it consistently
> overshoots outward. In my Stage 10 run it ends at 7.6× its starting radius after 40 orbits.
> Halving Δt halves the rate of the spiral but never removes it.
>
> On Verlet vs RK4: **"the relevant property isn't order, it's symplecticity."** Verlet is
> symplectic — by backward error analysis it solves a nearby *modified* Hamiltonian exactly, so
> its energy error is bounded and oscillates rather than drifting. RK4 is fourth order and far more
> accurate per step, but it isn't symplectic, so over very many orbits its energy drifts secularly.
> My data shows exactly that: over 200 orbits Verlet's final energy error is 2.5 × 10⁻¹² while
> RK4's is 3.3 × 10⁻⁸ — even though RK4 is thousands of times more accurate on a single orbit.
> I used RK4 anyway, because I only needed a few orbits and wanted maximum per-step accuracy.

### ❓ **[Hard]** Your Euler–Cromer measured as second order, but it's a first-order method. Explain.

**This is the best question you could be asked. You investigated it, so answer with relish.**

> **"That's real, and it took me a while to be sure it wasn't a bug."** The symplectic Euler method
> is *conjugate* to Störmer–Verlet — the two produce the same trajectory up to a fixed O(Δt) change
> of coordinates. That coordinate shift is periodic with the orbit, so if you sample the error
> after exactly one full period it cancels, and what's left is the underlying second-order
> behaviour.
>
> I tested it directly: I measured the convergence order at a quarter of an orbit as well as at a
> full orbit. At a quarter orbit I get 0.994 — first order, as theory says. At a full orbit I get
> 1.998. Both numbers are in `results/stage10_convergence.csv` and both panels are in the figure.
> It's a result in Hairer, Lubich and Wanner's *Geometric Numerical Integration*.

### ❓ Why AU and years instead of SI?

> **"Because in those units GM = 4π² exactly, and every quantity in the simulation is of order 1."**
> In SI I'd be squaring numbers around 10¹¹ inside every energy calculation and throwing away
> precision for nothing. It also makes errors instantly interpretable — "0.01 AU" is immediately
> 1% of the orbit.

### ❓ Is the two-body problem chaotic?

> **"No, and that's an important finding rather than a technicality."** It's integrable — as many
> conserved quantities as degrees of freedom. Nearby trajectories separate at worst *linearly*,
> driven by slightly different orbital periods, never exponentially. I measured this in Stages 11
> and 12 by perturbing exact initial conditions.
>
> It matters because it **rules out the convenient excuse**. If the ML rollout diverged and the
> system were chaotic, I could say "well, chaos". There's no chaos here to blame, so the divergence
> is entirely compounding model error.

### ❓ Did the model discover Newton's laws?

> **"No, and I'd resist that framing quite firmly."** I generated the training data *from* Newton's
> laws, so the information was in the data by construction. The two models that reproduce the
> physics exactly are the two I *handed* it to — the physics-feature model and the PINN — which is
> the opposite of discovery.
>
> Stage 8 is the evidence. A model that had genuinely induced x = v₀cos(θ)t would work at
> v₀ = 45 m/s. Every model that wasn't given the physics failed there. That's a model that learned
> the training region, not the law.
>
> If you want to see what discovery actually looks like, it's Schmidt and Lipson's 2009 *Science*
> paper, where symbolic regression recovers Hamiltonians in closed form from motion-tracking data —
> the output is an equation a physicist can read.

---

# Part 5 — Methodology and research design

### ❓ What kind of research is this?

> **Quantitative, computational, simulation-based, comparative experimental.** Quantitative because
> every claim is a number with units stored in a CSV. Computational because the object of study is
> an algorithm. Simulation-based because the data is generated, not observed. Comparative
> experimental because I manipulate one factor at a time and measure the response — comparing
> across five model families and two physical systems.

### ❓ Independent, dependent and controlled variables?

> **Independent** (manipulated): model family, training-set size, test region of parameter space,
> integrator, step size, eccentricity, problem formulation, prediction horizon, number of labelled
> points, presence of the physics term, and physical system.
>
> **Dependent** (measured): MAE, RMSE, R², radial error, relative error, usable horizon, energy
> drift, angular-momentum drift, physics residual, convergence order, training and inference time.
>
> **Controlled**: g and GM, all random seeds, the test set (reserved before training, always), all
> model hyperparameters (module-level constants so it's checkable), hardware and library versions.

### ❓ How did you avoid confounding?

> **"One rule: change exactly one thing at a time, and say what was held fixed."** Three concrete
> examples:
>
> In Stage 7 the test set is split off **once**, before the loop — otherwise a change in RMSE
> could be the test set rather than the training size.
>
> In Stage 12 I had a real confound: Part I used a closed-form map and Part II used a stepped flow
> map, so comparing them would confound the *system* with the *formulation*. I fixed it by
> re-expressing projectile motion as a one-step flow map too, so only the physics differs.
>
> In Stage 13 the "plain network" is the same class with `physics_weight = 0` — same architecture,
> same initialisation, same optimiser, same epochs. Any gap is the loss function and nothing else.

### ❓ Did you tune hyperparameters?

> **"No, deliberately."** They're fixed module constants chosen once from standard defaults. If I'd
> tuned them against test performance my test set would no longer be held out. It costs me some
> absolute accuracy and buys an uncompromised evaluation, and for a study about *where models fail*
> that's the right trade.

### ❓ How do you know your ground truth is correct?

> **"I validated it four independent ways, and I can quote the numbers."**
> 1. Against the analytic Kepler solution: RK4 agrees to 4.6 × 10⁻¹⁴ AU at e = 0 and 8.1 × 10⁻¹³
>    at e = 0.6, over two orbits.
> 2. Convergence orders come out 0.97, 0.99, 1.99, 3.92 against theory 1, 1, 2, 4 — a wrong slope
>    is the standard signature of an implementation bug.
> 3. Conserved quantities: Verlet holds energy to 1.5 × 10⁻⁸ over 200 orbits with no trend.
> 4. Kepler's third law emerges with fitted slope 1.500000, from a simulator only told the
>    inverse-square law.
>
> The reference error over my whole 5-year horizon is 3 × 10⁻¹¹ AU and my smallest ML error is
> 6.4 × 10⁻⁵ AU — six orders of magnitude larger. So I can say the reference is exact *for this
> purpose*, quantitatively.

### ❓ You wrote the PINN by hand. How do you know the gradients are right?

> **"I checked every one against central finite differences of the loss."** Data loss: max gradient
> error 6 × 10⁻¹⁰. Initial-condition loss: 5 × 10⁻⁹. Physics loss: 6 × 10⁻⁶ absolute, which is
> 8.6 × 10⁻⁷ *relative* — larger only because that loss contains a factor 1/h² = 10⁴ that amplifies
> both the true gradient and the finite-difference noise. That relative figure is the expected floor
> for this kind of check.

### ❓ Why finite differences in the PINN instead of automatic differentiation?

> **"Because the physics residual needs the second derivative of the network's output with respect
> to its own input, and doing that with autodiff means differentiating twice through the network."**
> With a central difference, the residual is built from three ordinary *forward* passes, so the
> gradient with respect to the weights is plain backpropagation. Everything stays first-order and
> hand-checkable, which was the point of writing it myself.
>
> The truncation error is O(h²) ≈ 10⁻⁴ with h = 0.01 s — thousands of times smaller than the model
> error I'm trying to reduce. The real trade-off is scaling: finite differences cost an extra pair
> of forward passes per derivative direction, so they'd be a bad choice for a PDE in many
> dimensions. For one direction — time — they're strictly simpler.

---

# Part 6 — Results and interpretation

### ❓ Why did linear regression fail so badly on projectile motion?

> **"Because it's structurally incapable of representing the answer."** A linear model computes a
> weighted sum: w₁v₀ + w₂θ + w₃y₀ + w₄t + b. The true answer is x = v₀ · cos(θ) · t — a product of
> three inputs — and y contains t². No choice of weights produces a product. It's not a tuning
> problem or a data problem; it's a hypothesis-space problem.
>
> What it *did* get right is informative: y is linear in y₀ with coefficient 1, and linear
> regression recovers that correctly.

### ❓ Explain the Random Forest's flat line in Stage 8.

> **"A forest predicts the average of the training targets in the leaf a point falls into. Beyond
> the last split threshold there are no new leaves, so every input past that boundary lands in the
> same leaf and gets the same answer."** The prediction is literally constant. It's not a bug or a
> failure to converge — it's the defining behaviour of the model family, and it's why tree
> ensembles must never be trusted outside their training envelope.
>
> I chose that figure's slice so the true response is *exactly* a straight line, which means every
> deviation you see is purely the model's inductive bias, with nothing else mixed in.

### ❓ Your best model was the worst at extrapolating. Explain.

> **"Flexibility and extrapolation pull in opposite directions."** The MLP got 0.11 m
> in-distribution — the best of anything I tested that wasn't given the physics. Outside, it
> degraded by 286×, the worst of any model. A flexible model has many ways to fit the training
> region well while behaving arbitrarily outside it; the data doesn't constrain it there, so its
> own smoothness prior decides, and that prior knows nothing about projectile motion.
>
> Linear regression only degraded 5.9× — but it was starting from 11 m, so that's faint praise.

### ❓ **[Central]** Your one-step error is 6 × 10⁻⁵ AU. Why does the rollout fail so fast?

> **"Because every prediction becomes the input to the next one, so the errors compound rather
> than merely persist."** After one step I'm slightly off. Step two starts from a state that was
> never on the true trajectory, so it inherits that error *and* adds its own — and the dynamics
> then act on the wrong state. My numbers: 6.4 × 10⁻⁵ AU per step, but past 0.01 AU within 0.37
> years, which is about 90 steps and a third of an orbit.
>
> It's the same mechanism that limits weather forecasting, and Sanchez-Gonzalez et al. identify it
> as the central obstacle in learned simulation. **Crucially, in my case it is not chaos** — I
> measured the perturbation growth of the exact dynamics and it's linear. This is compounding
> model error.

### ❓ Wouldn't more training data fix it?

> **"Not effectively, and I measured the exchange rate."** More data does reduce the one-step
> error — that curve falls steadily. But the usable horizon grows only about logarithmically:
> cutting the per-step error tenfold buys a fixed *additive* extension, not a tenfold one, because
> the error at time t is the per-step error amplified by the dynamics. That's the two panels of
> `stage11_learning_curve.png`, and it's why the structural fixes — Hamiltonian networks, Neural
> ODEs, multi-step training losses — matter more than data volume.

### ❓ Approach B had no error accumulation. Why not just use it?

> **"Because it's barely dynamics, and it only exists for systems we've already solved."** It maps
> (a, e, t) → (x, y) with no feedback loop, so yes — flat error curve. But it requires the orbit to
> be labelled by parameters that already summarise the entire solution, and those parameters exist
> only because this system is integrable. For a chaotic or non-integrable system there's nothing to
> put in place of (a, e).
>
> It also has its own failure: time enters as a raw input, so asking for t beyond the training
> window is an extrapolation in exactly the Stage 8 sense, and it collapses there. Breen et al.
> used this formulation for the three-body problem and are explicit that it only holds over the
> trained time interval.

### ❓ Your model has good position error but a physics residual of 2,000 m/s². What does that mean?

> **"That the Random Forest's predictions are in roughly the right places but describe motion no
> object could actually perform."** Its output is piecewise constant in t — it jumps between leaf
> values — so the numerical second derivative is enormous, even where the positions are fine.
>
> That's the point of measuring the residual separately. Position error asks "is it in the right
> place"; the residual asks "could anything obeying Newton's laws move like this". They come apart,
> and only one of them is usually reported.

### ❓ Isn't ML supposed to be faster than simulation?

> **"Per query, yes — and here it doesn't matter."** Learned inference is cheaper per prediction
> than stepping the integrator. But the classical computation was already fast enough, and it's ten
> orders of magnitude more accurate. You'd be trading essentially all your accuracy for a speed-up
> you didn't need.
>
> The trade becomes interesting when the classical computation is the expensive part. Breen et al.
> got ~10⁵× on the three-body problem where converged solutions need arbitrary-precision
> arithmetic. That's the regime where surrogates earn their place — and it isn't mine.

---

# Part 7 — Hostile questions

### ❓ "This is just curve fitting with extra steps."

> **"For the unconstrained models, yes — and demonstrating that is a result rather than a
> concession."** Stage 8 is the demonstration: a model that had learned the law would work at
> v₀ = 45 m/s, and none of them did. Calling it curve fitting is my conclusion, not an objection
> to it.
>
> The interesting part is what *isn't* just curve fitting. The physics-feature model and the PINN
> are exact or near-exact everywhere, including far outside the training range, because they were
> given structure rather than more points. That contrast is the argument of the project.

### ❓ "Your data is fake, so your results are meaningless."

> **"My data is generated, and that's the design, not an oversight."** The research question is
> about how an algorithm behaves, not about how projectiles move. Exactly-known truth is what makes
> every error attributable — with real data I'd have measurement noise, air resistance and
> calibration error all mixed into the same number, and none of the Stage 8, 11 or 12 measurements
> would be possible.
>
> I do state the cost plainly: a model trained on simulated data inherits every assumption of the
> simulation, so it learned to reproduce a simulation, not nature. Repeating the study with added
> noise is the single most valuable extension and I name it as such.

### ❓ "Any first-year student could run `sklearn.fit()`. What's the research contribution?"

> **"The fitting took about ten lines. The contribution is the experimental design around it."**
> Specifically: a controlled interpolation/extrapolation decomposition with a mechanism, not just a
> number — I plot a slice where the true response is exactly linear so each model's bias is
> visible. An error-accumulation budget relating per-step accuracy to usable horizon. A measurement
> ruling out chaos as the explanation for rollout divergence. Conservation laws used as an
> evaluation metric. And a three-way comparison of hard constraints, soft constraints and no
> constraints under identical conditions.
>
> There's also validation work that isn't machine learning at all: four integrators verified
> against an analytic Kepler solution, with measured convergence orders and Kepler's third law
> recovered to six decimal places.

### ❓ "Why should I believe your numbers?"

> **"Everything is seeded and reproducible — `python run_all.py` regenerates every dataset, table
> and figure, about 49 minutes."** Every number in the report is in a CSV in `results/`, and every
> figure is generated from those CSVs by a script; none was drawn or adjusted by hand.
>
> And the ground truth is independently validated — the RK4 reference agrees with an analytic
> solution obtained by a completely different route to 10⁻¹³ AU.

### ❓ "You got a result that contradicted theory and you kept it. Isn't that sloppy?"

> **"I had two, and investigating them rather than smoothing them over turned both into
> findings."** Euler–Cromer measuring as second order turned out to be the known conjugacy to
> Verlet, which I confirmed by measuring at a quarter period as well as a full one. And the
> polynomial model's rollout diverged to infinity and crashed my pipeline — so I made divergence an
> explicitly recorded outcome rather than excluding the model, because dropping an unstable model's
> worst runs would flatter it.

### ❓ "GraphCast beats supercomputers at weather. Doesn't that refute your conclusions?"

> **"No — it constrains their scope, and I cite it in the report for exactly that reason."** The
> differences are instructive. GraphCast trains on multi-step rollouts, so accumulated error is
> inside the objective; I trained on single steps, which is the baseline. The atmosphere has no
> cheap exact solver to compete against; a two-body orbit does. And GraphCast's useful horizon is a
> few tens of steps, not thousands.
>
> My conclusion isn't "learned simulation doesn't work" — it's "here is the baseline phenomenon,
> measured cleanly, that all the published improvements are designed to overcome".

### ❓ "Why didn't you use PyTorch?"

> **"For the standard models, scikit-learn was the right tool and PyTorch would have added a
> dependency with no benefit."** For the PINN I needed derivatives of the output with respect to
> the input inside the loss, which scikit-learn can't express — so I wrote the network in NumPy:
> forward pass, backprop and Adam, about 120 lines, every gradient verified against finite
> differences. That was a deliberate choice: PyTorch would have hidden the one mechanism the stage
> exists to explain.

---

# Part 8 — Questions about your own code

You must be able to open any file and explain it. Know this map:

| File | What it does | Key thing to be able to explain |
|---|---|---|
| `data/generation/projectileDataGeneration.py` | Projectile datasets | Why t is sampled as a *fraction of each sample's own flight time* |
| `data/generation/planetaryDataGeneration.py` | Orbital trajectories, three ML tables | Why one-step pairs must never cross a `traj_id` boundary |
| `src/planetary_physics.py` | Four integrators + analytic Kepler | Why `simulate_reference` separates Δt_store from Δt_integrate |
| `src/planetary_ml.py` | Flow-map / history / direct models, rollout | Why delta targets, and how divergence is recorded as NaN |
| `src/pinn.py` | NumPy network, backprop, Adam, three losses | Why tanh not ReLU; why the IC loss is mandatory |
| `src/models.py` | Model factory | Why the physics-feature basis makes the problem linear |
| `src/evaluation.py` | Metrics | Why targets are scored per coordinate, never averaged |
| `experiments/stage*.py` | One experiment each | The design block at the top of each file |

### ❓ Why tanh rather than ReLU in the PINN?

> **"Because a ReLU network is piecewise linear, so its second derivative is zero almost
> everywhere."** It is structurally incapable of satisfying d²y/dt² = −g, which demands a specific
> non-zero curvature. tanh is smooth and infinitely differentiable, which is what a physics
> residual needs.

### ❓ Why does the PINN need an initial-condition loss? Isn't the equation enough?

> **"No — the equation alone is satisfied by infinitely many trajectories."** d²y/dt² = −g is true
> of *every* parabola with the right curvature, regardless of where it starts or how fast. The
> initial conditions are what pick out one solution. Omitting them is the most common way a
> hand-written PINN silently fails — it converges to something with perfect physics loss and
> completely wrong positions.

### ❓ Why predict the change in state rather than the next state?

> **"Because over one step the planet moves about 0.025 AU while sitting at r ≈ 1 AU."** In
> absolute form the model has to output ~1.0 to a precision of 10⁻⁴ — it spends nearly all its
> capacity reproducing the identity map, and only the last digits carry the dynamics. In delta form
> the identity is free and every bit of capacity goes on the physics.
>
> Measured: 28–29× better for the forest and the MLP. And **exactly unchanged** for linear and
> polynomial models — which is the confirmation the reasoning is right, because a linear model can
> already represent the identity, so the reparametrisation is one it absorbs exactly.

### ❓ Why did you separate Δt_store from Δt_integrate?

> **"Because they're under opposite pressure."** Δt_store is what the ML model steps over, so it
> has to be reasonably large or a five-orbit rollout would need tens of thousands of model calls.
> Δt_integrate is what RK4 uses, so it has to be small or my ground truth would carry visible error.
> Using one value for both forces a bad compromise. I use Δt_store = 0.004 yr with ten internal
> substeps, giving a reference accurate to ~4 × 10⁻¹¹ AU per orbit.

---

# Part 9 — If you get stuck

**If you don't know:**
> "I didn't test that. The experiment that would settle it is ___."

Then name the experiment. This is a *good* answer — it shows you know the boundary of your
evidence.

**If you're asked for a number you've forgotten:**
> "It's in `results/stage__.csv` — may I open it?"

Knowing where a number lives is nearly as good as knowing the number.

**If you're challenged and you think you're right:**
> "I think it does hold, and here's why — ___. But I can see why it looks wrong: ___."

Acknowledge the reading, then defend with evidence.

**If you're challenged and you think you're wrong:**
> "You're right, I hadn't considered that. It would affect ___, and I'd need to ___ to check."

Concede cleanly and specifically. Defending a wrong position loses far more credit than conceding
a right objection.

**If they ask about something beyond your scope (GPUs, transformers, GR):**
> "That's outside what I tested. What I can say from my results is ___."

Redirect to your evidence. Never speculate to fill silence.

---

## The two sentences to have ready at all times

Opening, if asked what the project is about:

> *"I applied a standard machine-learning pipeline to two physical systems whose exact answers I
> already knew, so that I could measure precisely where it stops working and why."*

Closing, if asked what you found:

> *"That it interpolates well and extrapolates badly, that in stepped prediction the errors compound
> faster than accuracy improves, and that the thing which actually fixes both is giving the model
> the physics rather than more data."*
