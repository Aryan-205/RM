# HANDOFF — project state as of 12 September 2026

Read this first. It is written for a fresh agent (or for Aryan) picking the project up cold.
It records what is **done**, what is **running**, what is **left**, and the traps already hit
so they are not hit again.

---

## 1. One-paragraph status

All twelve remaining stages (7–13) have been **implemented, run, and interpreted**. All code is
written and working. All datasets, results tables and figures exist in `results/`. Five of the
six documents are complete. **The only substantive work left is finishing `docs/RESEARCH_PAPER.md`
— Sections 11, 12 and 14 are not yet written** (see §5). Two experiments were re-launched in the
background right before this handoff and need their outputs checked (see §3).

---

## 2. What is complete

### Code — all written, all runs clean

| File | Status |
|---|---|
| `run_all.py` | ✅ Reproduces the whole study, `--list` / `--only` / `--skip` |
| `src/evaluation.py` | ✅ Shared metrics |
| `src/models.py` | ✅ Model factory + `ProjectilePhysicsModel` (hard physics constraints) |
| `src/planetary_physics.py` | ✅ 4 integrators, analytic Kepler solver, `simulate_reference` |
| `src/planetary_ml.py` | ✅ `OneStepModel` / `HistoryModel` / `DirectModel`, batched rollout, divergence handling |
| `src/pinn.py` | ✅ NumPy MLP + backprop + Adam + 3 loss terms; **all gradients verified vs finite differences** |
| `src/projectile_ml.py` | ✅ Unchanged from before (Stages 3–6) |
| `data/generation/projectileDataGeneration.py` | ✅ Extended: `generateProjectileDatasetInRange`, `timeOfFlight`, state-space form for Stage 12 |
| `data/generation/planetaryDataGeneration.py` | ✅ New: trajectories + the three ML tables (A/B/C) |
| `plots/style.py` | ✅ Shared palette + `readable_log_axes` |
| `plots/{generalization,planetary,planetary_ml,pinn,comparison}.py` | ✅ One module per stage |
| `plots/learning_curve.py` | ✅ Refactored to use the shared palette |
| `experiments/stage{7,8,10,11,12,13}_*.py` | ✅ All written and run |

Every script's docstring opens with its experimental design (independent variable, dependent
variable, what was controlled). **Keep that convention** — the docstrings are quoted in the report.

### Documents

| File | Status |
|---|---|
| `README.md` | ✅ Complete — rewritten as the repo front door |
| `docs/METHODOLOGY.md` | ✅ Complete (610 lines) |
| `docs/LITERATURE_REVIEW.md` | ✅ Complete (649 lines, 24 sources, **all verified against publisher records via web search — do not add unverified citations**) |
| `docs/PRESENTATION_OUTLINE.md` | ✅ Complete (18 slides + 8 appendix slides) |
| `docs/VIVA_PREP.md` | ✅ Complete (617 lines) |
| `docs/RESEARCH_PAPER.md` | ⚠️ **INCOMPLETE — see §5** |
| `docs/RESEARCH_PAPER.tex` | ❌ **Stale.** Still the old Stages 0–6 version. Either regenerate from the .md or delete it |

---

## 3. What is running right now (CHECK THESE FIRST)

Two background processes were launched just before this handoff:

```bash
# Stage 11 — relaunched to pick up two figure fixes (~25 min)
python3 -u experiments/stage11_planetary_ml.py > /tmp/stage11b.log 2>&1 &

# Stage 12 — relaunched after a real bug fix (~10 min)
python3 -u experiments/stage12_comparison.py  > /tmp/stage12.log   2>&1 &
```

**To check:**
```bash
ps aux | grep "[s]tage1"           # still running?
tail -40 /tmp/stage11b.log
tail -40 /tmp/stage12.log
ls results/ | grep stage12          # should be 6 files when done
ls results/stage11_representative_rollout.csv   # new file from the rerun
```

If either died, just re-run it — both are deterministic.

**Why Stage 11 was relaunched:** two figure fixes were applied to `plots/planetary_ml.py` after
the first successful run — (a) the polynomial model's diverged panel printed `nan`, now it says
"ESCAPED the system after X yr"; (b) `stage11_error_growth.png` had its y-axis dragged down to
10⁻¹¹ by the RK4 reference line, squashing the whole ML story into the top fifth; the reference is
now stated in an annotation and the axis floors at 10⁻⁵. The experiment also now saves
`results/stage11_representative_rollout.csv` so those figures can be redrawn without a 25-minute
retrain. **The Stage 11 numbers themselves did not change** — the first run's CSVs are correct and
are the source of every number quoted below.

**Why Stage 12 was relaunched — this was a real bug, do not reintroduce it:**
`DIVERGENCE_RADIUS = 100` in `src/planetary_ml.py` is in **AU**. Stage 12 also rolls out
*projectile* trajectories, whose positions are in **metres** and routinely exceed 100. Every
projectile rollout was therefore being flagged as "escaped" and its error replaced by the
divergence sentinel, producing a nonsense `relative error after 1 period = 1.000e+01` for all
models. Fixed by making the threshold relative to each system's characteristic length
(`DIVERGENCE_FACTOR = 50.0` × mean system scale, passed into `rollout()`). The first lines of the
relaunched run confirm the fix: projectile linear one-step error 6.3 × 10⁻¹⁵, relative error after
one period 3.0 × 10⁻¹⁵.

---

## 4. Headline results (all measured, all in `results/*.csv`)

Quote these; do not re-derive them.

### Stage 7 — training-set size (projectile)
- Forest RMSE x: 16.37 ± 2.34 m (n=100) → **2.05 ± 0.02 m** (n=10,000); **still falling**, so 2,000 rows was insufficient.
- Linear RMSE x: 21.14 → 20.69 m across a 100× data increase — flat, because its functional form is the limit.
- Run-to-run SD shrinks from ±2.34 m to ±0.02 m.
- Whole-trajectory error: 14.85 m → **2.50 m**.

### Stage 8 — interpolation vs extrapolation (projectile) ★ the pivot of Part I
Mean radial error (m), same 8,000-row training set for all:

| Regime | linear | poly2 | forest | mlp | physics |
|---|---|---|---|---|---|
| interpolation (control) | 11.18 | 1.97 | 2.14 | **0.11** | 5.4e-14 |
| extrap v0 high | 36.26 | 8.15 | 31.25 | 18.53 | 5.9e-14 |
| **all variables outside** | 66.24 | 34.04 | 55.59 | 32.70 | 1.2e-13 |
| **degradation** | 5.9× | 17.2× | 26.0× | **286.4×** | none |

- Time extrapolation (train on first 60% of flight): MLP 0.12 m → **20.12 m** at t/t_flight = 0.95.
- `stage8_response_slice.png` is the single best figure in Part I — the forest goes **exactly flat**.

### Stage 10 — integrator validation (planetary)
- Measured orders **at ¼ orbit**: 0.974, 0.994, 1.994, 3.917 (theory 1, 1, 2, 4). ✅
- Euler after 40 orbits: final radius / initial = **7.640**.
- 200-orbit conservation, circular: Verlet final |ΔE/E| = 2.5e-12 (bounded); RK4 = 3.3e-8 (drifting).
- Kepler's third law fitted log–log slope = **1.500000**.
- Reference error over the 5-yr horizon ≈ **3 × 10⁻¹¹ AU**.
- Cost/step (µs): euler 5.6, euler_cromer 5.5, verlet 10.0, rk4 27.9.

**Known and explained anomaly — do not "fix" it:** Euler–Cromer measures as **order 2.00 at a full
orbit** and 0.99 at a quarter orbit. This is real: symplectic Euler is *conjugate* to Störmer–Verlet
(Hairer, Lubich & Wanner 2006), so its O(Δt) error is a periodic coordinate shift that cancels over
a whole period. Both measurements are reported deliberately.

### Stage 11 — ML for orbital motion ★ the central result of Part II
One-step position error (AU), absolute vs delta targets:

| model | absolute | delta | gain |
|---|---|---|---|
| linear | 5.258e-05 | 5.258e-05 | **1.00×** (exactly — linear models absorb the reparametrisation) |
| poly2 | 3.203e-05 | 3.203e-05 | **1.00×** |
| forest | 2.499e-02 | 8.691e-04 | 28.8× |
| mlp | 1.800e-03 | **6.448e-05** | 27.9× |

Median usable horizon (years) before exceeding a tolerance:

| model | 0.001 AU | 0.01 AU | 0.1 AU |
|---|---|---|---|
| linear | 0.020 | 0.062 | 0.192 |
| poly2 | 0.028 | 0.091 | 0.312 |
| forest | 0.007 | 0.076 | 0.386 |
| **mlp** | 0.039 | **0.372** | 0.973 |

- **Headline:** 6.4 × 10⁻⁵ AU per step, yet past 0.01 AU in **0.37 yr ≈ ⅓ of an orbit (~90 steps)**.
- 45% of poly2 rollouts **left the system entirely** (recorded as NaN beyond 100 AU, not dropped).
- Three distinct failure modes visible in `stage11_rollout_orbits.png`: linear **spirals in**, poly2
  **is ejected** (to ~45 AU), forest keeps the orbit with a wrong radius envelope, MLP **precesses**.
- Sensitivity of the *exact* dynamics: offsets 1e-10/1e-8/1e-6 AU amplify by only **45–47×** over
  5 years — **linear, not exponential. There is no chaos here to blame.**
- Learning curve: MLP one-step 2.9e-4 → 4.3e-5 AU (500 → 44,000 pairs) buys horizon 0.08 → 0.25 yr.
  Linear and poly2 do not improve at all — they are at their functional-form floor.

**Approach B (direct map (a,e,t)→state) nuance — get this right in the write-up.**
The summary CSV's `mean_error_in_window` (mlp: 0.301 AU) is misleading on its own. The actual curve
(`stage11_direct_map_curve.csv`) is **0.004–0.006 AU and flat from t ≈ 0.1 to t ≈ 2.2 yr**, then
climbs to 0.64 AU at 2.9 yr and ~1.3 AU beyond 3.5 yr. The reason: training trajectories are 3
*orbits*, and orbits with a ≈ 0.8 only reach t ≈ 2.15 yr, so time coverage thins out past ~2.2 yr
and the rest is extrapolation in t. So the honest statement is:
> Over the interval where training covers time densely (t ≲ 2.2 yr), Approach B's error is **flat**
> at ~0.005 AU — about **100× better than Approach A at t = 2 yr** (0.0064 vs 0.6396 AU) — and it
> collapses where time coverage thins, which is the same extrapolation failure as Stage 8.

Approach C (3-state history) final error 1.458 AU (forest) / 1.885 AU (mlp) — **no better than A**.

### Stage 13 — physics-informed ML
- Hard constraints recover **−4.905000** for the t² coefficient of y; −g/2 = −4.905000. Error ~1e-13 m everywhere.
- PINN with **0 labels: 1.392 ± 0.174 m**. Plain NN needs **~200 labels** to match.
- Ratio at 10/20/50/100 labels: 16.7× / 16.8× / 10.7× / 6.9×. At 2,000 labels they are **equivalent**
  (PINN 0.429 ± 0.047 vs plain 0.364 ± 0.060) — say so; it is the honest half of the result.
- Labels only in the first 40% of flight: beyond the window PINN 1.343 m vs plain 19.737 m = **14.7×**.
- Physics residual audit (g = 9.81 for scale): **Random Forest 2015.79 / 1494.89 m/s²** despite a
  2.65 m position error. Linear Regression: exactly 0.000 and exactly 9.810 (its own form implies both).
- Gradient checks: data 6.2e-10, physics 6.2e-6 (8.6e-7 relative), IC 5.0e-9.

---

## 5. What is LEFT TO DO

### 5.1 `docs/RESEARCH_PAPER.md` — the only substantive gap

Current section structure (verify with `grep -n "^# " docs/RESEARCH_PAPER.md`):

| § | Title | Status |
|---|---|---|
| — | Front matter, abstract, ToC | ✅ rewritten for the full study |
| 1–7 | Part I: intro → theory → ML from zero → methodology → implementation → results → discussion | ✅ pre-existing, still accurate |
| 8 | Stage 7 — How Much Data Do We Actually Need? | ✅ written |
| 9 | Stage 8 — Interpolation, Extrapolation... | ✅ written |
| 10 | Stages 9–10 — The Physics and the Numerical Baseline | ✅ written |
| **11** | **Stage 11 — Machine Learning for Orbital Motion** | ❌ **NOT WRITTEN** |
| **12** | **Stage 12 — The Controlled Comparison** | ❌ **NOT WRITTEN** (needs the running Stage 12 output) |
| 13 | Stage 13 — Physics-Informed Machine Learning | ✅ written |
| **14** | **General Discussion, Limitations and Conclusions** | ❌ **NOT WRITTEN** |
| 15 | How To Reproduce (currently still headed "# 10.") | ⚠️ **renumber to 15** |
| 16 | References (currently "# 11. References — Candidate Sources (To Be Verified)") | ⚠️ **replace** — point at `docs/LITERATURE_REVIEW.md`, which has the verified list |
| App. A–C | File listings / glossary / viva prep | ✅ exist; A is stale (pre-Part-II files only) |
| **App. D** | **Index of every figure and table** | ❌ promised in the ToC, not written |

**Insertion method that worked** (the file is ~2,400 lines; do not rewrite it wholesale):
```python
lines = Path("docs/RESEARCH_PAPER.md").read_text().split("\n")
at = next(i for i,l in enumerate(lines) if l.startswith("# 10. How To Reproduce"))
Path(...).write_text("\n".join(lines[:at]) + NEW_SECTION_TEXT + "\n".join(lines[at:]))
```
Sections 11 and 12 go **before** the `# PART III — SYNTHESIS` marker; Section 14 goes **after**
Section 13 and before `# 10. How To Reproduce`.

**Style to match** (Sections 8–10 and 13 are the template): numbered design table → results tables
with real numbers → an "interpretation" subsection that explains the *mechanism* → an explicit
"what this establishes / does not establish" close. Bold the key number in each table. Name the
figure file for every claim.

### 5.2 Smaller items

1. **`docs/RESEARCH_PAPER.tex` is stale.** Regenerate from the .md or delete it. Do not leave both.
2. **Appendix A** lists only the pre-Part-II files. Either extend it to the new modules or replace
   it with a pointer to the repo (preferred — the code is long).
3. **`.gitignore`** does not exist. Add one for `__pycache__/`, `.DS_Store`, `*.pyc`.
4. **Nothing is committed yet.** `git status` shows the entire Part II as untracked. Suggested
   commits: (a) shared infrastructure `src/evaluation.py`, `src/models.py`, `plots/style.py`;
   (b) Stage 8; (c) Stages 9–10 planetary physics; (d) Stage 11; (e) Stage 12; (f) Stage 13 PINN;
   (g) docs. Use the attribution lines from the session reminder.
5. **Check the two running jobs** (§3) and re-read `stage11_error_growth.png` and
   `stage11_rollout_orbits.png` to confirm the figure fixes landed.

---

## 6. Traps already hit — do not repeat

1. **Units in `DIVERGENCE_RADIUS`.** See §3. Any threshold compared against a position must be
   relative to the system's own scale.
2. **`ScalarFormatter` on a log axis renders 0.5 as "0".** Fixed in `plots/style.py` with a
   `FuncFormatter` using `%g`. Use `readable_log_axes()`, never `ScalarFormatter` directly.
3. **A log y-axis that reaches the RK4 reference (1e-11) squashes the ML story (1e-4 … 1e2).**
   Floor the axis and state the reference in an annotation.
4. **Plot ground truth FIRST, models after**, or an exactly-correct model is hidden under the black
   line and looks unplotted (hit in `plots/generalization.py`).
5. **`python3 -u`** or stdout buffers and a 25-minute job looks hung. `run_all.py` already does this.
6. **Fitting the convergence order over the wrong window** gives nonsense — Euler at coarse Δt is
   out of the asymptotic regime, RK4 at fine Δt is round-off-limited. `ORDER_FIT_BAND = (1e-12, 1e-1)`.
7. **One-step pairs must not cross a `traj_id` boundary**, or the dataset contains physically
   impossible transitions.
8. **`timeout` is not available on this macOS shell.** Use background jobs + polling.
9. **Monitor `grep` patterns that match the literal string `exited with code`** in a log file cause
   a notification storm. Filter more narrowly.
10. **MLP training is the bottleneck**: ~110 s per fit. Stage 11 ≈ 25 min, Stage 12 ≈ 10 min,
    Stage 13 ≈ 7 min. Budget for it; run in the background.

---

## 7. Non-negotiable framing (the user's brief requires this)

- **Never claim a model "learned physics".** Say "learned a function consistent with the physics".
- **Never claim ML beat classical physics.** It loses here by ~10 orders of magnitude; say so.
- **Never invent a citation.** Every source in `docs/LITERATURE_REVIEW.md` was verified by web
  search against publisher records. Anything new must be verified the same way.
- **Report inconvenient results.** Two were kept and turned into findings (Euler–Cromer conjugacy;
  the polynomial rollout diverging). Do not smooth over a third if one appears.
- The user is a **beginner in ML but a physics student**. Explanations should lead with the physics
  and with mechanism, not with ML jargon. Every figure caption in this project explains *what to
  read off the figure and why*, and that convention should continue.

---

## 8. Quick orientation commands

```bash
python3 run_all.py --list                     # stages + runtimes (~49 min total)
grep -n "^# " docs/RESEARCH_PAPER.md          # paper section map
ls results/                                   # 50+ figures and tables
python3 experiments/stage8_generalization.py  # fastest full experiment (~60 s), good smoke test
```
