#!/usr/bin/env python3
"""
Reproduce the entire study from scratch.

Regenerates every dataset, every results table and every figure, in dependency
order. Because every random step in the project is seeded, seeded scientific metrics should reproduce within numerical tolerance; timing columns vary.

    python run_all.py                # everything
    python run_all.py --list         # show the stages and their runtimes
    python run_all.py --only 8 11    # just those stages
    python run_all.py --skip 11 13   # everything except the slow ones

The stages are ordered by dependency, not by number: Stage 12 consumes the
planetary trajectories produced by the Stage 11 data generator, so that
generator runs first.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (stage number, description, script, approximate runtime in seconds)
STAGES = [
    (2, "Generate the projectile dataset",
     "data/generation/projectileDataGeneration.py", 1),
    (5, "Train, evaluate and visualise the projectile models (Stages 3-6)",
     "src/projectile_ml.py", 5),
    (7, "Effect of training-set size on projectile accuracy",
     "experiments/stage7_data_size.py", 90),
    (8, "Interpolation vs extrapolation",
     "experiments/stage8_generalization.py", 60),
    (9, "Generate the planetary trajectories (Stages 9-11 data)",
     "data/generation/planetaryDataGeneration.py", 40),
    (10, "Numerical integrator validation against the analytic Kepler solution",
     "experiments/stage10_integrators.py", 240),
    (11, "Machine learning for orbital motion",
     "experiments/stage11_planetary_ml.py", 1500),
    (12, "Projectile vs planetary, head to head",
     "experiments/stage12_comparison.py", 600),
    (13, "Physics-informed machine learning",
     "experiments/stage13_physics_informed.py", 420),
]


def run(stage, description, script, expected):
    print("\n" + "=" * 78)
    print(f"STAGE {stage}: {description}")
    print(f"  {script}   (expect roughly {expected} s)")
    print("=" * 78, flush=True)

    start = time.perf_counter()
    # -u so the child's output streams to the terminal instead of sitting in a
    # buffer until it finishes -- these stages take minutes and silence is
    # indistinguishable from a hang.
    result = subprocess.run([sys.executable, "-u", str(ROOT / script)], cwd=ROOT)
    elapsed = time.perf_counter() - start

    status = "OK" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"\n--- stage {stage}: {status} in {elapsed:.1f} s")
    return result.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", type=int, help="run only these stage numbers")
    parser.add_argument("--skip", nargs="*", type=int, default=[], help="skip these stages")
    parser.add_argument("--list", action="store_true", help="list the stages and exit")
    args = parser.parse_args()

    if args.list:
        total = sum(s[3] for s in STAGES)
        print(f"{'stage':>6}  {'~time':>7}  description")
        for stage, description, script, expected in STAGES:
            print(f"{stage:>6}  {expected:>6}s  {description}")
        print(f"\ntotal: about {total / 60:.0f} minutes on a laptop")
        return 0

    selected = [s for s in STAGES
                if (args.only is None or s[0] in args.only) and s[0] not in args.skip]

    print(f"Running {len(selected)} stage(s). Results land in results/, data in data/.")
    outcomes = []
    for stage, description, script, expected in selected:
        ok, elapsed = run(stage, description, script, expected)
        outcomes.append((stage, description, ok, elapsed))
        if not ok:
            print(f"\nStopping: stage {stage} failed. Later stages depend on it.")
            break

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for stage, description, ok, elapsed in outcomes:
        print(f"  {'OK  ' if ok else 'FAIL'}  stage {stage:>2}  {elapsed:7.1f} s  {description}")

    failures = sum(1 for *_, ok, _ in outcomes if not ok)
    total = sum(elapsed for *_, elapsed in outcomes)
    print(f"\n  {len(outcomes) - failures}/{len(outcomes)} stages succeeded "
          f"in {total / 60:.1f} minutes total")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
