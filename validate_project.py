"""Fast checks of delivered artifacts and core numerical invariants.

No experiment is retrained. Checks cover file completeness, readable figures,
report links, trajectory boundaries, the analytic orbital reference, the
projectile affine map, scale-aware escape handling, and PINN loss gradients.
Run: python validate_project.py
"""
from pathlib import Path
import ast
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def check_artifacts():
    names = {'actual_vs_predicted.png', 'residuals.png', 'trajectory_comparison.png',
             'stage7_learning_curve.png', 'stage7_stability.png',
             'stage7_trajectory_by_datasize.png'}
    for source in (ROOT / 'experiments').glob('*.py'):
        names.update(re.findall(r'[\"\'](stage\d+_[\w]+\.(?:png|csv))[\"\']', source.read_text()))
    for name in sorted(names):
        path = ROOT / 'results' / name
        check(path.is_file() and path.stat().st_size > 0, f'Missing result: {name}')
        if path.suffix == '.csv':
            check(len(pd.read_csv(path)) > 0, f'Empty table: {name}')
        else:
            pixels = mpimg.imread(path)
            check(pixels.shape[0] > 100 and pixels.shape[1] > 100, f'Invalid figure: {name}')
    text = (ROOT / 'docs/RESEARCH_PAPER.md').read_text()
    # Ignore code fences when checking document heading numbering.
    prose = re.sub(r'```.*?```', '', text, flags=re.S)
    numbers = [int(n) for n in re.findall(r'^# (\d+)\. ', prose, re.M)]
    check(numbers == list(range(1, 17)), f'Unexpected paper sections: {numbers}')
    check('# Appendix D' in prose, 'Missing artifact index')
    for ref in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', text):
        if '://' in ref or ref.startswith('#'):
            continue
        from urllib.parse import unquote
        check((ROOT / 'docs' / unquote(ref.split('#')[0])).exists(), f'Broken report link: {ref}')
    for source in ROOT.rglob('*.py'):
        if not any(part.startswith('.') for part in source.relative_to(ROOT).parts):
            ast.parse(source.read_text(), filename=str(source))
    print(f'PASS: {len(names)} result artifacts, report structure, links and Python syntax')


def check_physics():
    from src.planetary_physics import state_from_elements, simulate_reference, kepler_solution
    from data.generation.planetaryDataGeneration import make_one_step_pairs
    from data.generation.projectileDataGeneration import generateProjectileDatasetInRange
    from src.models import ProjectilePhysicsModel
    from src.planetary_ml import OneStepModel
    times, numeric = simulate_reference(state_from_elements(1.0, 0.4), 0.004, 250, 10)
    error = np.linalg.norm(numeric[:, :2] - kepler_solution(times, 1.0, 0.4)[:, :2], axis=1).max()
    check(error < 1e-8, f'Orbital reference mismatch: {error}')
    # Deliberately far-apart trajectories expose accidental global shifts.
    frame = pd.DataFrame({'traj_id': [0, 0, 1, 1], 't': [0, 1, 0, 1],
                          'a': [1]*4, 'e': [0]*4,
                          'x': [1, 2, 100, 101], 'y': [0]*4,
                          'vx': [1]*4, 'vy': [0]*4})
    pairs = make_one_step_pairs(frame)
    check(len(pairs) == 2 and np.all(pairs.next_x - pairs.x == 1), 'Pairs cross a trajectory boundary')
    train = generateProjectileDatasetInRange(100, seed=31)
    outside = generateProjectileDatasetInRange(40, v0_range=(60, 80), seed=32)
    fit = ProjectilePhysicsModel().fit(train, train[['x', 'y']])
    np.testing.assert_allclose(fit.predict(outside), outside[['x', 'y']], atol=1e-9, rtol=1e-10)
    rng = np.random.default_rng(12)
    states = rng.normal(size=(100, 4)) * 20
    dt = 0.02
    nxt = states.copy()
    nxt[:, 0] += states[:, 2] * dt
    nxt[:, 1] += states[:, 3] * dt - 0.5 * 9.81 * dt**2
    nxt[:, 3] -= 9.81 * dt
    model = OneStepModel(LinearRegression()).fit(states, nxt)
    initial = np.array([150., 10., 20., 5.])
    trajectory = model.rollout(initial, 100, divergence_radius=5000)
    t = np.arange(101)*dt
    exact = np.column_stack([150+20*t, 10+5*t-0.5*9.81*t*t,
                             np.full_like(t, 20), 5-9.81*t])
    np.testing.assert_allclose(trajectory, exact, atol=1e-9)
    escaped = model.rollout(initial, 3, divergence_radius=100)
    check(np.isnan(escaped[1:]).all(), 'Escaped states must remain explicitly non-finite')
    print(f'PASS: reference ({error:.2e} AU), pair boundaries, physics features and scaled rollout')


def check_gradients():
    from src.pinn import ProjectilePINN
    from data.generation.projectileDataGeneration import generateProjectileDatasetInRange
    frame = generateProjectileDatasetInRange(8, seed=91)
    X = frame[['v0', 'theta', 'y0', 't']].to_numpy()
    Y = frame[['x', 'y']].to_numpy()
    model = ProjectilePINN(hidden=(4,), seed=7, output_scale=2.)
    model._fit_scaler(X)
    for name, loss in [('data', lambda: model._data_loss(X, Y)),
                       ('physics', lambda: model._physics_loss(X)),
                       ('initial conditions', lambda: model._initial_condition_loss(X))]:
        _, cache, grad = loss()
        gw, gb = model.net.backward(cache, grad)
        actual, expected = [], []
        for param, derivative in zip(model.net.weights+model.net.biases, gw+gb):
            for index in np.ndindex(param.shape):
                old = param[index]
                epsilon = 1e-5
                param[index] = old + epsilon
                plus = loss()[0]
                param[index] = old - epsilon
                minus = loss()[0]
                param[index] = old
                actual.append((plus-minus)/(2*epsilon))
                expected.append(derivative[index])
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5,
                                   err_msg=f'{name} gradient mismatch')
    print('PASS: all parameters in data, physics and initial-condition gradient checks')


def main():
    checks = [check_artifacts, check_physics, check_gradients]
    failures = []
    for function in checks:
        try:
            function()
        except Exception as exc:
            failures.append(f'{function.__name__}: {exc}')
            print(f'FAIL: {failures[-1]}')
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
