import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .solver import sine_basis, solve_l1_sine_batch, solve_l1
from .reference import exact_sine_solution

ROOT = Path(__file__).resolve().parents[1]
TIMES = np.array([.01, .1, .5, 1.])


def fit_output_normalization(u, split):
    training = np.asarray(u)[np.asarray(split) == 'train']
    if training.ndim != 3 or len(training) == 0 or not np.isfinite(training).all():
        raise ValueError('normalization requires finite training trajectories')
    mean = training.mean(axis=(0, 2))[:, None]
    scale = training.std(axis=(0, 2))[:, None]
    return mean, np.maximum(scale, np.finfo(float).eps)


def generate(output=None, seed=20261016, batch_size=64):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be positive')
    destination = Path(output) if output else ROOT/'data'/'dataset.npz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    results = destination.parent/'results' if output else ROOT/'results'
    results.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, 128)
    coefficients = rng.normal(size=(1200, 32)) / np.arange(1, 33)**0.5
    basis = sine_basis(128, 32)
    coefficients /= np.max(np.abs(coefficients @ basis), axis=1, keepdims=True)
    u0 = coefficients @ basis
    random_alpha = rng.uniform(0.2, 0.9, 1100)
    held_out = np.isin(random_alpha, [.35, .75])
    while held_out.any():
        random_alpha[held_out] = rng.uniform(.2, .9, held_out.sum())
        held_out = np.isin(random_alpha, [.35, .75])
    alpha = np.concatenate((random_alpha, np.full(50, .35), np.full(50, .75)))
    split = np.array(['train']*1000 + ['test_random']*100 + ['test_alpha_035']*50 + ['test_alpha_075']*50)
    started = time.perf_counter()
    u = np.empty((1200, len(TIMES), len(x)))
    for start in range(0, 1200, batch_size):
        end = min(start+batch_size, 1200)
        for j, horizon in enumerate(TIMES):
            u[start:end, j] = solve_l1_sine_batch(coefficients[start:end], alpha[start:end], T=horizon)
        print(f'Generated {end}/1200', flush=True)
    duration = time.perf_counter()-started
    times = TIMES.copy()
    mean, scale = fit_output_normalization(u, split)
    metadata = dict(seed=seed, n_x=128, n_steps=500, T=1.0, n_modes=32,
                    times=times.tolist(), time_grids='independent uniform grids; 500 steps per horizon',
                    alpha_range=[.2, .9], coefficient_distribution='Normal(0,1)/sqrt(k); max(abs(u0))=1',
                    normalization='per-time mean and std over train samples and spatial nodes only',
                    method='implicit L1 in time, central differences in space; discrete sine diagonalization',
                    split_counts={str(s): int(np.sum(split==s)) for s in np.unique(split)})
    np.savez_compressed(destination, u0=u0, alpha=alpha, uT=u[:, -1], u=u, t=times,
                        x=x, split=split, coefficients=coefficients,
                        output_mean=mean, output_scale=scale,
                        metadata=np.array(json.dumps(metadata, ensure_ascii=False)))
    report = audit(destination, accuracy_output=results/'full_dataset_accuracy.csv')
    report['generation_seconds'] = duration
    report['sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (results/'dataset_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    for ax, idx in zip(axes, [0, 1100, 1150]):
        ax.plot(x, u0[idx], label='u0')
        for j, t in enumerate(times):
            ax.plot(x, u[idx, j], label=f't={t:g}')
        ax.set(title=f'{split[idx]}, alpha={alpha[idx]:.3f}', xlabel='x', ylabel='u')
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(results/'dataset_examples.png', dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(path, accuracy_output=None):
    with np.load(path, allow_pickle=False) as d:
        u0, a, uT, split = (d[k] for k in ('u0', 'alpha', 'uT', 'split'))
        _require(u0.shape == uT.shape == (1200, 128), 'invalid state shape')
        _require(a.shape == split.shape == (1200,), 'invalid alpha or split shape')
        u, t, x, coef = (d[k] for k in ('u', 't', 'x', 'coefficients'))
        _require(u.shape == (1200, 4, 128) and coef.shape == (1200, 32), 'invalid trajectory or coefficient shape')
        _require(x.shape == (128,) and np.allclose(x, np.linspace(0, 1, 128)), 'invalid spatial grid')
        _require(t.shape == (4,) and np.array_equal(t, TIMES), 'invalid sample times')
        _require(all(np.isfinite(v).all() for v in (u0, a, uT, u, coef)), 'non-finite data')
        _require(np.all((a >= .2) & (a <= .9)), 'alpha outside range')
        _require(np.count_nonzero(u0[:, [0, -1]]) == np.count_nonzero(u[:, :, [0, -1]]) == 0, 'nonzero boundaries')
        counts = {str(s): int(np.sum(split==s)) for s in np.unique(split)}
        _require(counts == dict(train=1000, test_random=100, test_alpha_035=50, test_alpha_075=50), 'invalid splits')
        _require(not np.isin(a[split=='train'], [.35, .75]).any(), 'held-out alpha in train')
        _require(np.all(a[split=='test_alpha_035']==.35) and np.all(a[split=='test_alpha_075']==.75), 'wrong held-out alpha')
        _require(len(np.unique(u0, axis=0)) == 1200, 'duplicate initial conditions')
        _require(np.allclose(coef @ sine_basis(128, 32), u0, rtol=0, atol=1e-14), 'coefficients mismatch')
        _require(np.allclose(np.max(np.abs(u0), axis=1), 1), 'initial conditions not normalized')
        _require(np.array_equal(u[:, -1], uT), 'uT differs from final snapshot')
        _require(np.max(np.abs(u)) <= 1 + 1e-12, 'maximum principle violated')
        mean, scale = fit_output_normalization(u, split)
        _require(d['output_mean'].shape == d['output_scale'].shape == (4, 1), 'invalid normalization shape')
        _require(np.allclose(d['output_mean'], mean) and np.allclose(d['output_scale'], scale), 'normalization mismatch')
        indices = [0, 499, 999, 1000, 1099, 1100, 1149, 1150, 1199]
        equivalence = []
        for i in indices:
            general = np.stack([solve_l1(u0[i], float(a[i]), T=horizon).u[-1] for horizon in t])
            equivalence.append(float(np.max(np.abs(general-u[i]))))
            _require(np.allclose(general, u[i], rtol=1e-10, atol=2e-12), 'batch differs from tridiagonal solver')
        errors = []
        for i in range(1200):
            exact = exact_sine_solution(x, t, a[i], coef[i])
            relative = np.linalg.norm(u[i]-exact, axis=1)/np.linalg.norm(exact, axis=1)
            errors.append(relative)
        errors = np.asarray(errors)
        if accuracy_output:
            with Path(accuracy_output).open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['index', 'split', 'alpha', 't', 'relative_l2'])
                for i in range(1200):
                    for j, value in enumerate(t):
                        writer.writerow([i, split[i], a[i], value, errors[i, j]])
        modal = np.einsum('btx,kx->btk', u, sine_basis(128, 32)) * (2/127)
        energy = np.sum(modal**2, axis=(0, 2))
        return dict(passed=True, shape_u0=list(u0.shape), shape_u=list(u.shape),
                    shape_alpha=list(a.shape), shape_uT=list(uT.shape), dtype=str(u0.dtype),
                    split_counts=counts, held_out_alphas_absent_from_train=True,
                    unique_initial_conditions=1200, zero_boundaries=True, all_finite=True,
                    checked_indices=indices, maximum_batch_vs_tridiagonal_error=max(equivalence),
                    times=t.tolist(), first_mode_energy_fraction=(np.sum(modal[:, :, 0]**2, axis=0)/energy).tolist(),
                    maximum_relative_l2_by_time=errors.max(axis=0).tolist(),
                    median_relative_l2_by_time=np.median(errors, axis=0).tolist(),
                    output_mean=mean[:, 0].tolist(), output_scale=scale[:, 0].tolist(),
                    size_bytes=Path(path).stat().st_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    generate(args.output, args.seed, args.batch_size)
