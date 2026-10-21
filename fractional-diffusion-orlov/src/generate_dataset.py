"""Deterministic 1000 train + 200 test dataset; run as a module."""
import argparse
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


def generate(output=None, seed=20261016, batch_size=64):
    destination = Path(output) if output else ROOT/'data'/'dataset.npz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, 128)
    coefficients = rng.normal(size=(1200, 8)) / np.arange(1, 9)**1.5
    basis = sine_basis()
    raw = coefficients @ basis
    coefficients /= np.max(np.abs(raw), axis=1, keepdims=True)
    u0 = coefficients @ basis
    alpha = np.concatenate((rng.uniform(0.2, 0.9, 1100), np.full(50, 0.35), np.full(50, 0.75)))
    split = np.array(['train']*1000 + ['test_random']*100 + ['test_alpha_035']*50 + ['test_alpha_075']*50)
    # Exact fixed test alpha values must never be present in random draws.
    assert not np.isin(alpha[:1100], [0.35, 0.75]).any()
    started = time.perf_counter()
    uT = np.empty_like(u0)
    for start in range(0, 1200, batch_size):
        end = min(start+batch_size, 1200)
        uT[start:end] = solve_l1_sine_batch(coefficients[start:end], alpha[start:end])
        print(f'Generated {end}/1200', flush=True)
    duration = time.perf_counter()-started
    metadata = dict(seed=seed, n_x=128, n_steps=500, T=1.0, n_modes=8,
                    alpha_range=[0.2, 0.9], coefficient_distribution='Normal(0,1)/k^1.5; max(abs(u0))=1',
                    method='implicit L1 in time, central differences in space; discrete sine diagonalization',
                    split_counts={str(s): int(np.sum(split==s)) for s in np.unique(split)})
    np.savez_compressed(destination, u0=u0, alpha=alpha, uT=uT, x=x, split=split,
                        coefficients=coefficients, metadata=np.array(json.dumps(metadata, ensure_ascii=False)))
    report = audit(destination)
    report['generation_seconds'] = duration
    report['sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (results/'dataset_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    for ax, idx in zip(axes, [0, 1100, 1150]):
        ax.plot(x, u0[idx], label='u0')
        ax.plot(x, uT[idx], label='u(T=1)')
        ax.set(title=f'{split[idx]}, alpha={alpha[idx]:.3f}', xlabel='x', ylabel='u')
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(results/'dataset_examples.png', dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


def audit(path):
    with np.load(path, allow_pickle=False) as d:
        u0, a, uT, split = (d[k] for k in ('u0', 'alpha', 'uT', 'split'))
        assert u0.shape == uT.shape == (1200, 128)
        assert a.shape == split.shape == (1200,)
        assert np.all(np.isfinite(u0)) and np.all(np.isfinite(uT)) and np.all(np.isfinite(a))
        assert np.all((a >= .2) & (a <= .9))
        assert np.count_nonzero(u0[:, [0,-1]]) == np.count_nonzero(uT[:, [0,-1]]) == 0
        assert set(np.unique(split)) == {'train','test_random','test_alpha_035','test_alpha_075'}
        assert np.sum(split=='train') == 1000
        assert np.sum(split=='test_random') == 100
        assert np.sum(split=='test_alpha_035') == np.sum(split=='test_alpha_075') == 50
        assert not np.isin(a[split=='train'], [.35,.75]).any()
        assert np.all(a[split=='test_alpha_035']==.35) and np.all(a[split=='test_alpha_075']==.75)
        hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in u0]
        assert len(set(hashes)) == 1200, 'duplicate initial conditions'
        np.testing.assert_allclose(d['coefficients'] @ sine_basis(), u0, atol=1e-14)
        assert np.max(np.abs(uT)) <= 1 + 1e-12
        indices = [0, 499, 999, 1000, 1099, 1100, 1149, 1150, 1199]
        equivalence, analytic = [], []
        for i in indices:
            general = solve_l1(u0[i], float(a[i])).u[-1]
            difference = float(np.max(np.abs(general-uT[i])))
            np.testing.assert_allclose(general, uT[i], rtol=1e-10, atol=2e-12)
            equivalence.append(difference)
            exact = exact_sine_solution(d['x'], [1.0], a[i], d['coefficients'][i])[0]
            analytic.append(float(np.linalg.norm(uT[i]-exact)/np.linalg.norm(exact)))
        return dict(passed=True, shape_u0=list(u0.shape), shape_alpha=list(a.shape), shape_uT=list(uT.shape),
                    dtype=str(u0.dtype), split_counts={str(s): int(np.sum(split==s)) for s in np.unique(split)},
                    held_out_alphas_absent_from_train=True, unique_initial_conditions=1200,
                    zero_boundaries=True, all_finite=True, checked_indices=indices,
                    maximum_batch_vs_tridiagonal_error=max(equivalence),
                    checked_multimode_final_relative_l2_errors=analytic,
                    size_bytes=Path(path).stat().st_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    generate(args.output, args.seed, args.batch_size)
