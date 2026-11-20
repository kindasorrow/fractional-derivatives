import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(output):
    directory = Path(output)
    report = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    with (directory/'history.csv').open(encoding='utf-8') as stream:
        history = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    epochs = [int(r['epoch']) for r in history]
    for label, field in [('fit', 'train_relative_squared_l2'), ('validation', 'validation_relative_squared_l2')]:
        axes[0].semilogy(epochs, [float(r[field]) for r in history], label=label)
    axes[0].axvline(report['best_epoch'], color='grey', linestyle=':', label='selected checkpoint')
    axes[0].set(xlabel='Epoch', ylabel='Mean squared relative L2', title='FNO training; selection by validation')
    rows = []
    times = [.01, .1, .5, 1.]
    names = ['test_random', 'test_alpha_035', 'test_alpha_075']
    for group in names:
        fno, baseline = [], []
        for horizon in times:
            metrics = report['metrics'][group][f'{horizon:g}']
            other = report['first_mode_baseline_metrics'][group][f'{horizon:g}']
            fno.append(metrics['mean_relative_l2']*100)
            baseline.append(other['mean_relative_l2']*100)
            rows.append(dict(split=group, t=horizon, fno_mean_relative_l2=metrics['mean_relative_l2'],
                             baseline_mean_relative_l2=other['mean_relative_l2'],
                             fno_aggregate_relative_l2=metrics['aggregate_relative_l2'],
                             fno_p95_relative_l2=metrics['p95_relative_l2']))
        axes[1].semilogx(times, fno, 'o-', label=f'FNO {group}')
        axes[1].semilogx(times, baseline, 'x--', label=f'1-mode {group}')
    axes[1].set(xlabel='t', ylabel='Mean relative L2 (%)', title='Held-out initial conditions')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(directory/'training_metrics.png', dpi=170)
    plt.close(fig)
    with (directory/'test_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    with np.load(directory/'predictions.npz', allow_pickle=False) as data:
        fig, axes = plt.subplots(3, 4, figsize=(14, 8), layout='constrained')
        for row, group in enumerate(names):
            ids = np.flatnonzero(data['split'] == group)
            errors = np.linalg.norm(data['prediction'][ids]-data['target'][ids], axis=-1)
            errors /= np.maximum(np.linalg.norm(data['target'][ids], axis=-1), 1e-16)
            for j, horizon in enumerate(times):
                local = np.argsort(errors[:, j])[len(ids)//2]
                idx = ids[local]
                ax = axes[row, j]
                ax.plot(data['x'], data['target'][idx, j], label='L1 target')
                ax.plot(data['x'], data['prediction'][idx, j], '--', label='FNO')
                ax.plot(data['x'], data['first_mode_baseline'][idx, j], ':', label='1-mode')
                ax.set(title=f'{group}, t={horizon:g}, id={idx}\nmedian-error sample: {errors[local,j]*100:.2f}%', xlabel='x')
                ax.grid(alpha=.25)
        axes[0, 0].legend(fontsize=7)
        fig.savefig(directory/'prediction_examples.png', dpi=170)
        plt.close(fig)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'results'/'fno')
    args = parser.parse_args()
    plot(args.output)
