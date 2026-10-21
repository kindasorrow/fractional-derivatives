"""Run with python -m src.validate from the project directory."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .solver import solve_l1
from .reference import exact_sine_solution

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.2, 0.35, 0.5, 0.75, 0.9)


def run_validation(output=None):
    out = Path(output) if output else ROOT / 'results'
    out.mkdir(parents=True, exist_ok=True)
    metrics = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    errorfig, erroraxes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    x = np.linspace(0, 1, 128)
    for ax, a in zip(axes.flat, ALPHAS):
        sol = solve_l1(np.sin(np.pi*x), a)
        exact = exact_sine_solution(x, sol.t, a)
        errors = np.max(np.abs(sol.u-exact), axis=1)
        rel = np.linalg.norm(sol.u[-1]-exact[-1]) / np.linalg.norm(exact[-1])
        metrics.append(dict(alpha=a, n_x=128, n_steps=500,
                            final_linf=float(errors[-1]), final_relative_l2=float(rel),
                            max_linf_over_time=float(errors.max()),
                            time_of_max_error=float(sol.t[errors.argmax()])))
        ax.plot(x, exact[-1], label='Mittag-Leffler', lw=2)
        ax.plot(x[::6], sol.u[-1, ::6], 'o', ms=3, label='L1 + central FD')
        ax.set(title=f'alpha = {a:g}, T = 1', xlabel='x', ylabel='u(x, 1)')
        ax.grid(alpha=0.25)
        erroraxes[0].loglog(sol.t[1:], errors[1:], label=f'alpha={a:g}')
        erroraxes[1].semilogy(sol.t[1:], errors[1:], label=f'alpha={a:g}')
    axes.flat[0].legend(fontsize=8)
    axes.flat[-1].axis('off')
    fig.suptitle('Time-fractional diffusion: final solution, 128 nodes / 500 steps')
    fig.savefig(out/'solution_comparison.png', dpi=170)
    plt.close(fig)
    for ax in erroraxes:
        ax.set(xlabel='t', ylabel='max_x |u_L1 - u_exact|')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    erroraxes[0].set_title('Early-time error (log time axis)')
    erroraxes[1].set_title('Error over the full time interval')
    errorfig.savefig(out/'error_vs_time.png', dpi=170)
    plt.close(errorfig)
    with (out/'validation_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    # Isolate temporal error with the exact solution of the discrete spatial ODE.
    # Separately isolate spatial error against the continuum at the same T.
    from pymittagleffler import mittag_leffler
    conv = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for a in ALPHAS:
        temporal = []
        for steps in (125, 250, 500, 1000):
            sol = solve_l1(np.sin(np.pi*x), a, n_steps=steps)
            lam = 4*127**2*np.sin(np.pi/(2*127))**2
            discrete_exact = float(np.real(mittag_leffler(-lam, a, 1))) * np.sin(np.pi*x)
            err = float(np.max(np.abs(sol.u[-1]-discrete_exact)))
            temporal.append(err)
            conv.append(dict(kind='time_at_T1', alpha=a, resolution=steps, error=err))
        if not np.all(np.diff(temporal) < 0):
            raise AssertionError(f'temporal convergence failed at alpha={a}')
        axes[0].loglog([1/125, 1/250, 1/500, 1/1000], temporal, 'o-', label=f'{a:g}')
        spatial = []
        for nodes in (32, 64, 128, 256):
            xx = np.linspace(0, 1, nodes)
            lam = 4*(nodes-1)**2*np.sin(np.pi/(2*(nodes-1)))**2
            discrete = float(np.real(mittag_leffler(-lam, a, 1)))*np.sin(np.pi*xx)
            exact = exact_sine_solution(xx, [1], a)[0]
            err = float(np.max(np.abs(discrete-exact)))
            spatial.append(err)
            conv.append(dict(kind='space_semidiscrete_T1', alpha=a, resolution=nodes, error=err))
        if not np.all(np.diff(spatial) < 0):
            raise AssertionError(f'spatial convergence failed at alpha={a}')
        axes[1].loglog([1/31, 1/63, 1/127, 1/255], spatial, 'o-', label=f'{a:g}')
    axes[0].set(title='Temporal error at T=1 vs spatially discrete reference', xlabel='dt', ylabel='max absolute error')
    axes[1].set(title='Spatial error at T=1 (exact time propagation)', xlabel='dx', ylabel='max absolute error')
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(title='alpha', fontsize=8)
    fig.savefig(out/'convergence.png', dpi=170)
    plt.close(fig)
    with (out/'convergence_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(conv[0]))
        writer.writeheader()
        writer.writerows(conv)
    print('Validation:')
    for row in metrics:
        print(row)
    return metrics


if __name__ == '__main__':
    run_validation()
