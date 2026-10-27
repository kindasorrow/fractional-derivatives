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


def convergence_errors(alpha, n_x=128, time_steps=(125, 250, 500, 1000),
                       space_nodes=(32, 64, 128, 256)):
    from pymittagleffler import mittag_leffler

    x = np.linspace(0, 1, n_x)
    n = len(x) - 1
    lam = 4*n**2*np.sin(np.pi/(2*n))**2
    discrete_exact = float(np.real(mittag_leffler(-lam, alpha, 1))) * np.sin(np.pi*x)
    temporal = []
    for steps in time_steps:
        sol = solve_l1(np.sin(np.pi*x), alpha, n_steps=steps)
        temporal.append(float(np.max(np.abs(sol.u[-1]-discrete_exact))))
    spatial = []
    for nodes in space_nodes:
        xx = np.linspace(0, 1, nodes)
        n = len(xx) - 1
        lam = 4*n**2*np.sin(np.pi/(2*n))**2
        discrete = float(np.real(mittag_leffler(-lam, alpha, 1)))*np.sin(np.pi*xx)
        exact = exact_sine_solution(xx, [1], alpha)[0]
        spatial.append(float(np.max(np.abs(discrete-exact))))
    return np.array(temporal), np.array(spatial)


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
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(metrics)

    conv = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    time_steps = (125, 250, 500, 1000)
    space_nodes = (32, 64, 128, 256)
    for a in ALPHAS:
        temporal, spatial = convergence_errors(a, n_x=len(x), time_steps=time_steps,
                                               space_nodes=space_nodes)
        for steps, err in zip(time_steps, temporal):
            conv.append(dict(kind='time_at_T1', alpha=a, resolution=steps, error=err))
        axes[0].loglog(1/np.array(time_steps), temporal, 'o-', label=f'{a:g}')
        for nodes, err in zip(space_nodes, spatial):
            conv.append(dict(kind='space_semidiscrete_T1', alpha=a, resolution=nodes, error=err))
        axes[1].loglog(1/(np.array(space_nodes)-1), spatial, 'o-', label=f'{a:g}')
    axes[0].set(title='Temporal error at T=1 vs spatially discrete reference', xlabel='dt', ylabel='max absolute error')
    axes[1].set(title='Spatial error at T=1 (exact time propagation)', xlabel='dx', ylabel='max absolute error')
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(title='alpha', fontsize=8)
    fig.savefig(out/'convergence.png', dpi=170)
    plt.close(fig)
    with (out/'convergence_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(conv[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(conv)
    print('Validation:')
    for row in metrics:
        print(row)
    return metrics


if __name__ == '__main__':
    run_validation()
