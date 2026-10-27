import argparse
import csv
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymittagleffler import mittag_leffler
from .advanced_solver import graded_time_grid, build_soe_kernel, solve_graded_l1, solve_soe


def _temporal_error(solution, alpha):
    x = solution.x
    eigenvalue = 4*(len(x)-1)**2*np.sin(np.pi/(2*(len(x)-1)))**2
    amplitude = np.real(mittag_leffler(-eigenvalue*solution.t**alpha, alpha, 1))
    exact = amplitude[:, None]*np.sin(np.pi*x)
    return float(np.max(np.abs(solution.u-exact)))


def run(output=None, repeats=2):
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be positive')
    destination = Path(output) if output else Path(__file__).resolve().parents[1]/'results'
    destination.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 1, 33)
    initial = np.sin(np.pi*x)
    tolerance = 1e-8
    rows = []
    for mesh, grading in (('uniform', 1.), ('graded', None)):
        for steps in (500, 2000, 10000):
            times = graded_time_grid(.5, steps, grading=grading)
            started = time.perf_counter()
            kernel = build_soe_kernel(.5, np.diff(times).min(), 1., tolerance)
            setup_seconds = time.perf_counter()-started
            full_seconds, fast_seconds = [], []
            for _ in range(repeats):
                started = time.perf_counter()
                full = solve_graded_l1(initial, .5, n_steps=steps, grading=grading)
                full_seconds.append(time.perf_counter()-started)
                started = time.perf_counter()
                fast = solve_soe(initial, .5, n_steps=steps, grading=grading,
                                 tolerance=tolerance, kernel=kernel)
                fast_seconds.append(time.perf_counter()-started)
            full_time, fast_time = float(np.median(full_seconds)), float(np.median(fast_seconds))
            rows.append(dict(mesh=mesh, alpha=.5, n_x=len(x), n_steps=steps,
                             full_l1_seconds=full_time, soe_seconds=fast_time,
                             kernel_setup_seconds=setup_seconds,
                             speedup_excluding_setup=full_time/fast_time,
                             speedup_including_setup=full_time/(fast_time+setup_seconds),
                             exponential_terms=len(kernel.rates),
                             sampled_kernel_relative_error=kernel.measured_relative_error,
                             maximum_soe_vs_full_l1=float(np.max(np.abs(fast.u-full.u))),
                             full_l1_temporal_linf=_temporal_error(full, .5),
                             soe_temporal_linf=_temporal_error(fast, .5),
                             full_history_bytes=steps*(len(x)-2)*8,
                             soe_history_bytes=len(kernel.rates)*(len(x)-2)*8))
            print(f'{mesh}, N={steps}: speedup={full_time/fast_time:.3f}, terms={len(kernel.rates)}', flush=True)
    convergence = []
    for alpha in (.2, .35, .5, .75, .9):
        for mesh, grading in (('uniform', 1.), ('graded', None)):
            previous_error = None
            for steps in (128, 256, 512, 1024):
                solution = solve_graded_l1(initial, alpha, n_steps=steps, grading=grading)
                error = _temporal_error(solution, alpha)
                order = None if previous_error is None else float(np.log2(previous_error/error))
                convergence.append(dict(alpha=alpha, mesh=mesh, n_steps=steps,
                                        temporal_linf=error, observed_order=order,
                                        asymptotic_order=alpha if mesh=='uniform' else 2-alpha))
                previous_error = error
    report = dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, platform=platform.platform()),
                  repeats=repeats, timing_statistic='median', kernel_tolerance=tolerance,
                  reference='Exact Mittag-Leffler propagation of the discrete spatial first mode; all time nodes',
                  soe_method='Positive log-trapezoid Laplace quadrature for t^-alpha/Gamma(1-alpha); exact newest interval; recurrence for older intervals',
                  limitations=['Kernel tolerance is verified on a logarithmic sample grid, not an interval-arithmetic certificate.',
                               'Fast history storage is bounded by exponential terms; returned Solution still stores the entire trajectory.',
                               'SOE construction differs from Jiang et al. integration-by-parts quadrature for t^(-1-alpha).',
                               'Observed orders on finite meshes need not yet equal the asymptotic order.'],
                  sources=[dict(title='Stynes, O Riordan, Gracia (2017): Error Analysis of a Finite Difference Method on Graded Meshes for a Time-Fractional Diffusion Equation',
                                url='https://doi.org/10.1137/16M1082329'),
                           dict(title='Jiang, Zhang, Zhang, Zhang (2017): Fast evaluation of the Caputo fractional derivative and its applications to fractional diffusion equations',
                                url='https://arxiv.org/abs/1511.03453')],
                  runtime=rows, convergence=convergence)
    (destination/'solver_benchmark.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    for name, data in (('solver_benchmark.csv', rows), ('graded_convergence.csv', convergence)):
        with (destination/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for mesh in ('uniform', 'graded'):
        subset = [row for row in rows if row['mesh']==mesh]
        axes[0].loglog([r['n_steps'] for r in subset], [r['full_l1_seconds'] for r in subset], 'o-', label=f'L1 {mesh}')
        axes[0].loglog([r['n_steps'] for r in subset], [r['soe_seconds']+r['kernel_setup_seconds'] for r in subset], 's--', label=f'SOE {mesh}, with setup')
    for alpha in (.2, .35, .5, .75, .9):
        subset = [r for r in convergence if r['alpha']==alpha and r['mesh']=='graded']
        axes[1].loglog([r['n_steps'] for r in subset], [r['temporal_linf'] for r in subset], 'o-', label=f'alpha={alpha:g}')
    axes[0].set(xlabel='Time steps', ylabel='Seconds', title='Measured runtime, 33 spatial nodes')
    axes[1].set(xlabel='Time steps', ylabel='Full-trajectory temporal error', title='Graded L1 vs semidiscrete reference')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(destination/'solver_benchmark.png', dpi=170)
    plt.close(fig)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.repeats)
