### Accuracy vs sequence length (median relative $L_2$; trained only at L=1000)

| model | params | L=1000 | L=2000 | L=4000 | L=7000 | L=10000 | unseen a |
|---|---|---|---|---|---|---|---|
| Transformer (RoPE) | 76,577 | 2.39e-02 | 5.85e-01 | 8.04e-01 | 8.84e-01 | 9.40e-01 | 9.25e-01 |
| Transformer (NoPE) | 76,577 | 1.81e-01 | 2.49e-01 | 2.87e-01 | 4.12e-01 | 4.08e-01 | 4.02e-01 |
| Transformer (ALiBi) | 76,577 | 3.52e-02 | 4.09e-02 | 5.15e-02 | 5.47e-02 | 5.45e-02 | 4.52e-02 |
| S4D (deep, 2 blocks) | 68,129 | 1.62e-03 | 3.21e-03 | 1.23e-02 | 2.76e-02 | 3.98e-02 | 3.68e-02 |
| S4D-linear (log init) | 3,360 | 6.18e-04 | 1.75e-03 | 3.87e-03 | 5.38e-03 | 5.43e-03 | 9.68e-03 |
| S4D-linear (linear init) | 3,360 | 8.94e-04 | 2.44e-03 | 4.06e-03 | 5.81e-03 | 5.51e-03 | 1.04e-02 |
| S4D-linear (single-scale init) | 3,360 | 8.20e-04 | 1.91e-03 | 3.98e-03 | 5.52e-03 | 5.35e-03 | 6.93e-03 |
| S4D-linear (softplus rates) | 3,360 | 1.71e-03 | 3.07e-03 | 6.33e-03 | 9.63e-03 | 9.48e-03 | 1.23e-02 |
| S4D-linear (trained at L=10000) | 3,360 | 9.31e-04 | 1.12e-03 | 1.40e-03 | 1.55e-03 | 1.67e-03 | 2.30e-03 |
| S4D-linear (1-scale init, trained at L=10000) | 3,360 | 1.33e-03 | 1.87e-03 | 2.41e-03 | 2.55e-03 | 2.94e-03 | 4.13e-03 |
| SOE / Jiang (fixed exponents) | 3,296 | 9.11e-04 | 2.31e-03 | 6.24e-03 | 1.37e-02 | 1.61e-02 | 2.44e-02 |
| SOE / Jiang (per-alpha readout) | 512 | 7.09e-04 | 4.31e-03 | 1.59e-02 | 3.32e-02 | 3.93e-02 | 2.62e-01 |
| SOE / Jiang, train-horizon band | 3,296 | 4.74e-04 | 1.68e-03 | 6.34e-03 | 1.56e-02 | 1.88e-02 | 2.62e-02 |
| *Jiang nodes + LS weights (no training)* | -- | 3.09e-08 | 3.23e-08 | 3.41e-08 | 3.31e-08 | 3.22e-08 | -- |


### Inference cost (RTX 4060 Ti, batch 1, fp32)

| model | ms @1k | ms @10k | time exp. | MiB @1k | MiB @10k | mem exp. | train s | train MiB |
|---|---|---|---|---|---|---|---|---|
| Transformer (RoPE) | 2.55e+00 | 1.60e+01 | 0.78 | 11.6 | 32.0 | 0.44 | 1179 | 317 |
| Transformer (NoPE) | 1.57e+00 | 1.54e+01 | 1.02 | 11.1 | 26.6 | 0.38 | 1089 | 284 |
| Transformer (ALiBi) | 2.20e+00 | 1.11e+02 | 1.75 | 50.7 | 3837.2 | 1.89 | 1930 | 317 |
| S4D (deep, 2 blocks) | 2.81e+00 | 5.58e+00 | 0.31 | 42.0 | 64.4 | 0.16 | 435 | 317 |
| S4D-linear (log init) | 9.43e-01 | 1.07e+00 | 0.06 | 9.6 | 14.1 | 0.16 | 202 | 50 |
| S4D-linear (linear init) | 8.59e-01 | 9.78e-01 | 0.02 | 9.6 | 14.1 | 0.16 | 187 | 50 |
| S4D-linear (single-scale init) | 9.68e-01 | 9.85e-01 | 0.01 | 9.6 | 14.1 | 0.16 | 182 | 50 |
| S4D-linear (softplus rates) | 1.66e+00 | 1.02e+00 | -0.17 | 9.6 | 14.1 | 0.16 | 180 | 50 |
| S4D-linear (trained at L=10000) | 8.66e-01 | 9.53e-01 | 0.05 | 9.6 | 14.1 | 0.16 | 84 | 130 |
| S4D-linear (1-scale init, trained at L=10000) | 9.99e-01 | 9.83e-01 | 0.00 | 9.6 | 14.1 | 0.16 | 86 | 130 |
| SOE / Jiang (fixed exponents) | 8.21e-01 | 8.99e-01 | 0.05 | 9.6 | 14.1 | 0.16 | 136 | 50 |
| SOE / Jiang (per-alpha readout) | 6.57e-01 | 6.73e-01 | 0.00 | 9.6 | 14.1 | 0.16 | 112 | 50 |
| SOE / Jiang, train-horizon band | 9.72e-01 | 8.49e-01 | -0.04 | 9.6 | 14.1 | 0.16 | 136 | 50 |


### Classical solvers (CPU, numpy/scipy)

| L | L1 recursion [ms] | L1 via FFT [ms] | Jiang SOE recursion [ms] | SOE rel. L2 vs L1 |
|---|---|---|---|---|
| 1000 | 4.9 | 0.18 | 0.80 | 4.74e-11 |
| 2000 | 9.9 | 0.23 | 1.20 | 6.49e-11 |
| 4000 | 20.3 | 0.29 | 1.64 | 5.66e-11 |
| 8000 | 42.0 | 0.43 | 2.78 | 3.33e-11 |
| 16000 | 516.7 | 1.15 | 5.21 | 9.09e-11 |
| 32000 | 1764.2 | 2.56 | 9.82 | 5.52e-11 |
| 64000 | 4244.8 | 5.58 | 18.98 | 5.56e-10 |
| **exponent** | **1.75** | **0.85** | **0.76** |  |


### Error at L=10000 by forcing family

| model | white | ou | smooth | steps | bursts | chirp | ramp_noise | mix |
|---|---|---|---|---|---|---|---|---|
| Transformer (RoPE) | 9.23e-01 | 9.51e-01 | 9.94e-01 | 9.70e-01 | 8.34e-01 | 9.90e-01 | 8.12e-01 | 9.14e-01 |
| Transformer (NoPE) | 7.12e-01 | 5.63e-01 | 4.91e-01 | 4.74e-01 | 6.68e-01 | 3.43e-01 | 1.84e-01 | 4.55e-01 |
| Transformer (ALiBi) | 4.06e-02 | 3.68e-02 | 4.12e-02 | 7.49e-02 | 7.30e-02 | 1.56e-02 | 1.93e-01 | 5.55e-02 |
| S4D (deep, 2 blocks) | 5.72e-02 | 4.13e-02 | 3.02e-02 | 4.87e-02 | 1.21e-01 | 2.78e-02 | 3.13e-02 | 3.16e-02 |
| S4D-linear (log init) | 3.34e-03 | 9.12e-03 | 5.98e-03 | 1.34e-02 | 3.79e-03 | 2.05e-03 | 3.40e-02 | 1.17e-02 |
| S4D-linear (linear init) | 3.68e-03 | 9.62e-03 | 7.52e-03 | 1.40e-02 | 4.23e-03 | 2.80e-03 | 3.48e-02 | 1.10e-02 |
| S4D-linear (single-scale init) | 3.38e-03 | 9.32e-03 | 6.18e-03 | 1.36e-02 | 3.94e-03 | 2.23e-03 | 3.47e-02 | 1.14e-02 |
| S4D-linear (softplus rates) | 6.45e-03 | 1.23e-02 | 9.10e-03 | 1.39e-02 | 7.13e-03 | 5.22e-03 | 3.49e-02 | 1.46e-02 |
| S4D-linear (trained at L=10000) | 1.95e-03 | 1.86e-03 | 1.83e-03 | 2.19e-03 | 2.57e-03 | 1.57e-03 | 6.51e-04 | 1.88e-03 |
| S4D-linear (1-scale init, trained at L=10000) | 3.41e-03 | 3.29e-03 | 3.37e-03 | 4.30e-03 | 3.40e-03 | 2.53e-03 | 1.12e-03 | 3.36e-03 |
| SOE / Jiang (fixed exponents) | 9.07e-03 | 1.30e-02 | 1.09e-02 | 2.58e-02 | 7.78e-03 | 7.80e-03 | 4.86e-02 | 1.42e-02 |
| SOE / Jiang (per-alpha readout) | 3.39e-02 | 3.33e-02 | 2.26e-02 | 5.86e-02 | 1.96e-02 | 2.48e-02 | 1.41e-01 | 4.35e-02 |
| SOE / Jiang, train-horizon band | 1.13e-02 | 1.89e-02 | 1.53e-02 | 3.10e-02 | 8.61e-03 | 1.24e-02 | 7.38e-02 | 2.18e-02 |


### Learned relaxation rates vs Jiang's nodes

| model | modes | slowest s | fastest s | decades | modes/decade | slowest time scale | KS vs log-uniform |
|---|---|---|---|---|---|---|---|
| **Jiang nodes (theory)** | 64 | 1.51e-04 | 1.16e+03 | 6.9 | 9.3 | 6637 s | 0.183 |
| s4d_deep | 2031/8192 | 6.23e-03 | 5.63e+02 | 5.0 | 409.8 | 160 s | 0.250 |
| s4d_lin_log | 64/64 | 1.92e-01 | 2.86e+02 | 3.2 | 20.2 | 5 s | 0.373 |
| s4d_lin_linear | 64/64 | 2.00e-01 | 4.24e+02 | 3.3 | 19.2 | 5 s | 0.596 |
| s4d_lin_narrow | 64/64 | 1.95e-01 | 2.78e+02 | 3.2 | 20.3 | 5 s | 0.440 |
| s4d_lin_softplus | 61/64 | 1.99e-01 | 1.45e+02 | 2.9 | 21.3 | 5 s | 0.387 |
| s4d_lin_log_longtrain | 64/64 | 4.25e-02 | 2.63e+02 | 3.8 | 16.9 | 24 s | 0.374 |
| s4d_lin_narrow_longtrain | 64/64 | 5.81e-02 | 2.40e+02 | 3.6 | 17.7 | 17 s | 0.415 |
| soe_jiang | 64/64 | 1.51e-04 | 1.16e+03 | 6.9 | 9.3 | 6637 s | 0.183 |
| soe_jiang_table | 64/64 | 1.51e-04 | 1.16e+03 | 6.9 | 9.3 | 6637 s | 0.183 |
| soe_jiang_trainband | 64/64 | 1.51e-03 | 1.61e+03 | 6.0 | 10.6 | 664 s | 0.213 |


### Implied impulse response, relative $L_2$ error over the full 10^4 lags

| alpha | Jiang+LS (no training) | soe_jiang | s4d_lin_log | s4d_lin_narrow |
|---|---|---|---|---|
| 0.2 | 4.50e-11 | 3.08e-02 | 3.49e-03 | 3.58e-03 |
| 0.3 | 5.99e-11 | 8.59e-03 | 5.85e-03 | 5.94e-03 |
| 0.4 | 6.14e-11 | 7.37e-03 | 7.84e-03 | 7.97e-03 |
| 0.5 | 4.13e-11 | 1.06e-02 | 8.66e-03 | 8.81e-03 |
| 0.6 | 1.23e-10 | 7.87e-03 | 7.58e-03 | 7.68e-03 |
| 0.7 | 2.70e-10 | 8.23e-03 | 5.16e-03 | 5.22e-03 |
| 0.8 | 6.04e-10 | 1.72e-02 | 2.85e-03 | 3.08e-03 |
| 0.9 | 1.51e-09 | 1.07e-01 | 1.24e-03 | 1.40e-03 |
