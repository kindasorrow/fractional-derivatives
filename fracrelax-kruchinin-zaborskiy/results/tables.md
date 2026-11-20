> **Reference.** Every error below is relative to the **L1 solution** (the training target), not to the continuous equation. The L1 scheme itself deviates from the exact kernel K(t) by +70% at t=tau, 7.6% at t=0.1, 1.5% at t=0.7 and ~0.1% only by t~12 (`results/validation.json`), so these numbers say how well a model learns the *L1 operator*. For statements about the equation, regenerate the data on a 10x finer grid and decimate.

### Accuracy vs sequence length (median relative $L_2$ vs the L1 solution; trained only at L=1000)

| model | knows T=100? | params | seeds | L=1000 | L=2000 | L=4000 | L=7000 | L=10000 | unseen a |
|---|---|---|---|---|---|---|---|---|---|
| Transformer (RoPE) | no | 76,577 | 3 | 2.58e-02 ± 3.4e-03 | 5.65e-01 ± 7.1e-02 | 7.97e-01 ± 3.2e-02 | 8.72e-01 ± 2.6e-02 | 9.30e-01 ± 2.8e-02 | 9.03e-01 ± 1.5e-02 |
| Transformer (NoPE) | no | 76,577 | 1 | 1.82e-01 | 2.47e-01 | 3.20e-01 | 4.12e-01 | 4.07e-01 | 3.87e-01 |
| Transformer (ALiBi) | no | 76,577 | 1 | 3.67e-02 | 4.39e-02 | 4.63e-02 | 5.33e-02 | 5.38e-02 | 4.59e-02 |
| S4D (deep, 2 blocks) | no | 68,129 | 3 | 1.65e-03 ± 2.2e-04 | 3.06e-03 ± 2.4e-04 | 9.21e-03 ± 2.6e-03 | 1.82e-02 ± 8.2e-03 | 2.36e-02 ± 1.4e-02 | 2.50e-02 ± 1.0e-02 |
| S4D-linear (log init) | no | 3,360 | 3 | 6.22e-04 ± 7.7e-06 | 1.78e-03 ± 4.3e-05 | 3.88e-03 ± 7.7e-06 | 5.41e-03 ± 4.8e-05 | 5.48e-03 ± 5.4e-05 | 1.09e-02 ± 1.8e-03 |
| S4D-linear (linear init) | no | 3,360 | 1 | 8.94e-04 | 2.44e-03 | 4.06e-03 | 5.81e-03 | 5.51e-03 | 1.04e-02 |
| S4D-linear (single-scale init) | no | 3,360 | 1 | 8.20e-04 | 1.91e-03 | 3.98e-03 | 5.52e-03 | 5.35e-03 | 6.93e-03 |
| S4D-linear (softplus rates) | no | 3,360 | 1 | 1.61e-03 | 3.07e-03 | 6.40e-03 | 9.78e-03 | 9.51e-03 | 1.26e-02 |
| S4D-linear (trained at L=10000) | yes: trained on L=10^4 | 3,360 | 1 | 9.31e-04 | 1.12e-03 | 1.40e-03 | 1.55e-03 | 1.67e-03 | 2.30e-03 |
| S4D-linear (1-scale init, trained at L=10000) | yes: trained on L=10^4 | 3,360 | 1 | 1.33e-03 | 1.87e-03 | 2.41e-03 | 2.55e-03 | 2.94e-03 | 4.13e-03 |
| S4D-linear (log init, LS-initialised readout) | partly: LS weights fitted on L=1000 kernel | 3,360 | 1 | 4.84e+01 | 7.25e+01 | 6.93e+01 | 9.43e+01 | 1.21e+02 | 1.94e+02 |
| S4D-linear (log init, LS-init, per-alpha readout) | partly: LS weights fitted on L=1000 kernel | 576 | 1 | 2.84e-02 | 1.85e-01 | 2.85e+01 | 1.21e+02 | 1.17e+02 | 1.39e+03 |
| SOE / Jiang (fixed exponents) | yes: nodes built for T=100 | 3,296 | 3 | 9.24e-04 ± 7.2e-05 | 2.16e-03 ± 2.1e-04 | 4.86e-03 ± 2.0e-03 | 9.82e-03 ± 4.1e-03 | 1.35e-02 ± 4.1e-03 | 2.07e-02 ± 3.1e-03 |
| SOE / Jiang (per-alpha readout) | yes: nodes built for T=100 | 512 | 1 | 7.09e-04 | 4.31e-03 | 1.59e-02 | 3.32e-02 | 3.93e-02 | 2.62e-01 |
| SOE / Jiang, train-horizon band | no: nodes built for T=10 | 3,296 | 1 | 4.71e-04 | 1.69e-03 | 6.32e-03 | 1.57e-02 | 1.89e-02 | 2.64e-02 |
| SOE / Jiang (LS-initialised readout) | yes: nodes + LS weights for T=100 | 3,296 | 3 | 5.33e-05 ± 6.8e-06 | 7.29e-05 ± 1.4e-05 | 7.87e-05 ± 9.9e-06 | 9.92e-05 ± 1.9e-05 | 1.23e-04 ± 1.6e-05 | 3.39e-04 ± 7.7e-05 |
| SOE / Jiang (LS-init, per-alpha readout) | yes: nodes + LS weights for T=100 | 512 | 1 | 1.64e-07 | 1.82e-07 | 2.39e-07 | 1.75e-07 | 1.93e-07 | 2.60e-01 |
| *Jiang nodes + LS weights (no training)* | yes: nodes + LS weights for T=100 | -- | -- | 3.09e-08 | 3.23e-08 | 3.41e-08 | 3.31e-08 | 3.22e-08 | -- |


`knows T=100?` marks which models were built with, or trained on, the test horizon (T = 100 s): `soe_jiang*` get their nodes from Jiang's rule for T=100, the `*_longtrain` models are trained on L=10^4 sequences. Everything marked `no` only knows T=10 s. Rows with seeds > 1 are mean ± sd over seeds (n = the `seeds` column).


### Variable-length sequences (`mixed_lengths.npz`, lengths uniform in [1000, 10000], each run at its own length)

| model | median | len 1000-2999 | len 3000-5999 | len 6000-10000 |
|---|---|---|---|---|
| Transformer (RoPE) | 8.37e-01 | 5.72e-01 | 8.50e-01 | 8.78e-01 |
| Transformer (NoPE) | 3.35e-01 | 2.03e-01 | 3.53e-01 | 4.25e-01 |
| Transformer (ALiBi) | 4.73e-02 | 4.46e-02 | 4.72e-02 | 5.18e-02 |
| S4D (deep, 2 blocks) | 1.39e-02 | 3.48e-03 | 1.21e-02 | 2.89e-02 |
| S4D-linear (log init) | 3.42e-03 | 1.36e-03 | 4.15e-03 | 4.39e-03 |
| S4D-linear (linear init) | 3.64e-03 | 1.99e-03 | 4.46e-03 | 4.42e-03 |
| S4D-linear (single-scale init) | 3.52e-03 | 1.47e-03 | 4.30e-03 | 4.39e-03 |
| S4D-linear (softplus rates) | 6.14e-03 | 2.04e-03 | 6.90e-03 | 8.40e-03 |
| S4D-linear (trained at L=10000) | 1.40e-03 | 1.12e-03 | 1.47e-03 | 1.47e-03 |
| S4D-linear (1-scale init, trained at L=10000) | 2.46e-03 | 1.89e-03 | 2.50e-03 | 2.74e-03 |
| S4D-linear (log init, LS-initialised readout) | 1.09e+02 | 1.02e+02 | 1.03e+02 | 1.28e+02 |
| S4D-linear (log init, LS-init, per-alpha readout) | 1.95e+01 | 7.95e-02 | 3.02e+01 | 5.78e+01 |
| SOE / Jiang (fixed exponents) | 7.08e-03 | 2.20e-03 | 7.47e-03 | 1.48e-02 |
| SOE / Jiang (per-alpha readout) | 1.40e-02 | 3.63e-03 | 1.60e-02 | 2.90e-02 |
| SOE / Jiang, train-horizon band | 6.02e-03 | 1.44e-03 | 6.15e-03 | 1.28e-02 |
| SOE / Jiang (LS-initialised readout) | 9.49e-05 | 9.95e-05 | 8.72e-05 | 1.03e-04 |
| SOE / Jiang (LS-init, per-alpha readout) | 3.10e-07 | 2.53e-07 | 3.19e-07 | 3.41e-07 |


### Inference cost (RTX 4060 Ti, batch 1, fp32)

| model | ms @1k | ms @10k | time exp. | MiB @1k | MiB @10k | mem exp. | train s | train MiB |
|---|---|---|---|---|---|---|---|---|
| Transformer (RoPE) | 2.71e+00 | 1.54e+01 | 0.74 | 11.6 | 32.0 | 0.44 | 1126 | 316 |
| Transformer (NoPE) | 2.17e+00 | 1.53e+01 | 0.89 | 11.1 | 26.6 | 0.38 | 1054 | 284 |
| Transformer (ALiBi) † | 3.13e+00 | 6.69e+01 | 1.36 | 32.0 | 272.8 | 0.93 | 1560 | 323 |
| S4D (deep, 2 blocks) | 3.03e+00 | 6.09e+00 | 0.33 | 42.0 | 64.4 | 0.16 | 377 | 317 |
| S4D-linear (log init) | 1.20e+00 | 1.08e+00 | -0.04 | 9.6 | 14.1 | 0.16 | 151 | 50 |
| S4D-linear (linear init) | 1.02e+00 | 1.05e+00 | 0.01 | 9.6 | 14.1 | 0.16 | 147 | 50 |
| S4D-linear (single-scale init) | 9.32e-01 | 1.09e+00 | 0.03 | 9.6 | 14.1 | 0.16 | 157 | 50 |
| S4D-linear (softplus rates) | 1.06e+00 | 1.15e+00 | 0.01 | 9.6 | 14.1 | 0.16 | 148 | 50 |
| S4D-linear (trained at L=10000) | 1.01e+00 | 1.28e+00 | 0.06 | 9.6 | 14.1 | 0.16 | 74 | 130 |
| S4D-linear (1-scale init, trained at L=10000) | 1.15e+00 | 1.05e+00 | 0.02 | 9.6 | 14.1 | 0.16 | 74 | 130 |
| S4D-linear (log init, LS-initialised readout) | 1.08e+00 | 1.02e+00 | -0.02 | 9.6 | 14.1 | 0.16 | 157 | 50 |
| S4D-linear (log init, LS-init, per-alpha readout) | 1.07e+00 | 6.43e-01 | -0.16 | 9.6 | 14.1 | 0.16 | 125 | 50 |
| SOE / Jiang (fixed exponents) | 8.86e-01 | 1.05e+00 | 0.11 | 9.6 | 14.1 | 0.16 | 126 | 50 |
| SOE / Jiang (per-alpha readout) | 1.01e+00 | 6.68e-01 | -0.16 | 9.6 | 14.1 | 0.16 | 99 | 50 |
| SOE / Jiang, train-horizon band | 1.01e+00 | 1.08e+00 | 0.04 | 9.6 | 14.1 | 0.16 | 139 | 50 |
| SOE / Jiang (LS-initialised readout) | 1.11e+00 | 1.52e+00 | 0.06 | 9.6 | 14.1 | 0.16 | 126 | 50 |
| SOE / Jiang (LS-init, per-alpha readout) | 8.93e-01 | 6.51e-01 | -0.14 | 9.6 | 14.1 | 0.16 | 95 | 50 |


† ALiBi bias is built per block of 512 queries (`models.alibi_attention`), so memory is O(H·512·L). Compute is still O(L²): that part is the method's.


### Classical solvers (CPU, numpy/scipy)

| L | L1 recursion [ms] | L1 via FFT [ms] | Jiang SOE recursion [ms] | SOE rel. L2 vs L1 |
|---|---|---|---|---|
| 1000 | 5.8 | 0.20 | 0.93 | 4.74e-11 |
| 2000 | 12.0 | 0.38 | 1.05 | 6.49e-11 |
| 4000 | 21.6 | 0.27 | 1.76 | 5.66e-11 |
| 8000 | 47.3 | 0.56 | 3.01 | 3.33e-11 |
| 16000 | 598.5 | 0.98 | 5.22 | 9.09e-11 |
| 32000 | 2046.6 | 2.63 | 10.29 | 5.52e-11 |
| 64000 | 4952.0 | 4.85 | 21.96 | 5.56e-10 |
| **exponent** | **1.74** | **0.76** | **0.78** |  |


### Error at L=10000 by forcing family

| model | white | ou | smooth | steps | bursts | chirp | ramp_noise | mix |
|---|---|---|---|---|---|---|---|---|
| Transformer (RoPE) | 1.01e+00 | 9.45e-01 | 9.95e-01 | 9.61e-01 | 8.61e-01 | 9.88e-01 | 7.69e-01 | 9.28e-01 |
| Transformer (NoPE) | 7.03e-01 | 5.28e-01 | 4.64e-01 | 4.58e-01 | 7.13e-01 | 3.97e-01 | 1.72e-01 | 4.07e-01 |
| Transformer (ALiBi) | 4.18e-02 | 3.60e-02 | 3.88e-02 | 7.97e-02 | 6.58e-02 | 1.68e-02 | 2.08e-01 | 5.62e-02 |
| S4D (deep, 2 blocks) | 5.72e-02 | 4.13e-02 | 3.02e-02 | 4.87e-02 | 1.21e-01 | 2.78e-02 | 3.13e-02 | 3.16e-02 |
| S4D-linear (log init) | 3.34e-03 | 9.12e-03 | 5.98e-03 | 1.34e-02 | 3.79e-03 | 2.05e-03 | 3.40e-02 | 1.17e-02 |
| S4D-linear (linear init) | 3.68e-03 | 9.62e-03 | 7.52e-03 | 1.40e-02 | 4.23e-03 | 2.80e-03 | 3.48e-02 | 1.10e-02 |
| S4D-linear (single-scale init) | 3.38e-03 | 9.32e-03 | 6.18e-03 | 1.36e-02 | 3.94e-03 | 2.23e-03 | 3.47e-02 | 1.14e-02 |
| S4D-linear (softplus rates) | 6.49e-03 | 1.24e-02 | 9.15e-03 | 1.40e-02 | 7.16e-03 | 5.32e-03 | 3.51e-02 | 1.47e-02 |
| S4D-linear (trained at L=10000) | 1.95e-03 | 1.86e-03 | 1.83e-03 | 2.19e-03 | 2.57e-03 | 1.57e-03 | 6.51e-04 | 1.88e-03 |
| S4D-linear (1-scale init, trained at L=10000) | 3.41e-03 | 3.29e-03 | 3.37e-03 | 4.30e-03 | 3.40e-03 | 2.53e-03 | 1.12e-03 | 3.36e-03 |
| S4D-linear (log init, LS-initialised readout) | 6.69e+01 | 1.19e+02 | 1.35e+02 | 1.77e+02 | 6.35e+01 | 5.35e+01 | 2.64e+02 | 1.86e+02 |
| S4D-linear (log init, LS-init, per-alpha readout) | 6.25e+01 | 2.17e+02 | 9.89e+01 | 1.10e+02 | 1.09e+02 | 5.03e+01 | 3.66e+02 | 1.70e+02 |
| SOE / Jiang (fixed exponents) | 1.06e-02 | 1.47e-02 | 1.17e-02 | 3.03e-02 | 8.89e-03 | 8.78e-03 | 5.78e-02 | 1.65e-02 |
| SOE / Jiang (per-alpha readout) | 3.39e-02 | 3.33e-02 | 2.26e-02 | 5.86e-02 | 1.96e-02 | 2.48e-02 | 1.41e-01 | 4.35e-02 |
| SOE / Jiang, train-horizon band | 1.14e-02 | 1.90e-02 | 1.53e-02 | 3.13e-02 | 8.44e-03 | 1.23e-02 | 7.20e-02 | 2.15e-02 |
| SOE / Jiang (LS-initialised readout) | 9.82e-05 | 1.17e-04 | 1.20e-04 | 1.75e-04 | 7.22e-05 | 8.64e-05 | 3.21e-04 | 1.22e-04 |
| SOE / Jiang (LS-init, per-alpha readout) | 2.58e-07 | 1.81e-07 | 1.72e-07 | 1.73e-07 | 2.54e-07 | 2.01e-07 | 1.59e-07 | 1.77e-07 |


### Learned relaxation rates vs Jiang's nodes

| model | modes | slowest s | fastest s | decades | modes/decade | slowest time scale | KS vs log-uniform |
|---|---|---|---|---|---|---|---|
| **Jiang nodes (theory)** | 64 | 1.51e-04 | 1.16e+03 | 6.9 | 9.3 | 6637 s | 0.183 |
| s4d_deep | 2031/8192 | 6.23e-03 | 5.63e+02 | 5.0 | 409.8 | 160 s | 0.250 |
| s4d_lin_log | 64/64 | 1.92e-01 | 2.86e+02 | 3.2 | 20.2 | 5 s | 0.373 |
| s4d_lin_linear | 64/64 | 2.00e-01 | 4.24e+02 | 3.3 | 19.2 | 5 s | 0.596 |
| s4d_lin_narrow | 64/64 | 1.95e-01 | 2.78e+02 | 3.2 | 20.3 | 5 s | 0.440 |
| s4d_lin_softplus | 61/64 | 1.99e-01 | 1.45e+02 | 2.9 | 21.3 | 5 s | 0.386 |
| s4d_lin_log_longtrain | 64/64 | 4.25e-02 | 2.63e+02 | 3.8 | 16.9 | 24 s | 0.374 |
| s4d_lin_narrow_longtrain | 64/64 | 5.81e-02 | 2.40e+02 | 3.6 | 17.7 | 17 s | 0.415 |
| s4d_lin_log_lsinit | 63/64 | 8.84e-02 | 1.38e+02 | 3.2 | 19.7 | 11 s | 0.016 |
| soe_jiang | 64/64 | 1.51e-04 | 1.16e+03 | 6.9 | 9.3 | 6637 s | 0.183 |
| soe_jiang_table | 64/64 | 1.51e-04 | 1.16e+03 | 6.9 | 9.3 | 6637 s | 0.183 |
| soe_jiang_trainband | 64/64 | 1.51e-03 | 1.61e+03 | 6.0 | 10.6 | 664 s | 0.213 |
| soe_jiang_lsinit | 59/64 | 1.51e-04 | 5.96e+02 | 6.6 | 8.9 | 6637 s | 0.186 |


### Implied impulse response, relative $L_2$ error over the full 10^4 lags

| alpha | Jiang+LS (no training) | soe_jiang | s4d_lin_log | s4d_lin_narrow |
|---|---|---|---|---|
| 0.2 | 4.50e-11 | 3.03e-02 | 3.49e-03 | 3.58e-03 |
| 0.3 | 5.99e-11 | 7.70e-03 | 5.85e-03 | 5.94e-03 |
| 0.4 | 6.14e-11 | 8.58e-03 | 7.84e-03 | 7.97e-03 |
| 0.5 | 4.13e-11 | 1.20e-02 | 8.66e-03 | 8.81e-03 |
| 0.6 | 1.23e-10 | 9.80e-03 | 7.58e-03 | 7.68e-03 |
| 0.7 | 2.70e-10 | 1.11e-02 | 5.16e-03 | 5.22e-03 |
| 0.8 | 6.04e-10 | 2.17e-02 | 2.85e-03 | 3.08e-03 |
| 0.9 | 1.51e-09 | 1.04e-01 | 1.24e-03 | 1.40e-03 |
