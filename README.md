# Learning the fractional relaxation operator

Can a sequence model learn the solution operator of

$$D^{\alpha} y(t) \;=\; -\lambda\, y(t) \;+\; f(t), \qquad y(0)=0, \qquad 0<\alpha<1,$$

where $D^{\alpha}$ is the **Caputo** derivative — and if it can, *what does it learn*?

The project compares three sequence models on this map $f \mapsto y$: a small
causal **Transformer**, a diagonal **S4D** state-space model, and a model whose
exponents are **frozen at the nodes of Jiang's sum-of-exponentials formula**.
They are trained on short sequences, tested on sequences ten times longer, and
then opened up to see whether the relaxation rates a network discovers by
gradient descent coincide with the ones numerical analysis prescribes.

---

## Quick start

**1. Environment** (Python 3.11+; developed on 3.14, Windows, RTX 4060 Ti)

```bash
python -m venv frac
```

Activate it — PowerShell: `.\frac\Scripts\Activate.ps1`,
cmd: `.\frac\Scripts\activate.bat`, bash/Linux/macOS: `source frac/bin/activate` —
then:

```bash
pip install -r requirements.txt
```

`requirements.txt` pulls the **CUDA** build of torch from the PyTorch index
(~2.5 GB). On a machine without an NVIDIA GPU, delete the first two
non-comment lines of that file and `pip install torch` instead; everything
still runs, just slower (see *Running on CPU* below).

> **If `python -m venv frac` fails with**
> `Unable to copy ...\venvlauncher.exe to ...\frac\Scripts\python.exe`:
> the venv already exists and a process is running from it, so Windows has
> `python.exe` locked. Nothing is wrong — skip the step and just activate it.
> For a deliberate clean rebuild, stop anything using the venv (including
> Jupyter kernels) and run `python -m venv --clear frac`.

**2. Check the numerics, then build the data** (~1 min, then ~30 s, 82 MB)

```bash
python -m src.validate
```

```bash
python -m src.datagen
```

**3. Train, benchmark, analyse, plot** (~1.5 h on an RTX 4060 Ti)

Run these in order (separate commands, so they work the same in PowerShell,
cmd and bash):

```bash
python -m src.train --all
```

```bash
python -m src.bench_classical
```

```bash
python -m src.bench
```

```bash
python -m src.analyze
```

```bash
python -m src.analyze_kernels
```

```bash
python -m src.figures
```

```bash
python -m src.report
```

Figures land in `results/`, tables in `results/tables.md`.

**4. Or just read the notebooks**

```bash
jupyter notebook notebooks/
```

`01_theory_and_data.ipynb` derives the operator and validates the solver and
Jiang's rule — it needs only step 2. `02_results.ipynb` renders every table and
figure from the saved json and needs step 3 (or the committed `results/`).

Launched from the activated venv, the default kernel is already the right one.
The venv is also registered as a named kernel (**Python (fracrelax)**) so the
notebooks work from any Jupyter; re-register after a rebuild with

```bash
python -m ipykernel install --user --name fracrelax --display-name "Python (fracrelax)"
```

> Run each `python -m src.*` command from the **project root**, not from `src/`.
>
> `src.bench_classical` and `src.figures` are split out as torch-free modules.
> In the `frac` venv above that is merely tidy, but in an **Anaconda** environment
> it is required: torch's `libiomp5md.dll` and Anaconda's MKL cannot both
> initialise in one process (OMP error #15). Verified both ways on this
> machine.

**Running on CPU.** Everything works, but `src.train --all` goes from ~1.5 h to
the better part of a day. To sample the pipeline instead, cut the budget:
`python -m src.train --model soe_jiang --steps 4000`, then
`python -m src.bench --models soe_jiang --device cpu`. The numerics
(`src.validate`), the data generation and both notebooks never touch the GPU.

**Skipping training.** `results/` is committed, so `python -m src.figures` and
`python -m src.report` reproduce every figure and table in this README without
re-running anything.

---

## 1. Why this operator is the right test case

With $y(0)=0$ the map is a causal convolution,

$$y(t)=\int_0^t K(t-r)\,f(r)\,dr, \qquad K(t)=t^{\alpha-1}E_{\alpha,\alpha}(-\lambda t^{\alpha}), \qquad \hat K(p)=\frac{1}{p^{\alpha}+\lambda}.$$

$\hat K$ is a Stieltjes function, so $K$ is completely monotone and has an exact
representation as a **continuous superposition of exponentials**

$$K(t)=\int_0^{\infty} e^{-st}\,\mu(s)\,ds, \qquad
\mu(s)=\frac{1}{\pi}\cdot\frac{s^{\alpha}\sin(\pi\alpha)}{s^{2\alpha}+2\lambda s^{\alpha}\cos(\pi\alpha)+\lambda^{2}}.$$

The two limits of $\mu$ give the two power laws

$$K(t)\sim \frac{t^{\alpha-1}}{\Gamma(\alpha)}\ (t\to 0), \qquad\qquad
K(t)\sim \frac{\alpha}{\Gamma(1-\alpha)\,\lambda^{2}}\,t^{-1-\alpha}\ (t\to\infty).$$

So the memory is **algebraic, never exponential**: there is no single time
constant, and no finite context window is ever "enough". That is what makes the
train-short/test-long question sharp, and it is why the problem has a clean
theoretical answer to compare a network against.

**The bridge to sequence models.** Discretising the $s$-integral turns one node
$s_j$ into one first-order mode $h_j' = -s_j h_j + f$ — that is, one state of a
diagonal SSM with pole $A_j=e^{-s_j\tau}$. Choosing good nodes is exactly what
**Jiang's formula** does, and learning poles is exactly what **S4D** does. The
two are the same object, so they can be compared directly (figure 1).

### Jiang's formula

[Jiang, Zhang, Zhang & Zhang, *Commun. Comput. Phys.* **21**(3) 650–678
(2017)](https://doi.org/10.4208/cicp.OA-2016-0136). For $0<\beta<1$,

$$t^{-\beta}=\frac{1}{\Gamma(\beta)}\int_0^{\infty}e^{-st}s^{\beta-1}\,ds
\;\approx\;\sum_{j=1}^{N_{\exp}} w_j e^{-s_j t},\qquad t\in[\tau,T],$$

the quadrature being Gauss–Jacobi with weight $s^{\beta-1}$ on $[0,s_{\min}]$
(absorbing the singularity exactly), Gauss–Legendre on geometric panels up to
$s_{\max}$, and truncation beyond. The node count is
$N_{\exp}=O\!\big(\log\tfrac1\varepsilon \cdot \log\tfrac{T}{\tau}\big)$ —
**logarithmic in the dynamic range**, which is why 64 exponentials cover four
decades of time to roughly $10^{-9}$ (`src/jiang.py`, verified in
`src/validate.py`). Here the kernel exponent is $\beta=1-\alpha$.

---

## 2. Ground truth: the L1 scheme

On a uniform grid $t_n=n\tau$ the L1 approximation of the Caputo derivative is

$$D^{\alpha}y(t_n)\approx\frac{\tau^{-\alpha}}{\Gamma(2-\alpha)}\sum_{k=0}^{n-1}b_k\,(y_{n-k}-y_{n-k-1}),\qquad b_k=(k+1)^{1-\alpha}-k^{1-\alpha},$$

which rearranges into the explicit, unconditionally positive recursion used in
`src/l1.py`:

$$(1+\mu\lambda)\,y_n=\sum_{m=1}^{n-1} d_m\,y_{n-m}+b_{n-1}y_0+\mu f_n,
\qquad \mu=\tau^{\alpha}\Gamma(2-\alpha),\quad d_m=b_{m-1}-b_m>0,$$

with $\sum_{m=1}^{n-1}d_m+b_{n-1}=1$.

**Validation** (`python -m src.validate`, full output in
`results/validation.json`). `src/exact.py` computes references from the spectral
representation above and never touches the L1 scheme, so the agreement is
meaningful:

| check | result |
|---|---|
| $\int \mu(s)/(p+s)\,ds = 1/(p^{\alpha}+\lambda)$ | max rel. err. $3.6\times10^{-16}$ |
| L1 vs exact step response ($f\equiv1$, $\alpha=0.5$) | converges, observed order $1.00$ |
| L1 recursion vs FFT convolution (LTI property) | $5.0\times10^{-16}$ |
| L1 impulse response $g[n]$ vs exact $K(n\tau)$ | $+70\%$ at $t=\tau$, $6\times10^{-4}$ at $t=24$ |
| Jiang rule for $t^{-\beta}$, $T/\tau=10^4$, $\varepsilon=10^{-6}$ | 160 nodes, achieved $9.6\times10^{-7}$ |
| 64 Jiang nodes + LS weights vs the L1 kernel | $\le 2\times10^{-8}$ relative, every $\alpha$ |

The last two rows matter for the whole study. The $+70\%$ deviation at the very
first step is *correct*: $K$ has an integrable singularity at $t=0$ that no
one-step scheme resolves. **The L1 solution is the ground truth the models are
trained against**, so this is a property of the target, not an error in it.

Because $y(0)=0$ makes the scheme exactly linear and time-invariant, the whole
dataset could be generated by FFT convolution with the scheme's own impulse
response — $O(L\log L)$ instead of $O(L^2)$, verified to $5\times10^{-16}$
against the direct recursion.

---

## 3. Data (task 1)

`python -m src.datagen` → `data/*.npz` (~82 MB, git-ignored; `data/demo_small.npz`
is committed).

* $\tau=0.01$ is **fixed** and $\lambda=1$. A longer sequence therefore means a
  longer *physical horizon* $T=L\tau$ and more memory to resolve — "train short,
  test long" is a statement about the kernel, not a change of units.
* $\alpha\in\{0.2,0.3,\dots,0.9\}$ for training; $\{0.25,0.55,0.85\}$ held out.
* Eight forcing families with **length-independent statistics** (rates and
  frequencies in physical units, not per-window): `white`, `ou`, `smooth`,
  `steps`, `bursts`, `chirp`, `ramp_noise`, `mix`. $f$ is normalised to unit RMS
  per trajectory; the resulting $y$ has RMS 0.41–0.55 across $\alpha$.

| file | contents |
|---|---|
| `train_L1000.npz` / `val_L1000.npz` / `test_L1000.npz` | 3200 / 400 / 512 sequences, $L=1000$ |
| `test_L10000.npz` | 384 sequences, $L=10000$ — the long-sequence test set |
| `scale_L{1000,2000,4000,7000,10000}.npz` | 128 each, for the scaling sweep |
| `heldout_alpha_L10000.npz` | 96 sequences at unseen $\alpha$ |
| `mixed_lengths.npz` | 240 sequences of **random length in [1000, 10000]** (the literal task spec), zero-padded with an explicit `length` array |
| `kernels_L10000.npz` | the exact L1 impulse response per $\alpha$ |

---

## 4. Models (task 2)

All four are strictly causal (verified by perturbing the last input and checking
that earlier outputs do not move), take $(f,\alpha)$, and use the same
zero-order-hold convention $A=e^{-s\tau}$, $B=(1-A)/s$ — so a learned rate $s_j$
and a Jiang node $s_j$ are the *same kind of object*, a relaxation rate in
1/time, and can be plotted on one axis.

| model | what is learned | params |
|---|---|---|
| **`soe_jiang`** — fixed exponents | 64 poles **frozen** at Jiang nodes; only the readout $c_j(\alpha)$ trains | 3,296 |
| **`s4d_lin_*`** — same architecture, learned poles | $c_j(\alpha)$ **and** the 64 rates; three initialisations (`log`, `lin`, `narrow`) and two rate parameterisations | 3,360 |
| **`s4d_deep`** — the S4D entry | 2 blocks of [pre-norm → diagonal SSM (64 states × 64 channels) → GELU → MLP], residual, FiLM conditioning on $\alpha$ | 68,129 |
| **`transformer_*`** — the Transformer entry | 2 layers, 4 heads, $d_{\text{model}}=64$, causal; `rope` / `nope` / `alibi` | 76,577 |

`s4d_lin_*` is deliberately the *same hypothesis class* as `soe_jiang` with the
same budget — only the pole locations differ. That is what makes question 5
answerable rather than confounded by architecture.

Two design points worth stating, because they are the obvious objections:

* **The band is a prior.** `soe_jiang` is built for the deployment horizon
  $T=100$, i.e. it is told which rates a 100-second horizon needs. The SSMs are
  initialised knowing only the *training* horizon, $s\in[1/T_{\text{train}},1/\tau]=[0.1,100]$.
  The ablation `soe_jiang_trainband` gives the fixed-exponent model the narrow
  band too, which isolates how much of its advantage is the band prior.
* **Weight decay is off for every model.** The target is a linear operator whose
  optimal readout coefficients are $O(1\text{–}30)$; an $L_2$ pull toward zero
  would penalise exactly the structured models. Over-fitting is controlled by
  keeping the best-on-validation checkpoint.

Training: identical data, 40,000 steps, batch 32, AdamW, cosine schedule, and a
scale-free loss $\frac1B\sum_b \lVert \hat y_b-y_b\rVert^2/\lVert y_b\rVert^2$
(needed because $\lVert y \rVert$ varies by an order of magnitude across forcing
families).

---

## 5. Repository

```
README.md          this file — problem formulation and results
notebooks/
  01_theory_and_data.ipynb   the operator, solver validation, Jiang's rule, the data
  02_results.ipynb           every table and figure, from the saved json
src/
  l1.py          L1 scheme (direct recursion + the LTI/FFT shortcut)
  exact.py       spectral representation: exact kernel, step response, Mittag-Leffler
  jiang.py       Jiang's composite Gauss rule, SOE fitting and application
  bands.py       relaxation-rate bands and the cached node set (torch-free)
  datagen.py     dataset generation
  models.py      SOEFixed, S4DLinear, S4D, TinyTransformer
  registry.py    the 11 trained configurations
  train.py  bench.py  bench_classical.py  analyze.py  analyze_kernels.py
  figures.py  report.py  style.py  validate.py
data/              .npz datasets (git-ignored except the demo; regenerate below)
frac/              the virtualenv (git-ignored)
results/           figures, benchmark json, tables
```

What each stage produces (commands in *Quick start* above):

| stage | writes | cost |
|---|---|---|
| `src.validate` | `results/validation.json` | ~1 min, CPU |
| `src.datagen` | `data/*.npz`, 82 MB | ~30 s, CPU |
| `src.train --all` | `data/checkpoints/*.pt`, `results/train_*.json` | ~1.5 h, GPU |
| `src.bench_classical` | `results/bench_classical.json` | ~2 min, CPU |
| `src.bench` | `results/bench.json` | ~10 min, GPU |
| `src.analyze` + `src.analyze_kernels` | `results/analysis.json` | ~3 min |
| `src.figures` | `results/fig*.png` | ~2 min, CPU |
| `src.report` | `results/tables.md` | instant |

---
