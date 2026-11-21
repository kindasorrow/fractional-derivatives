import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
import time
import warnings
import csv

warnings.filterwarnings("ignore")

# ============================================================
# 1. ЯДРО И СЕТКА
# ============================================================
delta, T = 1e-3, 1.0
t = np.logspace(np.log10(delta), np.log10(T), 300)
alphas = np.round(np.arange(0.1, 0.95, 0.1), 1)
kernels = {a: t ** (-1.0 - a) for a in alphas}

S_MIN = 0.1
S_MAX = 10000.0   # вернули широкий диапазон

# ============================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ
# ============================================================
def eval_soe(t, w, s):
    return (w[None, :] * np.exp(-t[:, None] * s[None, :])).sum(axis=1)

def max_rel_error(t, k, w, s):
    k_fit = eval_soe(t, w, s)
    return np.max(np.abs(k_fit - k) / k)

# ============================================================
# 3. СПОСОБ A: least_squares
# ============================================================
def fit_least_squares(t, k, M):
    if M < 1:
        return None, None

    def resid(params):
        w, s = params[:M], params[M:]
        return (eval_soe(t, w, s) - k) / k

    s0 = np.logspace(np.log10(S_MIN), np.log10(S_MAX), M)
    A = np.exp(-t[:, None] * s0[None, :])
    try:
        w0, *_ = np.linalg.lstsq(A / np.sqrt(k)[:, None], k / np.sqrt(k), rcond=None)
    except Exception:
        w0 = np.ones(M) * k.mean()
    w0 = np.maximum(w0, 1e-6)
    p0 = np.concatenate([w0, s0])

    lb = np.concatenate([np.full(M, 1e-14), np.full(M, 1e-14)])
    ub = np.concatenate([np.full(M, 1e12), np.full(M, 1e6)])

    try:
        sol = least_squares(resid, p0, bounds=(lb, ub),
                            xtol=1e-10, ftol=1e-10, gtol=1e-10,
                            max_nfev=2000)
        return sol.x[:M], sol.x[M:]
    except Exception:
        return None, None

# ============================================================
# 4. СПОСОБ B: fixed nodes + лог-подгонка
# ============================================================
def fit_fixed_nodes(t, k, M):
    """
    s_k фиксированы на лог-сетке.
    Стартовые веса — из lstsq, затем нелинейная подгонка в лог-пространстве.
    """
    if M < 1:
        return None, None
    s = np.logspace(np.log10(S_MIN), np.log10(S_MAX), M)
    A = np.exp(-t[:, None] * s[None, :])

    # стартовые веса: lstsq с весом 1/sqrt(k)
    sqrt_k = np.sqrt(k)
    try:
        w0, *_ = np.linalg.lstsq(A / sqrt_k[:, None], k / sqrt_k, rcond=None)
    except Exception:
        return None, None
    w0 = np.maximum(np.abs(w0), 1e-10)

    # лог-подгонка: минимизируем log(k_fit) - log(k)
    def resid(log_w):
        w = np.exp(log_w)
        k_fit = eval_soe(t, w, s)
        k_fit = np.maximum(k_fit, 1e-300)
        return np.log(k_fit) - np.log(k)

    try:
        sol = least_squares(resid, np.log(w0), max_nfev=2000)
        w = np.exp(sol.x)
    except Exception:
        return None, None
    return w, s

# ============================================================
# 5. СПОСОБ C: curve_fit
# ============================================================
def fit_curve_fit(t, k, M):
    if M < 1:
        return None, None

    def soe_func(t_, *params):
        w, s = params[:M], params[M:]
        return eval_soe(t_, np.array(w), np.array(s))

    s0 = np.logspace(np.log10(S_MIN), np.log10(S_MAX), M)
    A = np.exp(-t[:, None] * s0[None, :])
    try:
        w0, *_ = np.linalg.lstsq(A / np.sqrt(k)[:, None], k / np.sqrt(k), rcond=None)
    except Exception:
        w0 = np.ones(M) * k.mean()
    w0 = np.maximum(w0, 1e-6)
    p0 = np.concatenate([w0, s0])

    lb = np.concatenate([np.full(M, 1e-14), np.full(M, 1e-14)])
    ub = np.concatenate([np.full(M, 1e12), np.full(M, 1e6)])

    try:
        popt, _ = curve_fit(soe_func, t, k, p0=p0, bounds=(lb, ub),
                            maxfev=3000, method="trf")
        return popt[:M], popt[M:]
    except Exception:
        return None, None

# ============================================================
# 6. Jiang (заглушка)
# ============================================================
def fit_jiang(t, k, M):
    return None, None

# ============================================================
# 7. Поиск минимального M
# ============================================================
def find_min_M(fitter, t, k, M_max=20, epsilons=(1e-2, 1e-4, 1e-6)):
    result = {eps: None for eps in epsilons}
    for M in range(1, M_max + 1):
        w, s = fitter(t, k, M)
        if w is None:
            continue
        err = max_rel_error(t, k, w, s)
        if not np.isfinite(err):
            continue
        for eps in epsilons:
            if result[eps] is None and err < eps:
                result[eps] = M
        if all(v is not None for v in result.values()):
            break
    return result

# ============================================================
# 8. Основной цикл
# ============================================================
methods = {
    "least_squares": fit_least_squares,
    "fixed_nodes":   fit_fixed_nodes,
    "curve_fit":     fit_curve_fit,
    "jiang":         fit_jiang,
}
epsilons = (1e-2, 1e-4, 1e-6)

print("Запуск бенчмарка...")
print(f"{'Метод':<16} {'alpha':<6} {'ε=1e-2':<8} {'ε=1e-4':<8} {'ε=1e-6':<8}")
print("-" * 52)

results = {}
t_start = time.time()
for name, fitter in methods.items():
    for a in alphas:
        k = kernels[a]
        res = find_min_M(fitter, t, k, M_max=20, epsilons=epsilons)
        results[(name, a)] = res
        row = [f"{res[e] if res[e] is not None else '-'}" for e in epsilons]
        print(f"{name:<16} {a:<6.1f} {row[0]:<8} {row[1]:<8} {row[2]:<8}")
    print()
print(f"Время: {time.time() - t_start:.1f} с")

# ============================================================
# 9. Графики
# ============================================================

# (а) M(α) при фиксированном ε
for eps in epsilons:
    plt.figure(figsize=(8, 5))
    for name in methods:
        ys = [results[(name, a)][eps] for a in alphas]
        ys = [y if y is not None else np.nan for y in ys]
        plt.plot(alphas, ys, "o-", label=name, markersize=7)
    plt.xlabel(r"\(\alpha\)", fontsize=12)
    plt.ylabel("Минимальное M", fontsize=12)
    plt.title(fr"M(\(\alpha\)) при \(\varepsilon = {eps:.0e}\)")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"M_vs_alpha_eps_{eps:.0e}.png", dpi=150)
    plt.close()

# (б) M(ε) при фиксированном α
alpha_fix = 0.5
plt.figure(figsize=(8, 5))
for name in methods:
    ys = []
    for eps in epsilons:
        m = results[(name, alpha_fix)][eps]
        ys.append(m if m is not None else np.nan)
    plt.plot(epsilons, ys, "s-", label=name, markersize=8)
plt.xscale("log")
plt.xlabel(r"\(\varepsilon\)", fontsize=12)
plt.ylabel("Минимальное M", fontsize=12)
plt.title(fr"M(\(\varepsilon\)) при \(\alpha = {alpha_fix}\)")
plt.grid(True, which="both", ls=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("M_vs_eps.png", dpi=150)
plt.close()

# (в) Ошибка по t для одного примера
alpha_demo = 0.5
M_demo = 8
k_demo = kernels[alpha_demo]
t_fine = np.logspace(np.log10(delta), np.log10(T), 2000)
k_fine = t_fine ** (-1.0 - alpha_demo)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, fitter) in zip(axes, methods.items()):
    w, s = fitter(t, k_demo, M_demo)
    if w is None:
        ax.text(0.5, 0.5, "не удалось", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title(name)
        continue
    k_fit_fine = eval_soe(t_fine, w, s)
    rel_err = np.abs(k_fit_fine - k_fine) / k_fine
    ax.loglog(t_fine, rel_err, "b-", lw=2)
    for eps in epsilons:
        ax.axhline(eps, color="r", ls="--", alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("Отн. ошибка")
    ax.set_title(f"{name}\n(M={M_demo}, α={alpha_demo})")
    ax.grid(True, which="both", ls=":", alpha=0.6)

fig.suptitle("Ошибка аппроксимации по t", fontsize=14)
fig.tight_layout()
fig.savefig("error_vs_t.png", dpi=150)
plt.close()

# (г) Ядро vs приближения
fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(t_fine, k_fine, "k-", lw=2.5, label=r"$k(t)=t^{-1.5}$")
colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
for (name, fitter), c in zip(methods.items(), colors):
    w, s = fitter(t, k_demo, M_demo)
    if w is None:
        continue
    k_fit_fine = eval_soe(t_fine, w, s)
    ax.loglog(t_fine, k_fit_fine, "--", lw=1.8, color=c, label=f"{name}, M={M_demo}")
ax.set_xlabel("t")
ax.set_ylabel("k(t)")
ax.set_title(fr"Ядро и приближения (α = {alpha_demo}, M = {M_demo})")
ax.grid(True, which="both", ls=":", alpha=0.6)
ax.legend()
fig.tight_layout()
fig.savefig("kernel_vs_approx.png", dpi=150)
plt.close()

# (д) CSV
with open("benchmark_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "alpha", "eps=1e-2", "eps=1e-4", "eps=1e-6"])
    for name in methods:
        for a in alphas:
            res = results[(name, a)]
            writer.writerow([name, a] + [res[e] if res[e] is not None else ""
                                          for e in epsilons])

print("\nСохранено:")
print("  benchmark_results.csv")
print("  M_vs_alpha_eps_1e-02.png, M_vs_alpha_eps_1e-04.png, M_vs_alpha_eps_1e-06.png")
print("  M_vs_eps.png")
print("  error_vs_t.png")
print("  kernel_vs_approx.png")