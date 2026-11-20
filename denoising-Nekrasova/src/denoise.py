"""Сравнение классической диффузии Перона-Малика и дробной схемы
Грюнвальда-Летникова с конечной маской

Для каждой картинки отдельно подбираются число итераций и порог kappa
для Перона-Малика и для каждого порядка alpha. Сравнивается лучшее с лучшим
Картинки сравнения и графики сохраняются в jpg

Данные ищутся сами в папке data (пж распакуйте с ними архив перед запуском кода)
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Sample:
    dataset: str
    name: str
    clean: np.ndarray       # H x W x 3, float32, range [0, 255]
    noisy: np.ndarray       # H x W x 3, float32, range [0, 255]
    sigma: float
    pm: np.ndarray | None = None
    pm_iters: int | None = None
    pm_kappa: float | None = None
    frac_by_alpha: dict[float, np.ndarray] = field(default_factory=dict)
    frac_params: dict[float, tuple[int, float]] = field(default_factory=dict)


def discover_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def choose_files(root: Path, count: int, seed: int) -> list[Path]:
    files = discover_images(root)
    if len(files) < count:
        raise RuntimeError(f"Found only {len(files)} images in {root}; need {count}.")
    rng = random.Random(seed)
    return sorted(rng.sample(files, count))


def read_rgb(path: Path, max_side: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def add_gaussian_noise(clean: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """ Gaussian noise and clip to [0, 255].

    
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, clean.shape).astype(np.float32)
    return np.clip(clean + noise, 0.0, 255.0).astype(np.float32)


def shift_prev(x: torch.Tensor, k: int, dim: str) -> torch.Tensor:
    """output[i] = x[i-k]"""
    if dim == "x":
        return F.pad(x, (k, 0, 0, 0), mode="replicate")[..., :x.shape[-1]]
    return F.pad(x, (0, 0, k, 0), mode="replicate")[..., :x.shape[-2], :]


def shift_next(x: torch.Tensor, k: int, dim: str) -> torch.Tensor:
    """ output[i] = x[i+k]"""
    if dim == "x":
        return F.pad(x, (0, k, 0, 0), mode="replicate")[..., k:k + x.shape[-1]]
    return F.pad(x, (0, 0, 0, k), mode="replicate")[..., k:k + x.shape[-2], :]


def perona_malik_repo_style(image: np.ndarray, iterations: int = 30,
                            dt: float = 0.14, kappa: float = 30.0) -> np.ndarray:
    """Port of the 8-neighbour explicit scheme in fubel/PeronaMalikDiffusion.

    Conductance is evaluated independently for each of the eight directions
    (cardinal weight 1, diagonal weight0.5).
    """
    u = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous().float()
    directions = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 0.5), (-1, 1, 0.5), (1, -1, 0.5), (1, 1, 0.5),
    ]
    for _ in range(iterations):
        terms = torch.zeros_like(u)
        for dy, dx, factor in directions:
            if dx == -1:
                nb = shift_prev(u, 1, "x")
            elif dx == 1:
                nb = shift_next(u, 1, "x")
            else:
                nb = u
            if dy == -1:
                nb = shift_prev(nb, 1, "y")
            elif dy == 1:
                nb = shift_next(nb, 1, "y")
            diff = nb - u
            magnitude = torch.mean(diff.square(), dim=0, keepdim=True).sqrt()
            conductance = 1.0 / (1.0 + (magnitude / kappa).square())
            terms.add_(conductance * diff, alpha=factor)
        u = u + dt * terms
        u.clamp_(0.0, 255.0)
    return u.permute(1, 2, 0).cpu().numpy()


def gl_weights(alpha: float, K: int) -> torch.Tensor:
    """

    c_0 = 1,
    c_k = c_{k-1} * (1 - (alpha+1)/k) for k = 1..K
    (c_k = (-1)^k * binom(alpha, k)) for 0 < alpha <= 1, c_k <= 0 for k >= 1
    Positive lag weights w_k = -c_k, k = 1..K, 
    sum_k w_k * k = 1.
    
    """
    coeff = [1.0]
    for k in range(1, K + 1):
        coeff.append(coeff[-1] * (1.0 - (alpha + 1.0) / k))
    lag = torch.tensor([-v for v in coeff[1:]], dtype=torch.float32)
    ks = torch.arange(1, K + 1, dtype=torch.float32)
    scale = (lag * ks).sum()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid GL weights for alpha={alpha}, K={K}")
    return lag / scale


def gl_gradient(u: torch.Tensor, weights: torch.Tensor, dim: str) -> torch.Tensor:
    result = torch.zeros_like(u)
    for k, weight in enumerate(weights, start=1):
        result.add_(u - shift_prev(u, k, dim), alpha=float(weight))
    return result


def gl_adjoint(flux: torch.Tensor, weights: torch.Tensor, dim: str) -> torch.Tensor:
    result = torch.zeros_like(flux)
    for k, weight in enumerate(weights, start=1):
        result.add_(flux - shift_next(flux, k, dim), alpha=float(weight))
    return result


def fractional_anisotropic_diffusion(image: np.ndarray, alpha: float, K: int = 5,
                                      iterations: int = 20, dt: float = 0.12,
                                      kappa: float = 30.0) -> np.ndarray:
    u = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous().float()
    weights = gl_weights(alpha, K)
    for _ in range(iterations):
        gx = gl_gradient(u, weights, "x")
        gy = gl_gradient(u, weights, "y")
        magnitude = torch.mean(gx.square() + gy.square(), dim=0, keepdim=True).sqrt()
        conductance = 1.0 / (1.0 + (magnitude / kappa).square())
        fx, fy = conductance * gx, conductance * gy
        divergence_adjoint = gl_adjoint(fx, weights, "x") + gl_adjoint(fy, weights, "y")
        u = u - dt * divergence_adjoint
        u.clamp_(0.0, 255.0)
    return u.permute(1, 2, 0).cpu().numpy()


def metrics(clean: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    est = np.clip(estimate, 0.0, 255.0).astype(np.float32)
    p = float(peak_signal_noise_ratio(clean, est, data_range=255.0))
    s = float(structural_similarity(clean, est, data_range=255.0, channel_axis=-1))
    return p, s


def tune_pm(noisy: np.ndarray, clean: np.ndarray,
            iter_grid: list[int], kappa_grid: list[float]) -> tuple[np.ndarray, int, float, float, float]:
    """Perona-Malik"""
    best_psnr = -1.0
    best_ssim = 0.0
    best_img = noisy
    best_iters = iter_grid[0]
    best_kappa = kappa_grid[0]
    for it in iter_grid:
        for kappa in kappa_grid:
            out = perona_malik_repo_style(noisy, iterations=it, kappa=kappa)
            p, s = metrics(clean, out)
            if p > best_psnr:
                best_psnr, best_ssim = p, s
                best_img = out
                best_iters, best_kappa = it, kappa
    return best_img, best_iters, best_kappa, best_psnr, best_ssim


def tune_frac(noisy: np.ndarray, clean: np.ndarray, alpha: float, K: int,
              iter_grid: list[int], kappa_grid: list[float]) -> tuple[np.ndarray, int, float, float, float]:
    """fixed alpha"""
    best_psnr = -1.0
    best_ssim = 0.0
    best_img = noisy
    best_iters = iter_grid[0]
    best_kappa = kappa_grid[0]
    for it in iter_grid:
        for kappa in kappa_grid:
            out = fractional_anisotropic_diffusion(
                noisy, alpha, K=K, iterations=it, kappa=kappa,
            )
            p, s = metrics(clean, out)
            if p > best_psnr:
                best_psnr, best_ssim = p, s
                best_img = out
                best_iters, best_kappa = it, kappa
    return best_img, best_iters, best_kappa, best_psnr, best_ssim


def add_row(rows: list[dict], sample: Sample, method: str,
            alpha: float | str, estimate: np.ndarray,
            iters: int | str = "", kappa: float | str = "") -> None:
    """Append a metrics row for the given estimate."""
    p, s = metrics(sample.clean, estimate)
    rows.append({
        "dataset": sample.dataset,
        "dataset_label": "BSD500 photos" if sample.dataset == "BSD500" else "DTD textures",
        "image": sample.name,
        "sigma": sample.sigma,
        "method": method,
        "alpha": alpha,
        "iterations": iters,
        "kappa": kappa,
        "psnr": p,
        "ssim": s,
    })


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def save_comparison(sample: Sample, alpha: float, out_dir: Path) -> None:
    assert sample.pm is not None and alpha in sample.frac_by_alpha
    frac = sample.frac_by_alpha[alpha]
    titles = [
        "Clean reference",
        f"Gaussian noise sigma={sample.sigma:g}",
        f"Perona-Malik (it={sample.pm_iters}, kappa={sample.pm_kappa:g})",
        f"Fractional GL alpha={alpha:g} (it={sample.frac_params[alpha][0]}, kappa={sample.frac_params[alpha][1]:g})",
    ]
    imgs = [sample.clean, sample.noisy, sample.pm, frac]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
    for ax, title, im in zip(axes, titles, imgs):
        ax.imshow(to_uint8(im))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    name = Path(sample.name).stem
    fig.suptitle(f"{sample.dataset} sigma={sample.sigma:g}: {name}", fontsize=11)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    fig.savefig(
        out_dir / f"comparison_{sample.dataset}_s{sample.sigma:g}_{safe_name}.jpg",
        dpi=140,
        format="jpg",
    )
    plt.close(fig)


def save_curve(rows: list[dict], metric_name: str, sigma: float, out_path: Path) -> None:
    groups = ["BSD500 photos", "DTD textures"]
    colors = {"BSD500 photos": "C0", "DTD textures": "C1"}
    styles = {"BSD500 photos": "o-", "DTD textures": "s-"}

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)

    for group in groups:
        subset = [r for r in rows
                  if r["dataset_label"] == group
                  and r["method"] == "Fractional GL"
                  and float(r["sigma"]) == sigma]
        by_alpha: dict[float, list[float]] = {}
        for r in subset:
            by_alpha.setdefault(float(r["alpha"]), []).append(float(r[metric_name]))
        if not by_alpha:
            continue
        xs = sorted(by_alpha)
        ys = [float(np.mean(by_alpha[x])) for x in xs]
        ax.plot(xs, ys, styles[group], color=colors[group],
                linewidth=2, markersize=5, label=group)

    for group in groups:
        vals = [float(r[metric_name]) for r in rows
                if r["dataset_label"] == group
                and r["method"] == "Noisy"
                and float(r["sigma"]) == sigma]
        if vals:
            ax.axhline(float(np.mean(vals)), color=colors[group],
                       linestyle="--", linewidth=1.2,
                       label=f"{group} — noisy input")

    for group in groups:
        vals = [float(r[metric_name]) for r in rows
                if r["dataset_label"] == group
                and r["method"] == "Perona-Malik"
                and float(r["sigma"]) == sigma]
        if vals:
            ax.axhline(float(np.mean(vals)), color=colors[group],
                       linestyle="-", linewidth=1.5, alpha=0.7,
                       label=f"{group} — best Perona–Malik")

    ax.set_xlabel("Fractional order alpha")
    ax.set_ylabel("PSNR (dB)" if metric_name == "psnr" else "SSIM")
    ax.set_title(f"{metric_name.upper()} vs alpha (sigma={sigma:g}, mean over images, best-tuned)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=160, format="jpg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bsd_dir", type=Path, default=Path("./data/bsd"),
                        help="Folder containing BSDS500 images (preferably test split)")
    parser.add_argument("--dtd_dir", type=Path, default=Path("./data/dtd"),
                        help="Folder containing DTD images (preferably test split)")
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    parser.add_argument("--count", type=int, default=5,
                        help="Images per dataset (default: 5)")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[25.0],
                        help="One or more Gaussian noise standard deviations (default: 25)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_side", type=int, default=384,
                        help="Resize longest image side to at most this size")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--K", type=int, default=5, help="Number of GL lag weights")
    parser.add_argument("--show_alpha", type=float, default=0.8,
                        help="Fixed alpha used in side-by-side comparison images")
    parser.add_argument("--pm_iters", type=int, nargs="+",
                        default=[5, 10, 15, 25, 35],
                        help="Iteration grid for Perona–Malik")
    parser.add_argument("--pm_kappas", type=float, nargs="+",
                        default=[15.0, 20.0, 25.0, 35.0, 50.0],
                        help="kappa grid for Perona-Malik")
    parser.add_argument("--frac_iters", type=int, nargs="+",
                        default=[5, 10, 20, 30],
                        help="Iteration grid for fractional GL (per alpha)")
    parser.add_argument("--frac_kappas", type=float, nargs="+",
                        default=[15.0, 20.0, 25.0, 35.0, 50.0],
                        help="kappa grid for fractional GL (per alpha)")
    args = parser.parse_args()

    if args.count < 1 or args.max_side < 16 or args.K < 1:
        parser.error("count and K must be >=1; max_side must be >=16")
    if not args.alphas or any(not (0.0 < a <= 1.0) for a in args.alphas):
        parser.error("all alpha values must be in (0, 1]")
    if args.show_alpha not in args.alphas:
        parser.error("--show_alpha must be one of the values supplied to --alphas")
    if not args.sigmas or any(s <= 0 for s in args.sigmas):
        parser.error("all sigma values must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bsd_files = choose_files(args.bsd_dir, args.count, args.seed)
    dtd_files = choose_files(args.dtd_dir, args.count, args.seed + 1)

    rows: list[dict] = []
    all_samples: list[Sample] = []

    print(f"Selected {args.count} BSD500 photos and {args.count} DTD textures.")
    print(f"Sigmas: {args.sigmas}")
    print(f"Alphas: {args.alphas}, K={args.K}")
    print(f"PM grid: iters={args.pm_iters}, kappa={args.pm_kappas}")
    print(f"Frac grid (per alpha): iters={args.frac_iters}, kappa={args.frac_kappas}")
    print("Tuning is independent: best (iters, kappa) for PM vs best (iters, kappa) for each alpha.\n")

    for sigma in args.sigmas:
        print(f"=== sigma = {sigma:g} ===")
        samples: list[Sample] = []
        for dataset, paths in [("BSD500", bsd_files), ("DTD", dtd_files)]:
            for i, path in enumerate(paths):
                clean = read_rgb(path, args.max_side)
                noisy = add_gaussian_noise(
                    clean, sigma,
                    args.seed + 1000 * (1 if dataset == "BSD500" else 2) + i + int(sigma * 10),
                )
                samples.append(Sample(dataset, path.name, clean, noisy, sigma=sigma))

        for idx, sample in enumerate(samples, start=1):
            print(f"  [{idx}/{len(samples)}] {sample.dataset}/{sample.name}", flush=True)

            # Noisy baseline
            add_row(rows, sample, "Noisy", "", sample.noisy)

            # --- Best Perona-Malik (own iters + kappa) ---
            pm_img, pm_it, pm_k, pm_p, pm_s = tune_pm(
                sample.noisy, sample.clean, args.pm_iters, args.pm_kappas,
            )
            sample.pm = pm_img
            sample.pm_iters = pm_it
            sample.pm_kappa = pm_k
            add_row(rows, sample, "Perona-Malik", "", pm_img, iters=pm_it, kappa=pm_k)
            print(f"      PM best: it={pm_it}, kappa={pm_k:g} -> PSNR={pm_p:.2f} SSIM={pm_s:.4f}")

            # --- Best fractional for each alpha (own iters + kappa) ---
            for alpha in args.alphas:
                frac_img, f_it, f_k, f_p, f_s = tune_frac(
                    sample.noisy, sample.clean, alpha, args.K,
                    args.frac_iters, args.frac_kappas,
                )
                sample.frac_by_alpha[alpha] = frac_img
                sample.frac_params[alpha] = (f_it, f_k)
                add_row(rows, sample, "Fractional GL", alpha, frac_img,
                        iters=f_it, kappa=f_k)
                print(f"      alpha={alpha:g} best: it={f_it}, kappa={f_k:g} -> PSNR={f_p:.2f} SSIM={f_s:.4f}")

            save_comparison(sample, args.show_alpha, args.out_dir)
            all_samples.append(sample)

        # Curves for this sigma
        save_curve(rows, "psnr", sigma, args.out_dir / f"psnr_vs_alpha_s{sigma:g}.jpg")
        save_curve(rows, "ssim", sigma, args.out_dir / f"ssim_vs_alpha_s{sigma:g}.jpg")

  
    csv_path = args.out_dir / "metrics.csv"
    fieldnames = [
        "dataset", "dataset_label", "image", "sigma", "method", "alpha",
        "iterations", "kappa", "psnr", "ssim",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary: best-vs-best means
    print("\n" + "=" * 72)
    print("Mean metrics (best-tuned parameters, averaged over images)")
    print("=" * 72)
    for sigma in args.sigmas:
        print(f"\n--- sigma = {sigma:g} ---")
        for dlabel in ["BSD500 photos", "DTD textures"]:
            print(f"  {dlabel}")
            # Noisy
            subset = [r for r in rows
                      if r["dataset_label"] == dlabel
                      and r["method"] == "Noisy"
                      and float(r["sigma"]) == sigma]
            if subset:
                print(f"    Noisy            : PSNR={np.mean([r['psnr'] for r in subset]):6.2f}  "
                      f"SSIM={np.mean([r['ssim'] for r in subset]):.4f}")

            # Best PM
            subset = [r for r in rows
                      if r["dataset_label"] == dlabel
                      and r["method"] == "Perona-Malik"
                      and float(r["sigma"]) == sigma]
            if subset:
                print(f"    Best Perona-Malik: PSNR={np.mean([r['psnr'] for r in subset]):6.2f}  "
                      f"SSIM={np.mean([r['ssim'] for r in subset]):.4f}")

            # Best fractional per alpha
            for alpha in args.alphas:
                subset = [r for r in rows
                          if r["dataset_label"] == dlabel
                          and r["method"] == "Fractional GL"
                          and float(r["alpha"]) == alpha
                          and float(r["sigma"]) == sigma]
                if subset:
                    print(f"    Best Frac alpha={alpha:g}  : PSNR={np.mean([r['psnr'] for r in subset]):6.2f}  "
                          f"SSIM={np.mean([r['ssim'] for r in subset]):.4f}")

    print("\nNote: fixed large iteration counts over-smooth classical PM and make")
    print("fractional look better by accident. With independent tuning the fair")
    print("comparison is best-PM vs best-fractional; on Gaussian noise a well-tuned")
    print("PM is typically at least as good as any alpha (alpha=1 recovers ordinary PM up")
    print("to the stencil difference).")
    print(f"\nSaved: {csv_path}")
    print(f"       psnr_vs_alpha_s*.jpg, ssim_vs_alpha_s*.jpg")
    print(f"       comparison_*.jpg    {args.out_dir}")


if __name__ == "__main__":
    main()
