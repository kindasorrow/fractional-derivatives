import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .fno import FNO1d

ROOT = Path(__file__).resolve().parents[1]


def split_training_ids(split, seed=20261028, validation_fraction=.1):
    ids = np.flatnonzero(np.asarray(split) == 'train')
    if len(ids) < 2 or not 0 < validation_fraction < 1:
        raise ValueError('at least two train examples and a validation fraction in (0,1) are required')
    ids = np.random.default_rng(seed).permutation(ids)
    n_validation = min(len(ids)-1, max(1, round(len(ids)*validation_fraction)))
    return np.sort(ids[n_validation:]), np.sort(ids[:n_validation])


def fit_normalization(outputs, fit_ids):
    training = np.asarray(outputs)[fit_ids]
    if training.ndim != 3 or len(training) == 0 or not np.isfinite(training).all():
        raise ValueError('normalization requires finite fitting data')
    return (training.mean(axis=(0, 2)).astype(np.float32)[:, None],
            np.maximum(training.std(axis=(0, 2)), 1e-8).astype(np.float32)[:, None])


def decode_output(normalized, mean, scale):
    physical = normalized * scale + mean
    boundary = torch.zeros_like(physical[:, :1])
    return torch.cat((boundary, physical[:, 1:-1], boundary), dim=1)


def _dataset(data, ids, mean, scale):
    n_times = len(data['t'])
    sample_ids = np.repeat(ids, n_times)
    time_ids = np.tile(np.arange(n_times), len(ids))
    values = (data['u0'][sample_ids], data['alpha'][sample_ids], data['t'][time_ids],
              data['u'][sample_ids, time_ids], mean[time_ids], scale[time_ids])
    return TensorDataset(*(torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in values))


def _relative_squared(prediction, target):
    return torch.sum((prediction-target)**2, dim=-1) / torch.sum(target**2, dim=-1).clamp_min(1e-16)


def _evaluate_loss(model, loader, device):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for batch in loader:
            initial, alpha, times, target, mean, scale = (v.to(device) for v in batch)
            prediction = decode_output(model(initial, alpha, times), mean, scale)
            losses = _relative_squared(prediction, target)
            total += losses.sum().item()
            count += len(losses)
    return total/count


def error_metrics(prediction, target):
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    residual = prediction-target
    norms = np.linalg.norm(target, axis=-1)
    relative = np.linalg.norm(residual, axis=-1) / np.maximum(norms, 1e-16)
    return dict(mean_relative_l2=float(relative.mean()), median_relative_l2=float(np.median(relative)),
                p95_relative_l2=float(np.quantile(relative, .95)), max_relative_l2=float(relative.max()),
                aggregate_relative_l2=float(np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-16)),
                mae=float(np.abs(residual).mean()), count=len(target))


def train(data_path=None, output=None, epochs=300, batch_size=64, width=24, modes=24,
          depth=4, lr=.001, seed=20261028, threads=4, device='cpu'):
    if min(epochs, batch_size, threads) < 1 or lr <= 0:
        raise ValueError('epochs, batch_size, threads and lr must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(device)
    path = Path(data_path) if data_path else ROOT/'data'/'dataset.npz'
    out = Path(output) if output else ROOT/'results'/'fno'
    out.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ('u0', 'alpha', 'u', 't', 'split', 'x')}
    fit_ids, validation_ids = split_training_ids(data['split'], seed)
    mean, scale = fit_normalization(data['u'], fit_ids)
    fit_data = _dataset(data, fit_ids, mean, scale)
    validation_data = _dataset(data, validation_ids, mean, scale)
    generator = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(fit_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    model = FNO1d(width, modes, depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*.05)
    config = dict(epochs=epochs, batch_size=batch_size, width=width, modes=modes, depth=depth,
                  lr=lr, seed=seed, threads=threads, device=str(device))
    checkpoint = out/'best.pt'
    best_loss, best_epoch = float('inf'), None
    started = time.perf_counter()
    with (out/'history.csv').open('w', newline='', encoding='utf-8') as history:
        writer = csv.DictWriter(history, fieldnames=['epoch', 'train_relative_squared_l2',
                                                    'validation_relative_squared_l2', 'lr', 'elapsed_seconds'],
                                lineterminator='\n')
        writer.writeheader()
        for epoch in range(1, epochs+1):
            model.train()
            total, count = 0., 0
            for batch in fit_loader:
                initial, alpha, times, target, batch_mean, batch_scale = (v.to(device) for v in batch)
                optimizer.zero_grad(set_to_none=True)
                predicted = decode_output(model(initial, alpha, times), batch_mean, batch_scale)
                losses = _relative_squared(predicted, target)
                loss = losses.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += losses.detach().sum().item()
                count += len(losses)
            validation_loss = _evaluate_loss(model, validation_loader, device)
            if not np.isfinite(validation_loss):
                raise RuntimeError('non-finite validation loss')
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                torch.save(dict(model_state=model.state_dict(), config=config, epoch=epoch,
                                validation_relative_squared_l2=validation_loss,
                                output_mean=torch.from_numpy(mean), output_scale=torch.from_numpy(scale),
                                times=torch.from_numpy(data['t'].copy()),
                                fit_ids=torch.from_numpy(fit_ids), validation_ids=torch.from_numpy(validation_ids)),
                           checkpoint)
            writer.writerow(dict(epoch=epoch, train_relative_squared_l2=total/count,
                                 validation_relative_squared_l2=validation_loss,
                                 lr=optimizer.param_groups[0]['lr'], elapsed_seconds=time.perf_counter()-started))
            history.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f'Epoch {epoch}/{epochs}: train={total/count:.6g}, validation={validation_loss:.6g}', flush=True)
            scheduler.step()
    selected = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(selected['model_state'])
    model.eval()
    all_data = _dataset(data, np.arange(len(data['u0'])), mean, scale)
    predictions = []
    with torch.no_grad():
        for batch in DataLoader(all_data, batch_size=batch_size):
            initial, alpha, times, _, batch_mean, batch_scale = (v.to(device) for v in batch)
            predictions.append(decode_output(model(initial, alpha, times), batch_mean, batch_scale).cpu().numpy())
    predictions = np.concatenate(predictions).reshape(data['u'].shape)
    groups = dict(train_fit=fit_ids, validation=validation_ids)
    groups.update({str(name): np.flatnonzero(data['split'] == name)
                   for name in np.unique(data['split']) if name != 'train'})
    from .baselines import fit_first_mode, predict_first_mode

    baseline_coefficients = fit_first_mode(data['u0'][fit_ids], data['alpha'][fit_ids], data['u'][fit_ids])
    baseline = predict_first_mode(data['u0'], data['alpha'], baseline_coefficients)
    metrics = {}
    baseline_metrics = {}
    for name, ids in groups.items():
        metrics[name] = {f'{horizon:g}': error_metrics(predictions[ids, j], data['u'][ids, j])
                         for j, horizon in enumerate(data['t'])}
        baseline_metrics[name] = {f'{horizon:g}': error_metrics(baseline[ids, j], data['u'][ids, j])
                                  for j, horizon in enumerate(data['t'])}
    result = dict(config=config, best_epoch=best_epoch, best_validation_relative_squared_l2=best_loss,
                  elapsed_seconds=time.perf_counter()-started, data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  fit_ids=fit_ids.tolist(), validation_ids=validation_ids.tolist(),
                  output_mean=mean.tolist(), output_scale=scale.tolist(),
                  parameter_count=sum(p.numel() for p in model.parameters()), metrics=metrics,
                  first_mode_baseline='degree-5 polynomial alpha-to-damping fit on fitting IDs only',
                  first_mode_baseline_metrics=baseline_metrics)
    (out/'metrics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(out/'predictions.npz', prediction=predictions, target=data['u'],
                        t=data['t'], x=data['x'], alpha=data['alpha'], split=data['split'],
                        fit_ids=fit_ids, validation_ids=validation_ids,
                        first_mode_baseline=baseline, baseline_coefficients=baseline_coefficients)
    print(f'Best epoch: {best_epoch}; results: {out}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--modes', type=int, default=24)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    train(data_path=args.data, output=args.output, epochs=args.epochs, batch_size=args.batch_size,
          width=args.width, modes=args.modes, depth=args.depth, lr=args.lr, seed=args.seed,
          threads=args.threads, device=args.device)
