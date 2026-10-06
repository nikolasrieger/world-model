"""Headless diagnostics on held-out recorded episodes; no simulator execution."""
import json
from pathlib import Path
import numpy as np
import torch
from .adapters import MLPAdapter, LatentAdapter, check_action
from .mlp_model import StateMLP
from .latent_model import LatentModel, load_checkpoint, is_latent_checkpoint


def mse(prediction, target):
    return float((prediction - target).square().mean())


def motion_metrics(prediction, target, previous, latent, ram_metrics=None):
    if latent:
        mask = (target - previous).abs() > 1 / 255
        return {'changed_pixel_mse': mse(prediction[mask], target[mask]) if mask.any() else None,
                'changed_pixel_count': int(mask.sum())}
    return ram_metrics(prediction, target) if ram_metrics is not None else {}


def select_windows(episodes, latent, history, horizon, limit, observation_shape, num_actions):
    windows = []
    for index, episode in enumerate(episodes):
        observations, actions = (episode['frames'], episode['actions']) if latent else episode
        if len(observations) != len(actions) + 1:
            raise ValueError(f'Episode {index}: expected one more observation than actions.')
        expected = tuple(observation_shape)
        if tuple(observations.shape[1:]) != expected:
            raise ValueError(f'Episode {index}: expected observation shape {expected}.')
        check_action(actions, len(actions), num_actions)
        windows.extend((index, start) for start in range(len(observations) - history - horizon + 1))
    if not windows:
        raise ValueError('No validation episodes are long enough for history + horizon.')
    indices = np.linspace(0, len(windows) - 1, min(limit, len(windows)), dtype=int)
    return [windows[i] for i in indices]


@torch.no_grad()
def diagnose_window(adapter, observations, actions, history, horizon, seed, latent, ram_metrics=None):
    generator = torch.Generator().manual_seed(seed)

    def noise(time=None):
        if not latent:
            return None
        shape = (1, adapter.model.latent) if time is None else (1, time, adapter.model.latent)
        uniform = torch.rand(shape, generator=generator).clamp(1e-6, 1-1e-6)
        return -torch.log(-torch.log(uniform))

    context = dict(observations=observations[None, :history], actions=actions[None, :history-1])
    state = adapter.initialize(context, noise=noise(history))
    initial = adapter.observe(state)
    target_initial = observations[None, history-1].to(initial.device)
    rows = [{'stage': 'initialization', 'horizon': 0, 'mse': mse(initial, target_initial),
             **motion_metrics(initial, target_initial, observations[None, max(0, history-2)].to(initial.device), latent, ram_metrics)}]
    corrected = state
    for step in range(horizon):
        action = actions[history-1+step:history+step]
        shared_noise = noise()
        # Same noise for both paths and all action probes; differences cannot be
        # explained by different random draws at this transition.
        if step == 0:
            target = observations[None, history].to(initial.device)
            probes = [adapter.observe(adapter.step(state, [a], noise=shared_noise)) for a in range(adapter.model.num_actions)]
            factual = int(action.item())
            errors = [mse(p, target) for p in probes]
            rows.append({'stage': 'action_response', 'horizon': 1,
                         'recorded_action': factual, 'recorded_action_mse': errors[factual],
                         'other_actions_mean_mse': float(np.mean([e for a, e in enumerate(errors) if a != factual])),
                         'recorded_action_advantage': float(np.mean([e for a, e in enumerate(errors) if a != factual]) - errors[factual]),
                         'action_sensitivity_mse': float(torch.stack(probes).var(dim=0, unbiased=False).mean())})
        state = adapter.step(state, action, noise=shared_noise)
        corrected = adapter.step(corrected, action, noise=shared_noise)
        pred, teacher = adapter.observe(state), adapter.observe(corrected)
        target = observations[None, history+step].to(pred.device)
        previous = observations[None, history+step-1].to(pred.device)
        for stage, value in [('open_loop', pred), ('corrected_one_step', teacher), ('persistence', target_initial)]:
            rows.append({'stage': stage, 'horizon': step+1, 'mse': mse(value, target),
                         **motion_metrics(value, target, previous, latent, ram_metrics),
                         **({'mse_excess_over_corrected': mse(pred, target) - mse(teacher, target)}
                            if stage == 'open_loop' else {})})
        if step + 1 < horizon:
            if latent:
                corrected = adapter.model.posterior(corrected, target, noise=noise())
            else:
                corrected = adapter.initialize({'observations': target[:, None]})
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['stage'], row['horizon'])
        groups.setdefault(key, []).append(row)
    summary = []
    for (stage, horizon), group in groups.items():
        entry = dict(stage=stage, horizon=horizon)
        metrics = set().union(*(row.keys() for row in group)) - {'stage', 'horizon', 'seed', 'episode', 'start', 'recorded_action'}
        for metric in sorted(metrics):
            valid = [r for r in group if r.get(metric) is not None]
            by_seed = {}
            for row in valid:
                by_seed.setdefault(row['seed'], []).append(row[metric])
            means = [np.mean(values) for values in by_seed.values()]
            entry[metric] = dict(mean=float(np.mean(means)) if means else None,
                                 seed_std=float(np.std(means)) if means else None,
                                 samples=len(valid))
        summary.append(entry)
    return summary


def diagnose(args):
    if not args.ckpt:
        raise ValueError('diagnose requires ckpt=PATH.')
    if min(args.history, args.horizon, args.val_windows) < 1:
        raise ValueError('history, horizon and val_windows must be positive.')
    seeds = list(dict.fromkeys(int(seed) for seed in args.diagnostic_seeds))
    if not seeds:
        raise ValueError('diagnostic_seeds must not be empty.')
    ckpt = load_checkpoint(args.ckpt)
    latent = is_latent_checkpoint(ckpt)
    from .envs.atari import environment_spec, default_data_path, validate_metadata, object_metrics
    spec = environment_spec(args, ckpt)
    path = Path(args.data or ckpt.get('data') or default_data_path(spec, latent))
    data = torch.load(path, map_location='cpu', weights_only=True)
    if 'val' not in data or not data['val']:
        raise ValueError('Dataset must contain nonempty held-out val episodes.')
    validate_metadata(data, spec)
    episodes = data['val']
    adapter = LatentAdapter(LatentModel.load(ckpt, args.device).eval()) if latent else MLPAdapter(StateMLP.load(args.ckpt, args.device))
    windows = select_windows(episodes, latent, args.history, args.horizon, args.val_windows,
                             adapter.model.observation_shape, adapter.model.num_actions)
    if not latent:
        seeds = seeds[:1]  # Deterministic predictions do not need repeated sampling.
    rows = []
    for index, start in windows:
        episode = episodes[index]
        observations, actions = (episode['frames'], episode['actions']) if latent else episode
        observations = observations[start:start+args.history+args.horizon].float()
        if latent:
            observations = observations / 255
        if not torch.isfinite(observations).all() or observations.min() < 0 or observations.max() > 1:
            raise ValueError('Expected finite normalized observations in [0, 1].')
        actions = actions[start:start+args.history+args.horizon-1]
        for seed in seeds:
            for row in diagnose_window(adapter, observations, actions, args.history, args.horizon, seed, latent, object_metrics(spec)):
                rows.append(dict(episode=index, start=start, seed=seed, **row))
    summary = summarize(rows)
    report = dict(schema_version=1, model='latent' if latent else 'mlp',
                  checkpoint=str(Path(args.ckpt).resolve()), data=str(path.resolve()), split='val',
                  environment=spec, device=str(args.device), torch_version=str(torch.__version__), history=args.history, horizon=args.horizon,
                  windows=len(windows), sampling_seeds=seeds, summary=summary, records=rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Diagnostics: {report["model"]}, {len(windows)} validation windows, seeds={seeds}')
    for entry in summary:
        name = 'recorded_action_mse' if entry['stage'] == 'action_response' else 'mse'
        metric = entry[name]
        print(f'{entry["stage"]:20s} h={entry["horizon"]:2d} MSE={metric["mean"]:.6g} seed_std={metric["seed_std"]:.3g}')
    action = next(entry for entry in summary if entry['stage'] == 'action_response')
    print(f'Action sensitivity MSE: {action["action_sensitivity_mse"]["mean"]:.6g}; '
          f'recorded-action advantage: {action["recorded_action_advantage"]["mean"]:.6g} (positive is better)')
    for entry in summary:
        if entry['stage'] != 'open_loop':
            continue
        keys = ['changed_pixel_mse'] if latent else [key for key in ('paddle_y_mae', 'ball_position_error', 'ball_visible_recall') if key in entry]
        values = ', '.join(f'{key}={entry[key]["mean"]}' for key in keys)
        print(f'h={entry["horizon"]}: ' + (values + '; ' if values else '') +
              f'excess MSE over corrected={entry["mse_excess_over_corrected"]["mean"]:.6g}')
    print(f'Saved diagnostic details to {output}')
