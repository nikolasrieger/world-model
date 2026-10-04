"""Atari pixel caching and joint discrete RSSM sequence training."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .envs.pixels import PixelPongEnv, PIXEL_CONFIG as CONFIG
from .latent import LatentModel, noise_sequence
from .adapters import check_action


def frame(env):
    return env.frame()


def collect_pixels(count, seed):
    env, rng = PixelPongEnv(), np.random.default_rng(seed)
    episodes, total = [], 0
    try:
        env.reset(seed=seed)
        images, actions, rewards, terminals = [frame(env)], [], [], []
        while total < count or actions:
            action = int(rng.integers(6))
            _, reward, terminated, truncated, _ = env.step(action)
            images.append(frame(env))
            actions.append(action)
            rewards.append(int(reward) + 1)
            terminals.append(terminated) 
            total += 1
            if terminated or truncated:
                episodes.append(dict(frames=torch.stack(images), actions=torch.tensor(actions),
                    rewards=torch.tensor(rewards), terminated=torch.tensor(terminals, dtype=torch.float32)))
                env.reset()
                images, actions, rewards, terminals = [frame(env)], [], [], []
    finally:
        env.close()
    return episodes


def save_file(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def pixel_dataset(args):
    path = Path(args.data)
    if path.exists():
        data = torch.load(path, map_location='cpu', weights_only=True)
        if not args.collect_more:
            print(f'Reusing pixel episodes from {path}; no collection.', flush=True)
            return data
        seed = max(data['collection_seeds']) + 1
        print(f'Appending pixel episodes (at least {args.collect_more} transitions, seed {seed}).', flush=True)
        data['train'].extend(collect_pixels(args.collect_more, seed))
        data['collection_seeds'].append(seed)
    else:
        if args.collect_more:
            raise ValueError('--collect-more requires an existing pixel cache.')
        print(f'Collecting at least {args.samples} training transitions as 64x64 grayscale images...', flush=True)
        data = dict(kind='pixels64', config=CONFIG, collection_seeds=[args.seed, args.seed+10000],
                    train=collect_pixels(args.samples, args.seed),
                    val=collect_pixels(max(256, args.samples//5), args.seed+10000))
    save_file(data, path)
    return data


class Windows:
    def __init__(self, episodes, history, horizon):
        if history < 1 or horizon < 1:
            raise ValueError("History and horizon must be positive.")
        for ep in episodes:
            check_action(ep["actions"], len(ep["actions"]))
        self.episodes, self.length = episodes, history - 1 + horizon
        self.indices = [(i, start) for i, ep in enumerate(episodes) for start in range(len(ep['actions']) - self.length + 1)]

    def __len__(self):
        return len(self.indices)

    def batch(self, indices, device):
        rows = []
        for index in indices:
            ep, start = self.indices[int(index)]
            data = self.episodes[ep]
            stop = start + self.length
            rows.append((data['frames'][start:stop+1], data['actions'][start:stop], data['rewards'][start:stop], data['terminated'][start:stop]))
        images, actions, rewards, terminals = (torch.stack(x).to(device) for x in zip(*rows))
        return images.float()/255., actions, rewards, terminals


def kl_divergence(posterior, prior, free_nats=0.):
    q, p = posterior['distr']['logits'], prior['distr']['logits']

    def divergence(q, p):
        logq, logp = q.log_softmax(-1), p.log_softmax(-1)
        return (logq.exp() * (logq - logp)).sum((-1, -2)).clamp_min(free_nats).mean()

    return .8 * divergence(q.detach(), p) + .2 * divergence(q, p.detach())


def motion_mse(pred, target, prev):
    # Supervision only: include both new and vacated pixels, plus a one-pixel border.
    mask = F.max_pool2d((target - prev).abs().gt(.02).float(), 3, 1, 1)
    error = (pred - target).square() * mask
    return (error.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).mean()


def position_loss(pred, target):
    losses = []
    for left, right in ((5, 9), (9, 55), (55, 59)):
        observed = target[:, :, 11:58, left:right].flatten(1)
        contrast = (observed - observed.median(1, keepdim=True).values - .02).clamp_min(0)
        mass = contrast.sum(1, keepdim=True)
        probability = contrast / mass.clamp_min(1e-8)
        logits = pred[:, :, 11:58, left:right].flatten(1) / .05
        divergence = (probability * (probability.clamp_min(1e-8).log() - logits.log_softmax(1))).sum(1)
        visible = (mass[:, 0] > .05).float()
        losses.append((divergence * visible).sum() / visible.sum().clamp_min(1))
    return sum(losses) / len(losses)


def motion_mae(pred, target, prev):
    mask = (target - prev).abs().gt(.02).float()
    error = (pred - target).abs() * mask
    return (error.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).mean()


def image_loss(pred, target, prev):
    return (pred - target).square().mean() + 2 * motion_mae(pred, target, prev)


def loss(model, batch, history, generator, return_metrics=False, free_nats=0.):
    images, actions, rewards, terminals = batch
    size, time = images.shape[:2]
    embeds = model.encoder(images.flatten(0, 1) - .5).reshape(size, time, -1)
    noises = noise_sequence(time, size, model.latent, images.device, generator)
    h = model.start_hidden(size)
    features, kls = [], []
    for t in range(time):
        if t:
            h = model.transition(state, actions[:, t-1], validate=False)
        prior = dict(distr=dict(logits=model.prior_net(h).reshape(size, model.stoch, model.classes)))
        state = model.posterior_embed(h, embeds[:, t], noises[t])
        kls.append(kl_divergence(state, prior, free_nats))
        features.append(model.features(state))
    features = torch.stack(features, 1)
    decoded = model.decoder(features.flatten(0, 1)).reshape_as(images)
    # Negative log likelihood of unit-variance Normal
    recon = .5 * (decoded-images).square().flatten(2).sum(-1).mean()
    values = model.heads(features[:, 1:].flatten(0, 1)).reshape(size, time-1, 2)
    reward = .5 * (values[..., 0] - (rewards.float()-1).tanh()).square().mean()
    terminal = F.binary_cross_entropy_with_logits(values[..., 1], terminals)
    kl = torch.stack(kls).mean()
    total = recon + .1 * kl + reward + 5 * terminal
    if return_metrics:
        return total, dict(recon=recon.detach(), kl=kl.detach(), reward=reward.detach(), terminal=terminal.detach())
    return total


@torch.no_grad()
def evaluate(model, windows, args):
    generator = torch.Generator().manual_seed(args.seed + 20000)
    totals = dict(rollout_mse=0., reward_accuracy=0., terminal_accuracy=0.)
    diagnostic_rng = torch.Generator().manual_seed(args.seed + 30000)
    totals.update(motion_mae=0., posterior_motion_mae=0., position_kl=0., posterior_position_kl=0., motion_mse=0., posterior_mse=0., posterior_motion_mse=0., persistence_motion_mse=0.)
    for indices in torch.arange(len(windows)).split(args.batch_size):
        images, actions, rewards, terminals = windows.batch(indices, args.device)
        state = model.initialize(dict(observations=images[:, :args.history], actions=actions[:, :args.history-1]), generator=generator)
        teacher = state
        noises = iter(noise_sequence(args.horizon, len(indices), model.latent, images.device, generator))
        diagnostics = iter(noise_sequence(args.horizon, len(indices), model.latent, images.device, diagnostic_rng))
        for t in range(args.history, images.shape[1]):
            state = model.prior(state, actions[:, t-1], noise=next(noises), validate=False)
            target = images[:, t]
            pred = model.observe(state)
            totals['rollout_mse'] += (pred-target).square().mean()*len(indices)
            prev = images[:, t-1]
            totals['position_kl'] += position_loss(pred, target)*len(indices)
            totals['motion_mae'] += motion_mae(pred, target, prev)*len(indices)
            totals['motion_mse'] += motion_mse(pred, target, prev)*len(indices)
            totals['persistence_motion_mse'] += motion_mse(images[:, args.history-1], target, prev)*len(indices)
            teacher_prior = model.prior(teacher, actions[:, t-1], noise=torch.zeros_like(teacher['z']), validate=False)
            teacher = model.posterior(teacher_prior['h'], target, noise=next(diagnostics))
            recon = model.observe(teacher)
            totals['posterior_position_kl'] += position_loss(recon, target)*len(indices)
            totals['posterior_motion_mae'] += motion_mae(recon, target, prev)*len(indices)
            totals['posterior_mse'] += (recon-target).square().mean()*len(indices)
            totals['posterior_motion_mse'] += motion_mse(recon, target, prev)*len(indices)
            logits = model.heads(model.features(state))
            totals['reward_accuracy'] += ((logits[:, 0] / torch.tanh(logits.new_tensor(1.))).round().clamp(-1, 1)==rewards[:, t-1]-1).float().sum()
            totals['terminal_accuracy'] += ((logits[:, 1]>=0)==terminals[:, t-1].bool()).float().sum()
    values = torch.stack(list(totals.values())).cpu().tolist()
    return {key: value/(len(windows)*args.horizon) for key, value in zip(totals, values)}


def train_latent(args):
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    model = LatentModel.load(ckpt, args.device) if ckpt else LatentModel().to(args.device)
    data = pixel_dataset(args)
    train = Windows(data['train'], 1, args.sequence_length - 1)
    val = Windows(data['val'], args.history, args.horizon)
    lr = 2e-4
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    sampling = torch.Generator().manual_seed(args.seed)
    shuffle = torch.Generator().manual_seed(args.seed+1)
    start_epoch = 0
    if ckpt:
        opt.load_state_dict(ckpt['opt'])
        for group in opt.param_groups:
            group['lr'] = lr
        sampling.set_state(ckpt['sampling_rng'])
        shuffle.set_state(ckpt['shuffle_rng'])
        start_epoch = ckpt['epoch']
    print(f'Latent model: {sum(p.numel() for p in model.parameters()):,} parameters, {len(train)} training windows; sequence={args.sequence_length}, history={args.history}, horizon={args.horizon}.', flush=True)

    def save(epoch, metrics, path=None):
        save_file(dict(updates=updates, model_type='latent', model_config=model.config, weights=model.state_dict(), opt=opt.state_dict(), sampling_rng=sampling.get_state(), shuffle_rng=shuffle.get_state(), epoch=epoch, history=args.history, horizon=args.horizon, sequence_length=args.sequence_length, free_nats=args.free_nats, data=str(args.data), metrics=metrics), path or args.output)

    def score(metrics):
        return metrics['rollout_mse']

    if args.val_windows and len(val) > args.val_windows:
        val.indices = [val.indices[i] for i in torch.linspace(0, len(val)-1, args.val_windows).long()]
    updates = ckpt.get('updates', 0) if ckpt else 0
    run_updates = 0
    best = float('inf')
    if ckpt:
        metrics = evaluate(model.eval(), val, args)
        best = score(metrics)
        save(start_epoch, metrics)
    for epoch in range(start_epoch+1, start_epoch+args.epochs+1):
        model.train()
        train_loss = torch.zeros((), device=args.device)
        started = last_report = perf_counter()
        batches = torch.randperm(len(train), generator=shuffle).split(args.batch_size)
        for step, indices in enumerate(batches, 1):
            obj, components = loss(model, train.batch(indices, args.device), args.history, sampling, return_metrics=True, free_nats=args.free_nats)
            if not torch.isfinite(obj):
                raise RuntimeError('Nonfinite latent training loss.')
            opt.zero_grad()
            obj.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            updates += 1
            run_updates += 1
            train_loss += obj.detach()*len(indices)
            if step == 1 or step == len(batches) or perf_counter() - last_report >= 10:
                if str(args.device).startswith('mps'):
                    torch.mps.synchronize()
                elapsed = perf_counter() - started
                print(f"epoch {epoch}: batch {step}/{len(batches)}, "
                      f"{elapsed/step:.3f}s/batch, updates={updates}, losses={ {k: round(v.item(), 4) for k, v in components.items()} }", flush=True)
                last_report = perf_counter()
            if args.eval_every and updates % args.eval_every == 0:
                metrics = evaluate(model.eval(), val, args)
                print(f"update {updates}: val={metrics}", flush=True)
                if score(metrics) < best:
                    best = score(metrics)
                    save(epoch, metrics)
                model.train()
            if args.max_updates and run_updates >= args.max_updates:
                break
        print(f"epoch {epoch}: validating {len(val)} windows...", flush=True)
        metrics = evaluate(model.eval(), val, args)
        print(f"epoch {epoch}: loss={train_loss/min(len(train), step*args.batch_size):.5f}, val={metrics}", flush=True)
        output = Path(args.output)
        save(epoch, metrics, output.with_name(output.stem + "-last" + output.suffix))
        val_score = score(metrics)
        if val_score < best:
            best = val_score
            save(epoch, metrics)
        if args.max_updates and run_updates >= args.max_updates:
            break
    print(f'Saved best latent ckpt to {args.output}.', flush=True)
