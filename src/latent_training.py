"""Pixel episode caching and sequence training for the compact latent model."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .envs import PongEnv
from .envs.pong import CONFIG
from .latent import LatentModel, noise_sequence
from .adapters import check_action


def frame(env):
    image = Image.fromarray(env.render()).convert('L').resize((64, 64), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image, copy=True))[None]


def collect_pixels(count, seed):
    env, rng = PongEnv(render_mode='rgb_array'), np.random.default_rng(seed)
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
        if data.get('kind') != 'pixels64' or data.get('config') != CONFIG:
            raise ValueError('Latent training needs a pixels64 cache, not the RAM dataset; choose another --data path.')
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
        data = dict(version=1, kind='pixels64', config=CONFIG, collection_seeds=[args.seed, args.seed+10000],
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


def kl_divergence(posterior, prior):
    q, p = posterior['distr'], prior['distr']

    def divergence(q_mean, q_std, p_mean, p_std):
        value = (p_std.log() - q_std.log() + (q_std.square() + (q_mean-p_mean).square())/(2*p_std.square()) - .5)
        return value.mean().clamp_min(.1)

    dyn = divergence(q['mean'].detach(), q['std'].detach(), p['mean'], p['std'])
    repr = divergence(q['mean'], q['std'], p['mean'].detach(), p['std'].detach())
    return .8*dyn + .2*repr


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


def loss(model, batch, history, generator):
    images, actions, rewards, terminals = batch
    size, time = images.shape[:2]
    device = images.device
    h = torch.zeros(size, model.hidden, device=device)
    noises = iter(noise_sequence(2*time - 1 + time - history, size, model.latent, device, generator))
    state = model.posterior(h, images[:, 0], noise=next(noises))
    recon = image_loss(model.observe(state), images[:, 0], images[:, 1])
    kl, heads = h.sum(), h.sum()
    context_state = state if history == 1 else None
    for t in range(1, time):
        prior = model.prior(state, actions[:, t-1], noise=next(noises), validate=False)
        state = model.posterior(prior['h'], images[:, t], noise=next(noises))
        kl = kl + kl_divergence(state, prior)
        recon = recon + image_loss(model.observe(state), images[:, t], images[:, t-1])
        logits = model.heads(model.features(state))
        heads = heads + F.cross_entropy(logits[:, :3], rewards[:, t-1])
        heads = heads + F.binary_cross_entropy_with_logits(logits[:, 3], terminals[:, t-1])
        if t == history - 1:
            context_state = state

    imagined, rollout, outcome_loss = context_state, h.sum(), h.sum()
    for t in range(history, time):
        imagined = model.prior(imagined, actions[:, t-1], noise=next(noises), validate=False)
        pred = model.observe(imagined)
        rollout = rollout + image_loss(pred, images[:, t], images[:, t-1])
        logits = model.heads(model.features(imagined))
        outcome_loss = outcome_loss + F.cross_entropy(logits[:, :3], rewards[:, t-1])
        outcome_loss = outcome_loss + F.binary_cross_entropy_with_logits(logits[:, 3], terminals[:, t-1])
    horizon = time - history
    total = recon/time + rollout/horizon + .01*kl/(time-1)
    return total + .1*(heads/(time-1) + outcome_loss/horizon)


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
            totals['reward_accuracy'] += (logits[:, :3].argmax(-1)==rewards[:, t-1]).float().sum()
            totals['terminal_accuracy'] += ((logits[:, 3]>=0)==terminals[:, t-1].bool()).float().sum()
    values = torch.stack(list(totals.values())).cpu().tolist()
    return {key: value/(len(windows)*args.horizon) for key, value in zip(totals, values)}


def train_latent(args):
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    model = LatentModel.load(ckpt, args.device) if ckpt else LatentModel(arch=args.arch).to(args.device)
    data = pixel_dataset(args)
    train = Windows(data['train'], args.history, args.horizon)
    val = Windows(data['val'], args.history, args.horizon)
    if not len(train) or not len(val):
        raise ValueError("Training and validation data must contain at least one complete window.")
    lr = 3e-4
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
    print(f'Latent model ({model.arch}): {sum(p.numel() for p in model.parameters()):,} parameters, {len(train)} training windows; history={args.history}, horizon={args.horizon}.', flush=True)

    def save(epoch, metrics):
        save_file(dict(version=1, model_type='latent', model_config=model.config, weights=model.state_dict(), opt=opt.state_dict(), sampling_rng=sampling.get_state(), shuffle_rng=shuffle.get_state(), epoch=epoch, history=args.history, horizon=args.horizon, data=str(args.data), metrics=metrics), args.output)

    def score(metrics):
        return metrics['rollout_mse'] + 2*metrics.get('motion_mae', 0.)

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
            obj = loss(model, train.batch(indices, args.device), args.history, sampling)
            if not torch.isfinite(obj):
                raise RuntimeError('Nonfinite latent training loss.')
            opt.zero_grad()
            obj.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            train_loss += obj.detach()*len(indices)
            if step == 1 or step == len(batches) or perf_counter() - last_report >= 10:
                if str(args.device).startswith('mps'):
                    torch.mps.synchronize()
                elapsed = perf_counter() - started
                print(f"epoch {epoch}: batch {step}/{len(batches)}, "
                      f"{elapsed/step:.3f}s/batch, train ETA {elapsed/step*(len(batches)-step):.0f}s", flush=True)
                last_report = perf_counter()
        print(f"epoch {epoch}: validating {len(val)} windows...", flush=True)
        metrics = evaluate(model.eval(), val, args)
        print(f"epoch {epoch}: loss={train_loss/len(train):.5f}, val={metrics}", flush=True)
        val_score = score(metrics)
        if val_score < best:
            best = val_score
            save(epoch, metrics)
    print(f'Saved best latent ckpt to {args.output}.', flush=True)
