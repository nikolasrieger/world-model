"""Shared training loop, with model-specific data, updates, and evaluation."""
from pathlib import Path
from time import perf_counter
import hashlib

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from .configs import load_model_config
from torch.nn import functional as F

from .envs import PongEnv
from .envs.pong import CONFIG as RAM_CONFIG
from .envs.pixels import PixelPongEnv, PIXEL_CONFIG
from .mlp_model import StateMLP
from .latent_model import LatentModel, load_checkpoint
from .adapters import check_action


def collect(count, seed):
    env, rng = PongEnv(), np.random.default_rng(seed)
    state, _ = env.reset(seed=seed)
    episodes, states, actions, total = [], [state], [], 0
    try:
        while total < count or actions: 
            action = int(rng.integers(6))
            state, _, done, truncated, _ = env.step(action)
            states.append(state)
            actions.append(action)
            total += 1
            if done or truncated:
                episodes.append((torch.tensor(np.asarray(states)), torch.tensor(actions)))
                states, actions = [env.reset()[0]], []
    finally:
        env.close()
    return episodes


def dataset(args):
    path = Path(args.data)
    if path.exists():
        print(f"Loading episodes from {path}.", flush=True)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if not args.collect_more:
            return saved
        used = saved.get("collection_seeds", [saved["seed"], saved["seed"] + 10000])
        seed = max(used) + 1
        saved["train"].extend(collect(args.collect_more, seed))
        saved["collection_seeds"] = [*used, seed]
    else:
        saved = {"config": RAM_CONFIG, "seed": args.seed,
                 "train": collect(args.samples, args.seed),
                 "val": collect(max(256, args.samples // 5), args.seed + 10000)}
    save_file(saved, path)
    return saved


def sequences(episodes, horizon, device):
    states, actions, targets, starts, offset = [], [], [], [], 0
    for observations, controls in episodes:
        length = len(controls)
        states.append(observations[:-1])
        actions.append(controls)
        targets.append(observations[1:])
        starts.extend(range(offset, offset + max(0, length - horizon + 1)))
        offset += length
    return (*(torch.cat(x).to(device) for x in (states, actions, targets)),
            torch.tensor(starts, device=device))


def rollout_error(model, data, starts, horizon, scale=1.):
    states, actions, targets, _ = data
    predicted, loss = states[starts], 0.
    for step in range(horizon):
        predicted = model(predicted, actions[starts + step]).clamp(0, 1)
        loss = loss + ((predicted - targets[starts + step]) / scale).square().mean()
    return loss / horizon 


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
        data = dict(kind='pixels64', config=PIXEL_CONFIG, collection_seeds=[args.seed, args.seed+10000],
                    train=collect_pixels(args.samples, args.seed),
                    val=collect_pixels(max(256, args.samples//5), args.seed+10000))
    save_file(data, path)
    return data


class Windows:
    def __init__(self, episodes, history, horizon):
        if history < 1 or horizon < 1:
            raise ValueError('History and horizon must be positive.')
        for ep in episodes:
            check_action(ep['actions'], len(ep['actions']))
        self.episodes, self.length = episodes, history + horizon
        self.indices = [(i, start) for i, ep in enumerate(episodes)
                        for start in range(len(ep['frames']) - self.length + 1)]

    def __len__(self):
        return len(self.indices)

    def batch(self, indices):
        rows = []
        for index in indices:
            ep, start = self.indices[int(index)]
            data = self.episodes[ep]
            stop = start + self.length
            rows.append((data['frames'][start:stop], data['actions'][start:stop-1],
                         data['rewards'][start:stop-1]-1, data['terminated'][start:stop-1]))
        images, actions, rewards, terminals = (torch.stack(x) for x in zip(*rows))
        controls = F.one_hot(actions.long(), 6).float()
        controls = torch.cat((torch.zeros(len(rows), 1, 6), controls), 1)
        first = torch.zeros(len(rows), self.length)
        first[:, 0] = 1
        return dict(image=images.permute(0, 1, 3, 4, 2).numpy(), action=controls.numpy(),
                    reward=torch.cat((torch.zeros(len(rows), 1), rewards), 1).numpy(),
                    is_terminal=torch.cat((torch.zeros(len(rows), 1), terminals), 1).numpy(),
                    is_first=first.numpy())


@torch.no_grad()
def evaluate(model, windows, indices, history, batch_size):
    # Evaluation must not change the subsequent training samples.
    cpu_rng = torch.get_rng_state()
    mps_rng = torch.mps.get_rng_state() if torch.device(model.config['device']).type == 'mps' else None
    totals = dict(rollout_mse=0., reconstruction_mse=0., persistence_mse=0.)
    try:
        torch.manual_seed(20000)
        for selected in indices.split(batch_size):
            data = model.preprocess(windows.batch(selected))
            post, _ = model.dynamics.observe(model.encoder(data), data['action'], data['is_first'])
            init = {k: v[:, history-1] for k, v in post.items()}
            future = model.dynamics.imagine_with_action(data['action'][:, history:], init)
            pred = model.decoder(model.features(future))['image'].mode()
            recon = model.decoder(model.features(post))['image'].mode()[:, history:]
            target = data['image'][:, history:]
            for key, value in [('rollout_mse', pred), ('reconstruction_mse', recon),
                               ('persistence_mse', data['image'][:, history-1:history])]:
                totals[key] += float((value-target).square().mean()) * len(selected)
        return {key: value / len(indices) for key, value in totals.items()}
    finally:
        torch.set_rng_state(cpu_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)


class MLPTraining:
    drop_last = False
    save_latest = False
    evaluation_interval = None

    def __init__(self, args):
        self.args = args
        saved = dataset(args)
        self.data = sequences(saved['train'], args.horizon, args.device)
        self.val = sequences(saved['val'], args.horizon, args.device)
        self.count = len(self.data[3])
        if not self.count or not len(self.val[3]):
            raise ValueError('Not enough RAM sequences for the requested horizon.')
        self.ckpt = load_checkpoint(args.resume) if args.resume else {}
        config = getattr(args, 'model_config', None) or load_model_config('mlp')
        self.model = StateMLP.load(args.resume, args.device) if args.resume else StateMLP(hidden=config['hidden']).to(args.device)
        if not args.resume:
            self.model.normalize(self.data[0], self.data[2])
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        if 'opt' in self.ckpt:
            self.optimizer.load_state_dict(self.ckpt['opt'])
        self.resume_order = not args.collect_more and self.ckpt.get('horizon') == args.horizon
        print(f'MLP: {len(self.data[0])} transitions / {self.count} training sequences; '
              f'{len(self.val[3])} validation sequences; horizon={args.horizon}.', flush=True)

    def train_step(self, indices):
        starts = self.data[3][indices.to(self.args.device)]
        loss = rollout_error(self.model, self.data, starts, self.args.horizon, self.model.delta_scale)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': float(loss.detach())}

    @torch.no_grad()
    def evaluate(self):
        total = sum(rollout_error(self.model, self.val, starts, self.args.horizon).item() * len(starts)
                    for starts in self.val[3].split(self.args.batch_size))
        return {'rollout_mse': total / len(self.val[3])}

    def checkpoint(self, metrics):
        return dict(env_name='pong', dimensions=(128, 6, self.model.net[0].out_features),
                    opt=self.optimizer.state_dict(), rng_state=torch.get_rng_state(),
                    val_mse=metrics['rollout_mse'], horizon=self.args.horizon)


class LatentTraining:
    drop_last = True
    save_latest = True

    def __init__(self, args):
        self.args = args
        self.ckpt = ckpt = load_checkpoint(args.resume) if args.resume else {}
        self.model = model = LatentModel.load(ckpt, args.device) if ckpt else LatentModel(config=getattr(args, 'model_config', None), device=args.device)
        config = model.config
        if args.sequence_length is not None:
            config['batch_length'] = args.sequence_length
        if args.free_nats is not None:
            config['kl_free'] = args.free_nats
        config['batch_size'] = args.batch_size
        model._config.batch_length = config['batch_length']
        model._config.kl_free = config['kl_free']
        if config['batch_length'] < 2 or config['kl_free'] < 0:
            raise ValueError('Sequence length must be at least 2; free nats must be nonnegative.')
        data = pixel_dataset(args)
        if data.get('kind') != 'pixels64' or data.get('config') != PIXEL_CONFIG:
            raise ValueError('Pixel cache preprocessing does not match PixelPongEnv.')
        self.data = Windows(data['train'], 1, config['batch_length']-1)
        self.val = Windows(data['val'], args.history, args.horizon)
        self.count = len(self.data)
        if self.count < args.batch_size or not len(self.val):
            raise ValueError('Not enough complete windows for the requested batch/context/horizon.')
        self.val_indices = torch.linspace(0, len(self.val)-1, min(args.val_windows, len(self.val))).long()
        with open(args.data, 'rb') as stream:
            self.dataset_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        original_hash = ckpt.get('manifest', {}).get('dataset_sha256')
        self.resume_order = original_hash == self.dataset_hash and ckpt.get('config', {}).get('batch_length') == config['batch_length']
        if original_hash and original_hash != self.dataset_hash and not args.collect_more:
            raise ValueError('Resume dataset differs from the checkpoint; use its original pixel cache.')
        self.optimizer = model._model_opt._opt
        if ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.evaluation_interval = args.eval_every
        print(f'DreamerV3: {sum(p.numel() for p in model.parameters()):,} parameters; '
              f'sequence {config["batch_length"]}; batch {args.batch_size}; '
              f'lr {self.optimizer.param_groups[0]["lr"]:g}.', flush=True)

    def train_step(self, indices):
        post, context, metrics = self.model._train(self.data.batch(indices))
        del post, context
        return {key: float(metrics[source]) for key, source in
                [('loss', 'model_loss'), ('image', 'image_loss'), ('kl', 'kl')]}

    def evaluate(self):
        return evaluate(self.model, self.val, self.val_indices, self.args.history, self.args.batch_size)

    def checkpoint(self, metrics):
        return dict(implementation='nm512-dreamerv3', optimizer=self.optimizer.state_dict(),
                    config=OmegaConf.to_container(self.model.config, resolve=True), history=self.args.history, horizon=self.args.horizon,
                    manifest=dict(upstream_commit='6ef8646d807cd10ce0c88e10a7e943211e7fc44c',
                                  dataset_sha256=self.dataset_hash, torch=str(torch.__version__)))


def train(args):
    """One scheduler for batching, updates, evaluation, logging, and saving."""
    torch.manual_seed(args.seed)
    trainer = {'mlp': MLPTraining, 'latent': LatentTraining}[args.model](args)
    ckpt = trainer.ckpt
    epoch, step = ckpt.get('epoch', 0), ckpt.get('step', 0)
    shuffle = torch.Generator().manual_seed(args.seed+1) if trainer.drop_last else None
    order, offset = None, 0
    if trainer.resume_order and ckpt.get('order') is not None:
        order, offset = ckpt['order'], ckpt['offset']
        if len(order) != trainer.count or not 0 <= offset <= len(order):
            raise ValueError('Saved data cursor does not match the training windows.')
        if shuffle is not None:
            shuffle.set_state(ckpt['shuffle_rng'])
    elif ckpt.get('order') is not None:
        print('Training window layout changed; starting a new data permutation.', flush=True)
    cpu_rng = ckpt.get('cpu_rng', ckpt.get('rng_state'))
    if cpu_rng is not None:
        torch.set_rng_state(cpu_rng)
    if torch.device(args.device).type == 'mps' and ckpt.get('mps_rng') is not None:
        torch.mps.set_rng_state(ckpt['mps_rng'])
    output = Path(args.output)

    def save(metrics, improved):
        if not (trainer.save_latest or improved):
            return
        payload = trainer.checkpoint(metrics)
        payload.update(model_type=args.model, weights=trainer.model.state_dict(),
                       step=step, epoch=epoch, batch_size=args.batch_size, data=str(args.data), metrics=metrics,
                       cpu_rng=torch.get_rng_state(),
                       mps_rng=torch.mps.get_rng_state() if torch.device(args.device).type == 'mps' else None,
                       order=order, offset=offset,
                       shuffle_rng=shuffle.get_state() if shuffle is not None else None)
        if isinstance(args, DictConfig):
            payload['run_config'] = OmegaConf.to_container(args, resolve=True)
        save_file(payload, output)
        if trainer.save_latest and improved:
            save_file(payload, output.with_name(output.stem + '.best' + output.suffix))

    best = float('inf')
    if args.resume and not trainer.save_latest:
        metrics = trainer.evaluate()
        best = metrics['rollout_mse']
        save(metrics, True)
    run_steps, completed_epochs = 0, 0
    started, last_log = perf_counter(), 0.
    minimum_batch = args.batch_size if trainer.drop_last else 1
    print(f'Starting at update {step}, epoch {epoch}.', flush=True)
    while completed_epochs < args.epochs and (args.max_updates is None or run_steps < args.max_updates):
        if order is None or offset + minimum_batch > len(order):
            order, offset = torch.randperm(trainer.count, generator=shuffle), 0
        selected = order[offset:offset+args.batch_size]
        offset += len(selected)
        metrics = trainer.train_step(selected)
        step += 1
        run_steps += 1
        now = perf_counter()
        if run_steps == 1 or now-last_log >= 10:
            values = ', '.join(f'{name} {value:.4f}' for name, value in metrics.items())
            print(f'update {step}: {values}; {now-started:.1f}s', flush=True)
            last_log = now
        epoch_end = offset + minimum_batch > len(order)
        if epoch_end:
            epoch += 1
            completed_epochs += 1
        finished = completed_epochs >= args.epochs or (args.max_updates is not None and run_steps >= args.max_updates)
        evaluation_due = trainer.evaluation_interval and step % trainer.evaluation_interval == 0
        if evaluation_due or epoch_end or finished:
            metrics = trainer.evaluate()
            improved = metrics['rollout_mse'] < best
            if improved:
                best = metrics['rollout_mse']
            save(metrics, improved)
            print(f'update {step}, epoch {epoch}: validation {metrics}; '
                  f'checkpoint {"saved" if trainer.save_latest or improved else "unchanged"}', flush=True)
    print(f'Completed {run_steps} additional updates (now {step}); checkpoint: {output}.', flush=True)
