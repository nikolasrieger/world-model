import argparse
from pathlib import Path
import numpy as np
import torch
from .envs import PongEnv
from .envs.pong import CONFIG
from .model import StateMLP
from .helper import gif
from .latent_training import train_latent


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
        if saved.get('kind') == 'pixels64' or saved.get('config') != CONFIG:
            raise ValueError('MLP training needs a matching RAM cache, not pixel episodes.')
        if not args.collect_more:
            return saved
        used = saved.get("collection_seeds", [saved["seed"], saved["seed"] + 10000])
        seed = max(used) + 1
        saved["train"].extend(collect(args.collect_more, seed))
        saved["collection_seeds"] = [*used, seed]
    else:
        saved = {"version": 1, "config": CONFIG, "seed": args.seed,
                 "train": collect(args.samples, args.seed),
                 "val": collect(max(256, args.samples // 5), args.seed + 10000)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)
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


def train(args):
    torch.manual_seed(args.seed)
    saved = dataset(args)
    data = sequences(saved["train"], args.horizon, args.device)
    val = sequences(saved["val"], args.horizon, args.device)
    print(f"Training on {len(data[0])} transitions / {len(data[3])} sequences; validating on {len(val[3])} sequences; horizon={args.horizon}.", flush=True)
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else {}
    model = StateMLP.load(args.resume, args.device) if args.resume else StateMLP().to(args.device)
    if not args.resume:
        model.normalize(data[0], data[2])
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    if "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"])
    start_epoch = ckpt.get("epoch", 0)

    @torch.no_grad()
    def evaluate():
        total = sum(rollout_error(model, val, starts, args.horizon).item() * len(starts) for starts in val[3].split(args.batch_size))
        return total / len(val[3])

    def save(epoch, mse):
        payload = {"version": 3, "env_name": "pong", "dimensions": (128, 6, model.net[0].out_features),
                   "weights": model.state_dict(), "opt": opt.state_dict(),
                   "rng_state": torch.get_rng_state(), "epoch": epoch, "val_mse": mse,
                   "horizon": args.horizon, "data": str(args.data)}
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    best = evaluate() if args.resume else float("inf")
    if args.resume:
        save(start_epoch, best)
    for epoch in range(args.epochs):
        for indices in torch.randperm(len(data[3])).split(args.batch_size):
            starts = data[3][indices.to(args.device)]
            loss = rollout_error(model, data, starts, args.horizon, model.delta_scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
        mse = evaluate()
        epoch_number = start_epoch + epoch + 1
        print(f"epoch {epoch_number}: {args.horizon}-step val MSE {mse:.6f}", flush=True)
        if mse < best:
            best = mse
            save(epoch_number, mse)
    print(f"Saved {args.output}; best {args.horizon}-step MSE {best:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--model", choices=("mlp", "latent"), default="mlp")
    fit.add_argument("--arch", choices=("spatial", "dense"), default="spatial")
    fit.add_argument("--history", type=int, default=4)
    fit.add_argument("--samples", type=int, default=100000)
    fit.add_argument("--resume")
    fit.add_argument("--collect-more", type=int, default=0, metavar="N")
    fit.add_argument("--horizon", type=int, default=5)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--batch-size", type=int)
    render = commands.add_parser("gif")
    render.add_argument("--ckpt")
    render.add_argument("--output")
    render.add_argument("--mode", choices=("open-loop", "one-step"), default="open-loop")
    render.add_argument("--history", type=int, default=4)
    for command in (fit, render):
        command.add_argument("--device", choices=("cpu", "mps"), default="cpu")
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu.")
    if args.command == "gif":
        gif(args)
        return
    ckpt = {}
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        kind = ckpt.get("model_type", "mlp")
        args.model = kind
    latent = args.model == "latent"
    args.data = ckpt.get("data") or ("artifacts/pong-pixels.pt" if latent else "artifacts/pong-episodes.pt")
    args.output = ("artifacts/pong-latent.pt" if latent else "artifacts/pong.pt")
    if args.batch_size is None:
        args.batch_size = 32 if latent else 256
    if min(args.history, args.horizon, args.epochs, args.batch_size, args.samples) < 1 or args.collect_more < 0:
        parser.error("Training sizes must be positive; --collect-more must be nonnegative.")
    (train_latent if latent else train)(args)


if __name__ == "__main__":
    main()
