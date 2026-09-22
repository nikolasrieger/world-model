import argparse
from pathlib import Path
import numpy as np
import torch
from .envs import PongEnv
from .envs.pong import CONFIG
from .model import StateMLP
from .helper import positive, gif


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
        saved = {"version": 1, "config": CONFIG, "seed": args.seed,
                 "train": collect(args.samples, args.seed),
                 "validation": collect(max(256, args.samples // 5), args.seed + 10000)}
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
    validation = sequences(saved["validation"], args.horizon, args.device)
    print(f"Training on {len(data[0])} transitions / {len(data[3])} sequences; "
          f"validating on {len(validation[3])} sequences; horizon={args.horizon}.", flush=True)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else {}
    model = StateMLP.load(args.resume, args.device) if args.resume else StateMLP().to(args.device)
    if not args.resume:
        model.normalize(data[0], data[2])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
    start_epoch = checkpoint.get("epoch", 0)

    @torch.no_grad()
    def evaluate():
        total = sum(rollout_error(model, validation, starts, args.horizon).item() * len(starts)
                    for starts in validation[3].split(args.batch_size))
        return total / len(validation[3])

    def save(epoch, mse):
        payload = {"version": 3, "env_name": "pong", "dimensions": (128, 6, model.net[0].out_features),
                   "weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "rng_state": torch.get_rng_state(), "epoch": epoch, "validation_mse": mse,
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mse = evaluate()
        epoch_number = start_epoch + epoch + 1
        print(f"epoch {epoch_number}: {args.horizon}-step validation MSE {mse:.6f}", flush=True)
        if mse < best:
            best = mse
            save(epoch_number, mse)
    print(f"Saved {args.output}; best {args.horizon}-step MSE {best:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--samples", type=positive, default=100000)
    fit.add_argument("--data", default="artifacts/pong-episodes.pt")
    fit.add_argument("--resume")
    fit.add_argument("--collect-more", type=positive, default=0, metavar="N")
    fit.add_argument("--horizon", type=positive, default=5)
    fit.add_argument("--epochs", type=positive, default=30)
    fit.add_argument("--batch-size", type=positive, default=256)
    fit.add_argument("--output", default="artifacts/pong.pt")
    render = commands.add_parser("gif")
    render.add_argument("--checkpoint", default="artifacts/pong.pt")
    render.add_argument("--output", default="artifacts/pong.gif")
    render.add_argument("--steps", type=positive, default=200)
    for command in (fit, render):
        command.add_argument("--device", choices=("cpu", "mps"), default="cpu")
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    (train if args.command == "train" else gif)(args)


if __name__ == "__main__":
    main()
