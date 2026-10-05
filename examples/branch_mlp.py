"""Export checkpoint predictions and verify snapshot replay for either adapter."""
import argparse
from collections import deque
from pathlib import Path
import numpy as np
import torch
from src.adapters import MLPAdapter, LatentAdapter
from src.mlp_model import StateMLP
from src.latent_model import LatentModel, load_checkpoint, is_latent_checkpoint
from src.runtime import WorldRuntime


def collect_context(env, *, seed, history, latent, warmup=30, max_steps=300):
    env.reset(seed=seed)
    base = env.env if latent else env
    images = deque(maxlen=history)
    previous_ball = None
    moving = False
    for step in range(1, max_steps + 1):
        observation, _, done, truncated, _ = env.step(1)
        if done or truncated:
            raise RuntimeError('Episode ended before active-rally context was collected.')
        ram = base.unwrapped.ale.getRAM().astype(int)
        left, right = ram[50] - 15, ram[51] - 13
        ball = (ram[49] - 49, ram[54] - 14)
        visible = (20 <= left <= 193 and 20 <= right <= 193
                   and 0 < ball[0] < 160 and 34 <= ball[1] <= 193)
        if step < warmup or not visible:
            images.clear()
            previous_ball = None
            moving = False
            continue
        moving = moving or (previous_ball is not None and ball != previous_ball)
        previous_ball = ball
        images.append(env.frame().float()/255 if latent else torch.as_tensor(observation).clone())
        if len(images) == history and moving:
            return dict(observations=torch.stack(list(images))[None],
                        actions=torch.ones((1, history - 1), dtype=torch.long)), step
    raise RuntimeError(f'No visible moving rally with {history} context frames within {max_steps} steps.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--history', type=int, default=4)
    parser.add_argument('--output', default='artifacts/branches.npz')
    args = parser.parse_args()
    if args.horizon < 1 or args.history < 1:
        parser.error('horizon and history must be positive')
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(args.ckpt)
    latent = is_latent_checkpoint(checkpoint)
    if latent:
        from src.envs.pixels import PixelPongEnv
        env = PixelPongEnv()
        adapter = LatentAdapter(LatentModel.load(checkpoint, args.device).eval(),
                                torch.Generator().manual_seed(args.seed))
    else:
        from src.envs import PongEnv
        env = PongEnv()
        adapter = MLPAdapter(StateMLP.load(args.ckpt, args.device))
    try:
        context, initialization_steps = collect_context(
            env, seed=args.seed, history=args.history, latent=latent)
        print(f'Collected {args.history} active-rally frames after {initialization_steps} simulator steps.')
        runtime = WorldRuntime(adapter)
        runtime.reset(context)
    finally:
        env.close()
    saved = runtime.snapshot()
    actions = np.stack([np.ones(args.horizon, dtype=np.int64), np.full(args.horizon, 3, dtype=np.int64)])
    children = [runtime.branch(), runtime.branch()]

    def rollout(child, sequence):
        return torch.stack([child.step([int(action)]).cpu() for action in sequence])

    predictions = [rollout(child, sequence) for child, sequence in zip(children, actions)]
    children[0].restore(saved)
    torch.testing.assert_close(rollout(children[0], actions[0]), predictions[0], rtol=0, atol=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, context=context['observations'].numpy(), history_actions=context['actions'].numpy(),
             actions=actions, predictions=torch.stack(predictions).numpy(), seed=args.seed,
             checkpoint=str(Path(args.ckpt).resolve()), device=args.device, horizon=args.horizon,
             initialization_steps=initialization_steps,
             prediction_kind='predicted grayscale pixels' if latent else 'predicted normalized RAM')
    print(f'Saved predictions to {output}; snapshot replay passed.')


if __name__ == '__main__':
    main()
