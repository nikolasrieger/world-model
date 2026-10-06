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
from src.envs.atari import environment_spec, make_env, startup_action, readiness_check


def collect_context(env, *, seed, history, latent, ready, warmup=30, max_steps=300):
    env.reset(seed=seed)
    initial_action = startup_action(env)
    images = deque(maxlen=history)
    for step in range(1, max_steps + 1):
        observation, _, done, truncated, _ = env.step(initial_action)
        if done or truncated:
            raise RuntimeError('Episode ended before initialization context was collected.')
        if step < warmup or not ready(env):
            images.clear()
            continue
        images.append(env.frame().float()/255 if latent else torch.as_tensor(observation).clone())
        if len(images) == history:
            return dict(observations=torch.stack(list(images))[None],
                        actions=torch.full((1, history - 1), initial_action, dtype=torch.long)), step
    raise RuntimeError(f'No suitable context with {history} context frames within {max_steps} steps.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--env-id', default=None)
    parser.add_argument('--actions', nargs=2, type=int, default=None, help='Repeated actions for the two branches.')
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
    spec = environment_spec(args, checkpoint)
    if latent:
        adapter = LatentAdapter(LatentModel.load(checkpoint, args.device).eval(),
                                torch.Generator().manual_seed(args.seed))
    else:
        adapter = MLPAdapter(StateMLP.load(args.ckpt, args.device))
    env = make_env(spec, latent)
    try:
        if env.action_space.n != adapter.model.num_actions:
            raise ValueError("Checkpoint action count does not match environment.")
        context, initialization_steps = collect_context(
            env, seed=args.seed, history=args.history, latent=latent, ready=readiness_check(spec))
        print(f'Collected {args.history} initialization frames after {initialization_steps} simulator steps.')
        runtime = WorldRuntime(adapter)
        runtime.reset(context)
    finally:
        env.close()
    saved = runtime.snapshot()
    branch_actions = args.actions or [0, adapter.model.num_actions - 1]
    if any(a < 0 or a >= adapter.model.num_actions for a in branch_actions):
        parser.error('Branch actions must be in the model action space.')
    actions = np.stack([np.full(args.horizon, a, dtype=np.int64) for a in branch_actions])
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
             initialization_steps=initialization_steps, environment_id=spec["id"],
             prediction_kind='predicted grayscale pixels' if latent else 'predicted normalized RAM')
    print(f'Saved predictions to {output}; snapshot replay passed.')


if __name__ == '__main__':
    main()
