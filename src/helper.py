from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .envs.atari import environment_spec, make_env, startup_action, ram_renderer
from .mlp_model import StateMLP
from .adapters import MLPAdapter, LatentAdapter
from .runtime import WorldRuntime


def _ram_frame(env, observation, args):
    actual = Image.fromarray(env.render())
    prediction = ram_renderer(args.environment)(observation[0].cpu().numpy()).resize(actual.size)
    width, height = actual.size
    frame = Image.new('RGB', (2 * width + 8, height + 28), (12, 18, 28))
    frame.paste(actual, (0, 28))
    frame.paste(prediction, (width + 8, 28))
    draw = ImageDraw.Draw(frame)
    draw.text((4, 3), 'ALE / real', fill='white')
    draw.text((width + 12, 3), 'Predicted RAM', fill='white')
    draw.text((width + 12, 15), args.mode, fill='white')
    return frame.resize((frame.width * 2, frame.height * 2), Image.Resampling.NEAREST)


def _latent_frame(real, observation, runtime, reward, step, args):
    pred = observation[0, 0]
    decoded = Image.fromarray((pred.clamp(0,1).cpu().numpy()*255).astype(np.uint8)).convert('RGB')
    actual = Image.fromarray(real[0].numpy()).convert('RGB')
    canvas = Image.new('RGB', (520, 296), (12, 18, 28))
    canvas.paste(actual.resize((256,256), Image.Resampling.NEAREST), (0,40))
    canvas.paste(decoded.resize((256,256), Image.Resampling.NEAREST), (264,40))
    draw = ImageDraw.Draw(canvas)
    draw.text((4,3), f'ALE / real   reward {reward:+.0f}', fill='white')
    draw.text((268,3), f'Latent decoder / {args.mode}', fill='white')
    draw.text((268,16), f'step {step+1}, sample seed {args.seed}', fill='white')
    outcomes = runtime.outcomes()
    if outcomes:
        draw.text((268,28), f"r={outcomes['reward'].item():+.2f}  P(end)={outcomes['termination_probability'].item():.2f}", fill='white')
    return canvas


@torch.no_grad()
def gif(args):
    from .latent_model import load_checkpoint, is_latent_checkpoint, LatentModel

    ckpt = load_checkpoint(args.ckpt)
    latent = is_latent_checkpoint(ckpt)
    args.environment = environment_spec(args, ckpt)
    if latent:
        model = LatentModel.load(ckpt, args.device).eval()
        adapter = LatentAdapter(model, torch.Generator().manual_seed(args.seed))
    else:
        adapter = MLPAdapter(StateMLP.load(args.ckpt, args.device))
    runtime = WorldRuntime(adapter)
    env = make_env(args.environment, latent, render_mode='rgb_array')
    rng = np.random.default_rng(args.seed)
    frames = []
    try:
        if env.action_space.n != adapter.model.num_actions:
            raise ValueError("Checkpoint action count does not match environment.")
        state, _ = env.reset(seed=args.seed)
        for _ in range(30):
            state, _, done, truncated, _ = env.step(startup_action(env))
            if done or truncated:
                raise ValueError('Episode ended during warm-up.')
        if latent:
            images, actions = [env.frame()], []
            for _ in range(args.history-1):
                action = int(rng.integers(env.action_space.n))
                _, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    raise ValueError('Episode ended during history initialization; use a shorter history.')
                images.append(env.frame())
                actions.append(action)
            context = dict(observations=torch.stack(images)[None].float()/255.,
                           actions=torch.tensor([actions]))
        else:
            context = {'observations': state[None, None]}
        runtime.reset(context)
        for step in range(200):
            action = int(rng.integers(env.action_space.n))
            state, reward, done, truncated, _ = env.step(action)
            observation = runtime.step([action])
            if latent:
                real = env.frame()
                frames.append(_latent_frame(real, observation, runtime, reward, step, args))
            else:
                frames.append(_ram_frame(env, observation, args))
            if done or truncated:
                break
            if args.mode == 'one-step':
                if latent:
                    runtime.correct(real[None].float()/255.)
                else:
                    runtime.reset({'observations': state[None, None]})
    finally:
        env.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=70, loop=0)
    print(f'Saved {len(frames)} frames to {args.output}.')
