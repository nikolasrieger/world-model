from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .envs import PongEnv
from .envs.pong import ram_image
from .mlp_model import StateMLP
from .adapters import MLPAdapter


@torch.no_grad()
def gif(args):
    from .latent_model import load_checkpoint, is_latent_checkpoint
    ckpt = load_checkpoint(args.ckpt)
    if is_latent_checkpoint(ckpt):
        return latent_gif(args, ckpt)
    adapter = MLPAdapter(StateMLP.load(args.ckpt, args.device))
    env, rng = PongEnv(render_mode="rgb_array"), np.random.default_rng(args.seed)
    frames = []
    try:
        state, _ = env.reset(seed=args.seed)
        for _ in range(30): 
            state, _, _, _, _ = env.step(1)
        predicted = adapter.initialize({'observations': state[None, None]})
        for _ in range(200):
            action = int(rng.integers(6))
            state, _, done, truncated, _ = env.step(action)
            predicted = adapter.step(predicted, [action])
            observation = adapter.observe(predicted)
            frame = Image.new("RGB", (328, 238), (12, 18, 28))
            frame.paste(Image.fromarray(env.render()), (0, 28))
            frame.paste(ram_image(observation[0].cpu().numpy()), (168, 28))
            draw = ImageDraw.Draw(frame)
            draw.text((4, 3), "ALE / real", fill="white")
            draw.text((172, 3), "Predicted RAM", fill="white")
            draw.text((172, 15), args.mode, fill="white")
            frames.append(frame.resize((656, 476), Image.Resampling.NEAREST))
            if done or truncated:
                break
            if args.mode == 'one-step':
                predicted = adapter.initialize({'observations': state[None, None]})
    finally:
        env.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=70, loop=0)
            

@torch.no_grad()
def latent_gif(args, ckpt):
    from .adapters import LatentAdapter
    from .latent_model import LatentModel
    from .training import frame
    from .envs.pixels import PixelPongEnv

    model = LatentModel.load(ckpt, args.device).eval()
    adapter = LatentAdapter(model, torch.Generator().manual_seed(args.seed))
    env, rng = PixelPongEnv(), np.random.default_rng(args.seed)
    frames = []
    try:
        env.reset(seed=args.seed)
        for _ in range(30):
            env.step(1)
        images, actions = [frame(env)], []
        for _ in range(args.history-1):
            action = int(rng.integers(6))
            _, _, done, truncated, _ = env.step(action)
            if done or truncated:
                raise ValueError('Episode ended during history initialization; use a shorter --history.')
            images.append(frame(env))
            actions.append(action)
        context = dict(observations=torch.stack(images)[None].float()/255., actions=torch.tensor([actions]))
        state = adapter.initialize(context)
        for step in range(200):
            action = int(rng.integers(6))
            _, reward, done, truncated, _ = env.step(action)
            state = adapter.step(state, torch.tensor([action], device=args.device))
            real = frame(env)
            pred = adapter.observe(state)[0, 0]
            decoded = Image.fromarray((pred.clamp(0,1).cpu().numpy()*255).astype(np.uint8)).convert('RGB')
            actual = Image.fromarray(real[0].numpy()).convert('RGB')
            canvas = Image.new('RGB', (520, 296), (12, 18, 28))
            canvas.paste(actual.resize((256,256), Image.Resampling.NEAREST), (0,40))
            canvas.paste(decoded.resize((256,256), Image.Resampling.NEAREST), (264,40))
            draw = ImageDraw.Draw(canvas)
            draw.text((4,3), f'ALE / real   reward {reward:+.0f}', fill='white')
            draw.text((268,3), f'Latent decoder / {args.mode}', fill='white')
            draw.text((268,16), f'step {step+1}, sample seed {args.seed}', fill='white')
            outcomes = adapter.outcomes(state)
            if outcomes:
                draw.text((268,28), f"r={outcomes['reward'].item():+.2f}  P(end)={outcomes['termination_probability'].item():.2f}", fill='white')
            frames.append(canvas)
            if done or truncated:
                break
            if args.mode == 'one-step':
                state = model.posterior(state, real[None].float().to(args.device)/255., generator=adapter.generator)
    finally:
        env.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=70, loop=0)
    print(f'Saved {len(frames)} decoded frames to {args.output}.')
