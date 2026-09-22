import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .envs import PongEnv
from .envs.pong import ram_image
from .model import StateMLP


@torch.no_grad()
def gif(args):
    model = StateMLP.load(args.checkpoint, args.device)
    env, rng = PongEnv(render_mode="rgb_array"), np.random.default_rng(args.seed)
    frames = []
    try:
        state, _ = env.reset(seed=args.seed)
        for _ in range(30): 
            state, _, _, _, _ = env.step(1)
        predicted = torch.as_tensor(state[None], device=args.device)
        for step in range(args.steps):
            action = int(rng.integers(6))
            state, _, done, truncated, _ = env.step(action)
            predicted = model(predicted, torch.tensor([action], device=args.device)).clamp(0, 1)
            if not torch.isfinite(predicted).all():
                raise RuntimeError("Model produced nonfinite RAM predictions.")
            frame = Image.new("RGB", (328, 238), (12, 18, 28))
            frame.paste(Image.fromarray(env.render()), (0, 28))
            frame.paste(ram_image(predicted[0].cpu().numpy()), (168, 28))
            draw = ImageDraw.Draw(frame)
            draw.text((4, 3), "ALE / real", fill="white")
            draw.text((172, 3), "Predicted RAM", fill="white")
            draw.text((172, 15), fill=(150, 170, 190))
            frames.append(frame.resize((656, 476), Image.Resampling.NEAREST))
            if done or truncated:
                break
    finally:
        env.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=70, loop=0)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive.")
    return number