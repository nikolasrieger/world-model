import ale_py
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

ENV_ID = "ALE/Pong-v5"
DEFAULT_HORIZON = 10000
CONFIG = dict(obs_type="ram", frameskip=4, repeat_action_probability=0.0, full_action_space=False)
gym.register_envs(ale_py)


class PongEnv(gym.ObservationWrapper):
    def __init__(self, render_mode=None, max_steps=DEFAULT_HORIZON):
        super().__init__(gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=max_steps, **CONFIG))
        self.observation_space = gym.spaces.Box(0., 1., (128,), np.float32)

    def observation(self, observation):
        return observation.astype(np.float32) / 255.


def ram_image(state):
    ram = np.rint(np.clip(state, 0, 1) * 255).astype(int)
    image = Image.new("RGB", (160, 210), (12, 18, 28))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 34, 159, 193), outline=(50, 60, 75))
    for x, y, color in [(16, ram[50] - 15, (213, 130, 74)), (140, ram[51] - 13, (92, 186, 92))]:
        top, bottom = max(34, y), min(193, y + 14)
        if bottom >= top:
            draw.rectangle((x, top, x + 3, bottom), fill=color)
    if ram[54] and ram[49] > 49:
        x, y = ram[49] - 49, ram[54] - 14
        draw.rectangle((x, y, x + 1, y + 3), fill="white")
    return image
