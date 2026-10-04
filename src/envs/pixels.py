"""Atari image preprocessing shared by training and visualization."""
import gymnasium as gym
import numpy as np
import torch
from .pong import ENV_ID

PIXEL_CONFIG = dict(preprocessing='atari-maxpool-v1', frameskip=4, size=64,
                    grayscale=True, noop_max=30, repeat_action_probability=.25,
                    full_action_space=False, terminal_on_life_loss=False)


class PixelPongEnv:
    def __init__(self):
        base = gym.make(ENV_ID, obs_type='rgb', frameskip=1,
                        repeat_action_probability=.25, full_action_space=False)
        self.env = gym.wrappers.AtariPreprocessing(
            base, noop_max=30, frame_skip=4, screen_size=64,
            terminal_on_life_loss=False, grayscale_obs=True)
        self.image = None

    def reset(self, seed=None):
        self.image, info = self.env.reset(seed=seed)
        return self.image, info

    def step(self, action):
        self.image, reward, done, truncated, info = self.env.step(action)
        return self.image, reward, done, truncated, info

    def frame(self):
        return torch.from_numpy(np.array(self.image, copy=True))[None]

    def close(self):
        self.env.close()
