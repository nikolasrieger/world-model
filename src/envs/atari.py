import re
import ale_py
import gymnasium as gym
import numpy as np
import torch

gym.register_envs(ale_py)
RAM_CONFIG = dict(obs_type='ram', frameskip=4, repeat_action_probability=0., full_action_space=False)
PIXEL_CONFIG = dict(preprocessing='atari-maxpool-v1', frameskip=4, size=64,
                    grayscale=True, noop_max=30, repeat_action_probability=.25,
                    full_action_space=False, terminal_on_life_loss=False)


def environment_spec(args=None, checkpoint=None):
    spec = dict(id='ALE/Pong-v5', image_size=64, full_action_space=False)
    spec.update((checkpoint or {}).get('environment', {}))
    for key, arg in [('id', 'env_id'), ('image_size', 'image_size'), ('full_action_space', 'full_action_space')]:
        value = getattr(args, arg, None)
        if value is not None:
            if checkpoint and value != spec[key]:
                raise ValueError(f'{arg} conflicts with checkpoint environment metadata.')
            spec[key] = value
    size = spec['image_size']
    if not isinstance(size, int) or size < 16 or size & (size - 1):
        raise ValueError('image_size must be a power of two >= 16.')
    if not spec['id'].startswith('ALE/'):
        raise ValueError('The built-in collectors support ALE environments.')
    return spec


def default_data_path(spec, latent):
    slug = re.sub(r'[^a-z0-9]+', '-', spec['id'].split('/')[-1].split('-v')[0].lower())
    suffix = 'atari-pixels' if latent else 'episodes'
    if spec['image_size'] != 64 or spec['full_action_space']:
        slug += f'-{spec["image_size"]}-full{int(spec["full_action_space"])}'
    return f'artifacts/{slug}-{suffix}.pt'


class AtariRAMEnv(gym.ObservationWrapper):
    def __init__(self, spec, render_mode=None, max_steps=10000):
        super().__init__(gym.make(spec['id'], obs_type='ram', frameskip=4,
                                  repeat_action_probability=0., full_action_space=spec['full_action_space'],
                                  render_mode=render_mode, max_episode_steps=max_steps))
        self.observation_space = gym.spaces.Box(0., 1., self.observation_space.shape, np.float32)

    def observation(self, observation):
        return observation.astype(np.float32) / 255.


class AtariPixelEnv:
    def __init__(self, spec):
        base = gym.make(spec['id'], obs_type='rgb', frameskip=1,
                        repeat_action_probability=.25, full_action_space=spec['full_action_space'])
        self.env = gym.wrappers.AtariPreprocessing(base, noop_max=30, frame_skip=4,
                    screen_size=spec['image_size'], terminal_on_life_loss=False, grayscale_obs=True)
        self.action_space = self.env.action_space
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


def make_env(spec, latent=False, render_mode=None):
    return AtariPixelEnv(spec) if latent else AtariRAMEnv(spec, render_mode)


def startup_action(env):
    base = env.env if isinstance(env, AtariPixelEnv) else env
    meanings = base.unwrapped.get_action_meanings()
    return meanings.index('FIRE') if 'FIRE' in meanings else meanings.index('NOOP')


def validate_metadata(data, spec):
    actual = data.get('environment', dict(id='ALE/Pong-v5', image_size=64, full_action_space=False))
    assert actual == spec, 'Dataset environment/preprocessing differs from requested environment.'


def object_metrics(spec):
    if spec['id'] == 'ALE/Pong-v5':
        from .pong import ram_metrics
        return ram_metrics
    return None


def readiness_check(spec):
    if spec['id'] == 'ALE/Pong-v5':
        from .pong import RallyReadiness
        return RallyReadiness()
    return lambda env: True


def ram_renderer(spec):
    if spec['id'] == 'ALE/Breakout-v5':
        from .breakout import ram_image
        return ram_image
    if spec['id'] == 'ALE/Pong-v5':
        from .pong import ram_image
        return ram_image
    from PIL import Image
    def heatmap(state):
        values = np.rint(np.clip(state, 0, 1) * 255).astype(np.uint8)
        width = 16
        padded = np.pad(values, (0, (-len(values)) % width))
        return Image.fromarray(padded.reshape(-1, width)).convert('RGB')
    return heatmap
