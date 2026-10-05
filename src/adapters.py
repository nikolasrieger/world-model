from typing import Protocol
import torch


def check_action(action, batch_size):
    if action.shape != (batch_size,) or ((action < 0) | (action >= 6) | (action != action.long())).any():
        raise ValueError('Expected one ALE action (0..5) per batch element.')


def check_history(obs):
    if obs.ndim != 3 or obs.shape[1] < 1 or obs.shape[2] != 128:
        raise ValueError('Expected RAM history [batch, time, 128].')


def map_state(state, fn, strict=False):
    if isinstance(state, torch.Tensor):
        return fn(state)
    if isinstance(state, dict):
        if strict and any(type(k) not in (type(None), bool, int, float, complex, str, bytes) for k in state):
            raise TypeError('State dictionary keys must be immutable scalars.')
        return {k: map_state(v, fn, strict) for k, v in state.items()}
    if isinstance(state, (tuple, list)):
        return type(state)(map_state(v, fn, strict) for v in state)
    if strict and not isinstance(state, (type(None), bool, int, float, complex, str, bytes)):
        raise TypeError(f'Unsupported state leaf: {type(state).__name__}')
    return state


def clone_state(state):
    return map_state(state, lambda x: x.detach().clone())


def move_state(state, device):
    return map_state(state, lambda x: x.to(device))


class WorldModelAdapter(Protocol):
    capabilities: dict
    @property
    def device(self): ...
    def initialize(self, context, noise=None): ...
    def step(self, state, action, noise=None): ...
    def observe(self, state): ...
    def sample(self, state, action, noise=None): ...


class MLPAdapter:
    capabilities = dict(deterministic=True, recurrent=False, gradients=True, decoder=False, heads=False)

    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    def initialize(self, context, noise=None):
        obs = torch.as_tensor(context['observations'], device=self.device, dtype=torch.float32)
        check_history(obs)
        return {'ram': obs[:, -1].clone()}

    def step(self, state, action, noise=None):
        action = torch.as_tensor(action, device=self.device)
        check_action(action, len(state['ram']))
        return {'ram': self.model(state['ram'], action).clamp(0, 1)}

    def observe(self, state):
        return state['ram'].clone()

    sample = step


class LatentAdapter:
    def __init__(self, model, generator=None):
        self.model, self.generator = model, generator
        self.capabilities = dict(deterministic=False, explicit_rng=True, recurrent=True, gradients=True, decoder=model.decoder is not None, heads=model.heads is not None)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def initialize(self, context, noise=None, generator=None):
        return self.model.initialize(context, noise=noise, generator=generator if generator is not None else self.generator)

    def step(self, state, action, noise=None, generator=None):
        return self.model.prior(state, action, noise=noise, generator=generator if generator is not None else self.generator)

    def correct(self, state, observation, generator=None):
        return self.model.posterior(state, observation, generator=generator if generator is not None else self.generator)

    def observe(self, state):
        return self.model.observe(state)

    def outcomes(self, state, reward_fn=None):
        native = self.model.outcomes(state)
        if reward_fn is not None:
            native['reward'] = reward_fn(state)
        return native

    sample = step
