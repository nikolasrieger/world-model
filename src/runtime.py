from dataclasses import dataclass
import torch
from .adapters import map_state


def _copy(value):
    return map_state(value, lambda tensor: tensor.detach().clone(), strict=True)


@dataclass(frozen=True, eq=False)
class RuntimeSnapshot:
    """Opaque owned state."""
    _family: object
    _state: object
    _random_state: object


class WorldRuntime:
    def __init__(self, adapter):
        caps = getattr(adapter, 'capabilities', {})
        self._adapter = adapter
        self._family = object()
        self._state = None
        self._initialized = False
        self._random_state = None
        if caps.get('explicit_rng'):
            generator = getattr(adapter, 'generator', None)
            if generator is not None and generator.device.type != 'cpu':
                raise ValueError('Latent sampling requires a CPU generator.')
            self._random_state = (generator or torch.Generator().manual_seed(torch.initial_seed())).get_state().clone()

    def _require_state(self):
        assert self._initialized, 'Runtime must be initialized first.'

    @torch.no_grad()
    def _update(self, method, *args):
        kwargs = {}
        if self._random_state is not None:
            generator = torch.Generator().set_state(self._random_state)
            kwargs['generator'] = generator
        state = _copy(method(*args, **kwargs))
        observation = _copy(self._adapter.observe(state))
        self._state = state
        self._initialized = True
        if kwargs:
            self._random_state = generator.get_state().clone()
        return observation

    def reset(self, context):
        return self._update(self._adapter.initialize, context)

    def step(self, action):
        self._require_state()
        return self._update(self._adapter.step, self._state, action)

    def correct(self, observation):
        self._require_state()
        method = getattr(self._adapter, 'correct', None)
        return self._update(method, self._state, observation)

    @torch.no_grad()
    def observe(self):
        self._require_state()
        return _copy(self._adapter.observe(self._state))

    @torch.no_grad()
    def outcomes(self):
        self._require_state()
        method = getattr(self._adapter, 'outcomes', None)
        return {} if method is None else _copy(method(self._state))

    def snapshot(self):
        self._require_state()
        return RuntimeSnapshot(self._family, _copy(self._state), _copy(self._random_state))

    @torch.no_grad()
    def restore(self, snapshot):
        assert isinstance(snapshot, RuntimeSnapshot), "Expected RuntimeSnapshot."
        state = _copy(snapshot._state)
        observation = _copy(self._adapter.observe(state))
        self._state = state
        self._initialized = True
        self._random_state = _copy(snapshot._random_state)
        return observation

    def branch(self):
        saved = self.snapshot()
        child = WorldRuntime(self._adapter)
        child._family = self._family
        child.restore(saved)
        return child
