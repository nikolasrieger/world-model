"""Compact recurrent Gaussian latent model, not a Dreamer reproduction."""
import torch
from torch import nn
from torch.nn import functional as F
from .adapters import check_action


def normal_noise(shape, device, generator=None):
    return torch.randn(shape, generator=generator, device='cpu').to(device)


def noise_sequence(count, size, latent, device, generator=None):
    # Keep the same CPU RNG draws while transferring the whole sequence once.
    return torch.stack([torch.randn(size, latent, generator=generator)
                        for _ in range(count)]).to(device)


class SpatialDecoder(nn.Module):
    def __init__(self, hidden, channels):
        super().__init__()
        self.hidden, self.channels = hidden, channels
        self.net = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, features):
        z = features[:, self.hidden:].reshape(-1, self.channels, 32, 32)
        return self.net(z)


class LatentModel(nn.Module):
    def __init__(self, hidden=128, latent=None, arch='spatial'):
        super().__init__()
        if arch not in ('dense', 'spatial'):
            raise ValueError('Unknown latent arch.')
        latent = latent if latent is not None else (2048 if arch == 'spatial' else 32)
        self.arch = arch
        self.config = dict(hidden=hidden, latent=latent, arch=arch)
        self.hidden, self.latent = hidden, latent
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(), nn.Flatten(), nn.Linear(1024, 128), nn.SiLU())
        self.gru = nn.GRUCell(latent + 6, hidden)
        self.prior_net = nn.Linear(hidden, 2 * latent)
        self.posterior_net = nn.Sequential(nn.Linear(hidden + 128, 128), nn.SiLU(), nn.Linear(128, 2 * latent))
        features = hidden + latent
        self.decoder = nn.Sequential(nn.Linear(features, 1024), nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid())
        self.heads = nn.Sequential(nn.Linear(features, 64), nn.SiLU(), nn.Linear(64, 4))
        if arch == 'spatial':
            if latent % 1024:
                raise ValueError('Spatial latent size must be divisible by 1024 (32x32 grid).')
            channels = latent // 1024
            self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.SiLU(),
                nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
            self.posterior_net = nn.Conv2d(32, 2*channels, 1)
            self.posterior_context = nn.Linear(hidden, 2*latent)
            self.decoder = SpatialDecoder(hidden, channels)

    def state(self, h, stats, noise=None, generator=None):
        mean, raw = stats.chunk(2, -1)
        std = F.softplus(raw) + .1
        noise = normal_noise(mean.shape, mean.device, generator) if noise is None else noise.to(device=mean.device, dtype=mean.dtype)
        return {'h': h, 'z': mean + std * noise, 'distr': {'mean': mean, 'std': std}}

    def prior(self, state, action, noise=None, generator=None, *, validate=True):
        action = torch.as_tensor(action, device=state['h'].device)
        if validate:
            check_action(action, len(state['h']))
        h = self.gru(torch.cat((state['z'], F.one_hot(action.long(), 6)), -1), state['h'])
        return self.state(h, self.prior_net(h), noise, generator)

    def posterior(self, h, image, noise=None, generator=None):
        encoded = self.encoder(image)
        if self.arch == 'spatial':
            stats = self.posterior_net(encoded).flatten(1) + self.posterior_context(h)
        else:
            stats = self.posterior_net(torch.cat((h, encoded), -1))
        return self.state(h, stats, noise, generator)

    def initialize(self, context, noise=None, generator=None):
        device = next(self.parameters()).device
        images = torch.as_tensor(context['observations'], device=device, dtype=torch.float32)
        if images.ndim != 5 or images.shape[2:] != (1, 64, 64) or images.shape[1] < 1:
            raise ValueError('Expected pixel history [batch, time, 1, 64, 64].')
        batch, time = images.shape[:2]
        actions = torch.as_tensor(context.get('actions', torch.empty(batch, 0)), device=device)
        if actions.shape != (batch, time - 1):
            raise ValueError('Expected history actions [batch, time - 1].')
        check_action(actions.reshape(-1), batch * (time - 1))
        if noise is None:
            noise = noise_sequence(time, batch, self.latent, device, generator).transpose(0, 1)
        elif noise.shape != (batch, time, self.latent):
            raise ValueError('Expected noise [batch, time, latent].')
        state = self.posterior(torch.zeros(batch, self.hidden, device=device), images[:, 0], None if noise is None else noise[:, 0], generator)
        for t in range(1, time):
            prior = self.prior(state, actions[:, t-1], noise=torch.zeros_like(state['z']), validate=False)
            state = self.posterior(prior['h'], images[:, t], None if noise is None else noise[:, t], generator)
        return state

    def features(self, state):
        return torch.cat((state['h'], state['z']), -1)

    def observe(self, state):
        feats = self.features(state)
        return self.decoder(feats)

    def outcomes(self, state):
        logits = self.heads(self.features(state))
        return {'reward': (logits[:, :3].softmax(-1) * logits.new_tensor([-1, 0, 1])).sum(-1), 'termination_probability': logits[:, 3].sigmoid()}

    @classmethod
    def load(cls, ckpt, device='cpu'):
        config = dict(ckpt['model_config'])
        model = cls(**config).to(device)
        model.load_state_dict(ckpt['weights'])
        return model
