import torch
from torch import nn
from torch.nn import functional as F
from .adapters import check_action


def noise_sequence(count, size, latent, device, generator=None):
    uniform = torch.rand(count, size, latent, generator=generator, device='cpu')
    return -torch.log(-torch.log(uniform.clamp(1e-6, 1-1e-6))).to(device)


class RSSMCell(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(2 * hidden, 3 * hidden)
        self.norm = nn.LayerNorm(3 * hidden, eps=1e-3)

    def forward(self, x, h):
        reset, candidate, update = self.norm(self.linear(torch.cat((x, h), -1))).chunk(3, -1)
        candidate = (reset.sigmoid() * candidate).tanh()
        update = (update - 1).sigmoid()
        return update * candidate + (1-update) * h


class ImageDecoder(nn.Module):
    def __init__(self, features, depth):
        super().__init__()
        self.input = nn.Linear(features, 32 * depth)
        self.net = nn.Sequential(
            nn.Unflatten(1, (32 * depth, 1, 1)),
            nn.ConvTranspose2d(32 * depth, 4 * depth, 5, 2), nn.ELU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, 2), nn.ELU(),
            nn.ConvTranspose2d(2 * depth, depth, 6, 2), nn.ELU(),
            nn.ConvTranspose2d(depth, 1, 6, 2))

    def forward(self, features):
        # Unit-variance Gaussian mean; no sigmoid saturation.
        return self.net(self.input(features)) + .5


def head(features, units):
    layers = []
    for _ in range(4):
        layers.extend((nn.Linear(features, units), nn.ELU()))
        features = units
    return nn.Sequential(*layers, nn.Linear(units, 1))


class OutcomeHeads(nn.Module):
    def __init__(self, features, units):
        super().__init__()
        self.reward = head(features, units)
        self.terminal = head(features, units)

    def forward(self, features):
        return torch.cat((self.reward(features), self.terminal(features)), -1)


class LatentModel(nn.Module):
    def __init__(self, hidden=600, stoch=32, classes=32, depth=48,
                 head_units=400):
        super().__init__()
        if min(hidden, stoch, depth, head_units) < 1 or classes < 2:
            raise ValueError('Model dimensions must be positive; classes must be at least 2.')
        self.hidden, self.stoch, self.classes = hidden, stoch, classes
        self.latent = stoch * classes
        self.config = dict(hidden=hidden, stoch=stoch, classes=classes, depth=depth,
                           head_units=head_units)
        layers, channels = [], 1
        for width in (depth, 2*depth, 4*depth, 8*depth):
            layers.extend((nn.Conv2d(channels, width, 4, 2), nn.ELU()))
            channels = width
        self.encoder = nn.Sequential(*layers, nn.Flatten())
        self.input_net = nn.Sequential(nn.Linear(self.latent + 6, hidden), nn.ELU())
        self.gru = RSSMCell(hidden)
        self.prior_net = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, self.latent))
        self.posterior_net = nn.Sequential(nn.Linear(hidden + 32*depth, hidden), nn.ELU(), nn.Linear(hidden, self.latent))
        self.decoder = ImageDecoder(hidden + self.latent, depth)
        self.heads = OutcomeHeads(hidden + self.latent, head_units)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(layer):
        if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def initial_hidden(self, batch, device=None):
        p = next(self.parameters())
        return torch.zeros(batch, self.hidden, device=p.device if device is None else device, dtype=p.dtype)

    def state(self, h, logits, noise=None, generator=None):
        logits = logits.reshape(-1, self.stoch, self.classes)
        probs = logits.softmax(-1)
        if noise is None:
            noise = noise_sequence(1, len(h), self.latent, h.device, generator)[0]
        if noise.shape != (len(h), self.latent):
            raise ValueError('Expected Gumbel noise [batch, stoch * classes].')
        indices = (logits + noise.to(logits).reshape_as(logits)).argmax(-1)
        sample = F.one_hot(indices, self.classes).to(probs.dtype)
        z = sample + (probs - probs.detach())  # straight-through categorical sample
        return dict(h=h, z=z.flatten(1), distr=dict(logits=logits))

    def transition(self, state, action, *, validate=True):
        action = torch.as_tensor(action, device=state['h'].device)
        if validate:
            check_action(action, len(state['h']))
        controls = F.one_hot(action.long(), 6).to(state['z'].dtype)
        return self.gru(self.input_net(torch.cat((state['z'], controls), -1)), state['h'])

    def prior(self, state, action, noise=None, generator=None, *, validate=True):
        h = self.transition(state, action, validate=validate)
        return self.state(h, self.prior_net(h), noise, generator)

    def posterior(self, h, image, noise=None, generator=None):
        return self.posterior_embed(h, self.encoder(image - .5), noise, generator)

    def posterior_embed(self, h, embed, noise=None, generator=None):
        return self.state(h, self.posterior_net(torch.cat((h, embed), -1)), noise, generator)

    def start_hidden(self, batch):
        h = self.initial_hidden(batch)
        return self.gru(self.input_net(h.new_zeros(batch, self.latent + 6)), h)

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
            raise ValueError('Expected Gumbel noise [batch, time, stoch * classes].')
        embeds = self.encoder(images.flatten(0, 1) - .5).reshape(batch, time, -1)
        state = self.posterior_embed(self.start_hidden(batch), embeds[:, 0], noise[:, 0])
        for t in range(1, time):
            h = self.transition(state, actions[:, t-1], validate=False)
            state = self.posterior_embed(h, embeds[:, t], noise[:, t])
        return state

    def features(self, state):
        return torch.cat((state['h'], state['z']), -1)

    def observe(self, state):
        return self.decoder(self.features(state))

    def outcomes(self, state):
        values = self.heads(self.features(state))
        return dict(reward=values[:, 0], termination_probability=values[:, 1].sigmoid())

    @classmethod
    def load(cls, ckpt, device='cpu'):
        config = dict(ckpt['model_config'])
        model = cls(**config).to(device)
        model.load_state_dict(ckpt['weights'])
        return model
